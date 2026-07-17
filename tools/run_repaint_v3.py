#!/root/miniconda3/envs/musicgen/bin/python
"""
Repaint Diagnostic v3: 窗口局部 PER 评估。
只裁出重绘中央窗口 → 局部 ASR → 局部歌词 PER。

用法:
  python tools/run_repaint_v3.py verify   # 1歌1窗先验证
  python tools/run_repaint_v3.py generate # 全部生成
  python tools/run_repaint_v3.py evaluate # 窗口级 PER 评估
  python tools/run_repaint_v3.py all
"""
import argparse, json, os, pickle, subprocess, sys, time, re, shutil
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path("/root/ACE-Step-1.5")
EVAL_ROOT = Path("/root/autodl-tmp/tsm_eval_results") / "repaint_v3"
TEST_JSONL = PROJECT_ROOT / "Muse/infer/test.jsonl"
GEN_SCRIPT = PROJECT_ROOT / "gen_one.py"
PIPELINE_DIR = PROJECT_ROOT / "Muse/eval_pipeline"
QWEN_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-ASR-1.7B")
GT_LYRICS_DIR = PIPELINE_DIR / "gt_lyrics"
PYTHON = sys.executable
SEED = 42

WINDOW_SEC = 50
CROP_MARGIN_SEC = 3  # crop margin for crossfade transition removal

GROUPS = {
    "oneshot": {"task_type": "text2music", "mode": None, "strength": None},
    "repaint_conservative_01": {"task_type": "repaint", "mode": "conservative", "strength": 0.1},
    "repaint_conservative_025": {"task_type": "repaint", "mode": "conservative", "strength": 0.25},
    "repaint_balanced_05": {"task_type": "repaint", "mode": "balanced", "strength": 0.5},
}


def parse_entry(entry_idx):
    with open(TEST_JSONL) as f:
        for i, line in enumerate(f):
            if i == entry_idx: data = json.loads(line); break
    msgs = data["messages"]
    style = msgs[0]["content"].split("\n")[0].replace(
        "Please generate a song in the following style:", "").strip()
    seen = set(); parts = []
    for msg in msgs:
        c = msg.get("content", "")
        m = re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m: continue
        name = m.group(1)
        if name in seen: continue
        seen.add(name)
        rest = c[m.end():]; lyr = ""
        if "[lyrics:" in rest:
            lyr = rest.split("[lyrics:")[1].split("]")[0].strip()
        parts.append(m.group(0))
        if lyr: parts.append(lyr)
        parts.append("")
    return style, "\n".join(parts).strip()


def get_duration(entry_idx):
    with open(TEST_JSONL) as f:
        for i, line in enumerate(f):
            if i == entry_idx: data = json.loads(line); break
    msgs = data["messages"]; seen = set(); text = ""
    for msg in msgs:
        c = msg.get("content", "")
        m = re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m: continue
        name = m.group(1)
        if name in seen: continue
        seen.add(name)
        rest = c[m.end():]
        if "[lyrics:" in rest:
            lyr = rest.split("[lyrics:")[1].split("]")[0].strip()
            text += lyr
    return max(20, int(len(re.sub(r'\s', '', text)) * 0.45))


def get_section_boundaries(lyrics, total_dur):
    lines = lyrics.strip().split("\n")
    sections = []
    current_tag = "UNKNOWN"
    current_text = []
    for line in lines:
        m = re.match(r'\[([^\]]+)\]', line.strip())
        if m:
            if current_text and any(t.strip() for t in current_text):
                sections.append((current_tag, "\n".join(current_text).strip()))
            current_tag = m.group(1)
            current_text = []
        elif line.strip():
            current_text.append(line.strip())
    if current_text and any(t.strip() for t in current_text):
        sections.append((current_tag, "\n".join(current_text).strip()))
    if not sections: return []
    SILENCE = {'INTRO','OUTRO','INTERLUDE','INSTRUMENTAL'}
    weights = []
    for tag, text in sections:
        w = len(text) * (0.3 if tag.upper() in SILENCE else 1.0)
        weights.append(max(w, 1))
    total_w = sum(weights)
    boundaries = []; cum = 0.0
    for i, (tag, text) in enumerate(sections):
        dur = total_dur * weights[i] / total_w
        boundaries.append((tag, text, cum, cum+dur)); cum += dur
    return boundaries


def build_windows(total_dur):
    last_half = total_dur * 0.5
    windows = []; pos = last_half
    while pos < total_dur:
        end = min(pos + WINDOW_SEC, total_dur)
        windows.append((pos, end))
        pos = end
    return windows


def get_window_lyrics(sections, win_start, win_end):
    """Get only the lyrics that fall within a window."""
    result_lines = []
    for tag, text, s, e in sections:
        overlap = max(0, min(e, win_end) - max(s, win_start))
        if overlap > 1.0:
            result_lines.append(f"[{tag}]")
            result_lines.append(text)
    return "\n".join(result_lines).strip()


def gen_one(method, ckpt, caption, lyrics, out_dir, seed, duration, fidx, **extra):
    os.makedirs(out_dir, exist_ok=True)
    d = dict(method=method, ckpt=str(ckpt) if ckpt else "", caption=caption,
             lyrics=lyrics, out_dir=str(out_dir), seed=seed, duration=duration)
    d.update(extra)
    r = subprocess.run([PYTHON, str(GEN_SCRIPT)], input=pickle.dumps(d),
                       capture_output=True, timeout=600)
    if r.returncode != 0:
        return None, r.stderr.decode().strip()[-300:]
    for line in r.stdout.decode().split("\n"):
        if line.startswith("OK:"):
            gp = line[3:].strip()
            if gp and os.path.exists(gp):
                return gp, None
            return gp, None
    return None, "no OK:"


def crop_audio(wav_path, start_sec, end_sec, crop_margin=CROP_MARGIN_SEC):
    """Crop a window from audio, removing margin to avoid crossfade."""
    data, sr = sf.read(wav_path)
    s = int((start_sec + crop_margin) * sr)
    e = int((end_sec - crop_margin) * sr)
    s = max(0, s); e = min(len(data), e)
    if e <= s:
        s = int(start_sec * sr); e = int(end_sec * sr)
        s = max(0, s); e = min(len(data), e)
    return data[s:e], sr


def strip_section_tags(text):
    """Remove [SectionName] tags from lyrics for PER evaluation."""
    return re.sub(r'\[([^\]]+)\]\n*', '', text).strip()


def eval_window_per(gt_text, audio_segment, sr, out_dir, name):
    """Run ASR on a cropped audio segment and compute PER against gt_text."""
    # Strip section tags from GT for fair comparison
    clean_gt = strip_section_tags(gt_text)
    if not clean_gt:
        return {"per": None, "error": "empty_gt_after_strip"}

    # Save temp wav
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{name}.wav")
    sf.write(wav_path, audio_segment, sr)

    # Save gt lyrics as jsonl
    gt_lines = [json.dumps({"file_index": name, "text": clean_gt}, ensure_ascii=False)]
    gt_path = os.path.join(out_dir, f"gt_{name}.jsonl")
    with open(gt_path, "w") as f:
        f.write("\n".join(gt_lines))

    # Run ASR
    trans_path = os.path.join(out_dir, f"trans_{name}.jsonl")
    r = subprocess.run(
        [PYTHON, str(PIPELINE_DIR / "transcribe_local.py"),
         "--input_dir", os.path.dirname(wav_path),
         "--output", trans_path,
         "--model_path", str(QWEN_MODEL_PATH)],
        capture_output=True, timeout=300)
    if r.returncode != 0:
        return {"per": None, "error": r.stderr.decode()[-200:]}

    # Parse transcription: match by file name
    hyp_text = ""
    if os.path.exists(trans_path):
        for line in open(trans_path):
            d = json.loads(line)
            fname = os.path.basename(d.get("file_path", "") or d.get("file_name", ""))
            if fname.startswith(name):
                hyp_text = d.get("hyp_text", "")
                break

    if not hyp_text:
        return {"per": 1.0, "hyp_text": "", "gt_text": clean_gt,
                "S": 0, "D": len(clean_gt), "I": 0, "N": len(clean_gt),
                "hyp_len": 0, "gt_len": len(clean_gt)}

    # CER (Character Error Rate) - simpler and more robust than phoneme for Chinese
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, clean_gt, hyp_text)
    S = I = D = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            S += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            D += (i2 - i1)
        elif tag == 'insert':
            I += (j2 - j1)
    N = len(clean_gt)
    per = (S + D + I) / max(N, 1)

    return {"per": per, "hyp_text": hyp_text, "gt_text": clean_gt,
            "S": S, "D": D, "I": I, "N": N,
            "hyp_len": len(hyp_text), "gt_len": len(clean_gt)}


def run_cmd(cmd, desc, timeout=600):
    print(f"  {desc}...", end="", flush=True)
    t0=time.time(); r=subprocess.run(cmd,capture_output=True,timeout=timeout)
    ok=r.returncode==0; dt=time.time()-t0
    print(f" {'OK' if ok else 'FAIL'} ({dt:.0f}s)" if ok else f" FAIL ({r.stderr.decode()[-150:]})")
    return ok


# =========================================================================
#  Verify
# =========================================================================

def step_verify():
    """Single-song, single-window sanity check."""
    print("="*60)
    print("Verify: 1 song, 1 window")
    print("="*60)
    fidx = 0
    style, full_lyrics = parse_entry(fidx)
    dur = get_duration(fidx)
    sections = get_section_boundaries(full_lyrics, dur)
    windows = build_windows(dur)

    if not windows: return
    ws, we = windows[0]
    window_lyrics = get_window_lyrics(sections, ws, we)
    print(f"Song {fidx}: dur={dur:.0f}s, window=[{ws:.0f}-{we:.0f}]s")
    print(f"Window lyrics ({len(window_lyrics)}ch): {window_lyrics[:80]}...")

    # Generate oneshot
    os.makedirs(EVAL_ROOT / "oneshot" / "audio_zh", exist_ok=True)
    wav1 = EVAL_ROOT / "oneshot" / "audio_zh" / f"{fidx:06d}.flac"
    if not (wav1.exists() and wav1.stat().st_size > 1000):
        print("  [GEN] oneshot...", end="", flush=True)
        path, err = gen_one("baseline", None, style, full_lyrics, wav1.parent,
                            SEED+fidx, dur, fidx)
        if path:
            shutil.move(path, wav1) if os.path.abspath(path) != os.path.abspath(wav1) else None
            print(" OK")
        else:
            print(f" FAIL: {err}"); return
    else:
        print("  [SKIP] oneshot")

    # Crop oneshot window
    win_data, sr = crop_audio(wav1, ws, we)
    print(f"  Cropped window: {len(win_data)} samples ({len(win_data)/sr:.1f}s)")

    # Generate repaint with different modes
    for gname, gcfg in GROUPS.items():
        if gname == "oneshot": continue
        g_dir = EVAL_ROOT / gname / "audio_zh"
        g_dir.mkdir(parents=True, exist_ok=True)
        out_wav = g_dir / f"{fidx:06d}_w0.flac"

        if not (out_wav.exists() and out_wav.stat().st_size > 1000):
            print(f"  [GEN] {gname}...", end="", flush=True)
            path, err = gen_one("baseline", None, style, window_lyrics, g_dir,
                SEED+fidx+100, fidx+100, dur,
                task_type="repaint", src_audio=str(wav1),
                repainting_start=ws, repainting_end=we,
                repaint_mode=gcfg["mode"], repaint_strength=gcfg["strength"])
            if path:
                shutil.move(path, out_wav); print(" OK")
            else:
                print(f" FAIL: {err}"); continue
        else:
            print(f"  [SKIP] {gname}")

        # Crop repaint window
        rwin_data, r_sr = crop_audio(out_wav, ws, we)
        diff = np.abs(np.array(win_data, dtype=np.float64) -
                      np.array(rwin_data, dtype=np.float64))
        rms_orig = np.sqrt(np.mean(np.array(win_data, dtype=np.float64)**2))
        rms_rep = np.sqrt(np.mean(np.array(rwin_data, dtype=np.float64)**2))
        print(f"    RMS: orig={rms_orig:.4f} rep={rms_rep:.4f} "
              f"ratio={rms_rep/max(rms_orig,1e-12):.3f} "
              f"diff_max={diff.max():.4f}")

        # Local PER
        result = eval_window_per(window_lyrics, rwin_data, r_sr,
                                 EVAL_ROOT / "verify" / "per",
                                 f"{gname}_w0")
        if result["per"] is not None:
            print(f"    Window PER: {result['per']:.4f} "
                  f"(S={result['S']} D={result['D']} I={result['I']} N={result['N']})")
            print(f"    GT: {result['gt_text'][:60]}...")
            print(f"    ASR: {result['hyp_text'][:60]}...")

    print("\nVerify done.")


# =========================================================================
#  Full Generation
# =========================================================================

def step_generate():
    print("="*60)
    print(f"Generating {len(range(10))} songs × {len(GROUPS)} groups")
    print("="*60)
    t0 = time.time()

    for fidx in range(10):
        style, full_lyrics = parse_entry(fidx)
        dur = get_duration(fidx)
        sections = get_section_boundaries(full_lyrics, dur)
        windows = build_windows(dur)
        print(f"\n--- Song {fidx} (dur={dur:.0f}s, {len(windows)} windows) ---")

        # Oneshot
        g1_dir = EVAL_ROOT / "oneshot" / "audio_zh"
        g1_dir.mkdir(parents=True, exist_ok=True)
        wav1 = g1_dir / f"{fidx:06d}.flac"
        if not (wav1.exists() and wav1.stat().st_size > 1000):
            print(f"  [GEN] oneshot...", end="", flush=True)
            path, err = gen_one("baseline", None, style, full_lyrics, g1_dir,
                                SEED+fidx, dur, fidx)
            if path and path != "no_audio" and os.path.exists(path):
                if os.path.abspath(path) != os.path.abspath(wav1):
                    shutil.move(path, wav1)
                print(" OK")
            else:
                print(f" FAIL: {err or path}"); continue
        else:
            print(f"  [SKIP] oneshot")

        # Save window config
        win_cfg_dir = EVAL_ROOT / "windows"
        win_cfg_dir.mkdir(parents=True, exist_ok=True)
        win_configs = []

        for win_idx, (ws, we) in enumerate(windows):
            window_lyrics = get_window_lyrics(sections, ws, we)
            win_configs.append({
                "win_idx": win_idx, "song_idx": fidx,
                "win_start": ws, "win_end": we,
                "window_lyrics": window_lyrics,
                "window_lyrics_len": len(window_lyrics),
            })

            for gname, gcfg in GROUPS.items():
                if gname == "oneshot": continue
                g_dir = EVAL_ROOT / gname / "audio_zh"
                g_dir.mkdir(parents=True, exist_ok=True)
                out_wav = g_dir / f"{fidx:06d}_w{win_idx}.flac"

                if out_wav.exists() and out_wav.stat().st_size > 1000:
                    print(f"  [SKIP] {gname} w{win_idx}")
                    continue

                print(f"  [GEN] {gname} w{win_idx} ({ws:.0f}-{we:.0f}s, "
                      f"mode={gcfg['mode']}, strength={gcfg['strength']})...", end="", flush=True)
                path, err = gen_one("baseline", None, style, window_lyrics, g_dir,
                    SEED+fidx+win_idx*1000, fidx+win_idx*1000, dur,
                    task_type="repaint", src_audio=str(wav1),
                    repainting_start=ws, repainting_end=we,
                    repaint_mode=gcfg["mode"], repaint_strength=gcfg["strength"])
                if path and path != "no_audio" and os.path.exists(path):
                    shutil.move(path, out_wav); print(" OK")
                elif path == "no_audio" or (path and not os.path.exists(path)):
                    print(f" FAIL: {path}")
                else:
                    print(f" FAIL: {err}")

        json.dump(win_configs, open(win_cfg_dir / f"config_{fidx}.json", "w"),
                  indent=2, ensure_ascii=False)

    dt = time.time()-t0
    print(f"\nGenerated in {dt/60:.1f} min")
    for g in GROUPS:
        n = len(list((EVAL_ROOT / g / "audio_zh").glob("*.flac")))
        print(f"  {g}/zh: {n} files")


# =========================================================================
#  Window-level PER Evaluation
# =========================================================================

def step_evaluate():
    """Evaluate PER on each cropped window."""
    print("="*60)
    print("Window-level PER Evaluation")
    print("="*60)
    t0 = time.time()

    all_results = {g: [] for g in GROUPS}

    # Collect all windows
    win_cfg_dir = EVAL_ROOT / "windows"
    window_list = []
    for fidx in range(10):
        cfg_file = win_cfg_dir / f"config_{fidx}.json"
        if cfg_file.exists():
            for w in json.load(open(cfg_file)):
                window_list.append(w)

    print(f"Total windows: {len(window_list)}")

    for win in window_list:
        fidx = win["song_idx"]
        ws = win["win_start"]
        we = win["win_end"]
        win_lyrics = win["window_lyrics"]
        wname = f"s{fidx}_w{win['win_idx']}"

        if not win_lyrics.strip():
            for g in GROUPS:
                all_results[g].append({"song_idx": fidx, "win_start": ws, "win_end": we,
                                       "per": None, "N": 0, "reason": "empty_lyrics"})
            continue

        for gname in GROUPS:
            # Get audio: for oneshot, use original; for repaint, use the repaint output
            if gname == "oneshot":
                wav_path = EVAL_ROOT / "oneshot" / "audio_zh" / f"{fidx:06d}.flac"
            else:
                wav_path = EVAL_ROOT / gname / "audio_zh" / f"{fidx:06d}_w{win['win_idx']}.flac"

            if not wav_path.exists():
                all_results[gname].append({"song_idx": fidx, "win_start": ws, "win_end": we,
                                           "per": None, "reason": "no_audio"})
                continue

            # Crop to window (remove margin to avoid crossfade)
            crop_data, crop_sr = crop_audio(wav_path, ws, we)

            # Evaluate PER on this window
            per_dir = EVAL_ROOT / gname / "per_window"
            result = eval_window_per(win_lyrics, crop_data, crop_sr, per_dir, wname)
            result["song_idx"] = fidx
            result["win_start"] = ws
            result["win_end"] = we
            all_results[gname].append(result)

    # Summary
    print("\n" + "="*60)
    print("WINDOW PER SUMMARY")
    print("="*60)
    for gname in GROUPS:
        results = all_results[gname]
        pers = [r["per"] for r in results if r["per"] is not None]
        ns = [r["N"] for r in results if r.get("N", 0) > 0]
        if not pers:
            print(f"\n[{gname}] No valid PER results")
            continue
        # Aggregate S/D/I
        total_S = sum(r.get("S", 0) for r in results if r.get("S") is not None)
        total_D = sum(r.get("D", 0) for r in results if r.get("D") is not None)
        total_I = sum(r.get("I", 0) for r in results if r.get("I") is not None)
        total_N = sum(r.get("N", 0) for r in results if r.get("N") is not None)
        mean_per = sum(pers) / len(pers)
        agg_per = (total_S + total_D + total_I) / max(total_N, 1) * 100 if total_N > 0 else 0
        print(f"\n[{gname}] ({len(pers)} windows)")
        print(f"  Mean PER: {mean_per:.4f}")
        print(f"  Aggregated PER: {agg_per:.2f}%")
        print(f"  S={total_S} D={total_D} I={total_I} N={total_N}")

    # Per-song breakdown
    print("\n--- Per-song Window PER ---")
    print(f"{'Song':>4} {'oneshot':>8} {'cons_01':>8} {'cons_025':>8} {'bal_05':>8}")
    for fidx in range(10):
        line = f"{fidx:>4}"
        for gname in GROUPS:
            pers = [r["per"] for r in all_results[gname]
                    if r.get("song_idx") == fidx and r["per"] is not None]
            if pers:
                line += f" {sum(pers)/len(pers):>8.4f}"
            else:
                line += f" {'N/A':>8}"
        print(line)

    # Aggregate per window across songs
    print("\n--- Per-window position PER ---")
    max_windows = max(len([w for w in window_list if w["song_idx"] == f])
                      for f in range(10))
    for win_idx in range(max_windows):
        line = f"w{win_idx}:"
        for gname in GROUPS:
            pers = []
            for r in all_results[gname]:
                if r.get("win_idx", win_idx) == win_idx and r["per"] is not None:
                    pers.append(r["per"])
            if pers:
                line += f" {sum(pers)/len(pers):.4f}"
            else:
                # Use song_idx + win_start as backup
                pers = [r["per"] for r in all_results[gname]
                        if r.get("win_start") is not None and r["per"] is not None
                        and r["win_start"] == WINDOW_SEC * win_idx + 0.5 * win_idx * WINDOW_SEC]
                # simpler: just use index
                pers = [r["per"] for i, r in enumerate(all_results[gname])
                        if r["per"] is not None and i % max_windows == win_idx]
                if pers:
                    line += f" {sum(pers)/len(pers):.4f}"
                else:
                    line += "    N/A"
        print(line)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify", "generate", "evaluate", "all"])
    args = parser.parse_args()
    if args.mode == "verify": step_verify()
    elif args.mode == "generate": step_generate()
    elif args.mode == "evaluate": step_evaluate()
    elif args.mode == "all":
        step_generate()
        step_evaluate()

if __name__ == "__main__":
    main()
