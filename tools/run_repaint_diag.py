#!/root/miniconda3/envs/musicgen/bin/python
"""
Repaint Diagnostic v2: Independent repaint + PCM splice, no VAE chain.

用法:
  python tools/run_repaint_diag.py verify    # 单窗口 sanity check (1歌1窗)
  python tools/run_repaint_diag.py generate  # 完整生成 (10歌)
  python tools/run_repaint_diag.py evaluate  # 评估
"""
import argparse, json, os, pickle, subprocess, sys, time, re, shutil
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path("/root/ACE-Step-1.5")
EVAL_ROOT = Path("/root/autodl-tmp/tsm_eval_results") / "repaint_v2"
TEST_JSONL = PROJECT_ROOT / "Muse/infer/test.jsonl"
GEN_SCRIPT = PROJECT_ROOT / "gen_one.py"
PIPELINE_DIR = PROJECT_ROOT / "Muse/eval_pipeline"
SONGEVAL_CKPT = PROJECT_ROOT / "SongEval/ckpt/model.safetensors"
SONGEVAL_CONFIG = PROJECT_ROOT / "SongEval/config.yaml"
QWEN_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-ASR-1.7B")
GT_LYRICS_DIR = PIPELINE_DIR / "gt_lyrics"
PYTHON = sys.executable
SEED = 42

WINDOW_SEC = 50
CONTEXT_SEC = 10      # context on each side for repaint
CROPFADE_SEC = 2.0    # crossfade duration for splicing


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
    """Parse lyrics to [(section_name, text, start_time, end_time), ...]."""
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

    if not sections:
        return []

    SILENCE = {'INTRO', 'OUTRO', 'INTERLUDE', 'INSTRUMENTAL'}
    weights = []
    for tag, text in sections:
        w = len(text) * (0.3 if tag.upper() in SILENCE else 1.0)
        weights.append(max(w, 1))
    total_w = sum(weights)

    boundaries = []
    cum = 0.0
    for i, (tag, text) in enumerate(sections):
        dur = total_dur * weights[i] / total_w
        boundaries.append((tag, text, cum, cum + dur))
        cum += dur
    return boundaries


def build_windows(total_dur, window_sec=WINDOW_SEC):
    """Build windows for last ~50% of song. Returns [(start, end), ...]."""
    last_half = total_dur * 0.5
    windows = []
    pos = last_half
    while pos < total_dur:
        end = min(pos + window_sec, total_dur)
        windows.append((pos, end))
        pos = end
    return windows


def get_lyrics_for_window(sections, win_start, win_end):
    """Extract lyrics overlapping with [win_start, win_end]."""
    result = []
    for tag, text, s, e in sections:
        overlap = max(0, min(e, win_end) - max(s, win_start))
        if overlap > 1.0:
            result.append(f"[{tag}]")
            result.append(text)
    return "\n".join(result).strip()


def get_sinkhorn_lyrics(sections, win_start, win_end, total_dur):
    """
    Sinkhorn-inspired: allocate broader lyrics to each window.
    Include the target section AND the most similar (by cosine of section type)
    sections outside the window.
    """
    # Base: strict window lyrics
    base = get_lyrics_for_window(sections, win_start, win_end)

    # Find the main section tag(s) in this window
    window_tags = set()
    for tag, text, s, e in sections:
        if max(0, min(e, win_end) - max(s, win_start)) > 1.0:
            window_tags.add(tag.upper())

    # Allocate a broader budget: include sections of the SAME type within ±30% of duration
    result_lines = []
    included_tags = set()
    for tag, text, s, e in sections:
        overlap = max(0, min(e, win_end) - max(s, win_start))
        is_window = overlap > 1.0
        # Include if in window, OR same type and within extended range
        budget_start = max(0, win_start - total_dur * 0.15)
        budget_end = min(total_dur, win_end + total_dur * 0.15)
        is_nearby = (s < budget_end and e > budget_start)
        same_type = tag.upper() in window_tags and len(window_tags) > 0

        curr_tag = tag.upper()
        if curr_tag not in included_tags:
            if is_window or (is_nearby and same_type):
                result_lines.append(f"[{tag}]")
                result_lines.append(text)
                included_tags.add(curr_tag)

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
    return None, "no OK: in output"


def splice_repaint(orig_wav, repaint_wav, win_start, win_end, total_dur, sr,
                   context_sec=CONTEXT_SEC, crofade_sec=CROPFADE_SEC):
    """
    Splice the repainted central region into the original waveform.

    The repaint output covers [win_start-context, win_end+context].
    We extract the central region [win_start, win_end] from it and splice
    into original, with crossfade at the splice boundaries.

    Supports both mono and stereo audio.
    """
    orig = np.array(orig_wav, dtype=np.float64)
    repaint = np.array(repaint_wav, dtype=np.float64)
    is_stereo = len(orig.shape) > 1 and orig.shape[1] == 2
    fade_samples = int(crofade_sec * sr)

    ws = int(win_start * sr)
    we = int(win_end * sr)
    ws = max(0, ws)
    we = min(len(orig), we)

    out = orig.copy()

    def _blend(a, b, fade_curve):
        """Blend two arrays with a fade curve. Handles stereo."""
        if is_stereo:
            for c in range(2):
                out_part = out[:, c]
                rep_part = repaint[:, c]
                out_part_s = out_part if len(out_part) > 0 else np.array([])
                # We'll handle per-channel in the loops below
        return None  # Placeholder

    # Crossfade at start boundary
    if ws > 0 and fade_samples > 0:
        fade_start = max(0, ws - fade_samples)
        fade_len = ws - fade_start
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            if is_stereo:
                for c in range(2):
                    out[fade_start:ws, c] = (
                        orig[fade_start:ws, c] * (1 - fade_in) +
                        repaint[fade_start:ws, c] * fade_in
                    )
            else:
                out[fade_start:ws] = (
                    orig[fade_start:ws] * (1 - fade_in) +
                    repaint[fade_start:ws] * fade_in
                )

    # Replace central region
    replace_len = we - ws
    if replace_len > 0:
        if is_stereo:
            out[ws:we] = repaint[ws:we]
        else:
            out[ws:we] = repaint[ws:we]

    # Crossfade at end boundary
    if we < len(orig) and fade_samples > 0:
        fade_end = min(len(orig), we + fade_samples)
        fade_len = fade_end - we
        if fade_len > 0:
            fade_out = np.linspace(1, 0, fade_len)
            if is_stereo:
                for c in range(2):
                    out[we:fade_end, c] = (
                        orig[we:fade_end, c] * (1 - fade_out) +
                        repaint[we:fade_end, c] * fade_out
                    )
            else:
                out[we:fade_end] = (
                    orig[we:fade_end] * (1 - fade_out) +
                    repaint[we:fade_end] * fade_out
                )

    return out


def run_cmd(cmd, desc, timeout=600):
    print(f"  {desc}...", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    ok = r.returncode == 0
    dt = time.time() - t0
    print(f" {'OK' if ok else 'FAIL'} ({dt:.0f}s)" if ok else f" FAIL ({r.stderr.decode()[-150:]})")
    return ok


def analyze_window(audio, sr, win_start, win_end, name=""):
    """Analyze a window region for RMS and silence detection."""
    ws = int(win_start * sr)
    we = min(int(win_end * sr), len(audio))
    segment = np.array(audio[ws:we], dtype=np.float64)
    rms = np.sqrt(np.mean(segment ** 2)) if len(segment) > 0 else 0
    peak = np.max(np.abs(segment)) if len(segment) > 0 else 0
    silence_ratio = np.mean(np.abs(segment) < 0.01) if len(segment) > 0 else 1.0
    print(f"    {name}: rms={rms:.4f} peak={peak:.4f} silence_ratio={silence_ratio:.3f}")


# =========================================================================
#  Verify
# =========================================================================

def step_verify():
    """Single-song, single-window sanity check."""
    print("=" * 60)
    print("Sanity Check: 1 song, 1 window")
    print("=" * 60)

    fidx = 0
    style, full_lyrics = parse_entry(fidx)
    dur = get_duration(fidx)
    sections = get_section_boundaries(full_lyrics, dur)
    windows = build_windows(dur)

    if not windows:
        print("No windows!")
        return
    ws, we = windows[0]
    print(f"Song {fidx}: dur={dur:.0f}s, window=[{ws:.0f}-{we:.0f}]s")

    # Generate one-shot
    oneshot_dir = EVAL_ROOT / "oneshot" / "audio_zh"
    oneshot_dir.mkdir(parents=True, exist_ok=True)
    oneshot_wav = oneshot_dir / f"{fidx:06d}.flac"

    if not (oneshot_wav.exists() and oneshot_wav.stat().st_size > 1000):
        print("  [GEN] oneshot...", end="", flush=True)
        path, err = gen_one("baseline", None, style, full_lyrics, oneshot_dir, SEED + fidx, dur, fidx)
        if path:
            if os.path.abspath(path) != os.path.abspath(oneshot_wav):
                shutil.move(path, oneshot_wav)
            print(" OK")
        else:
            print(f" FAIL: {err}")
            return
    else:
        print("  [SKIP] oneshot")

    # Verify original audio
    orig_data, orig_sr = sf.read(oneshot_wav)
    print(f"  Original audio: sr={orig_sr}, len={len(orig_data)} ({len(orig_data)/orig_sr:.1f}s)")

    # Generate repaint for the window (once, with native lyrics)
    native_lyrics = get_lyrics_for_window(sections, ws, we)
    repaint_dir = EVAL_ROOT / "verify" / "audio_zh"
    repaint_dir.mkdir(parents=True, exist_ok=True)
    repaint_wav = repaint_dir / f"{fidx:06d}_w0.flac"

    if not (repaint_wav.exists() and repaint_wav.stat().st_size > 1000):
        print(f"  [GEN] repaint (lyrics={len(native_lyrics)}ch)...", end="", flush=True)
        path, err = gen_one("baseline", None, style, native_lyrics, repaint_dir,
            SEED + fidx + 100, fidx + 100, dur,
            task_type="repaint", src_audio=str(oneshot_wav),
            repainting_start=ws, repainting_end=we,
            repaint_mode="balanced", repaint_strength=0.5)
        if path:
            shutil.move(path, repaint_wav)
            print(" OK")
        else:
            print(f" FAIL: {err}")
            return
    else:
        print("  [SKIP] repaint")

    # Verify repaint audio
    rep_data, rep_sr = sf.read(repaint_wav)
    print(f"  Repaint audio: sr={rep_sr}, len={len(rep_data)} ({len(rep_data)/rep_sr:.1f}s)")

    # Analyze regions
    ctx_sec = CONTEXT_SEC
    print(f"\n  Region analysis (sr={orig_sr}):")
    analyze_window(orig_data, orig_sr, 0, ws, "pre-window (orig)")
    analyze_window(orig_data, orig_sr, ws, we, f"window [{ws:.0f}-{we:.0f}] (orig)")
    analyze_window(orig_data, orig_sr, we, len(orig_data)/orig_sr, "post-window (orig)")
    analyze_window(rep_data, rep_sr, 0, ws, "pre-window (repaint)")
    analyze_window(rep_data, rep_sr, ws, we, f"window [{ws:.0f}-{we:.0f}] (repaint)")
    analyze_window(rep_data, rep_sr, we, len(rep_data)/rep_sr, "post-window (repaint)")

    # Check pre-window: should be nearly identical (no VAE degradation in balance mode)
    pre_orig = np.array(orig_data[:int(ws * orig_sr)], dtype=np.float64)
    pre_rep = np.array(rep_data[:int(ws * orig_sr)], dtype=np.float64)
    if len(pre_orig) != len(pre_rep):
        print(f"\n  ⚠ LENGTH MISMATCH: orig={len(pre_orig)}, rep={len(pre_rep)}")
        pre_len = min(len(pre_orig), len(pre_rep))
        pre_orig, pre_rep = pre_orig[:pre_len], pre_rep[:pre_len]

    pre_diff = np.abs(pre_orig - pre_rep)
    print(f"\n  Pre-window diff: mean={pre_diff.mean():.6f} max={pre_diff.max():.6f}")
    print(f"  Pre-window identical: {np.max(pre_diff) < 1e-6}")

    # Window region analysis
    w_orig = np.array(orig_data[int(ws * orig_sr):int(we * orig_sr)], dtype=np.float64)
    w_rep = np.array(rep_data[int(ws * orig_sr):int(we * orig_sr)], dtype=np.float64)
    w_diff = np.abs(w_orig - w_rep)
    rms_orig = np.sqrt(np.mean(w_orig ** 2))
    rms_rep = np.sqrt(np.mean(w_rep ** 2))
    print(f"  Window diff: mean={w_diff.mean():.6f} max={w_diff.max():.6f}")
    print(f"  Window RMS: orig={rms_orig:.4f} rep={rms_rep:.4f} ratio={rms_rep/max(rms_orig,1e-12):.3f}")
    print(f"  Window changed: {rms_rep/max(rms_orig,1e-12) not in [0.95, 1.05]}")

    # Test splice
    result = splice_repaint(orig_data, rep_data, ws, we, dur, orig_sr)
    print(f"\n  Spliced audio: len={len(result)}, sr={orig_sr}")

    # Verify pre-window is bit-exact with original after splice
    pre_result = result[:int(ws * orig_sr)]
    pre_orig_s = np.array(orig_data[:int(ws * orig_sr)], dtype=np.float64)
    if len(pre_result) == len(pre_orig_s):
        splice_diff = np.max(np.abs(pre_result - pre_orig_s))
        print(f"  Spliced pre-window bit-exact: {splice_diff < 1e-10} (max_diff={splice_diff:.2e})")
    else:
        print(f"  Spliced pre-window length mismatch: {len(pre_result)} vs {len(pre_orig_s)}")

    # Test the window region
    w_result = result[int(ws * orig_sr):int(we * orig_sr)]
    w_rep_s = np.array(rep_data[int(ws * orig_sr):int(we * orig_sr)], dtype=np.float64)
    if len(w_result) == len(w_rep_s):
        w_splice_diff = np.max(np.abs(w_result - w_rep_s))
        print(f"  Spliced window matches repaint: {w_splice_diff < 1e-10} (max_diff={w_splice_diff:.2e})")
    else:
        print(f"  Spliced window length mismatch: {len(w_result)} vs {len(w_rep_s)}")

    # Save spliced output for inspection
    out_path = EVAL_ROOT / "verify" / f"spliced_{fidx:06d}.flac"
    sf.write(str(out_path), result, orig_sr)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if np.max(pre_diff) < 5e-5:
        print("  ✅ VAE boundary preservation: GOOD")
    elif np.max(pre_diff) < 1e-3:
        print("  ⚠ VAE boundary preservation: ACCEPTABLE (some encode-decode drift)")
    else:
        print("  ❌ VAE boundary preservation: POOR (significant drift)")

    if rms_rep / max(rms_orig, 1e-12) < 0.01:
        print("  ❌ Repaint region is NEAR SILENT — strength too conservative?")
    elif rms_rep / max(rms_orig, 1e-12) < 0.5:
        print("  ⚠ Repaint region amplitude significantly reduced")
    else:
        print("  ✅ Repaint region amplitude: NORMAL")


# =========================================================================
#  Full Generation
# =========================================================================

def step_generate():
    """Generate 10 songs × 3 groups with independent repaint + PCM splice."""
    print("=" * 60)
    print(f"Generating {len(range(10))} songs × 3 groups")
    print("=" * 60)
    t0 = time.time()

    for fidx in range(10):
        style, full_lyrics = parse_entry(fidx)
        dur = get_duration(fidx)
        sections = get_section_boundaries(full_lyrics, dur)
        windows = build_windows(dur)

        print(f"\n--- Song {fidx} (dur={dur:.0f}s, {len(windows)} windows) ---")

        # === Group 1: One-shot ===
        g1_dir = EVAL_ROOT / "oneshot" / "audio_zh"
        g1_dir.mkdir(parents=True, exist_ok=True)
        wav1 = g1_dir / f"{fidx:06d}.flac"
        if not (wav1.exists() and wav1.stat().st_size > 1000):
            print(f"  [GEN] oneshot...", end="", flush=True)
            path, err = gen_one("baseline", None, style, full_lyrics, g1_dir,
                                SEED + fidx, dur, fidx)
            if path:
                shutil.move(path, wav1) if os.path.abspath(path) != os.path.abspath(wav1) else None
                print(" OK")
            else:
                print(f" FAIL: {err}")
                continue
        else:
            print(f"  [SKIP] oneshot")

        orig_data, orig_sr = sf.read(wav1)
        assert orig_sr == 48000

        # === Prepare Groups 2 & 3 ===
        for group_name, lyrics_fn in [("repaint_native", get_lyrics_for_window),
                                       ("repaint_sinkhorn", get_sinkhorn_lyrics)]:
            g_dir = EVAL_ROOT / group_name / "audio_zh"
            g_dir.mkdir(parents=True, exist_ok=True)

            # Splice: start with original, iteratively replace windows
            spliced = np.array(orig_data, dtype=np.float64)

            for win_idx, (ws, we) in enumerate(windows):
                out_wav = g_dir / f"{fidx:06d}_w{win_idx}.flac"

                # Get lyrics for this window
                if lyrics_fn == get_sinkhorn_lyrics:
                    win_lyrics = lyrics_fn(sections, ws, we, dur)
                else:
                    win_lyrics = lyrics_fn(sections, ws, we)

                if out_wav.exists() and out_wav.stat().st_size > 1000:
                    print(f"  [SKIP] {group_name} w{win_idx} ({ws:.0f}-{we:.0f}s)")
                else:
                    print(f"  [GEN] {group_name} w{win_idx} ({ws:.0f}-{we:.0f}s, lyrics={len(win_lyrics)}ch)...",
                          end="", flush=True)
                    path, err = gen_one("baseline", None, style, win_lyrics, g_dir,
                        SEED + fidx + win_idx * 100, fidx + win_idx * 100, dur,
                        task_type="repaint", src_audio=str(wav1),
                        repainting_start=ws, repainting_end=we,
                        repaint_mode="balanced", repaint_strength=0.5)
                    if path:
                        shutil.move(path, out_wav)
                        print(" OK")
                    else:
                        print(f" FAIL: {err}")
                        continue

                # Read repaint result and splice
                rep_data, rep_sr = sf.read(out_wav)
                assert rep_sr == orig_sr

                # Splice
                spliced = splice_repaint(orig_data, rep_data, ws, we, dur, orig_sr)

            # Save final spliced audio
            final_path = g_dir / f"{fidx:06d}.flac"
            sf.write(str(final_path), spliced.astype(np.float32), orig_sr)
            print(f"  [SAVE] {group_name} final: {final_path}")

    dt = time.time() - t0
    print(f"\nGenerated in {dt/60:.1f} min")
    for g in ["oneshot", "repaint_native", "repaint_sinkhorn"]:
        n = len(list((EVAL_ROOT / g / "audio_zh").glob("[0-9]*.flac")))
        base = len([f for f in (EVAL_ROOT / g / "audio_zh").glob("[0-9]*.flac") if "_w" not in f.stem])
        print(f"  {g}/zh: {base} base files, {n} total")


# =========================================================================
#  Evaluation
# =========================================================================

def step_evaluate():
    """Evaluate all groups."""
    print("=" * 60)
    print("Evaluating")
    print("=" * 60)
    t0 = time.time()

    for group in ["oneshot", "repaint_native", "repaint_sinkhorn"]:
        audio_dir = EVAL_ROOT / group / "audio_zh"
        results_dir = EVAL_ROOT / group / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Only use base files (no _w prefix)
        base_files = sorted([
            f for f in audio_dir.glob("[0-9]*.flac")
            if "_w" not in f.stem
        ])
        n = len(base_files)
        if n == 0:
            print(f"  [SKIP] {group} (no base files)")
            continue

        # Create temp dir with symlinks
        tmp_eval = EVAL_ROOT / group / "tmp_eval"
        tmp_eval.mkdir(exist_ok=True)
        for f in tmp_eval.glob("*"):
            os.remove(f)
        for f in base_files:
            os.symlink(os.path.abspath(f), tmp_eval / f.name)

        print(f"\n--- {group} ({n} songs) ---")

        # SongEval
        run_cmd([PYTHON, str(PIPELINE_DIR / "eval_songeval.py"),
            "--input_dir", str(tmp_eval), "--model_name", f"{group}_zh",
            "--output", str(results_dir / f"songeval_{group}_zh.json"),
            "--ckpt", str(SONGEVAL_CKPT), "--config", str(SONGEVAL_CONFIG), "--gpu", "0"],
            f"SongEval [{group}]")

        # ASR
        trans_file = EVAL_ROOT / group / "transcriptions_zh.jsonl"
        ok = run_cmd([PYTHON, str(PIPELINE_DIR / "transcribe_local.py"),
            "--input_dir", str(tmp_eval), "--output", str(trans_file),
            "--model_path", str(QWEN_MODEL_PATH)], f"ASR [{group}]", timeout=3600)
        if not ok:
            continue

        # PER
        gt_file = GT_LYRICS_DIR / "zh.jsonl"
        per_out = EVAL_ROOT / group / "per_results_zh"
        run_cmd([PYTHON, str(PIPELINE_DIR / "calc_per_long.py"),
            "--hyp_file", str(trans_file), "--gt_file", str(gt_file),
            "--model_name", f"{group}_zh", "--output_dir", str(per_out)],
            f"PER [{group}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for group in ["oneshot", "repaint_native", "repaint_sinkhorn"]:
        csv = EVAL_ROOT / group / "per_results_zh" / "summary.csv"
        if csv.exists():
            print(f"\n[{group}]")
            for line in open(csv):
                line = line.strip()
                if line and not line.startswith("model"):
                    print(f"  {line}")
        songjson = EVAL_ROOT / group / "per_results_zh" / "songs.jsonl"
        if songjson.exists():
            # Compute early/middle/late S/D/I breakdown
            sd = {"early": {"S":0,"D":0,"I":0,"N":0}, "middle": {"S":0,"D":0,"I":0,"N":0}, "late": {"S":0,"D":0,"I":0,"N":0}}
            for line in open(songjson):
                d = json.loads(line)
                for split in ["early","middle","late"]:
                    for k in ["S","D","I","N"]:
                        sd[split][k] += d.get(f"{split}_phoneme_{k.lower()}", 0)
            print(f"  S/D/I breakdown:")
            for split in ["early","middle","late"]:
                s=sd[split]["S"]; d=sd[split]["D"]; i=sd[split]["I"]; n=max(sd[split]["N"],1)
                per = (s+d+i)/n*100
                print(f"    {split}: S={s} D={d} I={i} N={n} PER={per:.1f}%")

    # Compare SongEval
    print("\n--- SongEval Comparison ---")
    print(f"{'Group':<20} {'Coher':>7} {'Music':>7} {'Memor':>7} {'Clarity':>7} {'Natural':>7}")
    for group in ["oneshot", "repaint_native", "repaint_sinkhorn"]:
        se = EVAL_ROOT / group / "results" / f"songeval_{group}_zh.json"
        if se.exists():
            data = json.loads(open(se))
            m = data["metrics"]
            print(f"{group:<20} {m['Coherence']:>7.3f} {m['Musicality']:>7.3f} {m['Memorability']:>7.3f} {m['Clarity']:>7.3f} {m['Naturalness']:>7.3f}")

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify", "generate", "evaluate", "all"])
    args = parser.parse_args()
    if args.mode == "verify":
        step_verify()
    elif args.mode == "generate":
        step_generate()
    elif args.mode == "evaluate":
        step_evaluate()
    elif args.mode == "all":
        step_generate()
        step_evaluate()

if __name__ == "__main__":
    main()
