#!/root/miniconda3/envs/musicgen/bin/python
"""
TSM 评估脚本：批量生成 + SongEval + AudioBox + PER (含 Late PER + LDG)

用法：
  python tools/run_tsm_eval.py generate     # 只生成
  python tools/run_tsm_eval.py evaluate     # 只评估
  python tools/run_tsm_eval.py all          # 全部

生成输出：/root/autodl-tmp/tsm_eval_results/<method>/audio_{cn,en}/
评估使用 Muse/eval_pipeline/ 下的脚本：
  - eval_songeval.py   → 音乐质量 5 维度
  - eval_audiobox.py   → 音频美观度 4 维度
  - transcribe_local.py → Qwen3-ASR 转写
  - calc_per_long.py    → PER + Late PER + LDG
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path("/root/autodl-tmp/tsm_eval_results")
TEST_JSONL = PROJECT_ROOT / "Muse/infer/test.jsonl"
GEN_SCRIPT = PROJECT_ROOT / "gen_one.py"
PIPELINE_DIR = PROJECT_ROOT / "Muse/eval_pipeline"

SONGEVAL_CKPT = PROJECT_ROOT / "SongEval/ckpt/model.safetensors"
SONGEVAL_CONFIG = PROJECT_ROOT / "SongEval/config.yaml"
AUDIOBOX_CKPT = Path("/root/autodl-tmp/models/audiobox-aesthetics/checkpoint.pt")
QWEN_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-ASR-1.7B")
GT_LYRICS_DIR = PIPELINE_DIR / "gt_lyrics"

TRANSPORT_CKPT = Path("/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt")
TSM_CKPT = EVAL_ROOT.parent / "tsm_sinkhorn/checkpoints/best_loss/pm_retrieval.pt"

SEED = 42
DURATION = 120  # default, will be overridden by per-entry phoneme timing
FILE_INDEXES = {"zh": [0, 10, 20, 30, 40], "en": [0, 10, 20, 30, 40]}


def get_duration(entry_idx):
    """Compute duration from cleaned lyrics: CN chars ×0.45, EN words ×0.45."""
    import re as _re
    with open(TEST_JSONL) as f:
        for i, line in enumerate(f):
            if i == entry_idx:
                data = json.loads(line)
                break
    msgs = data["messages"]
    seen = set()
    text = ""
    for msg in msgs:
        c = msg.get("content", "")
        m = _re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m: continue
        name = m.group(1)
        if name in seen: continue
        seen.add(name)
        rest = c[m.end():]
        if "[lyrics:" in rest:
            lyr = rest.split("[lyrics:")[1].split("]")[0].strip()
            text += lyr
    has_cjk = any('一' <= ch <= '鿿' for ch in text)
    if has_cjk:
        chars = len(_re.sub(r'\s', '', text))
        return max(20, int(chars * 0.45))
    else:
        words = len(text.split())
        return max(20, int(words * 0.45))

METHODS = {
    "baseline": None,
    "transport_only": TRANSPORT_CKPT,
    "sinkhorn_tsm": TSM_CKPT,
}

# Reallocation variants (inference-only, no TSM, different lambdas)
REALLOCATION_LAMBDAS = {
    "realloc_lam005": 0.05,
    "realloc_lam010": 0.10,
}

# Late-only reallocation: λ=0 for first 50% audio, linear ramp to λmax for last 50%
LATE_ONLY_LAMBDAS = {
    "late_only_lam005": 0.05,  # baseline + late-only, no TSM
}

# TSM + late-only reallocation
TSM_LATE_ONLY_LAMBDAS = {
    "tsm_late_only_lam003": 0.03,  # TSM + late-only
}

# TSM + full (not late-only) reallocation
TSM_REALLOCATION_LAMBDAS = {
    "tsm_realloc_lam005": 0.05,  # TSM + full reallocation
}

# Step-gated reallocation: only active in q∈[step_start, step_end] denoising steps
# Format: name → (lam, step_start, step_end)
STEP_GATED_LAMBDAS = {
    "step_gated_lam003":  (0.03, 0.25, 0.65),
    "step_gated_lam005":  (0.05, 0.25, 0.65),
    # New windows with λ=0.03
    "step_gated_A_lam003": (0.03, 0.05, 0.35),  # q∈[0.05, 0.35]
    "step_gated_B_lam003": (0.03, 0.10, 0.45),  # q∈[0.10, 0.45]
}

# TSM + step-gated reallocation
TSM_STEP_GATED_LAMBDAS = {
    "tsm_step_gated_lam003": (0.03, 0.25, 0.65),  # TSM + step-gated
}

# TSM residual gain variants: amplify TSM output only
TSM_GAINS = {
    "tsm_gain1": 1.0,   # baseline TSM
    "tsm_gain2": 2.0,
    "tsm_gain4": 4.0,
    "tsm_gain8": 8.0,
}

# Progressive λ within step-gated window: λ(p) = lam_min + (lam - lam_min) * p
STEP_GATED_PROGRESSIVE_LAMBDAS = {
    "step_gated_prog_lam003": (0.03, 0.01, 0.25, 0.65),  # (lam, lam_min, s_start, s_end)
}

PYTHON = sys.executable


# =========================================================================
#  Parse test.jsonl
# =========================================================================

def parse_entry(entry_idx):
    """Parse test.jsonl entry idx → (style_tags, full_lyrics_with_sections)"""
    with open(TEST_JSONL) as f:
        for i, line in enumerate(f):
            if i == entry_idx:
                data = json.loads(line)
                break
    msgs = data["messages"]
    style = msgs[0]["content"].split("\n")[0].replace(
        "Please generate a song in the following style:", "").strip()

    # Collect section tags + lyrics, strip desc/phoneme inner tags
    import re as _re
    seen = set()
    lyrics_parts = []
    for msg in msgs:
        c = msg.get("content", "")
        # Find the outermost [SectionName] tag (before any inner [desc:/lyrics:/phoneme:])
        m = _re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m:
            continue
        tag = m.group(0)
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        # Extract lyrics if present: content after [desc:...][lyrics:...]
        rest = c[m.end():]
        lyr = ""
        if "[lyrics:" in rest:
            lyr = rest.split("[lyrics:")[1].split("]")[0].strip()
        lyrics_parts.append(tag)
        if lyr:
            lyrics_parts.append(lyr)
        lyrics_parts.append("")

    lyrics = "\n".join(lyrics_parts).strip()
    return style, lyrics


# =========================================================================
#  Generation
# =========================================================================

def gen_one(method, ckpt, caption, lyrics, out_dir, seed, duration, fidx, **extra):
    """Call gen_one.py via subprocess with pickle'd args on stdin."""
    os.makedirs(out_dir, exist_ok=True)
    args_dict = dict(method=method, ckpt=str(ckpt) if ckpt else "", caption=caption,
                     lyrics=lyrics, out_dir=str(out_dir), seed=seed, duration=duration)
    args_dict.update(extra)
    r = subprocess.run(
        [PYTHON, str(GEN_SCRIPT)], input=pickle.dumps(args_dict),
        capture_output=True, timeout=600,
    )
    if r.returncode != 0:
        return None, r.stderr.decode().strip()[-300:]
    for line in r.stdout.decode().split("\n"):
        if line.startswith("OK:"):
            gen_path = line[3:].strip()
            if gen_path and os.path.exists(gen_path):
                new_name = f"{fidx:06d}.flac"
                new_path = os.path.join(out_dir, new_name)
                if os.path.abspath(gen_path) != os.path.abspath(new_path):
                    os.rename(gen_path, new_path)
                return new_path, None
            return gen_path, None
    return None, "no OK: in output"


def step_generate():
    print("=" * 60)
    print("Batch Generation")
    print("=" * 60)
    t0 = time.time()

    for method, ckpt in METHODS.items():
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / method / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {method}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {method}/{lang} {fidx} ({dur:.0f}s, {len(lyrics)} chars)...", end="")
                sys.stdout.flush()
                path, err = gen_one(method, ckpt, style, lyrics, out_dir, SEED + fidx, dur, outf)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # Reallocation variants (inference-only, no ckpt)
    for method, lam in REALLOCATION_LAMBDAS.items():
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / method / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {method}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {method}/{lang} {fidx} ({dur:.0f}s, {len(lyrics)} chars, λ={lam})...", end="")
                sys.stdout.flush()
                path, err = gen_one("baseline", None, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf, reallocation_lambda=lam)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # Late-only reallocation variants (baseline + late-only, no TSM)
    for name, lam_max in LATE_ONLY_LAMBDAS.items():
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, late-only λmax={lam_max})...", end="")
                sys.stdout.flush()
                path, err = gen_one("baseline", None, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    reallocation_lambda=lam_max,
                                    late_only_reallocation=True)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # TSM + late-only reallocation variants
    for name, lam_max in TSM_LATE_ONLY_LAMBDAS.items():
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, TSM+late-only λmax={lam_max})...", end="")
                sys.stdout.flush()
                path, err = gen_one("sinkhorn_tsm", TSM_CKPT, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    reallocation_lambda=lam_max,
                                    late_only_reallocation=True)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # TSM + full reallocation variants
    for name, lam in TSM_REALLOCATION_LAMBDAS.items():
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, TSM+full λ={lam})...", end="")
                sys.stdout.flush()
                path, err = gen_one("sinkhorn_tsm", TSM_CKPT, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    reallocation_lambda=lam)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # Step-gated reallocation variants (baseline + step-gated)
    for name, params in STEP_GATED_LAMBDAS.items():
        if isinstance(params, tuple):
            lam, s_start, s_end = params
        else:
            lam, s_start, s_end = params, 0.25, 0.65  # legacy: just a float
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, step-gated λ={lam} q∈[{s_start},{s_end}])...", end="")
                sys.stdout.flush()
                path, err = gen_one("baseline", None, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    reallocation_lambda=lam,
                                    step_gated_reallocation=True,
                                    step_gated_start=s_start,
                                    step_gated_end=s_end)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # TSM + step-gated reallocation variants
    for name, params in TSM_STEP_GATED_LAMBDAS.items():
        lam, s_start, s_end = params
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, TSM+step-gated λ={lam} q∈[{s_start},{s_end}])...", end="")
                sys.stdout.flush()
                path, err = gen_one("sinkhorn_tsm", TSM_CKPT, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    reallocation_lambda=lam,
                                    step_gated_reallocation=True,
                                    step_gated_start=s_start,
                                    step_gated_end=s_end)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # TSM residual gain variants
    for name, gain in TSM_GAINS.items():
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, TSM gain={gain})...", end="")
                sys.stdout.flush()
                path, err = gen_one("sinkhorn_tsm", TSM_CKPT, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    tsm_gain=gain)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    # Progressive λ step-gated variants: λ(p) = lam_min → lam across song time
    for name, params in STEP_GATED_PROGRESSIVE_LAMBDAS.items():
        lam, lam_min, s_start, s_end = params
        for lang in ("zh", "en"):
            out_dir = EVAL_ROOT / name / f"audio_{lang}"
            os.makedirs(out_dir, exist_ok=True)
            for fidx in FILE_INDEXES[lang]:
                entry = fidx if lang == "zh" else fidx + 50
                outf = entry
                wav = out_dir / f"{outf:06d}.flac"
                if wav.exists() and wav.stat().st_size > 1000:
                    print(f"  [SKIP] {name}/{lang} {fidx}")
                    continue
                style, lyrics = parse_entry(entry)
                dur = get_duration(entry)
                print(f"  [GEN]  {name}/{lang} {fidx} ({dur:.0f}s, prog λ={lam_min}→{lam} q∈[{s_start},{s_end}])...", end="")
                sys.stdout.flush()
                path, err = gen_one("baseline", None, style, lyrics, out_dir,
                                    SEED + fidx, dur, outf,
                                    reallocation_lambda=lam,
                                    step_gated_reallocation=True,
                                    step_gated_start=s_start,
                                    step_gated_end=s_end,
                                    reallocation_lam_min=lam_min)
                if path:
                    print(f" OK")
                else:
                    print(f" FAIL: {err}")

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min")
    all_ms = (list(METHODS.keys()) + list(REALLOCATION_LAMBDAS.keys())
              + list(LATE_ONLY_LAMBDAS.keys()) + list(TSM_LATE_ONLY_LAMBDAS.keys())
              + list(TSM_REALLOCATION_LAMBDAS.keys())
              + list(STEP_GATED_LAMBDAS.keys())
              + list(TSM_STEP_GATED_LAMBDAS.keys())
              + list(TSM_GAINS.keys())
              + list(STEP_GATED_PROGRESSIVE_LAMBDAS.keys()))
    for m in all_ms:
        for lang in ("zh", "en"):
            n = len(list((EVAL_ROOT / m / f"audio_{lang}").glob("*.flac")))
            print(f"  {m}/{lang}: {n} files")


# =========================================================================
#  Evaluation
# =========================================================================

def run_cmd(cmd, desc, timeout=600):
    print(f"  {desc}...", end="")
    sys.stdout.flush()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    ok = r.returncode == 0
    dt = time.time() - t0
    print(f" {'OK' if ok else 'FAIL'} ({dt:.0f}s)" if ok else f" FAIL ({r.stderr.decode()[-150:]})")
    return ok


def step_evaluate():
    print("=" * 60)
    print("Batch Evaluation")
    print("=" * 60)
    t0 = time.time()

    all_methods = (list(METHODS.keys()) + list(REALLOCATION_LAMBDAS.keys())
                   + list(LATE_ONLY_LAMBDAS.keys()) + list(TSM_LATE_ONLY_LAMBDAS.keys())
                   + list(TSM_REALLOCATION_LAMBDAS.keys())
              + list(STEP_GATED_LAMBDAS.keys())
              + list(TSM_STEP_GATED_LAMBDAS.keys())
              + list(TSM_GAINS.keys())
              + list(STEP_GATED_PROGRESSIVE_LAMBDAS.keys()))
    for method in all_methods:
        print(f"\n--- {method} ---")
        results_dir = EVAL_ROOT / method / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for lang, lang_code in [("zh", "zh"), ("en", "en")]:
            audio_dir = EVAL_ROOT / method / f"audio_{lang}"
            if not audio_dir.exists() or not list(audio_dir.glob("*.flac")):
                continue
            model_name = f"{method}_{lang}"

            # ---- 1. SongEval ----
            run_cmd([
                PYTHON, str(PIPELINE_DIR / "eval_songeval.py"),
                "--input_dir", str(audio_dir),
                "--model_name", model_name,
                "--output", str(results_dir / f"songeval_{model_name}.json"),
                "--ckpt", str(SONGEVAL_CKPT),
                "--config", str(SONGEVAL_CONFIG),
                "--gpu", "0",
            ], f"SongEval [{model_name}]")

            # ---- 2. AudioBox Aesthetics ----
            run_cmd([
                PYTHON, str(PIPELINE_DIR / "eval_audiobox.py"),
                "--input_dir", str(audio_dir),
                "--model_name", model_name,
                "--output", str(results_dir / f"audiobox_{model_name}.json"),
                "--ckpt", str(AUDIOBOX_CKPT),
            ], f"AudioBox [{model_name}]", timeout=120)

            # ---- 3. ASR via Qwen3 (local) ----
            trans_file = EVAL_ROOT / method / f"transcriptions_{lang}.jsonl"
            ok = run_cmd([
                PYTHON, str(PIPELINE_DIR / "transcribe_local.py"),
                "--input_dir", str(audio_dir),
                "--output", str(trans_file),
                "--model_path", str(QWEN_MODEL_PATH),
            ], f"ASR [{model_name}]", timeout=1800)
            if not ok:
                continue

            # ---- 4. PER (long-form: overall + early/middle/late + LDG) ----
            gt_file = GT_LYRICS_DIR / f"{lang_code}.jsonl"
            per_out = EVAL_ROOT / method / f"per_results_{lang}"
            cmd = [
                PYTHON, str(PIPELINE_DIR / "calc_per_long.py"),
                "--hyp_file", str(trans_file),
                "--gt_file", str(gt_file),
                "--model_name", model_name,
                "--output_dir", str(per_out),
            ]
            if lang_code == "en":
                cmd += ["--offset", "50"]
            run_cmd(cmd, f"PER [{model_name}]")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    all_methods = (list(METHODS.keys()) + list(REALLOCATION_LAMBDAS.keys())
                   + list(LATE_ONLY_LAMBDAS.keys()) + list(TSM_LATE_ONLY_LAMBDAS.keys())
                   + list(TSM_REALLOCATION_LAMBDAS.keys())
              + list(STEP_GATED_LAMBDAS.keys())
              + list(TSM_STEP_GATED_LAMBDAS.keys())
              + list(TSM_GAINS.keys())
              + list(STEP_GATED_PROGRESSIVE_LAMBDAS.keys()))
    for method in all_methods:
        print(f"\n[{method}]")
        for lang in ["zh", "en"]:
            csv = EVAL_ROOT / method / f"per_results_{lang}" / "summary.csv"
            if csv.exists():
                for line in open(csv):
                    line = line.strip()
                    if line and not line.startswith("model"):
                        print(f"  {line}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="TSM Batch Eval")
    parser.add_argument("mode", choices=["generate", "evaluate", "all"],
                       help="generate=只生成, evaluate=只评估, all=全部")
    args = parser.parse_args()
    if args.mode in ("generate", "all"):
        step_generate()
    if args.mode in ("evaluate", "all"):
        step_evaluate()

if __name__ == "__main__":
    main()
