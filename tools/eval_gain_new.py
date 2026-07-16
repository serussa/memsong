#!/root/miniconda3/envs/musicgen/bin/python
"""Evaluate TSM gain=1 vs gain=8 on new samples only."""
import json, os, subprocess, sys, time
from pathlib import Path

EVAL_ROOT = Path("/root/autodl-tmp/tsm_eval_results")
PIPELINE_DIR = Path("/root/ACE-Step-1.5/Muse/eval_pipeline")
SONGEVAL_CKPT = Path("/root/ACE-Step-1.5/SongEval/ckpt/model.safetensors")
SONGEVAL_CONFIG = Path("/root/ACE-Step-1.5/SongEval/config.yaml")
AUDIOBOX_CKPT = Path("/root/autodl-tmp/models/audiobox-aesthetics/checkpoint.pt")
QWEN_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-ASR-1.7B")
GT_LYRICS_DIR = PIPELINE_DIR / "gt_lyrics"
PYTHON = sys.executable

METHODS = ["tsm_gain1_new", "tsm_gain8_new"]

def run_cmd(cmd, desc, timeout=600):
    print(f"  {desc}...", end=""); sys.stdout.flush()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    ok = r.returncode == 0; dt = time.time() - t0
    print(f" {'OK' if ok else 'FAIL'} ({dt:.0f}s)" if ok else f" FAIL ({r.stderr.decode()[-300:]})")
    return ok

for method in METHODS:
    print(f"\n=== {method} ===")
    results_dir = EVAL_ROOT / method / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for lang, lang_code in [("zh", "zh"), ("en", "en")]:
        audio_dir = EVAL_ROOT / method / f"audio_{lang}"
        if not audio_dir.exists() or not list(audio_dir.glob("*.flac")):
            continue
        model_name = f"{method}_{lang}"

        run_cmd([PYTHON, str(PIPELINE_DIR / "eval_songeval.py"),
            "--input_dir", str(audio_dir), "--model_name", model_name,
            "--output", str(results_dir / f"songeval_{model_name}.json"),
            "--ckpt", str(SONGEVAL_CKPT), "--config", str(SONGEVAL_CONFIG), "--gpu", "0"],
            f"SongEval [{model_name}]")

        run_cmd([PYTHON, str(PIPELINE_DIR / "eval_audiobox.py"),
            "--input_dir", str(audio_dir), "--model_name", model_name,
            "--output", str(results_dir / f"audiobox_{model_name}.json"),
            "--ckpt", str(AUDIOBOX_CKPT)], f"AudioBox [{model_name}]", timeout=120)

        trans_file = EVAL_ROOT / method / f"transcriptions_{lang}.jsonl"
        ok = run_cmd([PYTHON, str(PIPELINE_DIR / "transcribe_local.py"),
            "--input_dir", str(audio_dir), "--output", str(trans_file),
            "--model_path", str(QWEN_MODEL_PATH)], f"ASR [{model_name}]", timeout=1800)
        if not ok:
            continue

        gt_file = GT_LYRICS_DIR / f"{lang_code}.jsonl"
        per_out = EVAL_ROOT / method / f"per_results_{lang}"
        cmd = [PYTHON, str(PIPELINE_DIR / "calc_per_long.py"),
            "--hyp_file", str(trans_file), "--gt_file", str(gt_file),
            "--model_name", model_name, "--output_dir", str(per_out)]
        if lang_code == "en":
            cmd += ["--offset", "50"]
        run_cmd(cmd, f"PER [{model_name}]")

print("\nDone.")
