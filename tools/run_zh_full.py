#!/root/miniconda3/envs/musicgen/bin/python
"""Run full 50-song Chinese evaluation for baseline, transport_only, sinkhorn_tsm."""
import argparse, json, os, pickle, subprocess, sys, time, re
from pathlib import Path

PROJECT_ROOT = Path("/root/ACE-Step-1.5")
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

METHODS = {
    "baseline": None,
    "transport_only": TRANSPORT_CKPT,
    "sinkhorn_tsm": TSM_CKPT,
}
PYTHON = sys.executable
SEED = 42

def get_duration(entry_idx):
    with open(TEST_JSONL) as f:
        for i, line in enumerate(f):
            if i == entry_idx:
                data = json.loads(line); break
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
    has_cjk = any('一' <= ch <= '鿿' for ch in text)
    if has_cjk:
        return max(20, int(len(re.sub(r'\s', '', text)) * 0.45))
    else:
        return max(20, int(len(text.split()) * 0.55))

def parse_entry(entry_idx):
    with open(TEST_JSONL) as f:
        for i, line in enumerate(f):
            if i == entry_idx:
                data = json.loads(line); break
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
    lyrics = "\n".join(parts).strip()
    return style, lyrics

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
                new_name = f"{fidx:06d}.flac"
                new_path = os.path.join(out_dir, new_name)
                if os.path.abspath(gp) != os.path.abspath(new_path):
                    os.rename(gp, new_path)
                return new_path, None
            return gp, None
    return None, "no OK: in output"

def run_cmd(cmd, desc, timeout=600):
    print(f"  {desc}...", end=""); sys.stdout.flush()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    ok = r.returncode == 0; dt = time.time() - t0
    print(f" {'OK' if ok else 'FAIL'} ({dt:.0f}s)" if ok else f" FAIL ({r.stderr.decode()[-150:]})")
    return ok

def step_generate():
    print("=" * 60)
    print("Generate: 50 zh songs × 3 methods")
    print("=" * 60)
    t0 = time.time()
    for method, ckpt in METHODS.items():
        out_dir = EVAL_ROOT / method / "audio_zh"
        os.makedirs(out_dir, exist_ok=True)
        for fidx in range(50):
            wav = out_dir / f"{fidx:06d}.flac"
            if wav.exists() and wav.stat().st_size > 1000:
                print(f"  [SKIP] {method}/zh {fidx}")
                continue
            style, lyrics = parse_entry(fidx)
            dur = get_duration(fidx)
            print(f"  [GEN]  {method}/zh {fidx} ({dur:.0f}s)...", end="")
            sys.stdout.flush()
            path, err = gen_one(method, ckpt, style, lyrics, out_dir, SEED + fidx, dur, fidx)
            if path:
                print(f" OK")
            else:
                print(f" FAIL: {err}")
    dt = time.time() - t0
    print(f"\nGenerated in {dt/60:.1f} min")
    for m in METHODS:
        n = len(list((EVAL_ROOT / m / "audio_zh").glob("*.flac")))
        print(f"  {m}/zh: {n} files")

def step_evaluate():
    print("=" * 60)
    print("Evaluate: 50 zh songs × 3 methods")
    print("=" * 60)
    t0 = time.time()
    for method in METHODS:
        out_dir = EVAL_ROOT / method / "audio_zh"
        n_files = len(list(out_dir.glob("*.flac")))
        if n_files == 0:
            print(f"  [SKIP] {method} (no audio)")
            continue
        print(f"\n--- {method} ({n_files} files) ---")
        results_dir = EVAL_ROOT / method / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        run_cmd([PYTHON, str(PIPELINE_DIR / "eval_songeval.py"),
            "--input_dir", str(out_dir), "--model_name", f"{method}_zh",
            "--output", str(results_dir / f"songeval_{method}_zh.json"),
            "--ckpt", str(SONGEVAL_CKPT), "--config", str(SONGEVAL_CONFIG), "--gpu", "0"],
            f"SongEval [{method}]")

        run_cmd([PYTHON, str(PIPELINE_DIR / "eval_audiobox.py"),
            "--input_dir", str(out_dir), "--model_name", f"{method}_zh",
            "--output", str(results_dir / f"audiobox_{method}_zh.json"),
            "--ckpt", str(AUDIOBOX_CKPT)], f"AudioBox [{method}]", timeout=120)

        trans_file = EVAL_ROOT / method / "transcriptions_zh.jsonl"
        ok = run_cmd([PYTHON, str(PIPELINE_DIR / "transcribe_local.py"),
            "--input_dir", str(out_dir), "--output", str(trans_file),
            "--model_path", str(QWEN_MODEL_PATH)], f"ASR [{method}]", timeout=3600)
        if not ok:
            continue

        gt_file = GT_LYRICS_DIR / "zh.jsonl"
        per_out = EVAL_ROOT / method / "per_results_zh"
        run_cmd([PYTHON, str(PIPELINE_DIR / "calc_per_long.py"),
            "--hyp_file", str(trans_file), "--gt_file", str(gt_file),
            "--model_name", f"{method}_zh", "--output_dir", str(per_out)], f"PER [{method}]")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method in METHODS:
        csv = EVAL_ROOT / method / "per_results_zh" / "summary.csv"
        if csv.exists():
            print(f"\n[{method}]")
            for line in open(csv):
                line = line.strip()
                if line and not line.startswith("model"):
                    print(f"  {line}")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "evaluate", "all"])
    args = parser.parse_args()
    if args.mode in ("generate", "all"): step_generate()
    if args.mode in ("evaluate", "all"): step_evaluate()

if __name__ == "__main__":
    main()
