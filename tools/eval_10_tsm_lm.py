#!/root/miniconda3/envs/musicgen/bin/python
"""Evaluate sinkhorn_tsm with LM on 10 Chinese songs."""
import json, os, re, sys, time, shutil, pickle, subprocess, soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path("/root/ACE-Step-1.5")
TEST_JSONL = PROJECT_ROOT / "Muse/infer/test.jsonl"
GEN_SCRIPT = PROJECT_ROOT / "gen_one.py"
PIPELINE_DIR = PROJECT_ROOT / "Muse/eval_pipeline"
QWEN_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-ASR-1.7B")
GT_LYRICS_DIR = PIPELINE_DIR / "gt_lyrics"
TSM_CKPT = Path("/root/autodl-tmp/tsm_sinkhorn/checkpoints/best_loss/pm_retrieval.pt")
OUT_ROOT = Path("/root/autodl-tmp/tsm_eval_results") / "sinkhorn_tsm_lm"
PYTHON = sys.executable
SEED = 42

def run_cmd(cmd, desc, timeout=600):
    print(f"  {desc}...", end="", flush=True)
    t0=time.time(); r=subprocess.run(cmd,capture_output=True,timeout=timeout)
    ok=r.returncode==0; dt=time.time()-t0
    print(f" {'OK' if ok else 'FAIL'} ({dt:.0f}s)" if ok else f" FAIL ({r.stderr.decode()[-150:]})")
    return ok

print("="*60)
print("Generating 10 songs with sinkhorn_tsm + LM (duration fixed)")
print("="*60)
t0=time.time()

for fidx in range(10):
    with open(TEST_JSONL) as f:
        for i, l in enumerate(f):
            if i == fidx: d = json.loads(l); break
    msgs = d["messages"]
    style = msgs[0]["content"].split("\n")[0].replace(
        "Please generate a song in the following style:", "").strip()
    seen, parts = set(), []
    for msg in msgs:
        c = msg.get("content", "")
        m = re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m: continue
        n = m.group(1)
        if n in seen: continue
        seen.add(n)
        rest = c[m.end():]; l = ""
        if "[lyrics:" in rest: l = rest.split("[lyrics:")[1].split("]")[0].strip()
        parts.append(m.group(0))
        if l: parts.append(l)
        parts.append("")
    lyrics = "\n".join(parts).strip()

    audio_dir = OUT_ROOT / "audio_zh"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / f"{fidx:06d}.flac"

    if wav.exists() and wav.stat().st_size > 1000:
        print(f"  [SKIP] {fidx}")
        continue

    print(f"  [GEN] {fidx}...", end="", flush=True)
    d = dict(method="sinkhorn_tsm", ckpt=str(TSM_CKPT), caption=style, lyrics=lyrics,
             out_dir=str(audio_dir), seed=SEED+fidx, duration=180)
    r = subprocess.run([PYTHON, str(GEN_SCRIPT)], input=pickle.dumps(d),
                       capture_output=True, timeout=600)
    if r.returncode != 0:
        print(f" FAIL: {r.stderr.decode()[-200:]}")
        continue
    path = None
    for line in r.stdout.decode().split("\n"):
        if line.startswith("OK:"):
            path = line[3:].strip()
            break
    if path and os.path.exists(path):
        if os.path.abspath(path) != os.path.abspath(wav):
            shutil.move(path, wav)
        data, sr = sf.read(wav)
        print(f" OK ({len(data)/sr:.0f}s)")
    else:
        print(f" FAIL: no audio")

print(f"\nGenerated in {(time.time()-t0)/60:.1f} min")

# Evaluate
print("\n" + "="*60)
print("Evaluating")
print("="*60)
t0=time.time()

audio_dir = OUT_ROOT / "audio_zh"
results_dir = OUT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

run_cmd([PYTHON, str(PIPELINE_DIR / "eval_songeval.py"),
    "--input_dir", str(audio_dir), "--model_name", "sinkhorn_tsm_lm",
    "--output", str(results_dir / "songeval_sinkhorn_tsm_lm_zh.json"),
    "--ckpt", str(Path("/root/ACE-Step-1.5/SongEval/ckpt/model.safetensors")),
    "--config", str(Path("/root/ACE-Step-1.5/SongEval/config.yaml")), "--gpu", "0"],
    "SongEval")

trans_file = OUT_ROOT / "transcriptions_zh.jsonl"
run_cmd([PYTHON, str(PIPELINE_DIR / "transcribe_local.py"),
    "--input_dir", str(audio_dir), "--output", str(trans_file),
    "--model_path", str(QWEN_MODEL_PATH)], "ASR", timeout=3600)

gt_file = GT_LYRICS_DIR / "zh.jsonl"
per_out = OUT_ROOT / "per_results_zh"
run_cmd([PYTHON, str(PIPELINE_DIR / "calc_per_long.py"),
    "--hyp_file", str(trans_file), "--gt_file", str(gt_file),
    "--model_name", "sinkhorn_tsm_lm", "--output_dir", str(per_out)], "PER")

print("\n" + "="*60)
for line in open(per_out / "summary.csv"):
    print(line.strip())
print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
