#!/usr/bin/env python3
"""Run SongEval + Qwen3-ASR PER on TSM smoke test outputs."""
import json, os, sys, re, subprocess
from pathlib import Path

ACE_ROOT = Path("/root/ACE-Step-1.5")
SONGEVAL_DIR = ACE_ROOT / "SongEval"
AUDIO_DIRS = {
    "baseline":              ACE_ROOT / "outputs/tsm_smoke_test_no_pm/step3_baseline",
    "sinkhorn_only_no_pm":   ACE_ROOT / "outputs/tsm_smoke_test_no_pm/step3_sinkhorn_only_no_pm",
    "sinkhorn_pool_broadcast": ACE_ROOT / "outputs/tsm_smoke_test_no_pm/step3_sinkhorn_pool_broadcast",
    "sinkhorn_tsm":          ACE_ROOT / "outputs/tsm_smoke_test_no_pm/step3_sinkhorn_tsm",
}
EVAL_OUT = ACE_ROOT / "outputs/tsm_smoke_test_no_pm/eval_results"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

GT_LYRICS = """The morning light breaks through the clouds
I hear your voice calling out loud
Every step I take toward you
Feels like the world is brand new
We rise up high above the sky
Together we can learn to fly
No looking back no fear no doubt
This is what love is all about
The stars align when you are near
You make the darkness disappear"""

PYTHON = "/root/miniconda3/envs/musicgen/bin/python"

results = {}

# =========================================================================
# 1. SongEval
# =========================================================================
print("=" * 60)
print("SONGEVAL")
print("=" * 60)

for name, d in AUDIO_DIRS.items():
    flacs = sorted(d.glob("*.flac"))
    if not flacs:
        print(f"  {name}: NO FLAC FOUND")
        continue
    audio = flacs[0]
    out_dir = EVAL_OUT / "songeval" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    res_file = out_dir / "result.json"
    if not res_file.exists():
        subprocess.run(
            [PYTHON, str(SONGEVAL_DIR / "eval.py"),
             "-i", str(audio), "-o", str(out_dir)],
            cwd=str(SONGEVAL_DIR), capture_output=True,
        )

    if res_file.exists():
        with open(res_file) as f:
            data = json.load(f)
        fid = list(data.keys())[0]
        s = data[fid]
        results[name] = {
            "Coherence": s["Coherence"], "Musicality": s["Musicality"],
            "Memorability": s["Memorability"], "Clarity": s["Clarity"],
            "Naturalness": s["Naturalness"],
        }
        print(f"  {name:<28} Co={s['Coherence']:.4f} Mu={s['Musicality']:.4f} "
              f"Me={s['Memorability']:.4f} Cl={s['Clarity']:.4f} Na={s['Naturalness']:.4f}")
    else:
        print(f"  {name}: SongEval FAILED")
        results[name] = {}

# =========================================================================
# 2. Qwen3-ASR Transcription → PER
# =========================================================================
print("\n" + "=" * 60)
print("PER (Qwen3-ASR → phoneme error rate)")
print("=" * 60)

# Prepare audio dir for transcription (all 4 files → flat dir with indices)
PER_TMP = EVAL_OUT / "per_tmp"
PER_TMP.mkdir(parents=True, exist_ok=True)

# Copy files with indices
per_files = {}
for idx, (name, d) in enumerate(AUDIO_DIRS.items()):
    flacs = sorted(d.glob("*.flac"))
    if not flacs:
        continue
    src = flacs[0]
    dst = PER_TMP / f"audio_{idx:04d}.flac"
    if not dst.exists():
        import shutil
        shutil.copy2(src, dst)
    per_files[idx] = name

# Transcribe
trans_jsonl = EVAL_OUT / "transcription.jsonl"
if not trans_jsonl.exists():
    subprocess.run(
        [PYTHON, str(ACE_ROOT / "Muse/eval_pipeline/transcribe_local.py"),
         "--input_dir", str(PER_TMP), "--output", str(trans_jsonl)],
        capture_output=False,
    )

# Build GT lyrics file
gt_jsonl = EVAL_OUT / "gt_lyrics.jsonl"
with open(gt_jsonl, 'w') as f:
    for idx in sorted(per_files.keys()):
        f.write(json.dumps({"file_index": idx, "lyrics": GT_LYRICS}, ensure_ascii=False) + '\n')

# Calculate PER
per_output = EVAL_OUT / "per_result.json"
subprocess.run(
    [PYTHON, str(ACE_ROOT / "Muse/eval_pipeline/calc_per.py"),
     "--hyp_file", str(trans_jsonl), "--gt_file", str(gt_jsonl),
     "--model_name", "tsm_smoke", "--output", str(per_output)],
    capture_output=True,
)

if per_output.exists():
    per_data = json.loads(per_output.read_text())
    print(f"  Overall PER: {per_data['metrics']['PER']:.4f} ({per_data['count']} samples)")

    # Per-file PER from details
    details_file = per_output.parent / (per_output.stem + "_details.jsonl")
    if details_file.exists():
        with open(details_file) as f:
            for line in f:
                d = json.loads(line)
                name = per_files.get(d["idx"], f"idx={d['idx']}")
                if name in results:
                    results[name]["PER"] = d["per"]
                print(f"  {name:<28} PER={d['per']:.4f}")

# =========================================================================
# 3. Summary Table
# =========================================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

metrics = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness", "PER"]
header = f"{'Mode':<28}" + "".join(f"{m:>12}" for m in metrics)
print(header)
print("-" * 90)

baseline_se = results.get("baseline", {})
for name in ["baseline", "sinkhorn_only", "sinkhorn_pool_broadcast", "sinkhorn_tsm"]:
    r = results.get(name, {})
    if not r:
        continue
    row = f"{name:<28}"
    for m in metrics:
        val = r.get(m, "-")
        if isinstance(val, float):
            # Delta vs baseline for non-PER metrics
            if m != "PER" and baseline_se:
                delta = val - baseline_se.get(m, val)
                row += f"{val:>9.4f}{'+' if delta>=0 else ''}{delta:.4f} "
            else:
                row += f"{val:>12.4f}"
        else:
            row += f"{'>12s'}"
    print(row)

# Save full results
summary_path = EVAL_OUT / "summary.json"
json.dump(results, summary_path, indent=2, default=str)
print(f"\nFull results: {summary_path}")
