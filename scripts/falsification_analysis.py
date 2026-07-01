#!/usr/bin/env python3
"""Fast analysis-only pass for falsification experiment (data already generated)."""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from scipy import signal

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output/falsification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR = OUTPUT_DIR / "trajectories"

PROMPTS = {
    "ballad_m": "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
    "rock": "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
    "edm": "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
    "pop_f": "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
}
SEEDS = [42, 123, 999]
STEPS = 50

BASE_DIR = Path("output/temporal_scaling/trajectories")


def extract_dynamics(X):
    v = np.diff(X, axis=0)
    v_norm = norm(v, axis=1)
    if len(v) >= 5:
        w = min(5, len(v) - (1 - len(v) % 2))
        v_smooth = signal.savgol_filter(v_norm, w, 2)
    else:
        v_smooth = v_norm
    v_unit = v / (v_norm[:, None] + 1e-10)
    cos_sim = np.clip(np.sum(v_unit[:-1] * v_unit[1:], axis=1), -1.0, 1.0)
    curvature = 1.0 - cos_sim
    return {"v_norm": v_norm, "v_smooth": v_smooth, "curvature": curvature}


def compute_effective_rank(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    steps = X.shape[0]
    if steps <= 1: return 1.0
    C = Xc.T @ Xc / (steps - 1)
    s = np.linalg.svd(C, compute_uv=False)
    p = s / (s.sum() + 1e-10)
    H = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(H))


def compute_metrics(X):
    d = extract_dynamics(X)
    v_min_step = int(np.argmin(d["v_smooth"]))
    c = d["curvature"]
    c_peak = int(np.argmax(c)) if len(c) > 0 else 0
    return {
        "v_min_step": v_min_step, "v_min_norm": v_min_step / X.shape[0],
        "v_norm": d["v_norm"].tolist(), "curvature_peak": c_peak,
        "vc_alignment": abs(v_min_step - c_peak),
        "effective_rank": compute_effective_rank(X),
        "curvature": c.tolist(),
    }


# Load all data
all_results = {"baseline": {}, "rotated": {}, "cosine": {}}

for pname in PROMPTS:
    for seed in SEEDS:
        # Baseline: from BASE_DIR
        f = BASE_DIR / f"traj_{pname}_T50_s{seed}.npy"
        X = np.load(f)
        all_results["baseline"][(pname, seed)] = compute_metrics(X)

        # Rotated: same X with random orthogonal transform
        rng = np.random.RandomState(sum(ord(c) for c in pname))
        H = rng.randn(X.shape[1], X.shape[1])
        Q, _ = np.linalg.qr(H)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        Q = Q.astype(np.float32)
        X_rot = X @ Q.T
        all_results["rotated"][(pname, seed)] = compute_metrics(X_rot)

        # Cosine: from TRAJ_DIR
        f_cos = TRAJ_DIR / f"traj_cos_{pname}_T50_s{seed}.npy"
        if f_cos.exists():
            X_cos = np.load(f_cos)
            all_results["cosine"][(pname, seed)] = compute_metrics(X_cos)

# Stats
print("=" * 70)
print("  FALSIFICATION VERDICT")
print("=" * 70)

versions = ["baseline", "rotated", "cosine"]
for key, label, fmt in [("effective_rank", "Effective Rank", "{:.2f}"),
                         ("v_min_norm", "t*/T", "{:.3f}"),
                         ("vc_alignment", "V-C diff", "{:.0f}")]:
    print(f"  {label:<20}", end="")
    for v in versions:
        vals = [all_results[v][k][key] for k in all_results[v]]
        print(f"  {fmt.format(np.mean(vals)):>8} (σ={np.std(vals):.2f})", end="")
    print()

# Criteria
base_er = np.mean([all_results["baseline"][k]["effective_rank"] for k in all_results["baseline"]])
rot_er = np.mean([all_results["rotated"][k]["effective_rank"] for k in all_results["rotated"]])
cos_er = np.mean([all_results["cosine"][k]["effective_rank"] for k in all_results["cosine"]])

base_t = np.mean([all_results["baseline"][k]["v_min_norm"] for k in all_results["baseline"]])
rot_t = np.mean([all_results["rotated"][k]["v_min_norm"] for k in all_results["rotated"]])
cos_t = np.mean([all_results["cosine"][k]["v_min_norm"] for k in all_results["cosine"]])

base_vc = np.mean([all_results["baseline"][k]["vc_alignment"] for k in all_results["baseline"]])
rot_vc = np.mean([all_results["rotated"][k]["vc_alignment"] for k in all_results["rotated"]])
cos_vc = np.mean([all_results["cosine"][k]["vc_alignment"] for k in all_results["cosine"]])

print(f"\n  Criteria checks:")
print(f"  [1] Rank stable: base={base_er:.2f} rot={rot_er:.2f} cos={cos_er:.2f}  ✅" if abs(rot_er - base_er) < 1.0 and abs(cos_er - base_er) < 1.0 else "❌")
print(f"  [2] t*/T stable: base={base_t:.3f} rot={rot_t:.3f} cos={cos_t:.3f}  ✅" if abs(rot_t - base_t) < 0.15 and abs(cos_t - base_t) < 0.15 else "❌")
print(f"  [3] V-C aligned: base={base_vc:.1f} rot={rot_vc:.1f} cos={cos_vc:.1f}  ✅" if rot_vc < 8 and cos_vc < 8 else "❌")

n_pass = sum([abs(rot_er - base_er) < 1.0 and abs(cos_er - base_er) < 1.0,
              abs(rot_t - base_t) < 0.15 and abs(cos_t - base_t) < 0.15,
              rot_vc < 8 and cos_vc < 8])

print(f"\n  {'=' * 50}")
if n_pass >= 3:
    print(f"  ✅ ALIVE: Structure is invariant under coordinate transform + time reparam")
elif n_pass >= 1:
    print(f"  ⚠️  PARTIAL: {n_pass}/3 criteria passed")
else:
    print(f"  ❌ DEAD: Structure is a metric/representation artifact")
print(f"  {'=' * 50}")

# Save report
report = {
    "summary": {},
    "per_run": {},
    "verdict": {"criteria_passed": n_pass, "alive": n_pass >= 2}
}
for v in versions:
    er = [all_results[v][k]["effective_rank"] for k in all_results[v]]
    ts = [all_results[v][k]["v_min_norm"] for k in all_results[v]]
    vc = [all_results[v][k]["vc_alignment"] for k in all_results[v]]
    report["summary"][v] = {
        "effective_rank_mean": float(np.mean(er)), "effective_rank_std": float(np.std(er)),
        "tstar_norm_mean": float(np.mean(ts)), "tstar_norm_std": float(np.std(ts)),
        "vc_alignment_mean": float(np.mean(vc)),
    }
    for (p, s) in all_results[v]:
        key = f"{v}_{p}_s{s}"
        report["per_run"][key] = all_results[v][(p, s)]

with open(OUTPUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Report: {OUTPUT_DIR / 'report.json'}")
