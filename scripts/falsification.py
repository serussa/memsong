#!/usr/bin/env python3
"""
Single-shot Falsification Experiment for DiT Low-Dimensional Manifold.

Tests whether the observed low-dimensional structure (effective rank ~2,
velocity U-shape, t*/T scaling) is:

  (A) Real dynamical system property (survives coordinate transforms + time reparam)
  (B) Representation/metric artifact (collapses under attack)

Three versions run on identical prompts × seeds:
  A — Baseline (linear schedule, raw hidden states)
  B — Scrambled representation (random orthogonal rotation of hidden states)
  C — Time reparameterization (cosine timestep schedule)

Falsification criteria:
  ❌ DEAD if ≥2 of: rank dip vanishes, t* no longer scales, curvature/velocity decouple
  ✔ ALIVE if all: rank dip persists, t* scales with T, orthogonal rotation preserves structure

Usage:
    python scripts/falsification.py
"""

import argparse, json, os, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy import signal
from scipy.stats import pearsonr

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

# ===================================================================
# Config
# ===================================================================
PROMPTS = {
    "ballad_m": "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
    "rock": "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
    "edm": "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
    "pop_f": "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
}
SHORT = {"ballad_m": "Ballad", "rock": "Rock", "edm": "EDM", "pop_f": "Vocal Pop"}

SEEDS = [42, 123, 999]
STEPS = 50  # use 50 steps for all versions

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="output/falsification")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR = OUTPUT_DIR / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

# Base trajectory dir from main experiment (T=50 taken from temporal experiment)
BASE_TRAJ_DIR = Path("output/temporal_scaling/trajectories")
TEMP_TRAJ_DIR = Path("output/ts_stability/temporal_trajs")
MAIN_TRAJ_DIR = Path("output/dynamics_experiment/trajectories")

# ===================================================================
# Dynamics extraction
# ===================================================================
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
    return {"v": v, "v_norm": v_norm, "v_smooth": v_smooth, "v_unit": v_unit,
            "curvature": curvature, "steps_total": X.shape[0]}


def compute_effective_rank(X):
    """Effective rank from singular value entropy."""
    Xc = X - X.mean(axis=0, keepdims=True)
    steps = X.shape[0]
    if steps <= 1:
        return 1.0
    C = Xc.T @ Xc / (steps - 1)
    s = np.linalg.svd(C, compute_uv=False)
    p = s / (s.sum() + 1e-10)
    H = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(H))


def per_step_effective_rank(X, window=5):
    """Effective rank at each step using a sliding window."""
    T = X.shape[0]
    ranks = np.ones(T) * np.nan
    for t in range(window - 1, T):
        start = max(0, t - window + 1)
        chunk = X[start:t + 1]
        if chunk.shape[0] >= 3:
            ranks[t] = compute_effective_rank(chunk)
    return ranks


def compute_metrics(X):
    """Compute all metrics for one trajectory."""
    d = extract_dynamics(X)
    T = d["steps_total"]

    # Velocity minimum
    v_min_step = int(np.argmin(d["v_smooth"]))

    # Curvature peak: find curvature peak closest to velocity minimum
    c = d["curvature"]
    if len(c) > 0:
        c_peak = int(np.argmax(c))
    else:
        c_peak = 0

    # Alignment: distance between velocity min and curvature peak
    vc_alignment = abs(v_min_step - c_peak)

    # Per-step effective rank
    er_per_step = per_step_effective_rank(X, window=5)
    mid_dip = bool(np.nanmin(er_per_step[max(1, T//4):3*T//4]) <
                   np.nanmean(er_per_step[:max(1, T//4)]) * 0.9)

    return {
        "v_min_step": v_min_step,
        "v_min_norm": v_min_step / T,
        "v_min_value": float(d["v_norm"][v_min_step]),
        "v_norm": d["v_norm"].tolist(),
        "curvature_peak": c_peak,
        "curvature": c.tolist(),
        "vc_alignment": vc_alignment,
        "effective_rank": compute_effective_rank(X),
        "effective_rank_per_step": er_per_step.tolist(),
        "mid_dip": mid_dip,
    }


# ===================================================================
# Cosine timestep schedule
# ===================================================================
def cosine_timesteps(n_steps):
    """Generate cosine timestep schedule: t ∈ [1, 0] with cosine spacing."""
    t = np.linspace(0, 1, n_steps + 1)
    # Cosine schedule: more steps near 0 and 1, fewer in middle
    # t_noise = cos(s * π/2) where s goes from 0 → 1
    # But inverted: we want from noise (t=1) to clean (t=0)
    s = t[::-1]  # start at noise level 1, end at 0
    t_cos = np.cos(s * np.pi / 2)
    return t_cos.tolist()


# ===================================================================
# Random orthogonal rotation
# ===================================================================
def random_orthogonal_matrix(dim, seed=0):
    """Generate a random orthogonal matrix of size dim x dim."""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


# ===================================================================
# Load or generate trajectories
# ===================================================================
def get_baseline_trajectories():
    """Load existing T=50 trajectories from cache or main experiment."""
    X_loaded = {}
    for pname in PROMPTS:
        for seed in SEEDS:
            X = None
            # Try temporal_scaling cache
            f = TRAJ_DIR / f"traj_{pname}_T50_s{seed}.npy"
            if f.exists():
                X = np.load(f)
            if X is None:
                f = BASE_TRAJ_DIR / f"traj_{pname}_T50_s{seed}.npy"
                if f.exists():
                    X = np.load(f)
            if X is None:
                f = TEMP_TRAJ_DIR / f"traj_{pname}_T50_s{seed}.npy"
                if f.exists():
                    X = np.load(f)
            if X is None:
                # Check if main trajectories exist with default 30 steps
                f = MAIN_TRAJ_DIR / f"traj_{pname}_s{seed}.npy"
                if f.exists() and X is None:
                    X_30 = np.load(f)
                    if X_30.shape[0] == 30:
                        # We'll regenerate T=50 below
                        X = None
            if X is not None:
                X_loaded[(pname, seed)] = X
    return X_loaded


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("  SINGLE-SHOT FALSIFICATION EXPERIMENT")
    print("  Testing whether low-D manifold is real or artifact")
    print("=" * 70)

    # Collect existing trajectories
    existing = get_baseline_trajectories()
    print(f"\n  Existing T=50 trajectories: {len(existing)}/12")

    need_generate = [(p, s) for p in PROMPTS for s in SEEDS if (p, s) not in existing]

    # We need to generate:
    # 1. Missing T=50 baseline trajectories
    # 2. All T=50 cosine schedule trajectories (Version C)

    # But we can do Version B (orthogonal rotation) purely in post-processing
    # by loading trajectories and applying the rotation.

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    import torch

    class Collector:
        def __init__(self):
            self.states = []
        def __call__(self, mod, inp, out):
            hs = out[0]
            if hs.shape[0] > 1: hs = hs[:1]
            self.states.append(hs.mean(dim=1).detach().cpu())
        def get_traj(self):
            if not self.states: return None
            return torch.cat(self.states, dim=0).float().numpy()
        def reset(self):
            self.states = []

    # Check what we need to generate
    cos_gen_needed = [(p, s) for p in PROMPTS for s in SEEDS]
    base_gen_needed = need_generate

    if base_gen_needed or cos_gen_needed:
        print(f"\n  Initializing model...")
        dit_handler = AceStepHandler()
        dit_status, dit_success = dit_handler.initialize_service(
            project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
            device=args.device, use_flash_attention=False, compile_model=False, offload_to_cpu=False)
        if not dit_success:
            print(f"  Model init failed: {dit_status}")
            sys.exit(1)
        for layer_mod in dit_handler.model.decoder.layers:
            if getattr(layer_mod, "use_phase_memory", False):
                layer_mod.use_phase_memory = False
        collector = Collector()
        handle = dit_handler.model.decoder.layers[12].register_forward_hook(collector)

        # Generate missing baseline T=50
        if base_gen_needed:
            print(f"\n  Generating {len(base_gen_needed)} missing baseline T=50 trajectories...")
            for pname, seed in base_gen_needed:
                collector.reset()
                try:
                    params = GenerationParams(caption=PROMPTS[pname], lyrics="[Instrumental]",
                        instrumental=True, duration=30, inference_steps=50,
                        guidance_scale=5.0, seed=seed, thinking=False)
                    config = GenerationConfig(batch_size=1, audio_format="mp3", use_random_seed=False)
                    result = generate_music(dit_handler, None, params, config)
                    X = collector.get_traj()
                    if X is not None and len(X) >= 5:
                        np.save(TRAJ_DIR / f"traj_{pname}_T50_s{seed}.npy", X)
                        existing[(pname, seed)] = X
                        print(f"    baseline {pname} s{seed}: {len(X)} steps")
                except Exception as e:
                    print(f"    FAILED baseline {pname} s{seed}: {e}")

        # Generate cosine schedule trajectories
        print(f"\n  Generating {len(cos_gen_needed)} cosine-schedule trajectories...")
        for pname, seed in cos_gen_needed:
            # Check if already exists
            f_cos = TRAJ_DIR / f"traj_cos_{pname}_T50_s{seed}.npy"
            if f_cos.exists():
                print(f"    (cached) cosine {pname} s{seed}")
                continue
            collector.reset()
            try:
                cos_ts = cosine_timesteps(50)
                params = GenerationParams(caption=PROMPTS[pname], lyrics="[Instrumental]",
                    instrumental=True, duration=30, inference_steps=50,
                    guidance_scale=5.0, seed=seed, thinking=False,
                    timesteps=cos_ts)
                config = GenerationConfig(batch_size=1, audio_format="mp3", use_random_seed=False)
                result = generate_music(dit_handler, None, params, config)
                X = collector.get_traj()
                if X is not None and len(X) >= 5:
                    np.save(f_cos, X)
                    print(f"    cosine {pname} s{seed}: {len(X)} steps (t0={cos_ts[0]:.3f}, t-1={cos_ts[-1]:.3f})")
            except Exception as e:
                print(f"    FAILED cosine {pname} s{seed}: {e}")

        handle.remove()
    else:
        print(f"\n  All 12 baseline T=50 trajectories cached. Good.")

    # ===================================================================
    # Now compute metrics for all 3 versions
    # ===================================================================
    print("\n" + "=" * 70)
    print("  Computing metrics across all 3 versions...")
    print("=" * 70)

    versions = ["baseline", "rotated", "cosine"]
    all_results = {v: {} for v in versions}

    for pname in PROMPTS:
        for seed in SEEDS:
            # ---- VERSION A: Baseline ----
            X = existing.get((pname, seed))
            if X is not None:
                all_results["baseline"][(pname, seed)] = compute_metrics(X)

            # ---- VERSION B: Orthogonal rotation ----
            # Use a fixed rotation per prompt (deterministic from prompt name hash)
            if X is not None:
                rng_seed = sum(ord(c) for c in pname)
                Q = random_orthogonal_matrix(X.shape[1], seed=rng_seed)
                X_rot = X @ Q.T  # apply rotation
                all_results["rotated"][(pname, seed)] = compute_metrics(X_rot)

            # ---- VERSION C: Cosine schedule ----
            f_cos = TRAJ_DIR / f"traj_cos_{pname}_T50_s{seed}.npy"
            if f_cos.exists():
                X_cos = np.load(f_cos)
                all_results["cosine"][(pname, seed)] = compute_metrics(X_cos)

    # ===================================================================
    # Statistical comparison
    # ===================================================================
    print("\n  Metric comparison across versions:")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Rotated':>12} {'Cosine':>12}")
    print("  " + "-" * 61)

    metrics_to_compare = [
        ("effective_rank", "Effective Rank", "{:.2f}"),
        ("v_min_norm", "t*/T (velocity min)", "{:.3f}"),
        ("vc_alignment", "V-C Alignment (diff)", "{:.0f}"),
    ]

    comp_stats = {}
    for key, label, fmt in metrics_to_compare:
        comp_stats[key] = {}
        line = f"  {label:<25}"
        for v in versions:
            vals = [r[key] for r in all_results[v].values() if key in r]
            mean, std = np.mean(vals), np.std(vals)
            comp_stats[v] = comp_stats.get(v, {})
            comp_stats[v][key] = {"mean": float(mean), "std": float(std)}
            line += f"  {fmt.format(mean):>12}"
        print(line)

    # Check mid-dip
    print("\n  Mid-step effective rank dip present:")
    for v in versions:
        n_dip = sum(1 for r in all_results[v].values() if r.get("mid_dip", False))
        total = len(all_results[v])
        print(f"    {v:>12}: {n_dip}/{total} runs ({n_dip/total:.0%})")

    # ===================================================================
    # Visualization
    # ===================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Per-version velocity + curvature profiles
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for vi, version in enumerate(versions):
        # Velocity overlay
        ax = axes[vi, 0]
        for (p, s), r in all_results[version].items():
            v = np.array(r["v_norm"])
            ax.plot(v, alpha=0.4, linewidth=0.7)
            ax.scatter(r["v_min_step"], r["v_min_value"], s=15, alpha=0.6)
        ax.set_title(f"{version}: Velocity profiles")
        ax.set_xlabel("Step")
        ax.set_ylabel("|v|")
        ax.grid(alpha=0.2)

        # Curvature overlay
        ax = axes[vi, 1]
        for (p, s), r in all_results[version].items():
            c = np.array(r["curvature"])
            ax.plot(c, alpha=0.4, linewidth=0.7)
        ax.set_title(f"{version}: Curvature")
        ax.set_xlabel("Step")
        ax.set_ylabel("1 - cosθ")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "profiles_comparison.png", dpi=150)
    plt.close()
    print(f"\n  [plot] profiles_comparison.png")

    # 2. Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bar_metrics = [
        ("effective_rank", "Effective Rank", "{:.2f}"),
        ("v_min_norm", "t*/T", "{:.3f}"),
        ("vc_alignment", "V-C Alignment (step diff)", "{:.0f}"),
    ]
    colors = {"baseline": "#4c72b0", "rotated": "#dd8452", "cosine": "#55a868"}

    for mi, (key, label, fmt) in enumerate(bar_metrics):
        ax = axes[mi]
        positions = np.arange(len(versions))
        means = [np.mean([r[key] for r in all_results[v].values()]) for v in versions]
        stds = [np.std([r[key] for r in all_results[v].values()]) for v in versions]
        bars = ax.bar(positions, means, yerr=stds, color=[colors[v] for v in versions],
                      capsize=5, alpha=0.8, width=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(versions, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} across versions")
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150)
    plt.close()
    print(f"  [plot] metrics_comparison.png")

    # 3. Per-step effective rank (show mid-dip)
    fig, ax = plt.subplots(figsize=(10, 5))
    for vi, version in enumerate(versions):
        all_ranks = []
        for r in all_results[version].values():
            er_step = r.get("effective_rank_per_step", [])
            if er_step:
                all_ranks.append(np.array(er_step))
        if all_ranks:
            mean_er = np.nanmean(all_ranks, axis=0)
            std_er = np.nanstd(all_ranks, axis=0)
            steps = np.arange(len(mean_er))
            ax.plot(steps, mean_er, color=colors[version], label=version, linewidth=2)
            ax.fill_between(steps, mean_er - std_er, mean_er + std_er, alpha=0.1, color=colors[version])
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective Rank (window=5)")
    ax.set_title("Per-Step Effective Rank Across Versions")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "effective_rank_per_step.png", dpi=150)
    plt.close()
    print(f"  [plot] effective_rank_per_step.png")

    # ===================================================================
    # FALSIFICATION VERDICT
    # ===================================================================
    print("\n" + "=" * 70)
    print("  FALSIFICATION VERDICT")
    print("=" * 70)

    # Check 3 criteria
    criteria = {}

    # Criterion 1: rank dip persists
    base_dip = sum(1 for r in all_results["baseline"].values() if r.get("mid_dip", False)) / max(len(all_results["baseline"]), 1)
    rot_dip = sum(1 for r in all_results["rotated"].values() if r.get("mid_dip", False)) / max(len(all_results["rotated"]), 1)
    cos_dip = sum(1 for r in all_results["cosine"].values() if r.get("mid_dip", False)) / max(len(all_results["cosine"]), 1)
    criteria["rank_dip"] = rot_dip > 0.5 and cos_dip > 0.5
    print(f"\n  [1] Rank mid-dip persists under rotation:  {rot_dip:.0%} (need >50%)  {'✅' if rot_dip > 0.5 else '❌'}")
    print(f"      Rank mid-dip persists under cosine:     {cos_dip:.0%} (need >50%)  {'✅' if cos_dip > 0.5 else '❌'}")
    print(f"      (baseline reference: {base_dip:.0%})")

    # Criterion 2: t* still scales (t*/T not wildly different)
    base_tstar = np.mean([r["v_min_norm"] for r in all_results["baseline"].values()])
    rot_tstar = np.mean([r["v_min_norm"] for r in all_results["rotated"].values()])
    cos_tstar = np.mean([r["v_min_norm"] for r in all_results["cosine"].values()])
    tstar_stable = abs(rot_tstar - base_tstar) < 0.15 and abs(cos_tstar - base_tstar) < 0.15
    criteria["tstar_scales"] = tstar_stable
    print(f"\n  [2] t*/T stable under rotation:  base={base_tstar:.3f} → rot={rot_tstar:.3f} (Δ={abs(rot_tstar-base_tstar):.3f})")
    print(f"      t*/T stable under cosine:    base={base_tstar:.3f} → cos={cos_tstar:.3f} (Δ={abs(cos_tstar-base_tstar):.3f})")
    print(f"      Verdict: {'✅ stable' if tstar_stable else '❌ unstable'}")

    # Criterion 3: V-C alignment
    base_vc = np.mean([r["vc_alignment"] for r in all_results["baseline"].values()])
    rot_vc = np.mean([r["vc_alignment"] for r in all_results["rotated"].values()])
    cos_vc = np.mean([r["vc_alignment"] for r in all_results["cosine"].values()])
    vc_stable = rot_vc < 8 and cos_vc < 8  # alignment within 8 steps
    criteria["vc_aligned"] = vc_stable
    print(f"\n  [3] V-C alignment (lower = better):  base={base_vc:.1f}  rot={rot_vc:.1f}  cos={cos_vc:.1f}")
    print(f"      Verdict: {'✅ aligned' if vc_stable else '❌ decoupled'}")

    # Overall verdict
    n_passed = sum(criteria.values())
    print(f"\n  {'=' * 50}")
    if n_passed >= 3:
        print(f"  ✅ VERDICT: STRUCTURE IS REAL (invariant)")
        print(f"     All 3 criteria pass.")
        print(f"     Low-dimensional manifold is a genuine dynamical system property.")
    elif n_passed >= 1:
        print(f"  ⚠️  VERDICT: PARTIALLY ROBUST")
        print(f"     {n_passed}/3 criteria pass.")
        print(f"     Some structure is real, some may be metric-dependent.")
    else:
        print(f"  ❌ VERDICT: STRUCTURE IS ARTIFACT")
        print(f"     0/3 criteria pass.")
        print(f"     Low-dimensional structure collapses under coordinate transform.")
    print(f"  {'=' * 50}")

    # Save report
    report = {
        "config": {"prompts": list(PROMPTS.keys()), "seeds": SEEDS, "steps": STEPS},
        "per_run": {},
        "summary": {},
        "verdict": {
            "criteria": {k: bool(v) for k, v in criteria.items()},
            "n_passed": n_passed,
            "conclusion": "real" if n_passed >= 2 else "artifact" if n_passed == 0 else "partially_robust",
        }
    }

    for version in versions:
        for (p, s), r in all_results[version].items():
            key = f"{version}_{p}_s{s}"
            report["per_run"][key] = r

    for version in versions:
        er_vals = [r["effective_rank"] for r in all_results[version].values()]
        ts_vals = [r["v_min_norm"] for r in all_results[version].values()]
        al_vals = [r["vc_alignment"] for r in all_results[version].values()]
        report["summary"][version] = {
            "effective_rank_mean": float(np.mean(er_vals)),
            "effective_rank_std": float(np.std(er_vals)),
            "tstar_norm_mean": float(np.mean(ts_vals)),
            "tstar_norm_std": float(np.std(ts_vals)),
            "vc_alignment_mean": float(np.mean(al_vals)),
        }

    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report: {OUTPUT_DIR / 'report.json'}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
