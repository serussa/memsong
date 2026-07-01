#!/usr/bin/env python3
"""
Intrinsic Geometry Analysis: Arc-length Reparameterization of DiT Latent Dynamics.

Tests whether the observed low-dimensional structure in decoder.layers[12]
is an intrinsic geometric manifold independent of diffusion timestep parameterization.
"""

import argparse, json, os, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import norm

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

PROMPT_MAP = {
    "electro":   "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
    "rock":      "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
    "ballad_m":  "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
    "pop_f":     "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
    "folk":      "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major",
    "dancepop":  "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor",
    "cpop":      "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major",
    "edm":       "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
}
SHORT_LABEL_FULL = {
    "electro": "Electro Pop", "rock": "Rock", "ballad_m": "Ballad (M)",
    "pop_f": "Pop (F)", "folk": "Folk", "dancepop": "Dance Pop", "cpop": "C-Pop", "edm": "EDM",
}

N_S_GRID = 100

# =====================================================================
# Optimized arc-length reparameterization (uses np.interp, vectorized)
# =====================================================================
def arc_length_reparam(X):
    T, D = X.shape
    if T < 2:
        return None, None, None, None
    diffs = np.diff(X, axis=0)
    step_norms = norm(diffs, axis=1)
    cumulative = np.zeros(T)
    cumulative[1:] = np.cumsum(step_norms)
    total = cumulative[-1]
    if total < 1e-10:
        return None, None, None, None
    s_t = cumulative / total
    s_grid = np.linspace(0, 1, N_S_GRID)
    # Fast 1D interpolation using np.interp per dimension
    X_s = np.column_stack([np.interp(s_grid, s_t, X[:, d]) for d in range(D)])
    ds = s_grid[1] - s_grid[0]
    return s_t, X_s, ds, cumulative


def compute_effective_rank(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    T, D = Xc.shape
    if T <= 1:
        return 1.0
    # Use skinny SVD (min(T,D) singular values) instead of D×D covariance
    # This is much faster when T << D (sliding window with window=5, D=2048)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / (s.sum() + 1e-10)
    H = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(H))


def per_step_effective_rank_fast(X, window=5):
    """Faster sliding window effective rank using only eigen/svd of small window."""
    T = X.shape[0]
    ranks = np.ones(T) * np.nan
    for t in range(window - 1, T):
        start = max(0, t - window + 1)
        chunk = X[start:t + 1]
        if chunk.shape[0] >= 3:
            ranks[t] = compute_effective_rank(chunk)
    return ranks


def compute_curvature_proxy(X):
    if X.shape[0] < 3:
        return np.array([])
    d2 = X[2:] - 2 * X[1:-1] + X[:-2]
    return norm(d2, axis=1)


def compute_velocity(X):
    return norm(np.diff(X, axis=0), axis=1)


# =====================================================================
# Load trajectories
# =====================================================================
def load_all_trajectories():
    all_trajs = {}
    sources = {
        "dynamics": Path("output/dynamics_experiment/trajectories"),
        "temporal": Path("output/temporal_scaling/trajectories"),
    }
    for source_name, traj_dir in sources.items():
        if not traj_dir.exists():
            continue
        for fpath in sorted(traj_dir.glob("*.npy")):
            stem = fpath.stem.replace("traj_", "")
            X = np.load(fpath)
            schedule = "linear"
            fname = stem
            if "_T" in fname:
                parts = fname.split("_T")
                pname = parts[0]
                t_parts = parts[1].split("_s")
                n_steps = int(t_parts[0])
                seed = int(t_parts[1])
            elif "_s" in fname:
                parts = fname.split("_s")
                pname = parts[0]
                seed = int(parts[1])
                n_steps = X.shape[0]
            else:
                continue
            key = (pname, seed, n_steps, schedule)
            all_trajs[key] = X
    return all_trajs


# =====================================================================
# Main analysis
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/intrinsic_geometry")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  INTRINSIC GEOMETRY ANALYSIS", flush=True)
    print("  Arc-length Reparameterization of DiT Latent Dynamics", flush=True)
    print("=" * 70, flush=True)

    # ---- Load ----
    print("\n[1/4] Loading trajectories...", flush=True)
    all_trajs = load_all_trajectories()
    print(f"  {len(all_trajs)} total trajectories loaded", flush=True)

    # Filter: only linear, T=30 or T=50
    linear_trajs = {}
    for key, X in all_trajs.items():
        pname, seed, n_steps, schedule = key
        if schedule == "linear" and n_steps in (30, 50):
            linear_trajs[key] = X
    print(f"  Linear T=30: {sum(1 for k in linear_trajs if k[2]==30)} runs", flush=True)
    print(f"  Linear T=50: {sum(1 for k in linear_trajs if k[2]==50)} runs", flush=True)

    # ---- Analyze ----
    print("\n[2/4] Analyzing trajectories...", flush=True)
    prompt_names = sorted(set(k[0] for k in linear_trajs))

    all_results = []
    for key, X in linear_trajs.items():
        pname, seed, n_steps, _ = key
        result = {"pname": pname, "seed": seed, "T": n_steps}

        # t-space: PCA
        from sklearn.decomposition import PCA
        n_comp = min(4, X.shape[0], X.shape[1])
        pca_t = PCA(n_components=n_comp)
        pca_t.fit(X)
        result["t_pca_var"] = pca_t.explained_variance_ratio_.tolist()
        result["t_eff_rank"] = compute_effective_rank(X)

        # t-space: velocity (smoothed for cleaner transition detection)
        from scipy import signal as scipy_signal
        v_t = compute_velocity(X)
        result["t_velocity"] = v_t.tolist()
        result["t_velocity_mean"] = float(np.mean(v_t))
        if len(v_t) >= 5:
            w = min(5, len(v_t) - (1 - len(v_t) % 2))
            v_smooth = scipy_signal.savgol_filter(v_t, w, 2)
        else:
            v_smooth = v_t
        if len(v_smooth) > 0:
            result["t_star_vel"] = int(np.argmin(v_smooth))
            result["t_star_vel_norm"] = result["t_star_vel"] / n_steps
        else:
            result["t_star_vel"] = None
            result["t_star_vel_norm"] = None

        # t-space: curvature proxy
        c_t = compute_curvature_proxy(X)
        result["t_curvature"] = c_t.tolist()
        if len(c_t) > 0:
            result["t_curvature_mean"] = float(np.mean(c_t))
            result["t_curvature_peak_idx"] = int(np.argmax(c_t)) + 1
        else:
            result["t_curvature_mean"] = 0.0
            result["t_curvature_peak_idx"] = None

        # s-space: arc-length reparameterization
        s_t, X_s, ds, cumulative = arc_length_reparam(X)
        if X_s is not None and X_s.shape[0] >= 3:
            result["arc_length_valid"] = True
            result["s_t"] = s_t.tolist()
            result["total_arc_length"] = float(cumulative[-1])

            # Nonlinearity of s(t)
            t_lin = np.linspace(0, 1, n_steps)
            result["s_nonlinearity"] = float(np.mean(np.abs(s_t - t_lin)))

            # s-space: PCA
            pca_s = PCA(n_components=min(4, X_s.shape[0], X_s.shape[1]))
            pca_s.fit(X_s)
            result["s_pca_var"] = pca_s.explained_variance_ratio_.tolist()
            result["s_eff_rank"] = compute_effective_rank(X_s)

            # s-space: velocity (should be ~constant by construction = 1/ds)
            v_s = compute_velocity(X_s)
            result["s_velocity"] = v_s.tolist()
            result["s_velocity_mean"] = float(np.mean(v_s))
            result["s_velocity_std"] = float(np.std(v_s))

            # s-space: curvature proxy
            c_s = compute_curvature_proxy(X_s)
            result["s_curvature"] = c_s.tolist()
            if len(c_s) > 0:
                result["s_curvature_mean"] = float(np.mean(c_s))
                result["s_curvature_peak_idx"] = int(np.argmax(c_s)) + 1
                result["s_star_curv_norm"] = (np.argmax(c_s) + 1) / N_S_GRID
            else:
                result["s_curvature_mean"] = 0.0
                result["s_curvature_peak_idx"] = None
                result["s_star_curv_norm"] = None

            # s(t*): map velocity minimum from t-space to s-coordinate
            # This directly tests if transition stabilizes in s-space
            if result.get("t_star_vel") is not None:
                tstar = result["t_star_vel"]
                s_at_tstar = float(s_t[tstar]) if tstar < len(s_t) else 1.0
                result["s_at_tstar"] = s_at_tstar
            else:
                result["s_at_tstar"] = None
        else:
            result["arc_length_valid"] = False

        all_results.append(result)

    print(f"  {len(all_results)} runs analyzed", flush=True)
    s_valid = [r for r in all_results if r.get("arc_length_valid")]
    print(f"  {len(s_valid)} valid s-space", flush=True)

    # ---- Variance reduction ----
    print("\n[3/4] Computing variance reduction...", flush=True)

    # Overall variance
    # PRIMARY: t*/T in t-space vs s(t*) in s-space (map same transition to arc-length)
    # SECONDARY: s* from curvature peak in s-space
    t_stars = [r["t_star_vel_norm"] for r in all_results
               if r.get("t_star_vel_norm") is not None]
    s_tstar = [r["s_at_tstar"] for r in all_results
               if r.get("arc_length_valid") and r.get("s_at_tstar") is not None]
    s_curv = [r["s_star_curv_norm"] for r in all_results
              if r.get("arc_length_valid") and r.get("s_star_curv_norm") is not None]

    var_report = {
        "t_space_tstar_norm": {
            "mean": float(np.mean(t_stars)), "std": float(np.std(t_stars)),
            "var": float(np.var(t_stars)), "n": len(t_stars),
        },
        "s_space_at_tstar": {
            "mean": float(np.mean(s_tstar)), "std": float(np.std(s_tstar)),
            "var": float(np.var(s_tstar)), "n": len(s_tstar),
        },
        "s_space_curvature_peak": {
            "mean": float(np.mean(s_curv)), "std": float(np.std(s_curv)),
            "var": float(np.var(s_curv)), "n": len(s_curv),
        },
    }
    var_report["variance_reduction_at_tstar_pct"] = (
        1 - var_report["s_space_at_tstar"]["var"] / var_report["t_space_tstar_norm"]["var"]
    ) * 100
    if len(s_curv) > 0:
        var_report["variance_reduction_curvature_peak_pct"] = (
            1 - var_report["s_space_curvature_peak"]["var"] / var_report["t_space_tstar_norm"]["var"]
        ) * 100

    # Per-prompt
    per_prompt = {}
    for p in sorted(set(r["pname"] for r in all_results)):
        t_vals = [r["t_star_vel_norm"] for r in all_results
                  if r["pname"] == p and r.get("t_star_vel_norm") is not None]
        s_t_vals = [r["s_at_tstar"] for r in all_results
                    if r.get("arc_length_valid") and r["pname"] == p
                    and r.get("s_at_tstar") is not None]
        s_c_vals = [r["s_star_curv_norm"] for r in all_results
                    if r.get("arc_length_valid") and r["pname"] == p
                    and r.get("s_star_curv_norm") is not None]
        per_prompt[p] = {
            "t_star_mean": float(np.mean(t_vals)) if t_vals else None,
            "t_star_std": float(np.std(t_vals)) if t_vals else None,
            "s_at_tstar_mean": float(np.mean(s_t_vals)) if s_t_vals else None,
            "s_at_tstar_std": float(np.std(s_t_vals)) if s_t_vals else None,
            "s_curv_peak_mean": float(np.mean(s_c_vals)) if s_c_vals else None,
            "s_curv_peak_std": float(np.std(s_c_vals)) if s_c_vals else None,
        }
    var_report["per_prompt"] = per_prompt

    # Cross-prompt variance
    t_means = [per_prompt[p]["t_star_mean"] for p in per_prompt
               if per_prompt[p]["t_star_mean"] is not None]
    s_t_means = [per_prompt[p]["s_at_tstar_mean"] for p in per_prompt
                 if per_prompt[p]["s_at_tstar_mean"] is not None]
    var_report["cross_prompt"] = {
        "t_space_var": float(np.var(t_means)) if t_means else None,
        "s_space_at_tstar_var": float(np.var(s_t_means)) if s_t_means else None,
    }
    if var_report["cross_prompt"]["t_space_var"] and var_report["cross_prompt"]["s_space_at_tstar_var"]:
        var_report["cross_prompt"]["variance_reduction_pct"] = (
            1 - var_report["cross_prompt"]["s_space_at_tstar_var"]
            / var_report["cross_prompt"]["t_space_var"]
        ) * 100
    else:
        var_report["cross_prompt"]["variance_reduction_pct"] = None

    # By T
    by_T = {}
    for T_val in sorted(set(r["T"] for r in all_results)):
        t_sub = [r["t_star_vel_norm"] for r in all_results
                 if r["T"] == T_val and r.get("t_star_vel_norm") is not None]
        s_t_sub = [r["s_at_tstar"] for r in all_results
                   if r.get("arc_length_valid") and r["T"] == T_val
                   and r.get("s_at_tstar") is not None]
        by_T[str(T_val)] = {
            "t_space_mean": float(np.mean(t_sub)) if t_sub else None,
            "t_space_std": float(np.std(t_sub)) if t_sub else None,
            "s_at_tstar_mean": float(np.mean(s_t_sub)) if s_t_sub else None,
            "s_at_tstar_std": float(np.std(s_t_sub)) if s_t_sub else None,
        }
    var_report["by_T"] = by_T

    print(json.dumps(var_report, indent=2), flush=True)

    # ---- Per-step effective rank (subset for plots) ----
    print("\n  Computing per-step effective ranks (this may take a moment)...", flush=True)
    for r in all_results:
        if r["T"] in (30, 50):
            r["t_eff_rank_per_step"] = per_step_effective_rank_fast(
                linear_trajs[(r["pname"], r["seed"], r["T"], "linear")]
            ).tolist()
        if r.get("arc_length_valid"):
            _, X_s, _, _ = arc_length_reparam(
                linear_trajs[(r["pname"], r["seed"], r["T"], "linear")]
            )
            if X_s is not None:
                r["s_eff_rank_per_step"] = per_step_effective_rank_fast(X_s).tolist()
            else:
                r["s_eff_rank_per_step"] = None

    # ---- Verdict ----
    print("\n[4/4] Producing verdict...", flush=True)

    evidence = {}

    # Evidence 1: Overall variance reduction (s(t*) vs t*/T)
    vrp = var_report.get("variance_reduction_at_tstar_pct")
    evidence["variance_reduction_at_tstar_pct"] = vrp
    evidence["variance_reduced"] = vrp is not None and vrp > 20

    # Evidence 2: Cross-prompt variance reduction
    cp_vrp = var_report["cross_prompt"].get("variance_reduction_pct")
    evidence["cross_prompt_variance_reduction_pct"] = cp_vrp
    evidence["prompt_influence_weakens"] = cp_vrp is not None and cp_vrp > 15

    # Evidence 3: Effective rank convergence (T30 vs T50 in s-space)
    rank_30 = np.nanmean([r["s_eff_rank"] for r in all_results
                          if r.get("arc_length_valid") and r["T"] == 30])
    rank_50 = np.nanmean([r["s_eff_rank"] for r in all_results
                          if r.get("arc_length_valid") and r["T"] == 50])
    evidence["rank_T30_s"] = float(rank_30) if not np.isnan(rank_30) else None
    evidence["rank_T50_s"] = float(rank_50) if not np.isnan(rank_50) else None
    evidence["rank_converges"] = (
        evidence["rank_T30_s"] is not None and evidence["rank_T50_s"] is not None
        and abs(evidence["rank_T30_s"] - evidence["rank_T50_s"]) < 0.5
    )

    # Evidence 4: Arc-length convergence
    arc_T30 = [r["total_arc_length"] for r in all_results
               if r.get("arc_length_valid") and r["T"] == 30]
    arc_T50 = [r["total_arc_length"] for r in all_results
               if r.get("arc_length_valid") and r["T"] == 50]
    if arc_T30 and arc_T50:
        arc_ratio = np.mean(arc_T50) / np.mean(arc_T30)
        evidence["arc_length_ratio_T50_T30"] = float(arc_ratio)
        evidence["arc_converges"] = abs(arc_ratio - 50 / 30) < 0.3
    else:
        evidence["arc_converges"] = False

    n_support = sum([
        evidence.get("variance_reduced", False),
        evidence.get("prompt_influence_weakens", False),
        evidence.get("rank_converges", False),
        evidence.get("arc_converges", False),
    ])
    evidence["n_supporting"] = n_support
    evidence["n_total"] = 4

    if n_support >= 3:
        verdict, confidence = "intrinsic manifold", "high"
    elif n_support >= 2:
        verdict, confidence = "intrinsic manifold", "moderate"
    elif n_support >= 1:
        verdict, confidence = "partially intrinsic", "low"
    else:
        verdict, confidence = "parameterization artifact", "high"

    print(f"\n  Verdict: {verdict} (confidence: {confidence})", flush=True)
    def make_serializable(v):
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    evidence_serial = {k: make_serializable(v) for k, v in evidence.items()}
    print(f"  Evidence: {json.dumps(evidence_serial, indent=4)}", flush=True)

    # ---- Visualize ----
    print("\n  Generating plots...", flush=True)
    visualize_all(all_results, OUTPUT_DIR)
    visualize_temporal_comparison(all_results, OUTPUT_DIR)

    # ---- Save report ----
    report = {
        "config": {"n_s_grid": N_S_GRID},
        "data": {
            "n_runs": len(all_results),
            "n_valid_s": len([r for r in all_results if r.get("arc_length_valid")]),
            "prompts": sorted(set(r["pname"] for r in all_results)),
        },
        "variance_reduction": var_report,
        "evidence": evidence,
        "verdict": verdict,
        "confidence": confidence,
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return super().default(obj)
    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    # ---- Print summary ----
    print_summary(all_results, var_report, evidence, verdict, confidence, OUTPUT_DIR)


# =====================================================================
# Visualization
# =====================================================================
def visualize_all(all_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt_names = sorted(set(r["pname"] for r in all_results))
    prompt_colors = plt.cm.tab10(np.linspace(0, 1, len(prompt_names)))
    prompt_color_map = {p: prompt_colors[i] for i, p in enumerate(prompt_names)}
    def label_for(p):
        return SHORT_LABEL_FULL.get(p, p)

    # --- 1. s(t) curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        if r.get("arc_length_valid"):
            s_t = np.array(r["s_t"])
            ax.plot(np.arange(len(s_t)), s_t,
                    color=prompt_color_map.get(r["pname"], "gray"),
                    alpha=0.3, linewidth=0.8)
    ax.plot([0, 50], [0, 1], "k--", alpha=0.4, label="Linear")
    ax.set_xlabel("Diffusion Step t")
    ax.set_ylabel("Normalized Arc-length s(t)")
    ax.set_title("Arc-length vs Diffusion Step (each line = one run)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_s_of_t.png", dpi=150)
    plt.close()
    print("  [plot] 01_s_of_t.png", flush=True)

    # --- 2. Effective rank comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    t_ranks = [r["t_eff_rank"] for r in all_results]
    s_ranks = [r["s_eff_rank"] for r in all_results if r.get("arc_length_valid")]
    ax.scatter(t_ranks, s_ranks, alpha=0.6, s=30)
    lo, hi = min(t_ranks + s_ranks), max(t_ranks + s_ranks)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    ax.set_xlabel("Effective Rank (t-space)")
    ax.set_ylabel("Effective Rank (s-space)")
    ax.set_title(f"Overall Effective Rank")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for pname in prompt_names:
        s_ers = [np.array(r["s_eff_rank_per_step"]) for r in all_results
                 if r.get("arc_length_valid") and r["pname"] == pname
                 and r.get("s_eff_rank_per_step")]
        if s_ers:
            s_axis = np.linspace(0, 1, max(len(e) for e in s_ers))
            mean_er = np.nanmean(np.column_stack([
                np.pad(e, (0, max(0, len(s_axis) - len(e))),
                       constant_values=np.nan)[:len(s_axis)]
                for e in s_ers if len(e) > 0
            ]), axis=1)
            ax.plot(s_axis, mean_er, color=prompt_color_map[pname],
                    linewidth=1.5, label=label_for(pname))
    ax.set_xlabel("Normalized Arc-length s")
    ax.set_ylabel("Effective Rank (window=5)")
    ax.set_title("Per-Step Effective Rank (s-space)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "02_effective_rank.png", dpi=150)
    plt.close()
    print("  [plot] 02_effective_rank.png", flush=True)

    # --- 3. Velocity comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for r in all_results:
        v = np.array(r["t_velocity"])
        ax.plot(v, color=prompt_color_map.get(r["pname"], "gray"),
                alpha=0.3, linewidth=0.7)
    ax.set_xlabel("Diffusion Step t")
    ax.set_ylabel("Velocity ||dh/dt||")
    ax.set_title("Velocity Profiles (t-space)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in all_results:
        if r.get("arc_length_valid") and r.get("s_velocity"):
            v = np.array(r["s_velocity"])
            s_axis = np.linspace(0, 1, len(v))
            ax.plot(s_axis, v, color=prompt_color_map.get(r["pname"], "gray"),
                    alpha=0.3, linewidth=0.7)
    ax.set_xlabel("Normalized Arc-length s")
    ax.set_ylabel("Velocity ||dh/ds||")
    ax.set_title("Velocity Profiles (s-space)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_velocity_comparison.png", dpi=150)
    plt.close()
    print("  [plot] 03_velocity_comparison.png", flush=True)

    # --- 4. Curvature comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for r in all_results:
        c = np.array(r["t_curvature"])
        if len(c) > 0:
            ax.plot(np.arange(1, len(c) + 1), c,
                    color=prompt_color_map.get(r["pname"], "gray"),
                    alpha=0.3, linewidth=0.7)
    ax.set_xlabel("Diffusion Step t")
    ax.set_ylabel("Curvature Proxy ||Δ²h||")
    ax.set_title("Curvature Proxy (t-space)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in all_results:
        if r.get("arc_length_valid") and r.get("s_curvature"):
            c = np.array(r["s_curvature"])
            if len(c) > 0:
                s_axis = np.linspace(0, 1, len(c))
                ax.plot(s_axis, c,
                        color=prompt_color_map.get(r["pname"], "gray"),
                        alpha=0.3, linewidth=0.7)
    ax.set_xlabel("Normalized Arc-length s")
    ax.set_ylabel("Curvature Proxy ||Δ²h||")
    ax.set_title("Curvature Proxy (s-space)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "04_curvature_comparison.png", dpi=150)
    plt.close()
    print("  [plot] 04_curvature_comparison.png", flush=True)

    # --- 5. Transition point comparison ---
    # Panel: t-space (t*/T) | s-space (s at t*) | s-space (curv peak)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # t-space: t*/T
    ax = axes[0]
    t_by_p = {}
    for r in all_results:
        p = r["pname"]
        if r.get("t_star_vel_norm") is not None:
            t_by_p.setdefault(p, []).append(r["t_star_vel_norm"])
    p_list = sorted(t_by_p.keys())
    data_t = [t_by_p[p] for p in p_list]
    bp = ax.boxplot(data_t, labels=[label_for(p) for p in p_list], patch_artist=True)
    for patch, color in zip(bp["boxes"], [prompt_color_map[p] for p in p_list]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("t*/T")
    ax.set_title("Transition in t-space\n(t* = velocity min)")
    ax.grid(alpha=0.3, axis="y")

    # s-space: s(t*) -- main result
    ax = axes[1]
    s_by_p = {}
    for r in all_results:
        if r.get("arc_length_valid") and r.get("s_at_tstar") is not None:
            s_by_p.setdefault(r["pname"], []).append(r["s_at_tstar"])
    s_p_list = sorted(s_by_p.keys())
    data_s = [s_by_p[p] for p in s_p_list]
    bp2 = ax.boxplot(data_s, labels=[label_for(p) for p in s_p_list], patch_artist=True)
    for patch, color in zip(bp2["boxes"], [prompt_color_map[p] for p in s_p_list]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("s(t*)")
    ax.set_title("Transition in s-space\n(s(t*) = arc-length at t*)")
    ax.grid(alpha=0.3, axis="y")

    # s-space: curvature peak (secondary)
    ax = axes[2]
    sc_by_p = {}
    for r in all_results:
        if r.get("arc_length_valid") and r.get("s_star_curv_norm") is not None:
            sc_by_p.setdefault(r["pname"], []).append(r["s_star_curv_norm"])
    sc_p_list = sorted(sc_by_p.keys())
    data_sc = [sc_by_p[p] for p in sc_p_list]
    bp3 = ax.boxplot(data_sc, labels=[label_for(p) for p in sc_p_list], patch_artist=True)
    for patch, color in zip(bp3["boxes"], [prompt_color_map[p] for p in sc_p_list]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("s* (curvature peak)")
    ax.set_title("Transition in s-space\n(curvature proxy peak)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "05_transition_comparison.png", dpi=150)
    plt.close()
    print("  [plot] 05_transition_comparison.png", flush=True)

    # --- 6. PCA explained variance ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, space, var_key in [
        (axes[0], "t-space", "t_pca_var"),
        (axes[1], "s-space", "s_pca_var"),
    ]:
        pca_vars = np.array([r[var_key] for r in all_results
                             if r.get(space[0] + "_pca_var") is not None])
        if space == "s-space":
            pca_vars = np.array([r[var_key] for r in all_results
                                 if r.get("arc_length_valid")])
        mean_v = np.mean(pca_vars, axis=0)
        std_v = np.std(pca_vars, axis=0)
        nc = min(4, len(mean_v))
        ax.bar(range(nc), mean_v[:nc], yerr=std_v[:nc],
               color="steelblue", alpha=0.7, capsize=4)
        ax.set_xticks(range(nc))
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(f"PCA {space} — Cum: {sum(mean_v[:nc]):.1%}")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "06_pca_variance.png", dpi=150)
    plt.close()
    print("  [plot] 06_pca_variance.png", flush=True)

    # --- 7. Nonlinearity bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    nl_by_p = {}
    for r in all_results:
        if r.get("arc_length_valid"):
            nl_by_p.setdefault(r["pname"], []).append(r["s_nonlinearity"])
    p_list = sorted(nl_by_p.keys())
    means = [np.mean(nl_by_p[p]) for p in p_list]
    stds = [np.std(nl_by_p[p]) for p in p_list]
    ax.bar(range(len(p_list)), means, yerr=stds,
           color=[prompt_color_map[p] for p in p_list], alpha=0.7, capsize=4)
    ax.set_xticks(range(len(p_list)))
    ax.set_xticklabels([label_for(p) for p in p_list], fontsize=8, rotation=20)
    ax.set_ylabel("MAD from Linear")
    ax.set_title("s(t) Nonlinearity by Prompt")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "07_nonlinearity.png", dpi=150)
    plt.close()
    print("  [plot] 07_nonlinearity.png", flush=True)

    # --- 8. Rank comparison bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(prompt_names))
    w = 0.35
    t_means_b = [np.mean([r["t_eff_rank"] for r in all_results if r["pname"] == p])
                 for p in prompt_names]
    t_stds_b = [np.std([r["t_eff_rank"] for r in all_results if r["pname"] == p])
                for p in prompt_names]
    s_means_b = [np.mean([r["s_eff_rank"] for r in all_results
                          if r.get("arc_length_valid") and r["pname"] == p])
                 for p in prompt_names]
    s_stds_b = [np.std([r["s_eff_rank"] for r in all_results
                        if r.get("arc_length_valid") and r["pname"] == p])
                for p in prompt_names]
    ax.bar(x - w/2, t_means_b, w, yerr=t_stds_b, label="t-space",
           color="steelblue", alpha=0.7, capsize=4)
    ax.bar(x + w/2, s_means_b, w, yerr=s_stds_b, label="s-space",
           color="coral", alpha=0.7, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(p) for p in prompt_names], fontsize=8, rotation=20)
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank: t-space vs s-space")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "08_rank_by_prompt.png", dpi=150)
    plt.close()
    print("  [plot] 08_rank_by_prompt.png", flush=True)

    # --- 9. s(t) by prompt (panel) ---
    p_unique = sorted(set(r["pname"] for r in all_results if r.get("arc_length_valid")))
    n_cols = min(4, len(p_unique))
    n_rows = int(np.ceil(len(p_unique) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if len(p_unique) > 1 else [axes]
    for pi, pname in enumerate(p_unique):
        if pi >= len(axes_flat):
            break
        ax = axes_flat[pi]
        for r in all_results:
            if r["pname"] == pname and r.get("arc_length_valid"):
                ax.plot(np.arange(len(r["s_t"])), r["s_t"], alpha=0.6, linewidth=1.0)
        ax.set_title(label_for(pname))
        ax.set_xlabel("Step t")
        ax.set_ylabel("s(t)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    for pi in range(len(p_unique), len(axes_flat)):
        axes_flat[pi].set_visible(False)
    fig.suptitle("s(t) by Prompt", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "09_s_of_t_by_prompt.png", dpi=150)
    plt.close()
    print("  [plot] 09_s_of_t_by_prompt.png", flush=True)


def visualize_temporal_comparison(all_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt_names = sorted(set(r["pname"] for r in all_results))
    prompt_colors = plt.cm.tab10(np.linspace(0, 1, len(prompt_names)))
    prompt_color_map = {p: prompt_colors[i] for i, p in enumerate(prompt_names)}
    def label_for(p):
        return SHORT_LABEL_FULL.get(p, p)

    # --- s(t) comparison T=30 vs T=50 ---
    r_by_key = {}
    for r in all_results:
        r_by_key[(r["pname"], r["seed"], r["T"])] = r

    common = []
    for p in prompt_names:
        for s in sorted(set(r["seed"] for r in all_results if r["pname"] == p)):
            if (p, s, 30) in r_by_key and (p, s, 50) in r_by_key:
                common.append((p, s))
    common = list(set(common))

    if common:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
        for i, (pname, seed) in enumerate(common[:4]):
            ax = axes_flat[i]
            r30 = r_by_key[(pname, seed, 30)]
            r50 = r_by_key[(pname, seed, 50)]
            if r30.get("arc_length_valid") and r50.get("arc_length_valid"):
                ax.plot(np.arange(len(r30["s_t"])), r30["s_t"], "o-",
                        markersize=3, linewidth=1.2,
                        label=f"T=30 (arc={r30['total_arc_length']:.0f})")
                ax.plot(np.arange(len(r50["s_t"])), r50["s_t"], "s-",
                        markersize=3, linewidth=1.2,
                        label=f"T=50 (arc={r50['total_arc_length']:.0f})")
            ax.set_title(f"{label_for(pname)} s={seed}")
            ax.set_xlabel("Step t")
            ax.set_ylabel("s(t)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        for pi in range(len(common[:4]), 4):
            axes_flat[pi].set_visible(False)
        fig.suptitle("s(t): T=30 vs T=50", fontsize=13)
        plt.tight_layout()
        plt.savefig(output_dir / "10_s_of_t_T30_vs_T50.png", dpi=150)
        plt.close()
        print("  [plot] 10_s_of_t_T30_vs_T50.png", flush=True)

    # --- Transition by T ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    configs = [
        (axes[0], "t-space", "t_star_vel_norm", "t*/T"),
        (axes[1], "s-space (s at t*)", "s_at_tstar", "s(t*)"),
        (axes[2], "s-space (curv peak)", "s_star_curv_norm", "s*"),
    ]
    for ax, space, star_key, ylabel in configs:
        data_by_T = {}
        for r in all_results:
            T = r["T"]
            if star_key == "s_star_curv_norm":
                if not r.get("arc_length_valid") or r.get(star_key) is None:
                    continue
            elif star_key == "s_at_tstar":
                if not r.get("arc_length_valid") or r.get(star_key) is None:
                    continue
            elif r.get(star_key) is None:
                continue
            data_by_T.setdefault(T, []).append(r[star_key])
        Ts = sorted(data_by_T.keys())
        means = [np.mean(data_by_T[T]) for T in Ts]
        stds = [np.std(data_by_T[T]) for T in Ts]
        ax.errorbar(Ts, means, yerr=stds, fmt="o-", capsize=5,
                    linewidth=2, markersize=8, color="steelblue")
        ax.set_xlabel("Total Steps T")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Transition by T: {space}")
        ax.set_xticks(Ts)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "11_transition_by_T.png", dpi=150)
    plt.close()
    print("  [plot] 11_transition_by_T.png", flush=True)

    # --- Total arc-length convergence ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for pname, seed in common:
        r30 = r_by_key.get((pname, seed, 30))
        r50 = r_by_key.get((pname, seed, 50))
        if r30 and r50 and r30.get("arc_length_valid") and r50.get("arc_length_valid"):
            a3, a5 = r30["total_arc_length"], r50["total_arc_length"]
            col = prompt_color_map.get(pname, "gray")
            ax.scatter(a3, a5, color=col, s=50, alpha=0.7, zorder=5)
            ax.annotate(f"{label_for(pname)}s{seed}", (a3, a5), fontsize=6, alpha=0.7)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.4)
    ax.set_xlabel("Total Arc-length (T=30)")
    ax.set_ylabel("Total Arc-length (T=50)")
    ax.set_title("Arc-length Convergence (T=30 vs T=50)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "12_arc_convergence.png", dpi=150)
    plt.close()
    print("  [plot] 12_arc_convergence.png", flush=True)


# =====================================================================
# Summary
# =====================================================================
def print_summary(all_results, var_report, evidence, verdict, confidence, output_dir):
    prompt_names = sorted(set(r["pname"] for r in all_results))
    def label_for(p):
        return SHORT_LABEL_FULL.get(p, p)

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  INTRINSIC GEOMETRY ANALYSIS — SUMMARY")
    lines.append("  Arc-length reparameterization of DiT layer-12 dynamics")
    lines.append("=" * 70)

    n_T30 = sum(1 for r in all_results if r["T"] == 30)
    n_T50 = sum(1 for r in all_results if r["T"] == 50)
    lines.append(f"\n  Runs analyzed: {len(all_results)} (T=30: {n_T30}, T=50: {n_T50})")

    # s(t) nonlinearity
    nl = [r["s_nonlinearity"] for r in all_results if r.get("arc_length_valid")]
    lines.append(f"\n  s(t) nonlinearity:")
    lines.append(f"    Mean MAD from linear: {np.mean(nl):.4f} ± {np.std(nl):.4f}")

    # Effective rank
    t_rank = [r["t_eff_rank"] for r in all_results]
    s_rank = [r["s_eff_rank"] for r in all_results if r.get("arc_length_valid")]
    lines.append(f"\n  Effective rank:")
    lines.append(f"    t-space: {np.mean(t_rank):.2f} ± {np.std(t_rank):.2f}")
    lines.append(f"    s-space: {np.mean(s_rank):.2f} ± {np.std(s_rank):.2f}")

    # PCA
    t_pca = np.mean([r["t_pca_var"] for r in all_results], axis=0)
    s_pca = np.mean([r["s_pca_var"] for r in all_results if r.get("arc_length_valid")], axis=0)
    lines.append(f"\n  PCA explained variance (top-4):")
    lines.append(f"    t-space: {', '.join(f'PC{i+1}={t_pca[i]:.1%}' for i in range(min(4, len(t_pca))))}")
    lines.append(f"    s-space: {', '.join(f'PC{i+1}={s_pca[i]:.1%}' for i in range(min(4, len(s_pca))))}")

    # Transition variance
    vr = var_report
    lines.append(f"\n  Transition point variance:")
    lines.append(f"    t*/T (t-space): μ={vr['t_space_tstar_norm']['mean']:.4f}  σ²={vr['t_space_tstar_norm']['var']:.6f}")
    lines.append(f"    s(t*) (s-space): μ={vr['s_space_at_tstar']['mean']:.4f}  σ²={vr['s_space_at_tstar']['var']:.6f}")
    lines.append(f"    s* (curv peak): μ={vr['s_space_curvature_peak']['mean']:.4f}  σ²={vr['s_space_curvature_peak']['var']:.6f}")
    if vr.get("variance_reduction_at_tstar_pct") is not None:
        lines.append(f"    Variance reduction (s(t*) vs t*/T): {vr['variance_reduction_at_tstar_pct']:.1f}%")

    # Cross-prompt
    cp = vr.get("cross_prompt", {})
    if cp.get("t_space_var") is not None:
        lines.append(f"\n  Cross-prompt transition variance:")
        lines.append(f"    t-space: σ²={cp['t_space_var']:.6f}")
        lines.append(f"    s-space (s at t*): σ²={cp['s_space_at_tstar_var']:.6f}")
        if cp.get("variance_reduction_pct") is not None:
            lines.append(f"    Variance reduction: {cp['variance_reduction_pct']:.1f}%")

    # Per-prompt
    lines.append(f"\n  Per-prompt transition:")
    lines.append(f"  {'Prompt':<15} {'t*/T μ':>8} {'t*/T σ':>8} {'s(t*) μ':>8} {'s(t*) σ':>8} {'s*curv μ':>8}")
    lines.append("  " + "-" * 55)
    for p in sorted(vr.get("per_prompt", {}).keys()):
        pp = vr["per_prompt"][p]
        lines.append(f"  {label_for(p):<15} "
                     f"{pp['t_star_mean']:>8.4f} {pp['t_star_std']:>8.4f} "
                     f"{pp['s_at_tstar_mean']:>8.4f} {pp['s_at_tstar_std']:>8.4f} "
                     f"{pp['s_curv_peak_mean']:>8.4f}")

    # By T
    bt = vr.get("by_T", {})
    if bt:
        lines.append(f"\n  Transition by T:")
        lines.append(f"  {'T':>4}  {'t*/T μ':>8} {'t*/T σ':>8}  {'s(t*) μ':>8} {'s(t*) σ':>8}")
        lines.append("  " + "-" * 46)
        for T_val in sorted(bt.keys()):
            b = bt[T_val]
            lines.append(f"  {T_val:>4}  {b['t_space_mean']:>8.4f} {b['t_space_std']:>8.4f}  "
                         f"{b['s_at_tstar_mean']:>8.4f} {b['s_at_tstar_std']:>8.4f}")

    # Evidence
    lines.append(f"\n  Evidence:")
    for k, v in evidence.items():
        lines.append(f"    {k}: {v}")

    # Verdict
    lines.append("")
    lines.append("  " + "=" * 60)
    if verdict == "intrinsic manifold":
        lines.append(f"  ✅ VERDICT: INTRINSIC MANIFOLD (confidence: {confidence})")
        lines.append(f"     Low-dimensional geometric structure is intrinsic and")
        lines.append(f"     independent of diffusion timestep parameterization.")
    elif verdict == "partially intrinsic":
        lines.append(f"  ⚠️  VERDICT: PARTIALLY INTRINSIC")
        lines.append(f"     Some structure is intrinsic, some is parameterization-dependent.")
    else:
        lines.append(f"  ❌ VERDICT: PARAMETERIZATION ARTIFACT")
        lines.append(f"     Low-dimensional structure is an artifact of timestep sampling.")
    lines.append(f"  Supporting evidence: {evidence['n_supporting']}/{evidence['n_total']}")
    lines.append("  " + "=" * 60)
    lines.append(f"\n  Output: {output_dir}/")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    print(summary, flush=True)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
