#!/usr/bin/env python3
"""
t* Stability Analysis: Systematic Evaluation of Change-Point Detection.

Evaluates whether the latent anomaly peak t* = argmax S(t) is:
  (A) stable system property
  (B) noise-driven artifact
  (C) prompt-dependent control signal

Analyses:
  1. Stability across seeds (per-prompt variance)
  2. Stability across prompts (inter-prompt variance)
  3. Temporal consistency with diffusion length (20/30/40/50 steps)
  4. Permutation test against null (shuffled S)

Usage:
    python scripts/ts_stability.py [--dry-run] [--output-dir OUTPUT_DIR]
"""

import argparse, json, os, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy import signal as scipy_signal
from scipy.stats import pearsonr

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
TRAJ_DIR = Path("output/dynamics_experiment/trajectories")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--traj-dir", default=str(TRAJ_DIR))
parser.add_argument("--output-dir", default="output/ts_stability")
parser.add_argument("--device", default="cuda")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--temporal-seeds", type=int, default=3)
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSIENT = 3

PROMPT_MAP = {
    "electro": "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
    "rock": "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
    "ballad_m": "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
    "pop_f": "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
    "folk": "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major",
    "dancepop": "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor",
    "cpop": "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major",
}
SHORT = {v: k for k, v in PROMPT_MAP.items()}

# ====================================================================
# DYNAMICS EXTRACTION
# ====================================================================
def extract_dynamics(X):
    v = np.diff(X, axis=0)
    v_norm = norm(v, axis=1)
    v_unit = v / (v_norm[:, None] + 1e-10)
    a = np.diff(v, axis=0)
    a_norm = norm(a, axis=1)
    cos_sim = np.clip(np.sum(v_unit[:-1] * v_unit[1:], axis=1), -1.0, 1.0)
    curvature = 1.0 - cos_sim
    return {"v": v, "v_norm": v_norm, "v_unit": v_unit, "a": a, "a_norm": a_norm, "curvature": curvature}

# ====================================================================
# BUILD REFERENCE
# ====================================================================
def build_reference(dyn_dict, exp_keys):
    ref = {}
    for name, n_steps in [("v_norm", 29), ("curvature", 28), ("a_norm", 28)]:
        by_step = {}
        for t in range(n_steps):
            vals = []
            for p, s in exp_keys:
                d = dyn_dict[p][s]
                arr = {"v_norm": d["v_norm"], "curvature": d["curvature"], "a_norm": d["a_norm"]}[name]
                if t < len(arr):
                    vals.append(arr[t])
            vals = np.array(vals)
            by_step[t] = (vals.mean(), vals.std() + 1e-10, vals)
        ref[name] = by_step
    return ref

# ====================================================================
# COMPUTE S(t) AND t*
# ====================================================================
def compute_S_and_tstar(d, ref, steps_total=30):
    vn, c, an = d["v_norm"], d["curvature"], d["a_norm"]

    vn_z = np.array([abs(vn[t] - ref["v_norm"][t][0]) / ref["v_norm"][t][1] if t < len(vn) else 0 for t in range(steps_total - 1)])
    c_z = np.array([abs(c[t] - ref["curvature"][t][0]) / ref["curvature"][t][1] if t < len(c) else 0 for t in range(steps_total - 2)])
    an_z = np.array([abs(an[t] - ref["a_norm"][t][0]) / ref["a_norm"][t][1] if t < len(an) else 0 for t in range(steps_total - 2)])

    S_trans = np.zeros(steps_total - 1)
    S_trans += 0.35 * vn_z
    S_trans[1:] += 0.35 * c_z[:len(S_trans)-1]
    S_trans[1:] += 0.30 * an_z[:len(S_trans)-1]

    S_step = np.zeros(steps_total)
    S_step[0] = 0
    S_step[1:] = S_trans

    # t* = argmax in post-transient region
    steady = S_step[TRANSIENT:]
    t_star = int(np.argmax(steady)) + TRANSIENT
    S_max = float(steady.max())

    # Spectral entropy of S
    p = S_step / (S_step.sum() + 1e-10)
    entropy = float(-np.sum(p * np.log(p + 1e-10)))

    return {"S": S_step.tolist(), "t_star": t_star, "S_max": S_max,
            "entropy": entropy, "S_trans": S_trans.tolist()}

# ====================================================================
# LOAD DATA
# ====================================================================
def load_all(traj_dir):
    all_trajs, exp_keys = defaultdict(dict), []
    for fpath in sorted(Path(traj_dir).glob("*.npy")):
        stem = fpath.stem.replace("traj_", "")
        parts = stem.split("_s")
        if len(parts) != 2: continue
        pname, seed_str = parts
        prompt = PROMPT_MAP.get(pname, pname)
        seed = int(seed_str)
        all_trajs[prompt][seed] = np.load(fpath)
        exp_keys.append((prompt, seed))
    return all_trajs, exp_keys

# ====================================================================
# PERMUTATION TEST
# ====================================================================
def permutation_test(S_step_list, n_perm=10000):
    """Shuffle S within each run, recompute t*, compare with real."""
    real_t = np.array([s["t_star"] for s in S_step_list])
    real_mean = real_t.mean()
    perm_means = []
    for _ in range(n_perm):
        perm_t = []
        for s in S_step_list:
            S = np.array(s["S"])
            np.random.shuffle(S)
            steady = S[TRANSIENT:]
            perm_t.append(int(np.argmax(steady)) + TRANSIENT)
        perm_means.append(np.mean(perm_t))
    perm_means = np.array(perm_means)
    # Two-tailed p-value: fraction of permuted means as extreme as real
    p_val = float(np.mean(np.abs(perm_means - np.mean(perm_means)) >= np.abs(real_mean - np.mean(perm_means))))
    return {"real_mean": float(real_mean), "perm_mean": float(np.mean(perm_means)),
            "perm_std": float(np.std(perm_means)), "p_value": p_val}

# ====================================================================
# VISUALIZATION
# ====================================================================
def visualize(per_run_list, results, temporal_results, perm_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. t* per prompt × seed (heatmap style)
    fig, ax = plt.subplots(figsize=(10, 6))
    ts_data = per_run_list
    prompts = list(results["per_prompt"].keys())
    seeds = sorted(set(r["seed"] for r in ts_data))
    t_mat = np.full((len(prompts), len(seeds)), np.nan)
    for r in ts_data:
        pi = prompts.index(r["prompt_short"])
        si = seeds.index(r["seed"])
        t_mat[pi, si] = r["t_star"]
    im = ax.imshow(t_mat, cmap="viridis", aspect="auto", vmin=0, vmax=29)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds, fontsize=8)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts, fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Prompt")
    ax.set_title("t* = argmax S(t) by Prompt × Seed")
    plt.colorbar(im, ax=ax, label="t*", shrink=0.8)
    for pi in range(len(prompts)):
        for si in range(len(seeds)):
            if not np.isnan(t_mat[pi, si]):
                ax.text(si, pi, f"{int(t_mat[pi, si])}", ha="center", va="center", fontsize=7,
                       color="white" if t_mat[pi, si] < 15 else "black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tstar_heatmap.png", dpi=150)
    plt.close()
    print("  [plot] tstar_heatmap.png")

    # 2. t* distribution per prompt (boxplot)
    fig, ax = plt.subplots(figsize=(12, 5))
    bp_data = [results["per_prompt"][p]["tstar_list"] for p in prompts]
    bp = ax.boxplot(bp_data, labels=prompts, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(prompts)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("t*")
    ax.set_title("t* Distribution by Prompt (box = 4 seeds)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tstar_boxplot.png", dpi=150)
    plt.close()
    print("  [plot] tstar_boxplot.png")

    # 3. Temporal scaling: t* vs T (diffusion steps)
    if temporal_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, data in temporal_results.items():
            Ts = sorted(data.keys())
            ts_vals = [np.mean(data[T]) for T in Ts]
            ratios = [np.mean(data[T]) / T for T in Ts]
            ax.plot(Ts, ts_vals, "o-", label=label, linewidth=2)
        ax.set_xlabel("Total Diffusion Steps (T)")
        ax.set_ylabel("Mean t*")
        ax.set_title("t* Scaling with Diffusion Length")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tstar_temporal_scaling.png", dpi=150)
        plt.close()
        print("  [plot] tstar_temporal_scaling.png")

        # t*/T ratio
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, data in temporal_results.items():
            Ts = sorted(data.keys())
            ratios = [np.mean(data[T]) / T for T in Ts]
            ax.plot(Ts, ratios, "s-", label=label, linewidth=2)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="t*/T = 0.5")
        ax.set_xlabel("Total Diffusion Steps (T)")
        ax.set_ylabel("t* / T")
        ax.set_title("Normalized Change-Point (t*/T)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tstar_ratio_scaling.png", dpi=150)
        plt.close()
        print("  [plot] tstar_ratio_scaling.png")

    # 4. Permutation test
    if perm_result:
        fig, ax = plt.subplots(figsize=(8, 4))
        perm_means = perm_result.get("_perm_means", [perm_result["perm_mean"]])
        ax.hist(perm_means, bins=30, alpha=0.7, color="steelblue", edgecolor="white", label="Permuted")
        ax.axvline(perm_result["real_mean"], color="red", ls="--", lw=2, label=f"Real μ={perm_result['real_mean']:.1f}")
        ax.axvline(perm_result["perm_mean"], color="orange", ls="--", lw=1.5, label=f"Perm μ={perm_result['perm_mean']:.1f}")
        ax.set_xlabel("Mean t*")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation Test (p={perm_result['p_value']:.4f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "permutation_test.png", dpi=150)
        plt.close()
        print("  [plot] permutation_test.png")

    # 5. S(t) overlay with t* marked
    fig, ax = plt.subplots(figsize=(12, 5))
    for r in ts_data:
        ax.plot(np.arange(len(r["S"])), r["S"], alpha=0.3, linewidth=0.5, color="steelblue")
        ax.scatter(r["t_star"], r["S_max"], color="red", s=10, alpha=0.5, zorder=5)
    ax.axvline(TRANSIENT, color="gray", ls="--", alpha=0.5, label="transient end")
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("S(t)")
    ax.set_title("All S(t) Curves with t* Marked (red dots)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_S_with_tstar.png", dpi=150)
    plt.close()
    print("  [plot] all_S_with_tstar.png")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 70)
    print("  t* Stability Analysis")
    print("=" * 70)

    # ---- Step 1: Load trajectories and compute S(t) ----
    print("\n[1/4] Loading trajectories and computing S(t)...")
    all_trajs, exp_keys = load_all(args.traj_dir)
    print(f"  {len(exp_keys)} runs loaded")

    dyn_dict = defaultdict(dict)
    for p, s in exp_keys:
        dyn_dict[p][s] = extract_dynamics(all_trajs[p][s])

    ref = build_reference(dyn_dict, exp_keys)
    print(f"  Reference built: {len(ref['v_norm'])} velocity steps")

    # Compute S(t) and t* for each run
    per_run = []
    for p, s in exp_keys:
        result = compute_S_and_tstar(dyn_dict[p][s], ref, steps_total=30)
        per_run.append({
            "prompt": p, "seed": s, "prompt_short": SHORT.get(p, "?"),
            "t_star": result["t_star"], "S_max": result["S_max"],
            "entropy": result["entropy"], "S": result["S"],
        })

    # ---- Step 2: Stability analysis ----
    print("\n[2/4] Stability analysis...")

    # Per-prompt stats
    per_prompt = {}
    for r in per_run:
        key = r["prompt_short"]
        if key not in per_prompt:
            per_prompt[key] = {"tstar_list": [], "smax_list": [], "entropy_list": []}
        per_prompt[key]["tstar_list"].append(r["t_star"])
        per_prompt[key]["smax_list"].append(r["S_max"])
        per_prompt[key]["entropy_list"].append(r["entropy"])

    prompts = sorted(per_prompt.keys())
    tstars_all = np.array([r["t_star"] for r in per_run])

    # Cross-seed variance per prompt
    cross_seed_var = np.mean([np.var(per_prompt[p]["tstar_list"]) for p in prompts])
    cross_seed_std = np.sqrt(cross_seed_var)

    # Cross-prompt variance (variance of prompt means)
    prompt_means = np.array([np.mean(per_prompt[p]["tstar_list"]) for p in prompts])
    cross_prompt_var = float(np.var(prompt_means))
    cross_prompt_std = float(np.std(prompt_means))

    # ANOVA-like: F = cross_prompt_var / cross_seed_var
    anova_F = cross_prompt_var / (cross_seed_var + 1e-10)

    # PCA on S(t) profiles to check if shape is consistent
    S_matrix = np.array([r["S"] for r in per_run])
    from sklearn.decomposition import PCA
    pca_S = PCA(n_components=2)
    S_pca = pca_S.fit_transform(S_matrix)
    S_pca_var = pca_S.explained_variance_ratio_

    results = {
        "n_runs": len(per_run),
        "mean_tstar": float(tstars_all.mean()),
        "std_tstar": float(tstars_all.std()),
        "min_tstar": int(tstars_all.min()),
        "max_tstar": int(tstars_all.max()),
        "cross_seed_variance": float(cross_seed_var),
        "cross_seed_std": float(cross_seed_std),
        "cross_prompt_variance": cross_prompt_var,
        "cross_prompt_std": cross_prompt_std,
        "anova_F_ratio": float(anova_F),
        "S_pca_var_ratio": S_pca_var.tolist(),
        "per_run": [dict((k, v) for k, v in r.items() if k != "S") for r in per_run],  # stripped for JSON
        "per_prompt": {},
    }
    for p in prompts:
        results["per_prompt"][p] = {
            "mean_tstar": float(np.mean(per_prompt[p]["tstar_list"])),
            "std_tstar": float(np.std(per_prompt[p]["tstar_list"])),
            "var_tstar": float(np.var(per_prompt[p]["tstar_list"])),
            "mean_Smax": float(np.mean(per_prompt[p]["smax_list"])),
            "mean_entropy": float(np.mean(per_prompt[p]["entropy_list"])),
            "tstar_list": per_prompt[p]["tstar_list"],
        }

    print(f"  Mean t*: {results['mean_tstar']:.1f} ± {results['std_tstar']:.1f}")
    print(f"  Cross-seed variance: {cross_seed_var:.2f}")
    print(f"  Cross-prompt variance: {cross_prompt_var:.2f}")
    print(f"  F(inter/intra): {anova_F:.2f}")
    print(f"  S(t) PCA PC1: {S_pca_var[0]:.1%}")

    # ---- Step 3: Temporal consistency (20/30/40/50 steps) ----
    print("\n[3/4] Temporal consistency check...")
    temporal_results = {}
    temporal_agg = {}
    temporal_prompts = ["electro", "ballad_m"]  # 2 diverse prompts
    temporal_seeds = [42, 123, 999][:args.temporal_seeds]
    temporal_steps = [20, 30, 40, 50]
    TEMP_TRAJ_DIR = OUTPUT_DIR / "temporal_trajs"
    TEMP_TRAJ_DIR.mkdir(exist_ok=True)

    if not args.dry_run:
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        import torch

        class _Collector:
            def __init__(self):
                self.states = []
            def __call__(self, mod, inp, out):
                hs = out[0]
                if hs.shape[0] > 1:
                    hs = hs[:1]
                self.states.append(hs.mean(dim=1).detach().cpu())
            def get_traj(self):
                if not self.states:
                    return None
                return torch.cat(self.states, dim=0).float().numpy()
            def reset(self):
                self.states = []

        dit_handler = AceStepHandler()
        dit_status, dit_success = dit_handler.initialize_service(
            project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
            device=args.device, use_flash_attention=False, compile_model=False, offload_to_cpu=False)
        if not dit_success:
            print(f"  Model load failed: {dit_status}")
        else:
            for layer_mod in dit_handler.model.decoder.layers:
                if getattr(layer_mod, "use_phase_memory", False):
                    layer_mod.use_phase_memory = False

            collector = _Collector()
            handle = dit_handler.model.decoder.layers[12].register_forward_hook(collector)

            for pname in temporal_prompts:
                prompt = PROMPT_MAP[pname]
                for n_steps in temporal_steps:
                    key = f"{pname}_{n_steps}"
                    temporal_results[key] = {}
                    for seed in temporal_seeds:
                        collector.reset()
                        try:
                            params = GenerationParams(caption=prompt, lyrics="[Instrumental]",
                                instrumental=True, duration=30, inference_steps=n_steps,
                                guidance_scale=5.0, seed=seed, thinking=False)
                            config = GenerationConfig(batch_size=1, audio_format="mp3", use_random_seed=False)
                            result = generate_music(dit_handler, None, params, config)
                            X = collector.get_traj()
                            if X is not None and len(X) >= 4:
                                fname = f"traj_{pname}_T{n_steps}_s{seed}.npy"
                                np.save(TEMP_TRAJ_DIR / fname, X)

                                v = np.linalg.norm(np.diff(X, axis=0), axis=1)
                                if len(v) >= 5:
                                    v_smooth = scipy_signal.savgol_filter(v, min(5, len(v) - (1 - len(v) % 2)), 2)
                                else:
                                    v_smooth = v
                                v_min_step = int(np.argmin(v_smooth))
                                temporal_results[key][str(seed)] = {"t_star": v_min_step,
                                    "t_star_norm": v_min_step / n_steps, "v_min": float(v[v_min_step])}
                                print(f"    {pname} T={n_steps} s{seed}: t*={v_min_step} ({v_min_step/n_steps:.3f})")
                        except Exception as e:
                            print(f"    {pname} T={n_steps} s{seed}: FAILED {e}")
            handle.remove()
    else:
        # Load cached temporal trajectories — use velocity minimum (reference-free)
        for fpath in sorted(TEMP_TRAJ_DIR.glob("*.npy")):
            stem = fpath.stem.replace("traj_", "")
            parts = stem.split("_T")
            if len(parts) != 2: continue
            pname = parts[0]
            rest = parts[1].split("_s")
            if len(rest) != 2: continue
            n_steps = int(rest[0])
            seed = int(rest[1])
            key = f"{pname}_{n_steps}"
            if key not in temporal_results:
                temporal_results[key] = {}
            X = np.load(fpath)
            v = np.linalg.norm(np.diff(X, axis=0), axis=1)
            if len(v) >= 5:
                v_smooth = scipy_signal.savgol_filter(v, min(5, len(v) - (1 - len(v) % 2)), 2)
            else:
                v_smooth = v
            v_min_step = int(np.argmin(v_smooth))
            temporal_results[key][str(seed)] = {"t_star": v_min_step,
                "t_star_norm": v_min_step / n_steps, "v_min": float(v[v_min_step])}
        print(f"  Loaded {len(list(TEMP_TRAJ_DIR.glob('*.npy')))} cached temporal trajectories")

    # Aggregate temporal data
    temporal_agg = {}
    if temporal_results:
        for pname in temporal_prompts:
            temporal_agg[pname] = {}
            for ns in temporal_steps:
                key = f"{pname}_{ns}"
                vals = [v["t_star"] for v in temporal_results.get(key, {}).values()]
                if vals:
                    temporal_agg[pname][ns] = vals

    # ---- Step 4: Permutation test ----
    print("\n[4/4] Permutation test against null hypothesis...")
    perm_result = permutation_test(per_run, n_perm=5000)
    print(f"  Real mean t*: {perm_result['real_mean']:.2f}")
    print(f"  Permuted mean: {perm_result['perm_mean']:.2f} ± {perm_result['perm_std']:.2f}")
    print(f"  p-value: {perm_result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {perm_result['p_value'] < 0.05}")

    # ---- Visualize ----
    print("\n  Generating plots...")
    visualize(per_run, results, temporal_agg, perm_result)

    # ---- Summary ----
    lines = []
    lines.append("=" * 70)
    lines.append("  t* STABILITY ANALYSIS — SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\n  Runs analyzed: {len(per_run)}")
    lines.append(f"  Diffusion steps: 30 (default), 20/40/50 (temporal subset)")
    lines.append("")
    lines.append(f"  [1] t* = argmax S(t) (post-transient)")
    lines.append(f"    Mean t*:        {results['mean_tstar']:.1f}")
    lines.append(f"    Std t*:         {results['std_tstar']:.1f}")
    lines.append(f"    Range:          [{results['min_tstar']}, {results['max_tstar']}]")
    lines.append("")
    lines.append(f"  [2] Stability")
    lines.append(f"    Cross-seed variance:  {cross_seed_var:.2f} (σ={cross_seed_std:.2f})")
    lines.append(f"    Cross-prompt variance: {cross_prompt_var:.2f} (σ={cross_prompt_std:.2f})")
    lines.append(f"    F-ratio (inter/intra): {anova_F:.2f}")
    lines.append("")
    lines.append(f"  [3] Per-prompt t*")
    for p in prompts:
        pp = per_prompt[p]
        lines.append(f"    {p:>12}: μ={np.mean(pp['tstar_list']):.1f} σ={np.std(pp['tstar_list']):.1f} "
                     f"var={np.var(pp['tstar_list']):.1f} S_maxμ={np.mean(pp['smax_list']):.2f}")
    lines.append("")
    if temporal_agg:
        lines.append(f"  [4] Temporal scaling (t* / T)")
        for pname in temporal_prompts:
            if pname in temporal_agg:
                ratios = []
                for ns in sorted(temporal_agg[pname].keys()):
                    ts = temporal_agg[pname][ns]
                    r = np.mean(ts) / ns
                    ratios.append(r)
                    lines.append(f"    {pname:>10} T={ns:2d}: t*={np.mean(ts):.1f}  t*/T={r:.3f}")
                if len(ratios) >= 2:
                    lines.append(f"    {'':>10} constancy (std of t*/T): {np.std(ratios):.4f}")

    lines.append("")
    lines.append(f"  [5] Permutation test (n={5000}):")
    lines.append(f"    p-value: {perm_result['p_value']:.4f}")
    lines.append(f"    Significant: {perm_result['p_value'] < 0.05}")
    lines.append("")
    lines.append("  [6] S(t) PCA: PC1={:.1%}, PC2={:.1%}".format(S_pca_var[0], S_pca_var[1]))
    lines.append("")
    lines.append("  CONCLUSION:")
    if perm_result["p_value"] >= 0.05:
        lines.append("    t* is NOT distinguishable from random (null hypothesis NOT rejected)")
        lines.append("    → t* is a noise-driven artifact")
    elif cross_prompt_var > 2 * cross_seed_var:
        lines.append("    t* is prompt-dependent (cross-prompt variance dominates)")
        lines.append("    → t* is a prompt-dependent control signal")
    else:
        lines.append("    t* is stable across both seeds and prompts")
        lines.append("    → t* is a stable system property")
    lines.append("=" * 70)

    print("\n".join(lines))
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write("\n".join(lines))

    # Full report
    full_report = {
        "stats": {k: v for k, v in results.items() if k not in ("per_run",)},
        "per_run": per_run,
        "temporal": temporal_agg,
        "permutation": perm_result,
        "conclusion": {
            "is_significant": bool(perm_result["p_value"] < 0.05),
            "cross_seed_var": float(cross_seed_var),
            "cross_prompt_var": cross_prompt_var,
            "F_ratio": float(anova_F),
        }
    }
    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n  Report: {OUTPUT_DIR / 'report.json'}")
    print(f"  Summary: {OUTPUT_DIR / 'summary.txt'}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
