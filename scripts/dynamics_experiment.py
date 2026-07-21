#!/usr/bin/env python3
"""
Systematic DiT Hidden Dynamics Experiment Pipeline.

Runs controlled experiments across multiple prompts × seeds, collects
hidden state trajectories from decoder.layers[12], and analyzes whether
DiT exhibits stable low-dimensional temporal dynamics.

Usage:
    python scripts/dynamics_experiment.py [--output-dir OUTPUT_DIR]

Output:
    output/dynamics_experiment/
    ├── trajectories/          # Per-run .npy trajectory files
    ├── per_run/               # Per-run plots
    ├── pca_grid.png           # 4×3 PCA trajectories
    ├── velocity_overlay.png   # All velocity profiles
    ├── similarity_matrix.png  # Cross-run cosine similarity
    ├── pca_alignment.png      # PCA subspace alignment
    ├── variance_decomp.png    # Within/between-prompt variance
    ├── spectral_overlay.png   # FFT power spectra
    ├── report.json            # Full numeric results
    └── summary_table.txt      # Human-readable summary
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Experiment config — real-world music captions from /root/autodl-tmp/musicdata/
# ---------------------------------------------------------------------------
PROMPTS = [
    "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
    "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
    "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
    "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
    "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major",
    "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor",
    "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major",
]

SEEDS = [42, 123, 999, 7]

SHORT_LABELS = {
    "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major": "Electro Pop",
    "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor": "Rock",
    "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major": "Ballad (M)",
    "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major": "Pop (F)",
    "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major": "Folk",
    "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor": "Dance Pop",
    "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major": "C-Pop Ballad",
}
SHORT_FILENAME = {
    "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major": "electro",
    "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor": "rock",
    "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major": "ballad_m",
    "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major": "pop_f",
    "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major": "folk",
    "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor": "dancepop",
    "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major": "cpop",
}

LAYER = 12
STEPS = 30
GUIDANCE = 5.0
DURATION = 30

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


# ========== ARGS ==========
parser = argparse.ArgumentParser(description="DiT Dynamics Experiment")
parser.add_argument("--output-dir", type=str, default="output/dynamics_experiment")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dry-run", action="store_true", help="Skip generation, only analyze existing trajectories")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
TRAJ_DIR = OUTPUT_DIR / "trajectories"
PER_RUN_DIR = OUTPUT_DIR / "per_run"
for d in [OUTPUT_DIR, TRAJ_DIR, PER_RUN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ========== HOOK ==========
class HiddenStateCollector:
    """Collects hidden states from a forward hook, one per diffusion step."""

    def __init__(self):
        self.states = []

    def __call__(self, module, inp, out):
        hs = out[0]  # [B, T, D]
        if hs.shape[0] > 1:
            hs = hs[:1]  # take conditional half under CFG
        x_t = hs.mean(dim=1)  # [1, D]
        self.states.append(x_t.detach().cpu())

    @property
    def trajectory(self):
        if not self.states:
            return None
        return torch.cat(self.states, dim=0).float().numpy()  # [steps, D]

    def reset(self):
        self.states = []


# ========== PER-RUN METRICS ==========
def compute_per_run_metrics(X):
    """Compute all metrics for a single trajectory X ∈ [steps, D]."""
    from sklearn.decomposition import PCA

    steps, dim = X.shape
    metrics = {"steps": steps, "dim": dim}

    # --- PCA ---
    pca = PCA(n_components=min(4, steps, dim))
    X_pca = pca.fit_transform(X)
    metrics["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
    metrics["pca_components"] = pca.components_.tolist()
    metrics["pca_2d"] = X_pca[:, :2].tolist()

    # --- Effective rank ---
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc / (steps - 1)
    s = np.linalg.svd(C, compute_uv=False)
    p = s / (s.sum() + 1e-10)
    H = -np.sum(p * np.log(p + 1e-10))
    metrics["effective_rank"] = float(np.exp(H))
    metrics["singular_values"] = s.tolist()

    # --- Velocity ---
    diffs = np.diff(X, axis=0)
    v = np.linalg.norm(diffs, axis=1)
    metrics["velocity"] = v.tolist()
    metrics["velocity_mean"] = float(np.mean(v))
    metrics["velocity_std"] = float(np.std(v))
    metrics["velocity_max"] = float(np.max(v))
    metrics["velocity_peak_steps"] = [int(i) for i in np.argsort(v)[-5:][::-1]]

    # --- Acceleration ---
    acc = np.diff(v)
    metrics["acceleration"] = acc.tolist()
    metrics["acceleration_mean"] = float(np.mean(acc))

    # --- Curvature ---
    if steps >= 3:
        d1 = X[1:-1] - X[:-2]
        d2 = X[2:] - X[1:-1]
        dot = np.sum(d1 * d2, axis=1)
        norm = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-8
        kappa = 1.0 - (dot / norm)
        metrics["curvature"] = kappa.tolist()
        metrics["curvature_mean"] = float(np.mean(kappa))
        metrics["curvature_std"] = float(np.std(kappa))
        metrics["curvature_peak_steps"] = [int(i) for i in np.argsort(kappa)[-5:][::-1]]
    else:
        metrics["curvature_mean"] = 0.0

    # --- FFT ---
    pc1 = X_pca[:, 0]
    fft_vals = np.fft.rfft(pc1)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(steps)
    metrics["fft_freqs"] = freqs.tolist()
    metrics["fft_power"] = power.tolist()
    # Dominant (excluding DC)
    power_ac = power[1:]
    freqs_ac = freqs[1:]
    if len(power_ac) > 0:
        top_idx = np.argsort(power_ac)[-5:][::-1]
        total_power = power_ac.sum()
        metrics["dominant_fft_bins"] = freqs_ac[top_idx].tolist()
        metrics["dominant_fft_power_ratios"] = (power_ac[top_idx] / (total_power + 1e-10)).tolist()
        # Spectral entropy
        p_dist = power_ac / (total_power + 1e-10)
        metrics["spectral_entropy"] = float(-np.sum(p_dist * np.log(p_dist + 1e-10)))
    else:
        metrics["dominant_fft_bins"] = []
        metrics["spectral_entropy"] = 0.0

    # --- Signal-to-noise proxy: smoothness (ratio of low-freq power) ---
    if len(power_ac) > 2:
        low_cut = max(1, len(power_ac) // 4)  # lowest 25% of freqs
        low_power = power_ac[:low_cut].sum()
        high_power = power_ac[low_cut:].sum() + 1e-10
        metrics["smoothness_ratio"] = float(low_power / high_power)
    else:
        metrics["smoothness_ratio"] = 1.0

    return metrics


# ========== CROSS-RUN METRICS ==========
def cosine_similarity(X, Y):
    """Cosine similarity between trajectory matrices: trace-based."""
    Xf = X.reshape(X.shape[0], -1)
    Yf = Y.reshape(Y.shape[0], -1)
    Xn = Xf / (np.linalg.norm(Xf) + 1e-10)
    Yn = Yf / (np.linalg.norm(Yf) + 1e-10)
    return float(np.dot(Xn.ravel(), Yn.ravel()))


def pca_alignment_score(X, Y):
    """
    How well does PCA fit on X explain Y?

    Fit PCA on X, compute variance of Y explained by X's top-2 PCs.
    Returns a value in [0, 1].
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(4, X.shape[0], X.shape[1]))
    pca.fit(X)
    Y_proj = pca.transform(Y)
    # Total variance of Y = ||Y||²_F
    total_var = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2)
    # Explained = ||Y_proj @ components||²_F (reconstructed variance)
    Y_recon = Y_proj @ pca.components_[: Y_proj.shape[1]]
    Y_recon += Y.mean(axis=0, keepdims=True)  # add mean back
    explained_var = 1 - np.sum((Y - Y_recon) ** 2) / (total_var + 1e-10)
    return float(np.clip(explained_var, 0, 1))


def first_pc_similarity(X, Y):
    """Cosine similarity between PC1 directions of two trajectories."""
    from sklearn.decomposition import PCA

    pc1_x = PCA(n_components=1).fit(X).components_[0]
    pc1_y = PCA(n_components=1).fit(Y).components_[0]
    cos_sim = float(np.dot(pc1_x, pc1_y) / (np.linalg.norm(pc1_x) * np.linalg.norm(pc1_y) + 1e-10))
    return cos_sim


# ========== VISUALIZATION ==========
def visualize_experiment(all_trajs, all_metrics, exp_keys):
    """Generate all summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Color map for prompts
    prompt_names = list(dict.fromkeys([k[0] for k in exp_keys]))
    prompt_colors = plt.cm.tab10(np.linspace(0, 1, len(prompt_names)))
    prompt_color_map = {p: prompt_colors[i] for i, p in enumerate(prompt_names)}

    # Short labels
    short_labels = SHORT_LABELS

    n_runs = len(exp_keys)

    # ====================== 1. PCA trajectory grid ======================
    n_prompts = len(prompt_names)
    n_cols = min(3, n_prompts)
    n_rows = int(np.ceil(n_prompts / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_flat = axes.flatten() if n_prompts > 1 else [axes]
    seed_cmap = plt.cm.tab10
    for pi, prompt in enumerate(prompt_names):
        ax = axes_flat[pi]
        seeds_for_prompt = [s for p, s in exp_keys if p == prompt]
        for si, seed in enumerate(seeds_for_prompt):
            X = all_trajs[prompt][seed]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            color = seed_cmap(si / max(len(seeds_for_prompt), 1))
            ax.plot(X_pca[:, 0], X_pca[:, 1], color=color, alpha=0.6, linewidth=0.8)
            ax.scatter(X_pca[0, 0], X_pca[0, 1], color=color, s=15, marker="o", zorder=5)
            ax.scatter(X_pca[-1, 0], X_pca[-1, 1], color=color, s=15, marker="x", zorder=5)
        ax.set_title(short_labels.get(prompt, prompt[:20]), fontsize=12)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    for pi in range(n_prompts, len(axes_flat)):
        axes_flat[pi].set_visible(False)
    fig.suptitle("PCA Trajectories — All Seeds Overlaid per Prompt (o=start, x=end)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] pca_grid.png")

    # ====================== 2. Velocity overlay ======================
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (prompt, seed) in enumerate(exp_keys):
        v = all_metrics[prompt][seed]["velocity"]
        alpha = 0.5 if len(set([k[1] for k in exp_keys if k[0] == prompt])) > 1 else 1.0
        ax.plot(v, color=prompt_color_map[prompt], alpha=0.5, linewidth=1.0,
                label=short_labels.get(prompt, prompt[:20]) if seed == SEEDS[0] else "")
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Velocity (L2)")
    ax.set_title("Velocity Profiles Across All Runs (color = prompt)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "velocity_overlay.png", dpi=150)
    plt.close()
    print(f"  [plot] velocity_overlay.png")

    # ====================== 3. Similarity matrix ======================
    sim_mat = np.zeros((n_runs, n_runs))
    for i, (p_i, s_i) in enumerate(exp_keys):
        for j, (p_j, s_j) in enumerate(exp_keys):
            sim_mat[i, j] = cosine_similarity(all_trajs[p_i][s_i], all_trajs[p_j][s_j])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n_runs))
    ax.set_yticks(range(n_runs))
    labels = [f"{short_labels.get(p, p[:15])[:15]}\ns{p}" for p, s in exp_keys]
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, label="Cosine Similarity", shrink=0.8)
    ax.set_title("Cross-Run Trajectory Similarity", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "similarity_matrix.png", dpi=150)
    plt.close()
    print(f"  [plot] similarity_matrix.png")

    # ====================== 4. PCA subspace alignment ======================
    align_mat = np.zeros((n_runs, n_runs))
    for i, (p_i, s_i) in enumerate(exp_keys):
        for j, (p_j, s_j) in enumerate(exp_keys):
            align_mat[i, j] = pca_alignment_score(all_trajs[p_i][s_i], all_trajs[p_j][s_j])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(align_mat, cmap="plasma", vmin=0, vmax=1)
    ax.set_xticks(range(n_runs))
    ax.set_yticks(range(n_runs))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, label="PCA Alignment Score", shrink=0.8)
    ax.set_title("Cross-Run PCA Subspace Alignment", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_alignment.png", dpi=150)
    plt.close()
    print(f"  [plot] pca_alignment.png")

    # ====================== 5. Spectral overlay ======================
    n_spec_cols = min(4, n_prompts)
    n_spec_rows = int(np.ceil(n_prompts / n_spec_cols))
    fig, axes = plt.subplots(n_spec_rows, n_spec_cols, figsize=(5 * n_spec_cols, 4 * n_spec_rows))
    axes_spec = axes.flatten() if n_prompts > 1 else [axes]
    for pi, prompt in enumerate(prompt_names):
        ax = axes_spec[pi]
        for seed in SEEDS:
            if seed not in all_metrics[prompt]:
                continue
            m = all_metrics[prompt][seed]
            ax.plot(m["fft_freqs"][1:], m["fft_power"][1:], alpha=0.7, linewidth=1,
                    label=f"seed {seed}")
        ax.set_title(short_labels.get(prompt, prompt[:20]))
        ax.set_xlabel("Frequency (cycles/step)")
        ax.set_ylabel("Power")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for pi in range(n_prompts, len(axes_spec)):
        axes_spec[pi].set_visible(False)
    fig.suptitle("FFT Power Spectra (DC removed)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectral_overlay.png", dpi=150)
    plt.close()
    print(f"  [plot] spectral_overlay.png")

    # ====================== 6. Variance decomposition ======================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # a) Effective rank comparison
    ax = axes[0]
    for pi, prompt in enumerate(prompt_names):
        ranks = [all_metrics[prompt][s]["effective_rank"] for s in SEEDS if s in all_metrics[prompt]]
        ax.bar(pi, np.mean(ranks), color=prompt_colors[pi], alpha=0.7, width=0.5)
        ax.scatter([pi] * len(ranks), ranks, color="black", s=20, zorder=5)
    ax.set_xticks(range(len(prompt_names)))
    ax.set_xticklabels([short_labels.get(p, p[:15]) for p in prompt_names], fontsize=8, rotation=15)
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank by Prompt")
    ax.grid(alpha=0.3, axis="y")

    # b) Velocity mean comparison
    ax = axes[1]
    for pi, prompt in enumerate(prompt_names):
        vals = [all_metrics[prompt][s]["velocity_mean"] for s in SEEDS if s in all_metrics[prompt]]
        ax.bar(pi, np.mean(vals), color=prompt_colors[pi], alpha=0.7, width=0.5)
        ax.scatter([pi] * len(vals), vals, color="black", s=20, zorder=5)
    ax.set_xticks(range(len(prompt_names)))
    ax.set_xticklabels([short_labels.get(p, p[:15]) for p in prompt_names], fontsize=8, rotation=15)
    ax.set_ylabel("Mean Velocity")
    ax.set_title("Mean Velocity by Prompt")
    ax.grid(alpha=0.3, axis="y")

    # c) Spectral entropy comparison
    ax = axes[2]
    for pi, prompt in enumerate(prompt_names):
        vals = [all_metrics[prompt][s]["spectral_entropy"] for s in SEEDS if s in all_metrics[prompt]]
        ax.bar(pi, np.mean(vals), color=prompt_colors[pi], alpha=0.7, width=0.5)
        ax.scatter([pi] * len(vals), vals, color="black", s=20, zorder=5)
    ax.set_xticks(range(len(prompt_names)))
    ax.set_xticklabels([short_labels.get(p, p[:15]) for p in prompt_names], fontsize=8, rotation=15)
    ax.set_ylabel("Spectral Entropy")
    ax.set_title("Spectral Entropy by Prompt")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "variance_decomp.png", dpi=150)
    plt.close()
    print(f"  [plot] variance_decomp.png")


# ========== SUMMARY TABLE ==========
def format_summary(all_trajs, all_metrics, exp_keys, sim_mat):
    """Return a human-readable summary string."""
    prompt_names = list(dict.fromkeys([k[0] for k in exp_keys]))

    lines = []
    lines.append("=" * 80)
    lines.append("  DiT Hidden Dynamics Experiment — Summary Report")
    lines.append("=" * 80)

    # Per-prompt stats
    lines.append("")
    lines.append(f"{'Prompt':<20} {'Eff Rank':>10} {'Vel μ':>10} {'Vel σ':>10} {'Curv μ':>10} {'Smooth':>10} {'Dom Freq':>10}")
    lines.append("-" * 80)

    for prompt in prompt_names:
        seeds_data = [all_metrics[prompt][s] for s in SEEDS if s in all_metrics[prompt]]
        if not seeds_data:
            continue
        er = np.mean([m["effective_rank"] for m in seeds_data])
        vm = np.mean([m["velocity_mean"] for m in seeds_data])
        vs = np.mean([m["velocity_std"] for m in seeds_data])
        cm = np.mean([m["curvature_mean"] for m in seeds_data])
        sm = np.mean([m.get("smoothness_ratio", 0) for m in seeds_data])
        df = np.mean([m["dominant_fft_bins"][0] if m["dominant_fft_bins"] else 0 for m in seeds_data])
        label = SHORT_LABELS.get(prompt, prompt[:18])
        lines.append(f"{label:<20} {er:>10.2f} {vm:>10.1f} {vs:>10.1f} {cm:>10.4f} {sm:>10.1f} {df:>10.4f}")

    # Cross-run similarity stats
    lines.append("")
    lines.append("-" * 80)
    n = len(exp_keys)
    within_prompt = []
    between_prompt = []
    for i in range(n):
        for j in range(i + 1, n):
            val = sim_mat[i, j]
            if exp_keys[i][0] == exp_keys[j][0]:
                within_prompt.append(val)
            else:
                between_prompt.append(val)

    lines.append(f"Within-prompt trajectory similarity (mean ± std): {np.mean(within_prompt):.4f} ± {np.std(within_prompt):.4f}")
    lines.append(f"Between-prompt trajectory similarity (mean ± std): {np.mean(between_prompt):.4f} ± {np.std(between_prompt):.4f}")
    if np.std(within_prompt) > 0:
        effect_size = (np.mean(within_prompt) - np.mean(between_prompt)) / np.std(within_prompt)
        lines.append(f"Effect size (Cohen's d): {effect_size:.3f}")

    # PCA alignment stats
    align_within = []
    align_between = []
    align_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            X, Y = all_trajs[exp_keys[i][0]][exp_keys[i][1]], all_trajs[exp_keys[j][0]][exp_keys[j][1]]
            align_mat[i, j] = pca_alignment_score(X, Y)
            if i != j:
                if exp_keys[i][0] == exp_keys[j][0]:
                    align_within.append(align_mat[i, j])
                else:
                    align_between.append(align_mat[i, j])
    lines.append(f"Within-prompt PCA alignment: {np.mean(align_within):.4f} ± {np.std(align_within):.4f}")
    lines.append(f"Between-prompt PCA alignment: {np.mean(align_between):.4f} ± {np.std(align_between):.4f}")

    # PC1 direction stability
    pc1_sims_within = []
    pc1_sims_between = []
    for i in range(n):
        for j in range(i + 1, n):
            X, Y = all_trajs[exp_keys[i][0]][exp_keys[i][1]], all_trajs[exp_keys[j][0]][exp_keys[j][1]]
            pc1_sim = first_pc_similarity(X, Y)
            if exp_keys[i][0] == exp_keys[j][0]:
                pc1_sims_within.append(pc1_sim)
            else:
                pc1_sims_between.append(pc1_sim)
    lines.append("")
    lines.append(f"PC1 direction similarity (within-prompt): {np.mean(pc1_sims_within):.4f} ± {np.std(pc1_sims_within):.4f}")
    lines.append(f"PC1 direction similarity (between-prompt): {np.mean(pc1_sims_between):.4f} ± {np.std(pc1_sims_between):.4f}")

    # Variance decomposition
    # Total variance = mean over all trajectories of ||X - global_mean||^2
    # We approximate: compute pooled within-prompt var and between-prompt var
    lines.append("")
    lines.append("-" * 80)
    lines.append("Variance Decomposition (approximate):")
    all_flat = []
    prompt_means = {}
    for prompt in prompt_names:
        trajs = [all_trajs[prompt][s] for s in SEEDS if s in all_metrics[prompt]]
        if trajs:
            prompt_means[prompt] = np.mean(trajs, axis=0)
            all_flat.extend([t.ravel() for t in trajs])
    global_mean = np.mean(all_flat, axis=0) if all_flat else 0
    total_var = np.mean([np.sum((t.ravel() - global_mean) ** 2) for t in all_flat]) if all_flat else 0
    # Between-prompt: variance of prompt means around global mean
    between_var = np.mean([np.sum((pm.ravel() - global_mean) ** 2) for pm in prompt_means.values()]) if prompt_means else 0
    # Within-prompt: average variance within each prompt
    within_var = 0.0
    count = 0
    for prompt in prompt_names:
        trajs = [all_trajs[prompt][s] for s in SEEDS if s in all_metrics[prompt]]
        if len(trajs) >= 2:
            pm = prompt_means[prompt]
            within_var += np.mean([np.sum((t.ravel() - pm.ravel()) ** 2) for t in trajs])
            count += 1
    within_var = within_var / count if count > 0 else 0

    if total_var > 0:
        lines.append(f"  Between-prompt fraction: {between_var / total_var:.1%}")
        lines.append(f"  Within-prompt (seed) fraction: {within_var / total_var:.1%}")
        lines.append(f"  Residual fraction: {max(0, 1 - (between_var + within_var) / total_var):.1%}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ========== PER-RUN PLOTS ==========
def plot_single_run(X, metrics, prompt, seed, output_dir):
    """Generate a single-run analysis plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax = axes[0, 0]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(X_pca)), cmap="viridis", s=30)
    for i in range(len(X_pca) - 1):
        ax.plot(X_pca[i:i+2, 0], X_pca[i:i+2, 1], color=plt.cm.viridis(i / len(X_pca)), alpha=0.6)
    ax.set_title(f"PCA (eff rank = {metrics['effective_rank']:.2f})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(sc, ax=ax, label="step")

    # Velocity
    ax = axes[0, 1]
    v = np.array(metrics["velocity"])
    ax.plot(v, color="coral")
    ax.set_title(f"Velocity (μ={metrics['velocity_mean']:.1f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("L2")
    ax.grid(alpha=0.3)

    # Curvature
    ax = axes[1, 0]
    kappa = np.array(metrics.get("curvature", []))
    if len(kappa) > 0:
        ax.plot(kappa, color="mediumseagreen")
        ax.set_title(f"Curvature (μ={metrics['curvature_mean']:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("1 - cosθ")
    ax.grid(alpha=0.3)

    # FFT
    ax = axes[1, 1]
    freqs = np.array(metrics["fft_freqs"])[1:]
    power = np.array(metrics["fft_power"])[1:]
    ax.stem(freqs, power, basefmt=" ", markerfmt=".", linefmt="steelblue")
    ax.set_title(f"FFT (dom freq = {metrics['dominant_fft_bins'][0] if metrics['dominant_fft_bins'] else 'N/A':.4f})")
    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Prompt: {prompt[:50]}... | Seed: {seed}", fontsize=12)
    plt.tight_layout()
    fname = f"run_p{short_filename(prompt)}_s{seed}.png"
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()


def short_filename(prompt):
    """Create a short filesystem-safe label from a prompt."""
    return SHORT_FILENAME.get(prompt, prompt[:10].replace(",", "").replace(" ", "_"))


# ========== MAIN ==========
def main():
    print("=" * 80)
    print("  DiT Systematic Dynamics Experiment")
    print(f"  {len(PROMPTS)} prompts × {len(SEEDS)} seeds = {len(PROMPTS) * len(SEEDS)} runs")
    print("=" * 80)

    # Model load
    if not args.dry_run:
        print("\n[1/3] Initializing DiT model...")
        dit_handler = AceStepHandler()
        dit_status, dit_success = dit_handler.initialize_service(
            project_root=str(MODEL_ROOT),
            config_path="acestep-v15-sft",
            device=args.device,
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
        )
        if not dit_success:
            print(f"  FAILED: {dit_status}")
            sys.exit(1)
        model = dit_handler.model
        print(f"  Model: {type(model).__name__}, hidden={model.config.hidden_size}")

        # Disable PhaseMemory
        for layer_mod in model.decoder.layers:
            if getattr(layer_mod, "use_phase_memory", False):
                layer_mod.use_phase_memory = False
        print("  PhaseMemory disabled")
        llm_handler = None  # no LM needed

        # Hook
        collector = HiddenStateCollector()
        handle = model.decoder.layers[LAYER].register_forward_hook(collector)
        print(f"  Hook on decoder.layers[{LAYER}]")
    else:
        print("\n[1/3] Skipped (dry-run mode — loading existing trajectories)")

    # Run experiments
    print(f"\n[2/3] Running {len(PROMPTS) * len(SEEDS)} experiments...")

    all_trajs = defaultdict(dict)
    all_metrics = defaultdict(dict)
    exp_keys = []

    if not args.dry_run:
        for prompt in PROMPTS:
            for seed in SEEDS:
                print(f"  [{exp_keys.count((prompt, None)) if False else ''}] prompt={short_filename(prompt)}, seed={seed}...")
                collector.reset()

                params = GenerationParams(
                    task_type="text2music",
                    caption=prompt,
                    lyrics="[Instrumental]",
                    instrumental=True,
                    duration=DURATION,
                    inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    seed=seed,
                    thinking=False,
                    use_cot_caption=False,
                    use_cot_metas=False,
                )
                config = GenerationConfig(
                    batch_size=1,
                    audio_format="mp3",
                    use_random_seed=False,
                )

                try:
                    result = generate_music(
                        dit_handler=dit_handler,
                        llm_handler=llm_handler,
                        params=params,
                        config=config,
                        save_dir=None,  # no audio file
                    )
                    success = result.success
                except Exception as e:
                    print(f"    Exception: {e}")
                    success = False

                X = collector.trajectory
                if X is None or len(X) == 0:
                    print(f"    WARNING: no states collected!")
                    continue

                print(f"    -> collected {X.shape[0]} states")

                # Save trajectory
                fname = f"traj_{short_filename(prompt)}_s{seed}.npy"
                np.save(TRAJ_DIR / fname, X)

                all_trajs[prompt][seed] = X
                all_metrics[prompt][seed] = compute_per_run_metrics(X)
                exp_keys.append((prompt, seed))

                # Per-run plot
                plot_single_run(X, all_metrics[prompt][seed], prompt, seed, PER_RUN_DIR)

        handle.remove()
        print(f"  Hook removed. {len(exp_keys)} successful runs.")
    else:
        # Load from disk
        for prompt in PROMPTS:
            for seed in SEEDS:
                fname = f"traj_{short_filename(prompt)}_s{seed}.npy"
                path = TRAJ_DIR / fname
                if path.exists():
                    X = np.load(path)
                    all_trajs[prompt][seed] = X
                    all_metrics[prompt][seed] = compute_per_run_metrics(X)
                    exp_keys.append((prompt, seed))
                    print(f"  Loaded {fname}: {X.shape}")
                else:
                    print(f"  MISSING {fname}")

    n_runs = len(exp_keys)
    if n_runs == 0:
        print("ERROR: no runs collected. Aborting.")
        sys.exit(1)
    print(f"\n  Total successful runs: {n_runs}/{len(PROMPTS) * len(SEEDS)}")

    # ========== Cross-run analysis ==========
    print(f"\n[3/3] Computing cross-run metrics and generating visualizations...")

    # Similarity matrix
    sim_mat = np.zeros((n_runs, n_runs))
    for i, (p_i, s_i) in enumerate(exp_keys):
        for j, (p_j, s_j) in enumerate(exp_keys):
            sim_mat[i, j] = cosine_similarity(all_trajs[p_i][s_i], all_trajs[p_j][s_j])

    # Generate all plots
    visualize_experiment(all_trajs, all_metrics, exp_keys)

    # Summary table
    summary = format_summary(all_trajs, all_metrics, exp_keys, sim_mat)
    print(summary)
    with open(OUTPUT_DIR / "summary_table.txt", "w") as f:
        f.write(summary)

    # Build comprehensive report JSON
    report = {
        "config": {
            "prompts": PROMPTS,
            "seeds": SEEDS,
            "layer": LAYER,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "duration": DURATION,
        },
        "per_run": {},
        "cross_run": {
            "similarity_matrix": sim_mat.tolist(),
            "within_prompt_similarity": {},
            "between_prompt_similarity": {},
        },
    }

    for (prompt, seed) in exp_keys:
        key = f"{short_filename(prompt)}_s{seed}"
        report["per_run"][key] = all_metrics[prompt][seed]

    # Cross-run stats
    within = [sim_mat[i, j] for i in range(n_runs) for j in range(i + 1, n_runs)
              if exp_keys[i][0] == exp_keys[j][0]]
    between = [sim_mat[i, j] for i in range(n_runs) for j in range(i + 1, n_runs)
               if exp_keys[i][0] != exp_keys[j][0]]
    report["cross_run"]["within_prompt_similarity"] = {
        "mean": float(np.mean(within)) if within else None,
        "std": float(np.std(within)) if within else None,
    }
    report["cross_run"]["between_prompt_similarity"] = {
        "mean": float(np.mean(between)) if between else None,
        "std": float(np.std(between)) if between else None,
    }

    # Per-prompt summary stats
    prompt_names = list(dict.fromkeys([k[0] for k in exp_keys]))
    per_prompt = {}
    for prompt in prompt_names:
        seeds_data = [all_metrics[prompt][s] for s in SEEDS if s in all_metrics[prompt]]
        pl = SHORT_FILENAME.get(prompt, "unknown")
        per_prompt[pl] = {
            "effective_rank_mean": float(np.mean([m["effective_rank"] for m in seeds_data])),
            "effective_rank_std": float(np.std([m["effective_rank"] for m in seeds_data])),
            "velocity_mean": float(np.mean([m["velocity_mean"] for m in seeds_data])),
            "velocity_std": float(np.mean([m["velocity_std"] for m in seeds_data])),
            "curvature_mean": float(np.mean([m["curvature_mean"] for m in seeds_data])),
            "dominant_frequency": float(np.mean(
                [m["dominant_fft_bins"][0] if m["dominant_fft_bins"] else 0 for m in seeds_data]
            )),
            "spectral_entropy": float(np.mean([m["spectral_entropy"] for m in seeds_data])),
            "smoothness_ratio": float(np.mean([m.get("smoothness_ratio", 0) for m in seeds_data])),
            "pca_explained_variance_ratio": np.mean(
                [m["pca_explained_variance"][:2] for m in seeds_data], axis=0
            ).tolist(),
        }
    report["per_prompt"] = per_prompt

    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full report: {OUTPUT_DIR / 'report.json'}")
    print(f"  Summary: {OUTPUT_DIR / 'summary_table.txt'}")
    print(f"  Per-run plots: {PER_RUN_DIR}/")
    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
