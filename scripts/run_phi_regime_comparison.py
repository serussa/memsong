#!/usr/bin/env python3
"""
run_phi_regime_comparison.py
============================
Orchestrate φ collection under 3 regimes, compute standardized metrics,
and produce a cross-regime comparison table.

Usage:
    python scripts/run_phi_regime_comparison.py \
        --model-root /path/to/Ace-Step1.5 \
        --pm-dir /path/to/checkpoint/epoch_50 \
        --output-dir ./output/regime_comparison \
        --n-seeds 10 \
        --num-steps 50

Output:
    - phi_<regime>_<seed>.pt       — raw φ tensors
    - metrics_<regime>.json         — aggregated metrics per regime
    - comparison_table.txt          — LaTeX-style table
    - comparison_table.png          — bar chart
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch

# Setup matplotlib early (non-interactive)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root
PROJECT_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(PROJECT_ROOT))

from acestep.handler import AceStepHandler
from acestep.training.phase_memory_checkpoint import load_phase_memory_weights
from scripts.collect_phi_across_regimes import (
    PhaseMemoryHook,
    collect_regime_a,
    collect_regime_b,
    collect_regime_c,
    collect_regime_with_stats,
)
from scripts.phi_analysis_metrics import (
    structural_metrics_per_token,
    temporal_structure_metrics_per_token,
    ablation_metrics_per_token,
    pca_metrics_per_token,
)


# ==============================================================================
# METRIC AGGREGATION OVER SEEDS
# ==============================================================================

def compute_metrics_for_regime(phis: list, label: str) -> dict:
    """Compute all metric categories across seeds, aggregate mean±std.

    Args:
        phis: List of [T, S, D] tensors (one per seed).

    Returns:
        dict with per-category nested metrics and flat keys for table.
    """
    all_metrics = []
    for phi in phis:
        m = {}
        m.update(structural_metrics_per_token(phi))
        m.update(temporal_structure_metrics_per_token(phi))
        m.update(ablation_metrics_per_token(phi))
        m.update(pca_metrics_per_token(phi))
        all_metrics.append(m)

    # Aggregate over seeds
    flat = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        arr = np.array(vals)
        flat[key] = float(np.nanmean(arr))
        flat[key + "_seed_std"] = float(np.nanstd(arr))

    flat["regime"] = label
    flat["n_seeds"] = len(phis)
    return flat


# ==============================================================================
# LATEX TABLE GENERATOR
# ==============================================================================

REQUIRED_ROWS = [
    ("Monotonicity", "monotonicity_ratio", "{:.3f}", ""),
    ("Corr(φ, step)", "time_corr_mean", "{:.3f}", ""),
    ("Vel. variance", "velocity_var", "{:.4f}", ""),
    ("Accel. |Δ²φ|", "accel_abs_mean", "{:.4f}", ""),
    ("Short-range Coh.", "short_range_coherence_l1-5", "{:.3f}", ""),
    ("Long-range Coh.", "long_range_coherence_l10-25", "{:.3f}", ""),
    ("Self-sim. Entropy", "self_sim_entropy", "{:.3f}", ""),
    ("Ablation drop (shuffle)", "ablation_drop_shuffle", "{:.1%}", ""),
    ("Ablation drop (noise)", "ablation_drop_noise", "{:.1%}", ""),
    ("Orig short Coh.", "orig_short_coherence", "{:.3f}", ""),
    ("Noise short Coh.", "noise_short_coherence", "{:.3f}", ""),
    ("PCA effective rank", "pca_effective_rank", "{:.1f}", ""),
    ("PCA top-3 var", "pca_top3_var_explained", "{:.2%}", ""),
    ("PCA 1st var ratio", "pca_first_var_ratio", "{:.2%}", ""),
]


def build_comparison_table(regime_metrics: dict) -> str:
    """Build LaTeX-style comparison table.

    Args:
        regime_metrics: dict of regime_label -> flat metrics dict.

    Returns:
        String with formatted LaTeX table.
    """
    labels = list(regime_metrics.keys())
    if len(labels) < 2:
        return "Need at least 2 regimes for comparison."

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    col_spec = "l" + "c" * len(labels)
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    # Header
    header = "Metric" + " & " + " & ".join(labels) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for display_name, key, fmt_str, unit in REQUIRED_ROWS:
        row = display_name
        for lab in labels:
            m = regime_metrics[lab]
            val = m.get(key, None)
            std = m.get(key + "_seed_std", None)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                val_s = fmt_str.format(val)
                if std is not None and std > 0 and not math.isnan(std):
                    std_s = fmt_str.format(std)
                    row += f" & ${val_s} \\pm {std_s}$"
                else:
                    row += f" & ${val_s}$"
            else:
                row += " & ---"
        row += " \\\\"
        lines.append(row)

    lines.append("\\midrule")
    # Sample size row
    row = "Samples"
    for lab in labels:
        row += f" & N={regime_metrics[lab].get('n_seeds', '?')}"
    row += " \\\\"
    lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Phase coordinate $\\varphi$ comparison across " +
                 "diffusion regimes. Mean $\\pm$ std over seeds.}")
    lines.append("\\label{tab:phi_regime_comparison}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_coherence_comparison(regime_phis: dict, save_path: str):
    """Coherence decay curves for each regime (from first seed's first token)."""
    import matplotlib.pyplot as plt
    from scripts.phi_analysis_metrics import coherence_curve

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A": "steelblue", "B": "coral", "C": "seagreen"}
    markers = {"A": "Regime A (Decoder)", "B": "Regime B (ODE-only)", "C": "Regime C (Full)"}

    for label in ["A", "B", "C"]:
        phis = regime_phis.get(label, [])
        if not phis:
            continue
        phi = phis[0]  # first seed
        # Average over tokens, take first token's 2D slice for coherence
        phi_2d = phi[:, 0, :] if phi.dim() == 3 else phi
        curve = coherence_curve(phi_2d)
        ax.plot(np.arange(1, len(curve) + 1), curve,
                label=markers.get(label, label),
                color=colors.get(label, "gray"), linewidth=1.5)

    ax.set_xlabel("Lag (diffusion steps)")
    ax.set_ylabel("Mean cos(Δφ)")
    ax.set_title("Coherence Decay Across Regimes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ablation_drop_comparison(regime_metrics: dict, save_path: str):
    """Bar chart of ablation drops across regimes."""
    labels = []
    shuffle_drops = []
    noise_drops = []
    for lab in ["A", "B", "C"]:
        m = regime_metrics.get(lab, regime_metrics.get(
            {"A": "Regime A (Decoder)", "B": "Regime B (ODE-only)", "C": "Regime C (Full)"}.get(lab, lab)))
        # Find the actual label
        for k in regime_metrics:
            if k.startswith("Regime"):
                labels.append(k)
                shuffle_drops.append(regime_metrics[k].get("ablation_drop_shuffle", 0) * 100)
                noise_drops.append(regime_metrics[k].get("ablation_drop_noise", 0) * 100)
                break

    if len(labels) < 2:
        # Fallback: just use whatever keys exist
        labels = list(regime_metrics.keys())
        shuffle_drops = [regime_metrics[k].get("ablation_drop_shuffle", 0) * 100 for k in labels]
        noise_drops = [regime_metrics[k].get("ablation_drop_noise", 0) * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, shuffle_drops, width, label="Shuffle drop", color="coral", alpha=0.85)
    bars2 = ax.bar(x + width / 2, noise_drops, width, label="Noise drop", color="steelblue", alpha=0.85)
    ax.set_xlabel("Regime")
    ax.set_ylabel("Coherence drop (%)")
    ax.set_title("Ablation Sensitivity Across Regimes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pca_comparison(regime_metrics: dict, save_path: str):
    """Bar chart of PCA effective rank."""
    labels = list(regime_metrics.keys())
    ranks = [regime_metrics[k].get("pca_effective_rank", 0) for k in labels]
    top3 = [regime_metrics[k].get("pca_top3_var_explained", 0) * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(x, ranks, width, color="steelblue", alpha=0.85, label="Effective rank")
    ax1.set_ylabel("PCA Effective Rank (95% var)")
    ax1.set_xlabel("Regime")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, r in zip(bars, ranks):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{r:.1f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, top3, "ro-", linewidth=2, markersize=8, label="Top-3 var")
    ax2.set_ylabel("Top-3 variance ratio (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Phase Dimensionality Across Regimes")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-regime φ comparison: decoder-only vs ODE-only vs full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-root", type=str,
                        default="/root/autodl-tmp/Ace-Step1.5")
    parser.add_argument("--pm-dir", type=str,
                        default="/root/autodl-tmp/lyrics_checkpoints/checkpoints/epoch_50_loss_1.1069")
    parser.add_argument("--config", type=str, default="acestep-v15-sft")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "output" / "phi_regime_comparison"))
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random seeds to run per regime")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Diffusion steps per generation")
    parser.add_argument("--seq-len", type=int, default=750,
                        help="Sequence length (frames at 25Hz)")
    parser.add_argument("--guidance", type=float, default=7.0,
                        help="CFG guidance scale for regime C")
    parser.add_argument("--skip-a", action="store_true",
                        help="Skip regime A (decoder rollout)")
    parser.add_argument("--skip-b", action="store_true",
                        help="Skip regime B (ODE-only)")
    parser.add_argument("--skip-c", action="store_true",
                        help="Skip regime C (full pipeline)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print("=" * 64)
    print("  Loading model + PhaseMemory weights ...")
    print("=" * 64)
    os.environ["ACESTEP_OFFLINE"] = "1"
    os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=args.model_root,
        config_path=args.config,
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"Service init failed: {status}")

    load_phase_memory_weights(handler.model, args.pm_dir)
    model = handler.model

    # Find PhaseMemory module
    pm_module = None
    for name, mod in model.named_modules():
        if getattr(mod, "use_phase_memory", False) and hasattr(mod, "phase_memory"):
            pm_module = mod.phase_memory
            break
    if pm_module is None:
        raise RuntimeError("No PhaseMemory module found!")
    print(f"  PhaseMemory found on: {name}.phase_memory")
    print(f"  Model dtype: {model.dtype}")

    # ── Define regimes ──
    regimes = []
    if not args.skip_a:
        regimes.append(("A", "Regime A (Decoder)", collect_regime_a, {}))
    if not args.skip_b:
        regimes.append(("B", "Regime B (ODE-only)", collect_regime_b, {}))
    if not args.skip_c:
        regimes.append(("C", "Regime C (Full)", collect_regime_c,
                        {"guidance_scale": args.guidance}))

    if not regimes:
        print("All regimes skipped. Nothing to do.")
        return

    # ── Collect φ under each regime ──
    regime_phis = {}
    regime_metrics_dict = {}

    print("=" * 64)
    print(f"  Collecting φ across {len(regimes)} regimes × {args.n_seeds} seeds")
    print(f"  {args.num_steps} diffusion steps, seq_len={args.seq_len}")
    print("=" * 64)

    for short_label, label, collect_fn, extra_kwargs in regimes:
        print(f"\n>>> {label} <<<")
        phis = []
        for seed in range(1, args.n_seeds + 1):
            hook = PhaseMemoryHook(pm_module)
            with hook:
                phi = collect_fn(
                    model=model,
                    hook=hook,
                    num_steps=args.num_steps,
                    seq_len=args.seq_len,
                    seed=seed,
                    **extra_kwargs,
                )
            phis.append(phi)
            # Save per-seed φ
            fname = output_dir / f"phi_{short_label}_seed{seed}.pt"
            torch.save(phi, fname)
            print(f"    seed {seed:2d}: φ {tuple(phi.shape)}  "
                  f"range [{phi.min():.2f}, {phi.max():.2f}]  → {fname.name}")

        regime_phis[short_label] = phis
        # Compute metrics
        metrics = compute_metrics_for_regime(phis, label)
        regime_metrics_dict[label] = metrics

        # Save per-regime metrics
        mfile = output_dir / f"metrics_{short_label}.json"
        serializable = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str))}
        with open(mfile, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"    metrics → {mfile.name}")

    # ── Print comparison table ──
    print("\n" + "=" * 64)
    print("  CROSS-REGIME COMPARISON TABLE")
    print("=" * 64)

    table = build_comparison_table(regime_metrics_dict)
    print("\n" + table)

    # Also save as text
    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(table)
    print(f"\n  Table saved: {output_dir / 'comparison_table.txt'}")

    # ── Print human-readable summary ──
    print("\n" + "-" * 64)
    for label in sorted(regime_metrics_dict.keys()):
        m = regime_metrics_dict[label]
        print(f"  {label}:")
        print(f"    Monotonicity:        {m.get('monotonicity_ratio',float('nan')):.4f}")
        print(f"    Corr(φ, step):       {m.get('time_corr_mean',float('nan')):.4f}")
        print(f"    Short-range coh:     {m.get('short_range_coherence_l1-5',float('nan')):.4f}")
        print(f"    Ablation drop (shuf):{m.get('ablation_drop_shuffle',float('nan')):.1%}")
        print(f"    PCA effective rank:  {m.get('pca_effective_rank',float('nan')):.1f}")
        print(f"    Self-sim entropy:    {m.get('self_sim_entropy',float('nan')):.4f}")
    print("-" * 64)

    # ── Generate comparison plots ──
    print("\n  Generating comparison plots ...")
    try:
        plot_coherence_comparison(regime_phis,
                                   str(output_dir / "comparison_coherence_decay.png"))
    except Exception as e:
        print(f"    [SKIP] coherence plot: {e}")
        import traceback; traceback.print_exc()
    try:
        plot_ablation_drop_comparison(regime_metrics_dict,
                                       str(output_dir / "comparison_ablation_drop.png"))
    except Exception as e:
        print(f"    [SKIP] ablation plot: {e}")
    try:
        plot_pca_comparison(regime_metrics_dict,
                             str(output_dir / "comparison_pca_dimensionality.png"))
    except Exception as e:
        print(f"    [SKIP] PCA plot: {e}")

    # ── Save all metrics as single JSON ──
    all_metrics = {}
    for label, m in regime_metrics_dict.items():
        clean = {k: v for k, v in m.items()
                 if isinstance(v, (int, float, str)) and not isinstance(v, np.ndarray)}
        clean["regime"] = label
        all_metrics[label] = clean
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'=' * 64}")
    print(f"  All outputs → {output_dir}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
