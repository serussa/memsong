#!/usr/bin/env python3
"""
Generate paper figures from experiment metrics CSVs.

Figures
-------
  fig3_segment_missing.pdf  — Front / Middle / Late missing rate by variant.
  main_output_metrics.pdf   — Main output metrics bar chart.
  cumulative_coverage_error.pdf — Coverage error E_cum(p) over progress.
  ablation_late_error.pdf   — Ablation summary on late missing rate.

Optional:
  attention_staticity.pdf  — Attention staticity (if data available).
  transport_heatmap_case.pdf — Transport plan heatmap for a representative case.

All figures are saved as both PDF and PNG under ``--figures_dir``.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[WARN] matplotlib not available. Install with: pip install matplotlib")


# ===========================================================================
#  Matplotlib style setup
# ===========================================================================

def setup_style():
    """Configure matplotlib style for paper figures."""
    if not _HAS_MPL:
        return
    # Try Times New Roman, fall back to DejaVu Serif
    try:
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["DejaVu Serif"]
    rcParams["font.size"] = 9
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 10
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["legend.fontsize"] = 8
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["savefig.dpi"] = 150
    rcParams["figure.dpi"] = 100


# ===========================================================================
#  Color / marker palette
# ===========================================================================

# Method display names (consistent across figures)
METHOD_LABELS = {
    "baseline": "Baseline",
    "softmax_adapter": "Row-Softmax",
    "transport_only": "Sinkhorn (qk=0)",
    "full": "Full",
    "no_structural_units": "No-Struct-Units",
}

METHOD_COLORS = {
    "baseline": "#888888",
    "softmax_adapter": "#E08B6E",
    "transport_only": "#6BA3D6",
    "full": "#3B7A4D",
    "no_structural_units": "#B07AA2",
}

METHOD_MARKERS = {
    "baseline": "o",
    "softmax_adapter": "s",
    "transport_only": "D",
    "full": "^",
    "no_structural_units": "v",
}

# Paper-friendly labels
SEGMENT_LABELS = ["Front", "Middle", "Late"]
METRIC_LABELS = {
    "missing_rate": "Missing Rate",
    "repeat_rate": "Repeat Rate",
    "order_error": "Order Error",
    "late_missing_rate": "Late Missing Rate",
    "order_accuracy": "Order Accuracy",
}


# ===========================================================================
#  Data loading
# ===========================================================================

def load_summary_csv(path: Path) -> List[Dict]:
    """Load metrics summary CSV into list of dicts."""
    rows = []
    if not path.exists():
        print(f"  [WARN] Summary CSV not found: {path}")
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_cumulative_csv(path: Path) -> List[Dict]:
    """Load cumulative coverage CSV (long format)."""
    rows = []
    if not path.exists():
        print(f"  [WARN] Cumulative CSV not found: {path}")
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_per_sample_csv(path: Path) -> List[Dict]:
    """Load per-sample metrics CSV."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ===========================================================================
#  Figure 3a: Front / Middle / Late Missing Rate
# ===========================================================================

def plot_segment_missing(
    summary_rows: List[Dict],
    output_base: str,
):
    """Figure 3a: segment-level missing rate by variant."""

    # Extract data
    variants = []
    front_means, front_errs = [], []
    middle_means, middle_errs = [], []
    late_means, late_errs = [], []

    var_order = ["baseline", "softmax_adapter", "transport_only", "full"]

    for var in var_order:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            print(f"  [WARN] No summary data for variant '{var}' — skipping")
            continue
        r = matching[0]
        variants.append(METHOD_LABELS.get(var, var))

        for key, means_list, errs_list in [
            ("front_missing_rate", front_means, front_errs),
            ("middle_missing_rate", middle_means, middle_errs),
            ("late_missing_rate", late_means, late_errs),
        ]:
            means_list.append(float(r.get(f"{key}_mean", 0.0)))
            errs_list.append(float(r.get(f"{key}_stderr", 0.0)))

    if not variants:
        print("  [WARN] No variant data for segment_missing plot")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    x = np.arange(len(SEGMENT_LABELS))
    width = 0.18
    n_variants = len(variants)

    for i, (var_label, means, errs) in enumerate(zip(
        variants, [front_means, middle_means, late_means],
        [front_errs, middle_errs, late_errs],
    )):
        # This is wrong — need per-variant per-segment, not per-metric
        pass

    # Correct approach: per-variant, 3 bars (front, middle, late)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    x = np.arange(len(SEGMENT_LABELS))
    width = 0.20

    for i, var in enumerate(var_order):
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        means = [
            float(r.get("front_missing_rate_mean", 0.0)),
            float(r.get("middle_missing_rate_mean", 0.0)),
            float(r.get("late_missing_rate_mean", 0.0)),
        ]
        errs = [
            float(r.get("front_missing_rate_stderr", 0.0)),
            float(r.get("middle_missing_rate_stderr", 0.0)),
            float(r.get("late_missing_rate_stderr", 0.0)),
        ]
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=errs,
            label=METHOD_LABELS.get(var, var),
            color=METHOD_COLORS.get(var, "#888888"),
            capsize=2, error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SEGMENT_LABELS)
    ax.set_ylabel("Missing Rate")
    ax.set_title("Front / Middle / Late Missing Rate")
    ax.legend(loc="best", framealpha=0.8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=150)
        print(f"  → {path}")
    plt.close(fig)


# ===========================================================================
#  Figure 3b: Main output metrics bar
# ===========================================================================

def plot_main_metrics(
    summary_rows: List[Dict],
    output_base: str,
):
    """Figure 3b: main output metrics by variant."""
    metric_keys = ["missing_rate", "repeat_rate", "order_error", "late_missing_rate"]
    metric_labels_display = [METRIC_LABELS[k] for k in metric_keys]

    var_order = ["baseline", "softmax_adapter", "transport_only", "full"]
    n_variants = len(var_order)
    n_metrics = len(metric_keys)

    # Filter available variants
    available_vars = []
    for var in var_order:
        if any(r["variant"] == var for r in summary_rows):
            available_vars.append(var)

    if not available_vars:
        print("  [WARN] No variant data for main_metrics plot")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(n_metrics)
    width = 0.20

    for i, var in enumerate(available_vars):
        r = [row for row in summary_rows if row["variant"] == var][0]
        means = [float(r.get(f"{k}_mean", 0.0)) for k in metric_keys]
        errs = [float(r.get(f"{k}_stderr", 0.0)) for k in metric_keys]
        offset = (i - (len(available_vars) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, yerr=errs,
            label=METHOD_LABELS.get(var, var),
            color=METHOD_COLORS.get(var, "#888888"),
            capsize=2, error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels_display, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Main Output Metrics")
    ax.legend(loc="best", framealpha=0.8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=150)
        print(f"  → {path}")
    plt.close(fig)


# ===========================================================================
#  Figure 3c: Cumulative coverage error E_cum(p)
# ===========================================================================

def plot_cumulative_coverage(
    cum_rows: List[Dict],
    output_base: str,
):
    """Figure 3c: cumulative coverage error over song progress."""
    if not cum_rows:
        print("  [WARN] No cumulative coverage data — skipping plot")
        return

    # Aggregate by variant and progress
    var_data: Dict[str, Dict[float, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in cum_rows:
        var = r.get("variant", "")
        p_val = float(r.get("progress", 0.0))
        e_val = float(r.get("E_cum", 0.0))
        var_data[var][p_val].append(e_val)

    # Compute mean ± stderr
    var_order = ["softmax_adapter", "transport_only", "full"]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    has_data = False
    for var in var_order:
        if var not in var_data:
            continue
        p_vals = sorted(var_data[var].keys())
        if not p_vals:
            continue
        means = [np.mean(var_data[var][p]) for p in p_vals]
        stderrs = [np.std(var_data[var][p]) / max(np.sqrt(len(var_data[var][p])), 1)
                   for p in p_vals]
        ax.plot(p_vals, means,
                label=METHOD_LABELS.get(var, var),
                color=METHOD_COLORS.get(var, "#888888"),
                linewidth=1.5,
                marker=METHOD_MARKERS.get(var, "."),
                markersize=3,
                markevery=max(1, len(p_vals) // 10))
        ax.fill_between(p_vals,
                         np.array(means) - np.array(stderrs),
                         np.array(means) + np.array(stderrs),
                         alpha=0.15,
                         color=METHOD_COLORS.get(var, "#888888"))
        has_data = True

    if not has_data:
        print("  [WARN] No plottable cumulative coverage data")
        plt.close(fig)
        return

    ax.set_xlabel("Song Progress p")
    ax.set_ylabel(r"$E_{\mathrm{cum}}(p)$")
    ax.set_title("Cumulative Coverage Error")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=150)
        print(f"  → {path}")
    plt.close(fig)


# ===========================================================================
#  Figure 3d: Ablation summary
# ===========================================================================

def plot_ablation_late_error(
    summary_rows: List[Dict],
    output_base: str,
):
    """Figure 3d: ablation on late missing rate."""
    var_order = ["baseline", "softmax_adapter", "transport_only", "full",
                 "no_structural_units"]

    variants, late_means, late_errs = [], [], []
    for var in var_order:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        variants.append(METHOD_LABELS.get(var, var))
        late_means.append(float(r.get("late_missing_rate_mean", 0.0)))
        late_errs.append(float(r.get("late_missing_rate_stderr", 0.0)))

    if not variants:
        print("  [WARN] No ablation data")
        return

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(variants))

    bars = ax.bar(
        x, late_means, yerr=late_errs,
        color=[METHOD_COLORS.get(v.lower().replace(" ", "_"),
                                 "#888888") for v in variants],
        capsize=3, error_kw={"linewidth": 0.8},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel("Late Missing Rate")
    ax.set_title("Ablation: Late Missing Rate")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=150)
        print(f"  → {path}")
    plt.close(fig)


# ===========================================================================
#  Optional: Transport heatmap case
# ===========================================================================

def plot_transport_heatmap(
    generation_dir: str,
    output_base: str,
):
    """Optional: Pi transport plan heatmap for a representative case."""
    # Find a representative sample
    gen_dir = Path(generation_dir)
    candidates = list(gen_dir.glob("full/**/transport_diagnostics.npz"))
    if not candidates:
        print("  [WARN] No transport diagnostics found for heatmap")
        return

    # Use first available
    diag_path = candidates[0]
    data = np.load(diag_path)
    Pi = data.get("Pi")
    if Pi is None or Pi.size == 0:
        print("  [WARN] Invalid Pi matrix for heatmap")
        return
    if Pi.ndim == 3:
        Pi = Pi[0]  # CFG batch doubling; take first
    if Pi.ndim != 2:
        print("  [WARN] Invalid Pi matrix for heatmap")
        return

    unit_section_ids = data.get("unit_section_ids")
    unit_is_lyric = data.get("unit_is_lyric")
    p_audio = data.get("p_audio")
    mu = data.get("mu")

    # Subsample to manageable size
    T, K = Pi.shape
    max_T_display = 200
    if T > max_T_display:
        idx_t = np.linspace(0, T - 1, max_T_display, dtype=int)
        Pi = Pi[idx_t, :]
        if p_audio is not None:
            p_audio = p_audio[idx_t]

    max_K_display = 40
    if K > max_K_display:
        idx_k = np.linspace(0, K - 1, max_K_display, dtype=int)
        Pi = Pi[:, idx_k]
        if unit_section_ids is not None:
            unit_section_ids = unit_section_ids[idx_k]
        if unit_is_lyric is not None:
            unit_is_lyric = unit_is_lyric[idx_k]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(Pi.T, aspect="auto", origin="lower",
                    cmap="Blues", interpolation="nearest")

    # Annotate with section labels
    if unit_section_ids is not None:
        section_names = {0: "UNK", 1: "INTRO", 2: "VERSE", 3: "PRE",
                         4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INST"}
        tick_positions = np.arange(len(unit_section_ids))
        tick_labels = [section_names.get(int(sid), "?")
                       for sid in unit_section_ids]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)

    ax.set_xlabel("Audio Timestep")
    ax.set_ylabel("Condition Unit")
    ax.set_title(r"Transport Plan $\Pi$ (Full Method)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\Pi_{ij}$")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{output_base}.{ext}"
        fig.savefig(path, dpi=150)
        print(f"  → {path}")
    plt.close(fig)


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from experiment metrics."
    )
    parser.add_argument("--metrics_dir", type=str, default="metrics/paper_eval",
                        help="Directory with metrics CSVs")
    parser.add_argument("--figures_dir", type=str, default="figures/paper_eval",
                        help="Output directory for figures")
    parser.add_argument("--generation_dir", type=str, default=None,
                        help="Generation output directory (for heatmap)")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not _HAS_MPL:
        print("[FATAL] matplotlib required for plotting. Install with: pip install matplotlib")
        sys.exit(1)

    setup_style()

    # Load data
    summary_path = metrics_dir / "output_metrics_summary.csv"
    per_sample_path = metrics_dir / "output_metrics_per_sample.csv"
    cum_path = metrics_dir / "internal_cumulative_curve.csv"
    cum_agg_path = metrics_dir / "internal_cumulative_curve_agg.csv"

    summary_rows = load_summary_csv(summary_path)
    per_sample_rows = load_per_sample_csv(per_sample_path)
    cum_rows = load_cumulative_csv(cum_agg_path if cum_agg_path.exists() else cum_path)

    n_samples = len(per_sample_rows)
    n_summary = len(summary_rows)
    n_cum = len(cum_rows)
    print(f"Loaded {n_samples} per-sample, {n_summary} summary, {n_cum} cumulative rows")

    # ---- Figure 3a: Segment missing ----
    print("\n[Figure 3a] Segment-level missing rate...")
    if summary_rows:
        plot_segment_missing(
            summary_rows,
            str(figures_dir / "fig3_segment_missing"),
        )
    else:
        print("  [SKIP] No summary data")

    # ---- Figure 3b: Main metrics ----
    print("\n[Figure 3b] Main output metrics...")
    if summary_rows:
        plot_main_metrics(
            summary_rows,
            str(figures_dir / "main_output_metrics"),
        )
    else:
        print("  [SKIP] No summary data")

    # ---- Figure 3c: Cumulative coverage ----
    print("\n[Figure 3c] Cumulative coverage error...")
    if cum_rows:
        plot_cumulative_coverage(
            cum_rows,
            str(figures_dir / "cumulative_coverage_error"),
        )
    else:
        print("  [SKIP] No cumulative coverage data")

    # ---- Figure 3d: Ablation ----
    print("\n[Figure 3d] Ablation summary...")
    if summary_rows:
        plot_ablation_late_error(
            summary_rows,
            str(figures_dir / "ablation_late_error"),
        )
    else:
        print("  [SKIP] No summary data")

    # ---- Optional: Transport heatmap ----
    if args.generation_dir and Path(args.generation_dir).exists():
        print("\n[Optional] Transport heatmap...")
        plot_transport_heatmap(
            args.generation_dir,
            str(figures_dir / "transport_heatmap_case"),
        )
    else:
        print("\n[SKIP] Transport heatmap (no --generation_dir)")

    print(f"\nDone. Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
