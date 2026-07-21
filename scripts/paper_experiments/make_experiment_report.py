#!/usr/bin/env python3
"""
Generate experiment report from all metrics and figures.

Reads CSVs from ``--metrics_dir`` and generates a Markdown report
under ``--output`` with:

  1. Experiment configuration summary.
  2. Sample counts per variant.
  3. Main metrics table (Missing Rate, Repeat Rate, Order Error, etc.).
  4. Segment-level metrics table.
  5. Ablation results.
  6. Internal diagnostic results (marginal error, cumulative coverage).
  7. LaTeX tables ready for copy-pasting into the paper.
  8. Missing-data warnings.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_csv(path: Path) -> List[Dict]:
    """Load a CSV file into a list of dicts."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ===========================================================================
#  Formatting helpers
# ===========================================================================

def fmt_val(v, decimals: int = 4) -> str:
    """Format a metric value with fixed decimal places."""
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def fmt_mean_stderr(mean: str, stderr: str, decimals: int = 4) -> str:
    """Format as ``mean ± stderr``."""
    m = fmt_val(mean, decimals)
    s = fmt_val(stderr, decimals)
    return f"{m} ± {s}"


# ===========================================================================
#  Report sections
# ===========================================================================

METHOD_LABELS = {
    "baseline": "Baseline",
    "softmax_adapter": "Row-Softmax",
    "transport_only": "Sinkhorn (qk=0)",
    "full": "Full",
    "no_structural_units": "No-Struct-Units",
}

SECTION_ORDER = ["baseline", "softmax_adapter", "transport_only", "full",
                  "no_structural_units"]


def section_config() -> str:
    return """## 1. Experiment Configuration

- **Model**: ACE-Step 1.5 (DiT backbone, layer-12 injection)
- **Variants**: see Section 4
- **Seeds**: 0, 1 (per prompt)
- **Inference steps**: 50
- **Guidance scale**: 7.0
- **Duration**: Up to 240 seconds
"""


def section_samples(per_sample_rows: List[Dict]) -> str:
    """Count samples per variant."""
    variants = defaultdict(set)
    for r in per_sample_rows:
        variants[r["variant"]].add((r["prompt_id"], r["seed"]))

    lines = ["## 2. Sample Counts\n"]
    lines.append("| Variant | Prompts × Seeds | Total Samples |")
    lines.append("|---------|-----------------|---------------|")
    total = 0
    for var in SECTION_ORDER:
        if var in variants:
            n = len(variants[var])
            label = METHOD_LABELS.get(var, var)
            lines.append(f"| {label} | {n} | {n} |")
            total += n
    lines.append(f"| **Total** | | **{total}** |")
    lines.append("")
    return "\n".join(lines)


def section_main_metrics(summary_rows: List[Dict]) -> str:
    """Main metrics table."""
    keys = [
        ("missing_rate", "Missing ↓"),
        ("repeat_rate", "Repeat ↓"),
        ("order_error", "Order Error ↓"),
        ("late_missing_rate", "Late Missing ↓"),
        ("order_accuracy", "Order Accuracy ↑"),
    ]

    lines = ["## 3. Main Results\n"]
    header = "| Method | " + " | ".join(k[1] for k in keys) + " |"
    sep = "|" + "|".join("---" for _ in range(len(keys) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for var in SECTION_ORDER:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        vals = []
        for k, _ in keys:
            mean = r.get(f"{k}_mean", "N/A")
            stderr = r.get(f"{k}_stderr", "N/A")
            if mean != "N/A" and stderr != "N/A":
                vals.append(fmt_mean_stderr(mean, stderr))
            else:
                vals.append("N/A")
        label = METHOD_LABELS.get(var, var)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


def section_segment_metrics(summary_rows: List[Dict]) -> str:
    """Segment-level metrics table."""
    segments = ["front", "middle", "late"]
    lines = ["## 4. Segment-Level Metrics\n"]

    # Missing rate by segment
    lines.append("### Missing Rate by Segment\n")
    header = "| Method | " + " | ".join(s.capitalize() for s in segments) + " |"
    sep = "|" + "|".join("---" for _ in range(len(segments) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for var in SECTION_ORDER:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        vals = []
        for seg in segments:
            mean = r.get(f"{seg}_missing_rate_mean", "N/A")
            stderr = r.get(f"{seg}_missing_rate_stderr", "N/A")
            if mean != "N/A" and stderr != "N/A":
                vals.append(fmt_mean_stderr(mean, stderr))
            else:
                vals.append("N/A")
        label = METHOD_LABELS.get(var, var)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    # Order error by segment
    lines.append("\n### Order Error by Segment\n")
    header = "| Method | " + " | ".join(s.capitalize() for s in segments) + " |"
    lines.append(header)
    lines.append(sep)
    for var in SECTION_ORDER:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        vals = []
        for seg in segments:
            mean = r.get(f"{seg}_order_error_mean", "N/A")
            stderr = r.get(f"{seg}_order_error_stderr", "N/A")
            if mean != "N/A" and stderr != "N/A":
                vals.append(fmt_mean_stderr(mean, stderr))
            else:
                vals.append("N/A")
        label = METHOD_LABELS.get(var, var)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


def section_ablation(summary_rows: List[Dict]) -> str:
    """Ablation table."""
    keys = [
        ("late_missing_rate", "Late Missing ↓"),
        ("order_error", "Order Error ↓"),
    ]
    lines = ["## 5. Ablation Results\n"]
    header = ("| Method | Sinkhorn | Structural Units | State Cost | "
              + " | ".join(k[1] for k in keys) + " |")
    sep = "|" + "|".join("---" for _ in range(5)) + "|"
    lines.append(header)
    lines.append(sep)

    ablation_config = {
        "baseline": ("No", "N/A", "N/A"),
        "softmax_adapter": ("No (row-softmax)", "Yes", "Yes"),
        "transport_only": ("Yes", "Yes", "No (qk=0)"),
        "full": ("Yes", "Yes", "Yes"),
        "no_structural_units": ("Yes", "No (lyric only)", "Yes"),
    }

    for var in SECTION_ORDER:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        sinkhorn, struct, cost = ablation_config.get(var, ("?", "?", "?"))
        vals = [sinkhorn, struct, cost]
        for k, _ in keys:
            mean = r.get(f"{k}_mean", "N/A")
            stderr = r.get(f"{k}_stderr", "N/A")
            if mean != "N/A" and stderr != "N/A":
                vals.append(fmt_mean_stderr(mean, stderr))
            else:
                vals.append("N/A")
        label = METHOD_LABELS.get(var, var)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


def section_internal_diagnostics(metrics_dir: Path) -> str:
    """Internal diagnostic results."""
    lines = ["## 6. Internal Diagnostics\n"]

    # Marginal error
    marginal_path = metrics_dir / "internal_marginal_error.csv"
    marginal_rows = load_csv(marginal_path)
    if marginal_rows:
        lines.append("### Condition Marginal Error\n")
        lines.append("| Variant | Mean Marginal Error | Std |")
        lines.append("|---------|-------------------|-----|")
        by_variant = defaultdict(list)
        for r in marginal_rows:
            by_variant[r["variant"]].append(float(r["marginal_error"]))
        for var in SECTION_ORDER:
            if var in by_variant:
                vals = by_variant[var]
                mean_v = float(np.mean(vals)) if vals else 0.0
                std_v = float(np.std(vals)) if vals else 0.0
                label = METHOD_LABELS.get(var, var)
                lines.append(f"| {label} | {mean_v:.6f} | {std_v:.6f} |")
        lines.append("")

    # State cost contribution
    state_path = metrics_dir / "state_cost_contribution.csv"
    state_rows = load_csv(state_path)
    if state_rows:
        lines.append("### Marginal Error (State-Cost Proxy)\n")
        lines.append("| Variant | Mean | Std | N |")
        lines.append("|---------|------|-----|---|")
        for r in state_rows:
            var = r.get("variant", "?")
            label = METHOD_LABELS.get(var, var)
            m = fmt_val(r.get("mean_marginal_error", "N/A"))
            s = fmt_val(r.get("std_marginal_error", "N/A"))
            n = r.get("num_samples", "?")
            lines.append(f"| {label} | {m} | {s} | {n} |")
        note = state_rows[0].get("note", "") if state_rows else ""
        if note:
            lines.append(f"\n*Note*: {note}")
        lines.append("")

    # Check if cumulative coverage data exists
    cum_path = metrics_dir / "internal_cumulative_curve.csv"
    if cum_path.exists():
        cum_rows = load_csv(cum_path)
        lines.append(f"\n### Cumulative Coverage Error\n")
        lines.append(f"- Data available: {len(cum_rows)} rows across variants.")
        lines.append("- See Figure 3c for visualisation.\n")

    return "\n".join(lines)


def section_latex_tables(summary_rows: List[Dict]) -> str:
    """Generate LaTeX tables for the paper."""
    lines = ["## 7. LaTeX Tables\n"]

    # --- Table 1: Main Results ---
    lines.append("```latex")
    lines.append("% Table 1: Main Results")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results. $\downarrow$=lower is better, $\uparrow$=higher is better.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append("Method & "
                  r"Missing $\downarrow$ & Repeat $\downarrow$ & "
                  r"Order Error $\downarrow$ & Late Missing $\downarrow$ & "
                  r"Order Accuracy $\uparrow$ \\")
    lines.append(r"\midrule")

    main_keys = ["missing_rate", "repeat_rate", "order_error",
                  "late_missing_rate", "order_accuracy"]
    for var in SECTION_ORDER:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        vals = []
        for k in main_keys:
            mean = r.get(f"{k}_mean", "N/A")
            stderr = r.get(f"{k}_stderr", "N/A")
            if mean != "N/A" and stderr != "N/A":
                vals.append(f"${fmt_val(mean)} \\pm {fmt_val(stderr)}$")
            else:
                vals.append("N/A")
        label = METHOD_LABELS.get(var, var).replace("_", "\\_")
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```\n")

    # --- Table 2: Ablation ---
    lines.append("```latex")
    lines.append("% Table 2: Ablation")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation over method components.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append("Method & Sinkhorn & Structural Units & "
                  "State Cost & Late Missing $\\downarrow$ \\")
    lines.append(r"\midrule")

    ablation_config = {
        "baseline": ("---", "---", "---"),
        "softmax_adapter": ("Row-softmax", "Yes", "Yes"),
        "transport_only": ("Yes", "Yes", "No"),
        "full": ("Yes", "Yes", "Yes"),
        "no_structural_units": ("Yes", "No", "Yes"),
    }
    for var in SECTION_ORDER:
        matching = [r for r in summary_rows if r["variant"] == var]
        if not matching:
            continue
        r = matching[0]
        s, st, c = ablation_config.get(var, ("?", "?", "?"))
        late_mean = r.get("late_missing_rate_mean", "N/A")
        late_stderr = r.get("late_missing_rate_stderr", "N/A")
        if late_mean != "N/A" and late_stderr != "N/A":
            late_str = f"${fmt_val(late_mean)} \\pm {fmt_val(late_stderr)}$"
        else:
            late_str = "N/A"
        label = METHOD_LABELS.get(var, var).replace("_", "\\_")
        lines.append(f"{label} & {s} & {st} & {c} & {late_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```\n")

    return "\n".join(lines)


def section_warnings(per_sample_rows: List[Dict], summary_rows: List[Dict],
                      metrics_dir: Path, figures_dir: Path) -> str:
    """List missing-data warnings."""
    warnings = []
    lines = ["## 8. Warnings / Missing Data\n"]

    # Check samples
    if not per_sample_rows:
        warnings.append("No per-sample metrics available (output_metrics_per_sample.csv missing).")

    # Check summary
    if not summary_rows:
        warnings.append("No summary metrics available (output_metrics_summary.csv missing).")

    # Check internal diagnostics
    for fname in ["internal_marginal_error.csv", "internal_cumulative_curve.csv"]:
        if not (metrics_dir / fname).exists():
            warnings.append(f"No internal diagnostics: {fname} missing. "
                            "Run evaluate_internal_diagnostics.py first.")

    # Check figures
    for fname in ["fig3_segment_missing.pdf",
                   "main_output_metrics.pdf",
                   "cumulative_coverage_error.pdf",
                   "ablation_late_error.pdf"]:
        if not (figures_dir / fname).exists():
            warnings.append(f"Figure {fname} not found. Run plot_paper_figures.py first.")

    if warnings:
        for w in warnings:
            lines.append(f"- ⚠ {w}")
    else:
        lines.append("All required metrics and figures are present.")

    lines.append("")
    lines.append("### Attention Staticity Note\n")
    lines.append("Attention staticity diagnostic (inter-step attention distance, ")
    lines.append("attention entropy) was NOT computed because it requires a custom ")
    lines.append("cross-attention hook in the baseline model during generation. ")
    lines.append("If the paper needs this analysis:\n")
    lines.append("- Add a forward hook on ``model.decoder.layers[12].cross_attn`` ")
    lines.append("  to save attention maps during inference.\n")
    lines.append("- Compute ``D_attn(tau) = 1 - cosine(vec(A_{tau+1}), vec(A_{tau}))`` ")
    lines.append("  and ``H(A_tau) = -sum(A_tau * log(A_tau + eps))``.\n")
    lines.append("- The claim about near-static attention should be weakened to ")
    lines.append("  qualitative motivation if this diagnostic is absent.\n")

    return "\n".join(lines)


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper experiment report."
    )
    parser.add_argument("--metrics_dir", type=str, default="metrics/paper_eval",
                        help="Directory with metrics CSVs")
    parser.add_argument("--figures_dir", type=str, default="figures/paper_eval",
                        help="Figures directory")
    parser.add_argument("--output", type=str,
                        default="reports/paper_experiment_summary.md",
                        help="Output report path")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all data
    per_sample_rows = load_csv(metrics_dir / "output_metrics_per_sample.csv")
    summary_rows = load_csv(metrics_dir / "output_metrics_summary.csv")

    print(f"Loaded {len(per_sample_rows)} per-sample, {len(summary_rows)} summary rows")

    # Build report
    sections = [
        "# Paper Experiment Summary\n",
        section_config(),
        section_samples(per_sample_rows),
        section_main_metrics(summary_rows),
        section_segment_metrics(summary_rows),
        section_ablation(summary_rows),
        section_internal_diagnostics(metrics_dir),
        section_latex_tables(summary_rows),
        section_warnings(per_sample_rows, summary_rows, metrics_dir, figures_dir),
    ]

    report = "\n".join(sections)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport written to {output_path}")
    print(f"({len(report)} characters)")


if __name__ == "__main__":
    # Need numpy for internal diagnostics section
    try:
        import numpy as np
    except ImportError:
        import sys as _sys
        print("[WARN] numpy not available, internal diagnostics section may be incomplete",
              file=_sys.stderr)
        # Provide minimal stub
        import types
        np = types.ModuleType("numpy")
        np.mean = lambda x: sum(x) / len(x) if x else 0.0
        np.std = lambda x: (sum((v - sum(x)/len(x))**2 for v in x) / len(x))**0.5 if len(x) > 1 else 0.0
        sys.modules["numpy"] = np
    main()
