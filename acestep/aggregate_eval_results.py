#!/usr/bin/env python3
"""
ACE-Step PM-CTR 汇总统计脚本

读取 eval_outputs 目录，汇总输出:
  1. aggregate_transport_metrics.csv — 每个 method 一行平均指标
  2. method_comparison_table.md     — 论文用 markdown 表格
  3. failure_case_index.json        — 可疑样本列表
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_eval_dirs(output_root: str) -> List[Dict[str, Any]]:
    """Find all method-level eval directories."""
    results = []
    output_root = Path(output_root)
    if not output_root.is_dir():
        print(f"[ERROR] Output root not found: {output_root}")
        return results

    for prompt_dir in sorted(output_root.iterdir()):
        if not prompt_dir.is_dir():
            continue
        for seed_dir in sorted(prompt_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.replace("seed_", ""))
            for method_dir in sorted(seed_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                marker_file = method_dir / ".marker.json"
                config_file = method_dir / "inference_config.json"
                summary_file = method_dir / "transport_summary.json"
                units_file = method_dir / "transport_units.csv"
                heatmap_file = method_dir / "transport_heatmap.png"
                audio_file = list(method_dir.glob("*.flac"))
                audio_file = audio_file[0] if audio_file else None

                entry = {
                    "prompt": prompt_dir.name,
                    "seed": seed,
                    "method": method,
                    "path": str(method_dir),
                    "has_config": config_file.exists(),
                    "has_summary": summary_file.exists(),
                    "has_units": units_file.exists(),
                    "has_heatmap": heatmap_file.exists(),
                    "has_audio": audio_file is not None,
                    "audio_file": str(audio_file) if audio_file else None,
                }

                if config_file.exists():
                    with open(config_file) as f:
                        entry["config"] = json.load(f)

                if summary_file.exists():
                    with open(summary_file) as f:
                        entry["summary"] = json.load(f)

                if marker_file.exists():
                    with open(marker_file) as f:
                        entry["marker"] = json.load(f)

                results.append(entry)

    return results


def compute_aggregate_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute per-method aggregate metrics from transport summaries."""
    method_data = defaultdict(list)

    for e in entries:
        method = e["method"]
        s = e.get("summary")
        if s is not None:
            method_data[method].append(s)

    aggregates = {}
    for method, summaries in method_data.items():
        n = len(summaries)
        agg = {"count": n}
        keys_of_interest = [
            "mean_abs_usage_error", "max_abs_usage_error",
            "row_error", "col_error",
            "monotonic_center_violations",
            "total_lyric_usage", "total_control_usage",
            "total_silence_usage",
            "intro_usage", "inst_usage", "outro_usage",
            "lyric_frac", "control_frac", "silence_frac",
            "skipped_units", "overused_units",
            "qk_std", "transport_qk_ratio", "write_ratio",
        ]
        for key in keys_of_interest:
            vals = [s.get(key, float("nan")) for s in summaries if key in s and s[key] is not None]
            # Filter non-numeric
            numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not (v != v)]
            if numeric_vals:
                agg[f"mean_{key}"] = sum(numeric_vals) / len(numeric_vals)
                agg[f"std_{key}"] = (
                    (sum((v - agg[f"mean_{key}"]) ** 2 for v in numeric_vals) / len(numeric_vals)) ** 0.5
                    if len(numeric_vals) > 1 else 0.0
                )
                agg[f"min_{key}"] = min(numeric_vals)
                agg[f"max_{key}"] = max(numeric_vals)
                agg[f"n_{key}"] = len(numeric_vals)

        aggregates[method] = agg

    return aggregates


def save_aggregate_csv(aggregates: Dict[str, Dict[str, float]], output_path: str):
    """Save aggregate metrics to CSV."""
    methods = sorted(aggregates.keys())
    if not methods:
        print("[WARN] No methods to aggregate")
        return

    # Collect all metric keys
    all_keys = set()
    for agg in aggregates.values():
        all_keys.update(k for k in agg.keys() if k.startswith("mean_"))

    sorted_keys = sorted(all_keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method", "count"] + sorted_keys
        writer.writerow(header)
        for method in methods:
            agg = aggregates[method]
            row = [method, agg.get("count", 0)]
            for key in sorted_keys:
                row.append(agg.get(key, ""))
            writer.writerow(row)

    print(f"[OK] Aggregate CSV saved: {output_path}")


def save_comparison_markdown(aggregates: Dict[str, Dict[str, float]], output_path: str):
    """Save method comparison table as markdown."""
    methods = sorted(aggregates.keys())
    if not methods:
        print("[WARN] No methods to aggregate")
        return

    # Key metrics for the table
    metric_labels = [
        ("mean_mean_abs_usage_error", "Mean Abs Usage Error"),
        ("mean_monotonic_center_violations", "Monotonic Violations"),
        ("mean_total_control_usage", "Control Usage"),
        ("mean_intro_usage", "Intro Usage"),
        ("mean_inst_usage", "Inst Usage"),
        ("mean_outro_usage", "Outro Usage"),
        ("mean_qk_std", "QK Std"),
        ("mean_transport_qk_ratio", "QK Ratio"),
        ("mean_write_ratio", "Write Ratio"),
        ("mean_row_error", "Row Error"),
        ("mean_col_error", "Col Error"),
    ]

    lines = ["# PM-CTR Method Comparison Table\n"]
    lines.append(f"| Metric | {' | '.join(methods)} |")
    lines.append(f"| --- | {' | '.join(['---'] * len(methods))} |")

    for key, label in metric_labels:
        row = [f"**{label}**"]
        for method in methods:
            agg = aggregates.get(method, {})
            val = agg.get(key, None)
            if val is not None:
                row.append(f"{val:.6f}")
            else:
                row.append("-")
        lines.append(f"| {' | '.join(row)} |")

    # Add counts
    lines.append(f"\n**Sample counts:**")
    for method in methods:
        lines.append(f"- {method}: {aggregates.get(method, {}).get('count', 0)}")

    content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)

    print(f"[OK] Comparison table saved: {output_path}")


def save_failure_case_index(entries: List[Dict[str, Any]], output_path: str):
    """Identify suspicious samples and save index."""
    thresholds = {
        "high_monotonic_violations": 50,
        "high_control_usage": 0.3,
        "low_lyric_usage": 0.3,
        "low_qk_ratio": 0.01,
        "low_write_ratio": 0.001,
        "high_write_ratio": 2.0,
    }

    failures = []

    for e in entries:
        s = e.get("summary")
        if s is None:
            continue

        flags = []
        prompt = e["prompt"]
        seed = e["seed"]
        method = e["method"]

        mv = s.get("monotonic_center_violations", 0)
        if mv > thresholds["high_monotonic_violations"]:
            flags.append(f"monotonic_violations={mv}")

        ctrl = s.get("total_control_usage", 0)
        if ctrl > thresholds["high_control_usage"]:
            flags.append(f"control_usage={ctrl:.3f}")

        lyr = s.get("total_lyric_usage", 0)
        if lyr < thresholds["low_lyric_usage"]:
            flags.append(f"lyric_usage={lyr:.3f}")

        qkr = s.get("transport_qk_ratio", 1.0)
        if qkr < thresholds["low_qk_ratio"]:
            flags.append(f"qk_ratio={qkr:.6f}")

        wr = s.get("write_ratio", 0.5)
        if wr < thresholds["low_write_ratio"]:
            flags.append(f"write_ratio={wr:.6f}")
        if wr > thresholds["high_write_ratio"]:
            flags.append(f"write_ratio_high={wr:.3f}")

        if flags:
            failures.append({
                "prompt": prompt,
                "seed": seed,
                "method": method,
                "flags": flags,
                "path": e["path"],
            })

    with open(output_path, "w") as f:
        json.dump({
            "total_samples": len(entries),
            "suspicious_count": len(failures),
            "thresholds": thresholds,
            "suspicious_samples": failures,
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] Failure case index saved: {output_path}")
    print(f"     {len(failures)} suspicious samples out of {len(entries)} total")


def main():
    parser = argparse.ArgumentParser(description="ACE-Step PM-CTR Aggregate Results")
    parser.add_argument("--input-root", type=str, default="./eval_outputs",
                        help="Root directory of evaluation outputs")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs/aggregate",
                        help="Output directory for aggregate results")

    args = parser.parse_args()

    print("=" * 60)
    print("PM-CTR Aggregate Results")
    print("=" * 60)

    entries = find_eval_dirs(args.input_root)
    print(f"\nFound {len(entries)} eval entries")

    if not entries:
        print("[ERROR] No eval entries found")
        sys.exit(1)

    # Summary by method
    method_counts = defaultdict(int)
    for e in entries:
        method_counts[e["method"]] += 1
    print("\nEntries per method:")
    for method, count in sorted(method_counts.items()):
        with_summary = sum(1 for e in entries if e["method"] == method and e.get("summary"))
        print(f"  {method}: {count} total, {with_summary} with transport summary")

    # Compute aggregates
    aggregates = compute_aggregate_metrics(entries)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Aggregate transport metrics CSV
    csv_path = os.path.join(args.output_dir, "aggregate_transport_metrics.csv")
    save_aggregate_csv(aggregates, csv_path)

    # 2. Method comparison table
    md_path = os.path.join(args.output_dir, "method_comparison_table.md")
    save_comparison_markdown(aggregates, md_path)

    # 3. Failure case index
    json_path = os.path.join(args.output_dir, "failure_case_index.json")
    save_failure_case_index(entries, json_path)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
