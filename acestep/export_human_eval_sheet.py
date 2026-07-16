#!/usr/bin/env python3
"""
ACE-Step PM-CTR 人工听评表导出

从 eval_outputs 中抽样生成 human_eval_sheet.csv。

列:
  sample_id, prompt_id, seed, method_anonymized (M1/M2/...),
  audio_path, lyric_path,
  rate_lyric_order_1_5, rate_skip_1_5, rate_repeat_1_5,
  rate_section_stability_1_5, rate_audio_quality_1_5, notes

方法顺序随机打乱 (基于 seed 固定打乱)。
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def anonymize_methods(methods: List[str], seed: int = 42) -> Dict[str, str]:
    """Create a deterministic anonymized mapping from method names to M1/M2/..."""
    sorted_methods = sorted(methods)
    rng = random.Random(seed)
    rng.shuffle(sorted_methods)
    return {m: f"M{i + 1}" for i, m in enumerate(sorted_methods)}


def find_audio_files(output_root: str, max_per_prompt_seed: int = 1) -> List[Dict[str, Any]]:
    """Find audio files and organize by prompt, seed, method."""
    entries = []
    output_root = Path(output_root)

    if not output_root.is_dir():
        print(f"[ERROR] Output root not found: {output_root}")
        return entries

    for prompt_dir in sorted(output_root.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_id = prompt_dir.name

        # Find lyrics in the prompt directory (if any)
        lyric_file = None
        for f in prompt_dir.iterdir():
            if f.suffix == ".txt":
                lyric_file = str(f)

        for seed_dir in sorted(prompt_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = seed_dir.name.replace("seed_", "")
            for method_dir in sorted(seed_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                audio_files = sorted(method_dir.glob("*.flac"))
                if not audio_files:
                    audio_files = sorted(method_dir.glob("*.wav"))
                if not audio_files:
                    continue

                for audio_file in audio_files[:max_per_prompt_seed]:
                    entries.append({
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "method": method,
                        "audio_path": str(audio_file.absolute()),
                        "lyric_path": lyric_file or "",
                    })

    return entries


def main():
    parser = argparse.ArgumentParser(description="ACE-Step Human Eval Sheet Export")
    parser.add_argument("--input-root", type=str, default="./eval_outputs",
                        help="Root directory of evaluation outputs")
    parser.add_argument("--output-path", type=str,
                        default="./eval_outputs/human_eval_sheet.csv",
                        help="Output CSV path")
    parser.add_argument("--sample-per-prompt", type=int, default=1,
                        help="Samples per prompt-seed combination")
    parser.add_argument("--anonymize-seed", type=int, default=42,
                        help="Seed for anonymization shuffle")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max total samples in the sheet")
    parser.add_argument("--prompt-subset", type=int, default=None,
                        help="Only include first N prompts (for testing)")
    parser.add_argument("--style", type=str, default="wide",
                        choices=["wide", "long"],
                        help="CSV layout: wide (one row per sample) or long "
                             "(one row per method per sample)")

    args = parser.parse_args()

    print("=" * 60)
    print("PM-CTR Human Evaluation Sheet Export")
    print("=" * 60)

    # Find all entries
    entries = find_audio_files(args.input_root, max_per_prompt_seed=args.sample_per_prompt)
    print(f"\nFound {len(entries)} audio entries")

    if not entries:
        print("[ERROR] No audio entries found")
        return

    # Collect unique methods
    methods = sorted(set(e["method"] for e in entries))
    method_to_anon = anonymize_methods(methods, seed=args.anonymize_seed)
    print(f"Methods: {methods}")
    print(f"Anonymized: {method_to_anon}")

    # If prompt subset
    if args.prompt_subset is not None:
        seen_prompts = set()
        filtered = []
        for e in entries:
            if e["prompt_id"] not in seen_prompts and len(seen_prompts) < args.prompt_subset:
                filtered.append(e)
            elif e["prompt_id"] in seen_prompts:
                filtered.append(e)
            else:
                pass
            seen_prompts.add(e["prompt_id"])
        entries = filtered
        print(f"After prompt subset: {len(entries)} entries")

    # Limit total
    if len(entries) > args.max_samples:
        rng = random.Random(args.anonymize_seed)
        # Stratified: keep structure by grouping by (prompt_id, seed) then sample
        grouped = defaultdict(list)
        for e in entries:
            grouped[(e["prompt_id"], e["seed"])].append(e)

        all_groups = list(grouped.values())
        rng.shuffle(all_groups)

        selected = []
        for group in all_groups:
            if len(selected) + len(group) <= args.max_samples:
                selected.extend(group)
        entries = selected
        print(f"After max samples: {len(entries)} entries")

    # Write CSV
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "prompt_id", "seed", "method_anonymized",
            "audio_path", "lyric_path",
            "rate_lyric_order_1_5", "rate_skip_1_5", "rate_repeat_1_5",
            "rate_section_stability_1_5", "rate_audio_quality_1_5",
            "notes",
        ])

        for i, e in enumerate(entries):
            anon_method = method_to_anon.get(e["method"], e["method"])
            writer.writerow([
                i + 1,
                e["prompt_id"],
                e["seed"],
                anon_method,
                e["audio_path"],
                e["lyric_path"],
                "",  # rate_lyric_order_1_5
                "",  # rate_skip_1_5
                "",  # rate_repeat_1_5
                "",  # rate_section_stability_1_5
                "",  # rate_audio_quality_1_5
                "",  # notes
            ])

    print(f"\n[OK] Human eval sheet saved: {args.output_path}")
    print(f"     {len(entries)} rows")
    print(f"\nMethod mapping (keep confidential):")
    for method, anon in sorted(method_to_anon.items(), key=lambda x: x[1]):
        print(f"  {anon} = {method}")


if __name__ == "__main__":
    main()
