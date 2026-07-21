#!/usr/bin/env python3
"""
Output-level lyric evaluation metrics for paper experiments.

Computes lyric realization metrics by aligning ASR transcripts with
target lyrics.  Supports whisper/funasr/external_json backends.

Metrics
-------
  - Missing Rate: fraction of target lyric units not found in transcript.
  - Repeat Rate: fraction of matched units that are duplicates.
  - Order Accuracy: LCS-based alignment score.
  - Order Error: 1 - Order Accuracy.
  - Late Missing Rate: missing rate in the last third of target units.
  - Front / Middle / Late segment breakdowns.

Usage
-----
  python evaluate_lyrics.py --generation_dir outputs/paper_eval \\
      --asr_backend external_json --transcript_json data/transcripts.json \\
      --output_dir metrics/paper_eval
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root for parse_lyrics_to_units if available
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from acestep.phase_memory import parse_lyrics_to_units
    _HAS_PM = True
except ImportError:
    _HAS_PM = False


# ===========================================================================
#  Text normalisation
# ===========================================================================

def normalise_text(text: str, lang: str = "auto") -> str:
    """Normalise text for alignment: lowercase, strip punctuation, collapse spaces."""
    text = text.strip().lower()
    # Remove punctuation (keep Chinese characters, alphanumeric, spaces)
    text = re.sub(r"[^\w\s一-鿿]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise_text(text: str, lang: str = "auto") -> List[str]:
    """Tokenise text into comparison units.

    For Chinese: character-level tokens (each Chinese char is a token,
    English words are words).  For English: whitespace-split words.
    """
    text = normalise_text(text)
    if not text:
        return []

    # Check if text contains Chinese characters
    has_chinese = bool(re.search(r"[一-鿿]", text))

    if has_chinese:
        # Character-level: split Chinese chars, keep English words intact
        tokens = []
        for word in text.split():
            # If word is purely Chinese chars, split into characters
            if re.match(r"^[一-鿿]+$", word):
                tokens.extend(list(word))
            else:
                tokens.append(word)
        return tokens
    else:
        return text.split()


# ===========================================================================
#  Lyric unit parsing
# ===========================================================================

def parse_target_units(lyrics_text: str) -> List[Dict]:
    """Parse target lyrics into units for alignment.

    Returns list of dicts with ``text``, ``section``, ``is_lyric``.
    """
    # Simple line-based parsing without section IDs
    lines = [l.strip() for l in lyrics_text.strip().split("\n") if l.strip()]
    units = []
    current_section = "UNKNOWN"

    for line in lines:
        # Section tag?
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            tag = m.group(1).strip().upper()
            tag_clean = tag.replace("-", "").replace(" ", "")
            from acestep.phase_memory import SECTION_TAG_MAP, NON_LYRIC_SECTIONS
            mapped = SECTION_TAG_MAP.get(tag_clean, None)

            if mapped is not None:
                current_section = mapped
                if mapped in NON_LYRIC_SECTIONS:
                    units.append({"text": line, "section": mapped, "is_lyric": False})
                continue

            # Natural control line detection
            from acestep.phase_memory import is_natural_control_line
            if is_natural_control_line(line):
                units.append({"text": line, "section": current_section, "is_lyric": False})
                continue

            # Unknown tag, skip
            continue

        # Lyric line
        units.append({"text": line, "section": current_section, "is_lyric": True})

    return units


def extract_lyric_units(lyrics_text: str) -> List[str]:
    """Extract only sung lyric units (skip non-lyric)."""
    units = parse_target_units(lyrics_text)
    return [u["text"] for u in units if u["is_lyric"]]


# ===========================================================================
#  Fuzzy matching
# ===========================================================================

def compute_lcs(tokens_a: List[str], tokens_b: List[str]) -> List[Tuple[int, int]]:
    """Compute longest common subsequence between two token lists.

    Returns list of (i, j) pairs of matched positions.
    """
    m, n = len(tokens_a), len(tokens_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    matches = []
    i, j = m, n
    while i > 0 and j > 0:
        if tokens_a[i - 1] == tokens_b[j - 1]:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


def match_tokens(
    target_tokens: List[str],
    transcript_tokens: List[str],
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Match target tokens to transcript tokens using LCS.

    Returns
    -------
    matched_target : list of target token indices that matched.
    matched_trans : list of transcript token indices matched.
    lcs_pairs : list of (t_idx, s_idx) pairs from LCS.
    """
    lcs_pairs = compute_lcs(target_tokens, transcript_tokens)
    matched_target = [t for t, _ in lcs_pairs]
    matched_trans = [s for _, s in lcs_pairs]
    return matched_target, matched_trans, lcs_pairs


# ===========================================================================
#  Metrics computation
# ===========================================================================

def compute_metrics(
    target_units: List[str],
    transcript_text: str,
) -> Dict[str, float]:
    """Compute all lyric-level metrics for one sample.

    Parameters
    ----------
    target_units : list of str
        Target lyric unit texts (one per line).
    transcript_text : str
        ASR transcript text (raw string).

    Returns
    -------
    metrics : dict
        Keys: missing_rate, repeat_rate, order_accuracy, order_error,
              late_missing_rate, num_target_units, num_matched_units,
              front_missing, middle_missing, late_missing,
              front_order_error, middle_order_error, late_order_error,
              front_missing_rate, middle_missing_rate, late_missing_rate,
              segment_labels.
    """
    result: Dict[str, float] = {}
    result["num_target_units"] = len(target_units)

    if not target_units:
        result["missing_rate"] = 1.0
        result["repeat_rate"] = 0.0
        result["order_accuracy"] = 0.0
        result["order_error"] = 1.0
        result["late_missing_rate"] = 1.0
        result["num_matched_units"] = 0
        return result

    if not transcript_text.strip():
        result["missing_rate"] = 1.0
        result["repeat_rate"] = 0.0
        result["order_accuracy"] = 0.0
        result["order_error"] = 1.0
        result["late_missing_rate"] = 1.0
        result["num_matched_units"] = 0
        return result

    # Tokenise target and transcript
    target_text = " ".join(target_units)
    target_tokens = tokenise_text(target_text)
    transcript_tokens = tokenise_text(transcript_text)

    if not target_tokens:
        result["missing_rate"] = 1.0
        result["repeat_rate"] = 0.0
        result["order_accuracy"] = 0.0
        result["order_error"] = 1.0
        result["late_missing_rate"] = 1.0
        result["num_matched_units"] = 0
        return result

    # Match via LCS
    matched_target, matched_trans, lcs_pairs = match_tokens(target_tokens, transcript_tokens)

    # ---- Missing rate ----
    num_missing = len(target_tokens) - len(set(matched_target))
    missing_rate = num_missing / max(len(target_tokens), 1)
    result["missing_rate"] = missing_rate

    # ---- Repeat rate (transcript tokens that match multiple target tokens) ----
    if len(matched_trans) > 0:
        repeat_count = len(matched_trans) - len(set(matched_trans))
        repeat_rate = repeat_count / max(len(matched_trans), 1)
    else:
        repeat_rate = 0.0
    result["repeat_rate"] = repeat_rate

    # ---- Order accuracy (LCS-based) ----
    lcs_len = len(lcs_pairs)
    order_accuracy = lcs_len / max(len(target_tokens), 1)
    result["order_accuracy"] = order_accuracy
    result["order_error"] = 1.0 - order_accuracy

    # ---- Late missing rate (last third of target tokens) ----
    n = len(target_tokens)
    late_start = int(2 * n / 3)
    late_targets = set(range(late_start, n))
    matched_set = set(matched_target)
    late_missing = len(late_targets - matched_set)
    result["late_missing_rate"] = late_missing / max(len(late_targets), 1)

    # ---- Front / Middle / Late segment breakdown ----
    segments = {
        "front": (0, int(n / 3)),
        "middle": (int(n / 3), int(2 * n / 3)),
        "late": (int(2 * n / 3), n),
    }
    for seg_name, (s, e) in segments.items():
        seg_targets = set(range(s, e))
        seg_missing = len(seg_targets - matched_set)
        seg_len = max(e - s, 1)
        result[f"{seg_name}_missing"] = seg_missing
        result[f"{seg_name}_missing_rate"] = seg_missing / seg_len

        # Order error per segment
        seg_target_tokens = target_tokens[s:e]
        seg_matched = [t for t in matched_target if s <= t < e]
        # Count correct relative order within segment
        seg_correct = 0
        for idx_in_seg, t_idx in enumerate(seg_matched):
            expected = s + idx_in_seg
            if t_idx == expected or (idx_in_seg > 0 and t_idx > seg_matched[idx_in_seg - 1]):
                seg_correct += 1
        seg_order_err = 1.0 - (seg_correct / max(len(seg_target_tokens), 1))
        result[f"{seg_name}_order_error"] = seg_order_err

    result["num_matched_units"] = len(set(matched_target))
    result["segment_labels"] = "front/middle/late"
    result["lcs_length"] = lcs_len
    result["total_target_tokens"] = len(target_tokens)

    return result


# ===========================================================================
#  ASR backends
# ===========================================================================

def run_asr_external(transcript_json: str) -> Dict[str, str]:
    """Load pre-computed transcripts from JSON.

    Expected format:
    {
        "variant/prompt_id/seed_0": "transcript text ...",
        ...
    }
    Or:
    [
        {"variant": "...", "prompt_id": "...", "seed": 0, "transcript": "..."},
        ...
    ]
    """
    with open(transcript_json) as f:
        data = json.load(f)

    transcripts = {}
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        for item in data:
            key = f"{item.get('variant', '')}/{item.get('prompt_id', '')}/seed_{item.get('seed', 0)}"
            transcripts[key] = item.get("transcript", "")
        return transcripts
    return transcripts


def run_asr_whisper(audio_path: str) -> str:
    """Transcribe audio with Whisper (requires whisper installed)."""
    try:
        import whisper
    except ImportError:
        raise ImportError("whisper not installed. Use --asr_backend external_json instead.")

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()


def run_asr_funasr(audio_path: str) -> str:
    """Transcribe audio with FunASR (requires funasr installed)."""
    try:
        from funasr import AutoModel
    except ImportError:
        raise ImportError("funasr not installed. Use --asr_backend external_json instead.")

    model = AutoModel(model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    result = model.generate(input=audio_path)
    return result[0].get("text", "").strip() if result else ""


# ===========================================================================
#  Main evaluation
# ===========================================================================

def collect_generated_samples(
    generation_dir: str,
    variants: Optional[List[str]] = None,
) -> List[Dict]:
    """Walk generation output directory and collect sample info.

    Returns list of dicts with ``variant``, ``prompt_id``, ``seed``,
    ``audio_path``, ``lyrics``, ``metadata_path``.
    """
    samples = []
    gen_dir = Path(generation_dir)

    for variant_dir in sorted(gen_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        if variants and variant_dir.name not in variants:
            continue

        for prompt_dir in sorted(variant_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            prompt_id = prompt_dir.name

            for seed_dir in sorted(prompt_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                seed = int(seed_dir.name.replace("seed_", ""))

                audio_path = seed_dir / "audio.wav"
                if not audio_path.exists():
                    audio_path = seed_dir / "audio.flac"
                if not audio_path.exists():
                    continue

                metadata_path = seed_dir / "metadata.json"
                lyrics = ""
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        meta = json.load(f)
                        lyrics = meta.get("lyrics", "")

                transport_diag = seed_dir / "transport_diagnostics.npz"
                has_diag = transport_diag.exists()

                samples.append({
                    "variant": variant_dir.name,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "audio_path": str(audio_path),
                    "lyrics": lyrics,
                    "metadata_path": str(metadata_path),
                    "has_transport_diag": has_diag,
                })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Output-level lyric evaluation metrics."
    )
    parser.add_argument("--generation_dir", type=str, required=True,
                        help="Generation output directory")
    parser.add_argument("--asr_backend", type=str, default="external_json",
                        choices=["whisper", "funasr", "external_json", "none"],
                        help="ASR backend")
    parser.add_argument("--transcript_json", type=str, default=None,
                        help="Path to external transcript JSON")
    parser.add_argument("--output_dir", type=str, default="metrics/paper_eval",
                        help="Output directory for metrics CSVs")
    parser.add_argument("--variants", type=str, default=None,
                        help="Optional comma-separated filter for variants")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_filter = None
    if args.variants:
        variant_filter = [v.strip() for v in args.variants.split(",")]

    # Collect samples
    print("Collecting generated samples...")
    samples = collect_generated_samples(args.generation_dir, variant_filter)
    print(f"  Found {len(samples)} samples")

    if not samples:
        print("  No samples found. Exiting.")
        return

    # Load transcripts
    transcripts: Dict[str, str] = {}
    if args.asr_backend == "external_json":
        if not args.transcript_json:
            print("[ERROR] --transcript_json required for external_json backend")
            sys.exit(1)
        transcripts = run_asr_external(args.transcript_json)
        print(f"  Loaded {len(transcripts)} external transcripts")
    elif args.asr_backend == "none":
        print("  ASR backend: none — using empty transcripts")
    else:
        print(f"  ASR backend: {args.asr_backend}")

    # Compute metrics per sample
    per_sample = []
    variant_stats: Dict[str, List] = defaultdict(list)

    for sample in samples:
        key = f"{sample['variant']}/{sample['prompt_id']}/seed_{sample['seed']}"
        lyrics = sample["lyrics"]

        # Get transcript
        transcript = ""
        if args.asr_backend == "external_json":
            transcript = transcripts.get(key, transcripts.get(sample["audio_path"], ""))
        elif args.asr_backend == "whisper":
            transcript = run_asr_whisper(sample["audio_path"])
        elif args.asr_backend == "none":
            transcript = ""

        # Parse target lyrics into units
        target_units = extract_lyric_units(lyrics)

        # Compute metrics
        metrics = compute_metrics(target_units, transcript)

        row = {
            "variant": sample["variant"],
            "prompt_id": sample["prompt_id"],
            "seed": sample["seed"],
            "transcript_path": sample["audio_path"],
            "audio_path": sample["audio_path"],
            **metrics,
        }
        per_sample.append(row)
        variant_stats[sample["variant"]].append(metrics)

    # Write per-sample CSV
    per_sample_path = output_dir / "output_metrics_per_sample.csv"
    if per_sample:
        fieldnames = [
            "variant", "prompt_id", "seed",
            "missing_rate", "repeat_rate", "order_accuracy", "order_error",
            "late_missing_rate",
            "front_missing", "middle_missing", "late_missing",
            "front_missing_rate", "middle_missing_rate", "late_missing_rate",
            "front_order_error", "middle_order_error", "late_order_error",
            "num_target_units", "num_matched_units",
            "lcs_length", "total_target_tokens",
            "transcript_path", "audio_path",
        ]
        with open(per_sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_sample)
        print(f"  Per-sample metrics → {per_sample_path}")

    # Write summary CSV
    summary_path = output_dir / "output_metrics_summary.csv"
    summary_rows = []
    for variant, metrics_list in sorted(variant_stats.items()):
        row = {"variant": variant, "num_samples": len(metrics_list)}
        for key in [
            "missing_rate", "repeat_rate", "order_accuracy", "order_error",
            "late_missing_rate",
            "front_missing_rate", "middle_missing_rate", "late_missing_rate",
            "front_order_error", "middle_order_error", "late_order_error",
        ]:
            values = [m.get(key, 0.0) for m in metrics_list if key in m]
            if values:
                row[f"{key}_mean"] = float(np.mean(values))
                row[f"{key}_std"] = float(np.std(values))
                row[f"{key}_stderr"] = float(np.std(values) / np.sqrt(max(len(values), 1)))
            else:
                row[f"{key}_mean"] = 0.0
                row[f"{key}_std"] = 0.0
                row[f"{key}_stderr"] = 0.0
        summary_rows.append(row)

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary → {summary_path}")

    # Print quick summary
    print("\n" + "=" * 60)
    print("Quick Summary")
    print("=" * 60)
    for row in summary_rows:
        print(f"  {row['variant']}: n={row['num_samples']}")
        print(f"    Missing Rate: {row['missing_rate_mean']:.4f} ± {row['missing_rate_std']:.4f}")
        print(f"    Repeat Rate: {row['repeat_rate_mean']:.4f} ± {row['repeat_rate_std']:.4f}")
        print(f"    Order Error: {row['order_error_mean']:.4f} ± {row['order_error_std']:.4f}")
        print(f"    Late Missing: {row['late_missing_rate_mean']:.4f} ± {row['late_missing_rate_std']:.4f}")

    print(f"\nDone. Metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
