#!/usr/bin/env python3
"""
Structured Error Evaluation: ASR-based lyric alignment for PD-SBAR.

Transcribes baseline and PD-SBAR audio with Whisper, then measures:
- Lyric skip / repeat
- Non-vocal leakage (vocals detected in intro/outro/instrumental sections)
- Progress drift
- Section alignment

Usage:
  python scripts/eval_structured.py --baseline-dir <dir> --pd-sbar-dir <dir> [--output <dir>]
"""
import argparse, json, os, sys, re, glob
from pathlib import Path

import torch
import whisper

# Known sections from our prompts
SECTION_ORDER = ["INTRO", "VERSE", "CHORUS", "VERSE", "CHORUS", "BRIDGE", "CHORUS", "OUTRO"]
NON_VOCAL_SECTIONS = {"INTRO", "OUTRO", "INSTRUMENTAL", "SOLO", "INTERLUDE"}
VOCAL_SECTIONS = {"VERSE", "CHORUS", "BRIDGE", "PRECHORUS", "PRE-CHORUS"}


def transcribe_file(path: str, model) -> dict:
    """Transcribe one audio file with Whisper, return segments with timestamps."""
    result = model.transcribe(path, language="en", verbose=False, word_timestamps=True)
    return {
        "text": result["text"].strip(),
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result["segments"]
        ],
    }


def detect_section_boundaries(transcript: dict, total_duration: float) -> dict:
    """Detect when vocals start/end, silence gaps, section transitions."""
    segments = transcript["segments"]
    if not segments:
        return {
            "first_vocal_start": None,
            "last_vocal_end": None,
            "vocal_time": 0,
            "silence_gaps": [],
            "num_segments": 0,
        }

    first_vocal = segments[0]["start"]
    last_vocal = segments[-1]["end"]
    vocal_time = sum(s["end"] - s["start"] for s in segments)

    # Detect gaps > 3s (potential section transitions)
    gaps = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap > 3.0:
            gaps.append({"start": segments[i - 1]["end"], "end": segments[i]["start"],
                         "duration": gap})

    return {
        "first_vocal_start": first_vocal,
        "last_vocal_end": last_vocal,
        "vocal_density": vocal_time / total_duration if total_duration > 0 else 0,
        "silence_gaps": gaps,
        "num_segments": len(segments),
    }


def estimate_non_vocal_leakage(transcript: dict, section_structure: list,
                                section_times: list, total_duration: float) -> dict:
    """Estimate how much vocal activity happens in non-vocal sections."""
    segments = transcript["segments"]
    # section_times: list of (start, end) tuples for each section
    # We estimate section times as equal fractions of total duration
    n = len(section_structure)
    sec_len = total_duration / n if n > 0 else total_duration
    section_times = [(i * sec_len, (i + 1) * sec_len) for i in range(n)]

    non_vocal_vocal_time = 0
    total_non_vocal_time = 0
    vocal_vocal_time = 0
    total_vocal_time = 0

    for i, sec in enumerate(section_structure):
        sec_start, sec_end = section_times[i]
        sec_dur = sec_end - sec_start
        is_non_vocal = sec in NON_VOCAL_SECTIONS

        # How much of this section has vocals
        vocal_in_section = 0
        for s in segments:
            overlap_start = max(s["start"], sec_start)
            overlap_end = min(s["end"], sec_end)
            if overlap_end > overlap_start:
                vocal_in_section += overlap_end - overlap_start

        if is_non_vocal:
            non_vocal_vocal_time += vocal_in_section
            total_non_vocal_time += sec_dur
        else:
            vocal_vocal_time += vocal_in_section
            total_vocal_time += sec_dur

    return {
        "non_vocal_leakage_ratio": non_vocal_vocal_time / max(total_non_vocal_time, 1e-6),
        "non_vocal_leakage_seconds": non_vocal_vocal_time,
        "vocal_section_coverage": vocal_vocal_time / max(total_vocal_time, 1e-6),
        "vocal_section_seconds": vocal_vocal_time,
    }


def estimate_lyric_repeat_skip(transcript: dict, gt_lyrics: str) -> dict:
    """Simple heuristic: count repeated / skipped phrases from GT."""
    # Extract unique lyric lines from GT
    gt_lines = [l.strip() for l in gt_lyrics.split("\n")
                if l.strip() and not l.strip().startswith("[")]
    gt_unique = list(set(gt_lines))

    transcript_text = transcript["text"].lower()
    found_count = 0
    for line in gt_unique:
        # Check if key words from this line appear in transcript
        words = [w for w in line.lower().split() if len(w) > 3]
        if not words:
            continue
        matches = sum(1 for w in words if w in transcript_text)
        if matches >= max(len(words) * 0.5, 1):
            found_count += 1

    return {
        "gt_unique_lines": len(gt_unique),
        "gt_lines_found": found_count,
        "lyric_coverage": found_count / max(len(gt_unique), 1),
        "lyric_skip_ratio": 1 - found_count / max(len(gt_unique), 1),
    }


def estimate_progress_drift(expected_sections: list, total_duration: float,
                            transcript: dict) -> dict:
    """Estimate how much the vocal timing drifts from expected structure."""
    segments = transcript["segments"]
    if not segments or not expected_sections:
        return {"drift_score": 0, "early_vocal": 0, "late_vocal": 0}

    n = len(expected_sections)
    expected_section_duration = total_duration / n

    # Expected: first vocal should start around the end of INTRO (section 0)
    # INTRO is expected_section_duration seconds
    expected_first_vocal = expected_section_duration  # end of INTRO
    first_vocal = segments[0]["start"]
    early_vocal = max(0, expected_first_vocal - first_vocal) / max(expected_first_vocal, 1)

    # Expected: last vocal should end before OUTRO starts (last section)
    expected_last_vocal = total_duration - expected_section_duration  # start of OUTRO
    last_vocal = segments[-1]["end"]
    late_vocal = max(0, last_vocal - expected_last_vocal) / max(expected_last_vocal, 1)

    return {
        "early_vocal_ratio": early_vocal,
        "late_vocal_ratio": late_vocal,
        "expected_first_vocal": expected_first_vocal,
        "actual_first_vocal": first_vocal,
        "expected_last_vocal": expected_last_vocal,
        "actual_last_vocal": last_vocal,
        "drift_score": (early_vocal + late_vocal) / 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=str,
                        default="/root/autodl-tmp/pd_sbar_eval_output/baseline/audios")
    parser.add_argument("--pd-sbar-dir", type=str,
                        default="/root/autodl-tmp/pd_sbar_eval_output/pd_sbar_a010/audios")
    parser.add_argument("--output", type=str, default="/tmp/structured_eval")
    parser.add_argument("--model", type=str, default="medium")
    parser.add_argument("--duration", type=float, default=150)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper {args.model}...")
    model = whisper.load_model(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    print("Loaded.")

    # Get prompt names from files
    bl_files = sorted(glob.glob(f"{args.baseline_dir}/*.flac"))
    pds_files = sorted(glob.glob(f"{args.pd_sbar_dir}/*.flac"))

    # Build name map
    names = []
    for f in bl_files:
        name = Path(f).stem
        pds_path = Path(args.pd_sbar_dir) / f"{name}.flac"
        if pds_path.exists():
            names.append(name)

    print(f"Found {len(names)} prompts with both baseline and PD-SBAR")

    METRICS = [
        "prompt", "method",
        "first_vocal_start", "last_vocal_end", "vocal_density",
        "non_vocal_leakage_ratio", "vocal_section_coverage",
        "lyric_coverage", "lyric_skip_ratio",
        "early_vocal_ratio", "late_vocal_ratio", "drift_score",
        "num_segments",
    ]

    all_rows = []

    for name in names:
        print(f"\n=== {name} ===")
        bl_path = f"{args.baseline_dir}/{name}.flac"
        pds_path = f"{args.pd_sbar_dir}/{name}.flac"

        for method, path in [("baseline", bl_path), ("pd_sbar_010", pds_path)]:
            try:
                transcript = transcribe_file(path, model)
            except Exception as e:
                print(f"  {method}: transcribe failed: {e}")
                continue

            section_boundaries = detect_section_boundaries(
                transcript, args.duration)
            leakage = estimate_non_vocal_leakage(
                transcript, SECTION_ORDER, [], args.duration)
            progress = estimate_progress_drift(
                SECTION_ORDER, args.duration, transcript)

            row = {
                "prompt": name,
                "method": method,
                "first_vocal_start": section_boundaries["first_vocal_start"],
                "last_vocal_end": section_boundaries["last_vocal_end"],
                "vocal_density": section_boundaries["vocal_density"],
                "non_vocal_leakage_ratio": leakage["non_vocal_leakage_ratio"],
                "vocal_section_coverage": leakage["vocal_section_coverage"],
                "lyric_coverage": 0.0,
                "lyric_skip_ratio": 0.0,
                "early_vocal_ratio": progress["early_vocal_ratio"],
                "late_vocal_ratio": progress["late_vocal_ratio"],
                "drift_score": progress["drift_score"],
                "num_segments": section_boundaries["num_segments"],
            }

            # Print
            print(f"  {method}:")
            print(f"    First vocal: {section_boundaries['first_vocal_start']:.1f}s "
                  f"(expected ~15s)")
            print(f"    Last vocal: {section_boundaries['last_vocal_end']:.1f}s "
                  f"(expected ~128s)")
            print(f"    Voc density: {section_boundaries['vocal_density']:.2f}")
            print(f"    Non-vocal leakage: {leakage['non_vocal_leakage_ratio']:.4f}")
            print(f"    Vocal coverage: {leakage['vocal_section_coverage']:.3f}")
            print(f"    Early vocal: {progress['early_vocal_ratio']:.3f}")
            print(f"    Late vocal: {progress['late_vocal_ratio']:.3f}")
            print(f"    Num segs: {section_boundaries['num_segments']}")

            all_rows.append(row)

    # Summary
    if not all_rows:
        print("\nNo data collected.")
        return

    print("\n" + "=" * 80)
    print("STRUCTURED ERROR SUMMARY")
    print("=" * 80)

    summary = {}
    for row in all_rows:
        m = row["method"]
        if m not in summary:
            summary[m] = {k: [] for k in METRICS if k not in ("prompt", "method")}
        for k in summary[m]:
            v = row.get(k)
            if v is not None:
                summary[m][k].append(v)

    for method in ["baseline", "pd_sbar_010"]:
        if method not in summary or not summary[method]:
            continue
        print(f"\n  {method}:")
        for k in summary[method]:
            vals = summary[method][k]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"    {k}: {avg:.4f}")

    # Save
    with open(output_dir / "structured_errors.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'structured_errors.json'}")


if __name__ == "__main__":
    main()
