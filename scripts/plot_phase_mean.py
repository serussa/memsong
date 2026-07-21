#!/usr/bin/env python3
"""Plot unwrapped phase velocity and spike-time statistics from phase_history.pt."""

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze phase velocity and spike times")
    parser.add_argument(
        "--phase-path",
        type=str,
        required=True,
        help="Path to phase_history.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: alongside phase_history.pt)",
    )
    parser.add_argument(
        "--mark-t",
        type=int,
        default=10,
        help="Time step to mark with a vertical line",
    )
    parser.add_argument(
        "--peak-mode",
        type=str,
        default="abs",
        choices=["abs", "pos"],
        help="Spike time metric: abs=max|v|, pos=max v",
    )
    return parser.parse_args()


def main() -> None:
    """Load phase history, unwrap, and plot phase velocity over time."""
    args = parse_args()
    phase_path = Path(args.phase_path)
    if not phase_path.exists():
        raise FileNotFoundError(f"phase_history.pt not found: {phase_path}")

    phase_tensor = torch.load(phase_path, map_location="cpu")
    phase = np.asarray(phase_tensor, dtype=np.float32)

    phase_unwrapped = np.unwrap(phase, axis=0)
    velocity = np.diff(phase_unwrapped, axis=0)

    if args.peak_mode == "abs":
        peak_idx = np.argmax(np.abs(velocity), axis=0)
    else:
        peak_idx = np.argmax(velocity, axis=0)

    peak_mean = float(np.mean(peak_idx))
    peak_var = float(np.var(peak_idx))

    output_path = (
        Path(args.output)
        if args.output is not None
        else phase_path.with_name("phase_velocity.png")
    )

    hist_path = output_path.with_name("phase_velocity_peak_hist.png")
    align_path = output_path.with_name("phase_velocity_aligned.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    mean_velocity = velocity.mean(axis=1)
    ax.plot(np.arange(1, mean_velocity.shape[0] + 1), mean_velocity, linewidth=1.5)
    ax.axvline(args.mark_t, color="orange", linestyle="--", linewidth=1.0)
    ax.set_title("Phase Velocity Over Time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Velocity (radians per step)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, velocity.shape[0] + 2) - 0.5
    ax.hist(peak_idx, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(peak_mean, color="orange", linestyle="--", linewidth=1.0)
    ax.set_title("Spike Time Histogram")
    ax.set_xlabel("t_peak")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)

    max_shift = int(velocity.shape[0])
    aligned = np.full((velocity.shape[1], 2 * max_shift + 1), np.nan, dtype=np.float32)
    center = max_shift
    for i in range(velocity.shape[1]):
        shift = int(peak_idx[i])
        start = center - shift
        end = start + velocity.shape[0]
        aligned[i, start:end] = velocity[:, i]

    aligned_mean = np.nanmean(aligned, axis=0)
    x = np.arange(-max_shift, max_shift + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(aligned.shape[0]):
        ax.plot(x, aligned[i], color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(x, aligned_mean, color="black", linewidth=1.5)
    ax.axvline(0, color="orange", linestyle="--", linewidth=1.0)
    ax.set_title("Aligned Velocity Curves (t_peak=0)")
    ax.set_xlabel("Aligned time")
    ax.set_ylabel("Velocity (radians per step)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(align_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {align_path}")
    print(f"t_peak mean: {peak_mean:.3f}")
    print(f"t_peak var: {peak_var:.3f}")


if __name__ == "__main__":
    main()
