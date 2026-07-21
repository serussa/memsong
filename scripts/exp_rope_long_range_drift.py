#!/usr/bin/env python3
"""
Experiment 4: Long-range Self-similarity / Drift Analysis.

Analyzes generated audio for structural repetition and drift:
  1. Mel-spectrogram self-similarity matrix → detect repetition blocks
  2. Segment-wise cosine similarity → structural diversity
  3. Novelty curve → whether later parts produce new content
  4. Compare across durations

Usage:
    python scripts/exp_rope_long_range_drift.py <audio_path> [--output-dir output/rope_exp4]
    python scripts/exp_rope_long_range_drift.py --compare output/rope_exp*/audio_*.flac
"""

import sys, json, math
from pathlib import Path
import numpy as np

SRC = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(SRC))

try:
    import librosa
except ImportError:
    print("[ERROR] librosa required. pip install librosa")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_rope_analyze import (
    compute_self_similarity,
    detect_repetition_blocks,
    compute_novelty_curve,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(audio_path: str, sr: int = 22050):
    """Load audio and extract mel-spectrogram features.

    Returns dict with features, time axis, and metadata.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    dur = len(y) / sr

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000,
        hop_length=512, win_length=2048,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # [T, 128]

    # MFCCs for compression
    mfcc = librosa.feature.mfcc(S=mel_db.T, sr=sr, n_mfcc=20).T  # [T, 20]

    # Temporal segments (1s windows)
    seg_hop = sr  # 1 second
    n_segs = max(1, len(y) // seg_hop)
    seg_features = []
    for i in range(n_segs):
        seg = y[i * seg_hop:(i + 1) * seg_hop]
        if len(seg) < seg_hop // 2:
            break
        # Spectral centroid, rolloff, bandwidth
        centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)[0].mean()
        rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr)[0].mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)[0].mean()
        zcr = librosa.feature.zero_crossing_rate(seg)[0].mean()
        rms = librosa.feature.rms(y=seg)[0].mean()
        seg_features.append([centroid, rolloff, bandwidth, zcr, rms])
    seg_features = np.array(seg_features)  # [n_segs, 5]

    hop = 512  # hop_length for mel/mfcc
    return {
        "audio_path": str(audio_path),
        "duration_s": dur,
        "mel_db": mel_db,          # [T_mel, 128]
        "mel_t": np.arange(mel_db.shape[0]) * hop / sr,
        "mfcc": mfcc,              # [T_mfcc, 20]
        "mfcc_t": np.arange(mfcc.shape[0]) * hop / sr,
        "seg_features": seg_features,  # [n_segs, 5]
        "seg_t": np.arange(seg_features.shape[0]),  # in seconds
        "sr": sr,
        "hop_length": hop,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_self_similarity(features, output_dir: Path, prefix: str = ""):
    """Compute self-similarity matrix and detect repetition."""
    # Use mel-spectrogram, PCA-reduced to 20 dims for speed
    mel = features["mel_db"]
    # Downsample if too long (>2000 frames -> take every Nth)
    T = mel.shape[0]
    downsample = max(1, T // 1500)
    if downsample > 1:
        mel = mel[::downsample]
        t_axis = features["mel_t"][::downsample]
    else:
        t_axis = features["mel_t"]

    # Compute self-similarity matrix
    sim = compute_self_similarity(mel)  # [T', T']

    # Detect repetition blocks
    blocks = detect_repetition_blocks(sim, threshold=0.75, min_block=5)
    block_density = len(blocks) / sim.shape[0] * 100 if sim.shape[0] > 0 else 0

    # Novelty curve (using MFCC for smoother result)
    novelty = compute_novelty_curve(features["mfcc"])

    # Segment-wise similarity
    seg = features["seg_features"]
    seg_sim = compute_self_similarity(seg) if seg.shape[0] > 1 else np.eye(1)

    # Structural diversity: mean similarity of upper triangle (excluding diagonal)
    n = seg_sim.shape[0]
    if n > 2:
        triu = seg_sim[np.triu_indices(n, k=1)]
        diversity = 1.0 - triu.mean()
    else:
        diversity = 0.0

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Self-similarity matrix
    ax = axes[0, 0]
    im = ax.imshow(sim, aspect="auto", cmap="viridis", vmin=-0.5, vmax=1.0,
                   extent=[0, t_axis[-1], t_axis[-1], 0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Self-Similarity ({downsample}x downsampled)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Detected repetition blocks
    ax = axes[0, 1]
    ax.imshow(sim, aspect="auto", cmap="viridis", vmin=-0.5, vmax=1.0,
              extent=[0, t_axis[-1], t_axis[-1], 0])
    for s, e, offset in blocks:
        rect = plt.Rectangle(
            (t_axis[s].item(), t_axis[s + offset].item() if s + offset < len(t_axis) else 0),
            t_axis[e].item() - t_axis[s].item(),
            t_axis[e].item() - t_axis[s].item(),
            fill=False, edgecolor="red", linewidth=1, linestyle="--",
        )
        ax.add_patch(rect)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Repetition Blocks: {len(blocks)} found")

    # 3. Novelty curve
    ax = axes[0, 2]
    t_novel = features["mfcc_t"]
    ax.plot(t_novel, novelty)
    # Smooth with moving average
    if len(novelty) > 50:
        kernel = np.ones(20) / 20
        smooth = np.convolve(novelty, kernel, mode="same")
        ax.plot(t_novel, smooth, "r-", linewidth=2, label="Smoothed")
    ax.axvline(x=t_novel[len(t_novel)//2], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min Distance to Prior")
    ax.set_title("Novelty Curve (new content detection)")
    ax.legend()

    # 4. Segment similarity matrix
    ax = axes[1, 0]
    im = ax.imshow(seg_sim, aspect="auto", cmap="RdYlBu", vmin=-1, vmax=1)
    ax.set_xlabel("Segment (1s)")
    ax.set_ylabel("Segment (1s)")
    ax.set_title(f"Segment Similarity (diversity={diversity:.3f})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 5. Mel-spectrogram
    ax = axes[1, 1]
    hop = features["hop_length"]
    im = ax.imshow(features["mel_db"].T, aspect="auto", origin="lower",
                   cmap="magma",
                   extent=[0, features["duration_s"], 0, 128])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel band")
    ax.set_title("Mel-spectrogram")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary_lines = [
        f"Duration: {features['duration_s']:.1f}s",
        f"Self-sim blocks: {len(blocks)}",
        f"Block density: {block_density:.1f}%",
        f"Segment diversity: {diversity:.3f}",
        f"Novelty (mean last 10%): {novelty[-max(1,len(novelty)//10):].mean():.4f}",
        f"Novelty (mean first 10%): {novelty[:max(1,len(novelty)//10)].mean():.4f}",
        f"Novelty decay ratio: {novelty[-max(1,len(novelty)//10):].mean() / max(1e-8, novelty[:max(1,len(novelty)//10)].mean()):.3f}",
    ]
    ax.text(0.1, 0.9, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plot_path = output_dir / f"{prefix}self_similarity.png"
    plt.savefig(plot_path, dpi=120)
    print(f"  → saved {plot_path}")
    plt.close()

    # Return metrics
    return {
        "duration": features["duration_s"],
        "n_self_sim_blocks": len(blocks),
        "block_density_pct": block_density,
        "segment_diversity": diversity,
        "novelty_mean_first10pct": float(novelty[:max(1, len(novelty)//10)].mean()),
        "novelty_mean_last10pct": float(novelty[-max(1, len(novelty)//10):].mean()),
        "novelty_decay_ratio": float(
            novelty[-max(1, len(novelty)//10):].mean()
            / max(1e-8, novelty[:max(1, len(novelty)//10)].mean())
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Long-range drift analysis")
    parser.add_argument("audio_paths", nargs="+", help="Audio file(s) to analyze")
    parser.add_argument("--output-dir", default=str(SRC / "output" / "rope_exp4"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for i, audio_path in enumerate(args.audio_paths):
        print(f"\n[{i+1}/{len(args.audio_paths)}] Analyzing {audio_path}")
        if not Path(audio_path).exists():
            print(f"  ⚠ File not found, skipping")
            continue

        features = extract_features(audio_path)
        print(f"  Duration: {features['duration_s']:.1f}s, "
              f"mel_frames={features['mel_db'].shape[0]}, "
              f"segments={features['seg_features'].shape[0]}")

        prefix = Path(audio_path).stem + "_"
        metrics = analyze_self_similarity(features, output_dir, prefix=prefix)
        all_metrics.append(metrics)

    # Save combined metrics
    if all_metrics:
        json_path = output_dir / "drift_metrics.json"
        json_path.write_text(json.dumps(all_metrics, indent=2))
        print(f"\n→ saved {json_path}")

        # Duration comparison plot
        if len(all_metrics) > 1:
            durs = [m["duration"] for m in all_metrics]
            diversities = [m["segment_diversity"] for m in all_metrics]
            decays = [m["novelty_decay_ratio"] for m in all_metrics]
            blocks = [m["n_self_sim_blocks"] for m in all_metrics]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(durs, diversities, "bo-")
            axes[0].set_xlabel("Duration (s)")
            axes[0].set_ylabel("Segment Diversity")
            axes[0].set_title("Diversity vs Duration")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(durs, decays, "ro-")
            axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            axes[1].set_xlabel("Duration (s)")
            axes[1].set_ylabel("Novelty Decay Ratio")
            axes[1].set_title("Novelty Decay vs Duration")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(durs, blocks, "go-")
            axes[2].set_xlabel("Duration (s)")
            axes[2].set_ylabel("Repetition Blocks")
            axes[2].set_title("Repetition vs Duration")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(output_dir / "duration_comparison.png", dpi=120)
            print(f"→ saved {output_dir / 'duration_comparison.png'}")
            plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
