#!/usr/bin/env python3
"""Visualize mel spectrogram, chromagram, and time-time similarity for an audio file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import librosa
    import librosa.display
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    librosa = None


FIXED_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "audio_analysis"


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a visualization containing mel spectrogram, "
            "chromagram, and a time-time similarity matrix."
        )
    )
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=Path("/root/ACE-Step-1.5/output/e07e419d-8da6-2853-1c4a-77e207d6b3ca.flac"),
        help="Input audio path (wav/mp3/flac/...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Base output filename (always saved under output/audio_analysis)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Target sample rate for analysis (default: 22050)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for frame-based features (default: 512)",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Mel bins for mel spectrogram (default: 128)",
    )
    parser.add_argument(
        "--sync-step",
        type=int,
        default=10,
        help="Frame aggregation step for similarity matrix (default: 10)",
    )
    return parser


def _resolve_output_paths(audio_path: Path, output_path: Path | None) -> tuple[Path, Path]:
    """Resolve output paths under the fixed output directory."""
    if output_path is not None:
        stem = output_path.stem
    else:
        stem = audio_path.stem

    feature_path = FIXED_OUTPUT_DIR / f"{stem}_features.png"
    similarity_path = FIXED_OUTPUT_DIR / f"{stem}_similarity.png"
    return feature_path, similarity_path


def _compute_features(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    n_mels: int,
    sync_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mel spectrogram, chromagram, and chroma-based time-time similarity."""
    if librosa is None:
        raise ModuleNotFoundError("librosa")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    frame_count = chroma.shape[1]
    step = max(sync_step, 1)
    sync_boundaries = np.arange(0, frame_count, step)
    if sync_boundaries.size == 0:
        sync_boundaries = np.array([0])

    chroma_smoothed = librosa.util.sync(chroma, sync_boundaries)
    chroma_vectors = chroma_smoothed.T

    vector_norms = np.linalg.norm(chroma_vectors, axis=1, keepdims=True)
    vector_norms = np.maximum(vector_norms, 1e-10)
    chroma_normalized = chroma_vectors / vector_norms

    time_correlation_matrix = chroma_normalized @ chroma_normalized.T
    time_correlation_matrix = np.clip(time_correlation_matrix, 0.0, 1.0)
    frame_times = librosa.frames_to_time(
        sync_boundaries,
        sr=sr,
        hop_length=hop_length,
    )

    return mel_db, chroma, frame_times, time_correlation_matrix


def _plot_features(
    mel_db: np.ndarray,
    chroma: np.ndarray,
    frame_times: np.ndarray,
    time_correlation_matrix: np.ndarray,
    sr: int,
    hop_length: int,
    feature_output_path: Path,
    similarity_output_path: Path,
    title: str,
) -> None:
    """Render and save mel/chroma plot and similarity plot separately."""
    if librosa is None:
        raise ModuleNotFoundError("librosa")

    fig_features, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    img1 = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[0],
    )
    axes[0].set_title("Mel Spectrogram (dB)")
    fig_features.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(
        chroma,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="chroma",
        ax=axes[1],
    )
    axes[1].set_title("Chromagram")
    fig_features.colorbar(img2, ax=axes[1])

    fig_features.suptitle(title)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_features.savefig(feature_output_path, dpi=180)
    plt.close(fig_features)

    fig_similarity, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    img3 = librosa.display.specshow(
        time_correlation_matrix,
        x_coords=frame_times,
        y_coords=frame_times,
        x_axis="time",
        y_axis="time",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title("Chroma Self-Similarity Matrix")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Time (seconds)")
    fig_similarity.colorbar(img3, ax=ax)
    fig_similarity.suptitle(title)
    similarity_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_similarity.savefig(similarity_output_path, dpi=180)
    plt.close(fig_similarity)


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if librosa is None:
        raise RuntimeError(
            "This script requires librosa. Install it with: pip install librosa"
        )

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    y, sr = librosa.load(str(audio_path), sr=args.sr, mono=True)

    mel_db, chroma, frame_times, time_correlation_matrix = _compute_features(
        y=y,
        sr=sr,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        sync_step=args.sync_step,
    )

    feature_output_path, similarity_output_path = _resolve_output_paths(audio_path, args.output)
    _plot_features(
        mel_db=mel_db,
        chroma=chroma,
        frame_times=frame_times,
        time_correlation_matrix=time_correlation_matrix,
        sr=sr,
        hop_length=args.hop_length,
        feature_output_path=feature_output_path,
        similarity_output_path=similarity_output_path,
        title=f"Audio Analysis: {audio_path.name}",
    )

    print(f"Saved feature figure to: {feature_output_path}")
    print(f"Saved similarity figure to: {similarity_output_path}")


if __name__ == "__main__":
    main()
