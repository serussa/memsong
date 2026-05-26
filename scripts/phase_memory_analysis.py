#!/usr/bin/env python3
"""Phase Memory recurrence dynamics visualization.

Expected input:
- phase_history.pt (torch Tensor) with shape [T, D]
  containing phase angles in radians within [-pi, pi].

Outputs (in output_dir):
- recurrence_heatmap.png
- phase_trajectory.png
- recurrence_curve.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_phase_history(path: Path) -> torch.Tensor:
    """Load phase history from a .pt file.

    Args:
        path: Path to phase_history.pt.

    Returns:
        Tensor with shape [T, D] on CPU (float32).
    """
    phase = torch.load(path, map_location="cpu")
    if not isinstance(phase, torch.Tensor):
        raise TypeError("phase_history.pt must contain a torch.Tensor")
    if phase.ndim != 2:
        raise ValueError("phase_history must have shape [T, D]")
    return phase.float()


def compute_recurrence_matrix(phase: torch.Tensor) -> torch.Tensor:
    """Compute recurrence matrix R(i, j) = mean_k cos(theta_i_k - theta_j_k).

    Args:
        phase: Tensor [T, D] with phase angles in radians.

    Returns:
        Tensor [T, T] recurrence matrix.
    """
    # Expand to [T, 1, D] and [1, T, D], then compute cos(delta) and mean over D.
    delta = phase[:, None, :] - phase[None, :, :]
    recurrence = torch.cos(delta).mean(dim=-1)
    return recurrence


def plot_recurrence_heatmap(recurrence: torch.Tensor, out_path: Path) -> None:
    """Plot recurrence heatmap.

    Args:
        recurrence: Tensor [T, T].
        out_path: Output PNG path.
    """
    rec = recurrence.numpy()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(rec, cmap="magma", origin="lower", aspect="auto")
    plt.colorbar(im, label="Mean cos phase diff")
    plt.title("Phase Recurrence Heatmap")
    plt.xlabel("Time Step")
    plt.ylabel("Time Step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_phase_trajectories(phase: torch.Tensor, out_path: Path, num_dims: int = 4) -> None:
    """Plot phase orbit trajectories for the first few dimensions.

    Args:
        phase: Tensor [T, D].
        out_path: Output PNG path.
        num_dims: Number of dimensions to plot (default: 4).
    """
    t, d = phase.shape
    num_dims = min(num_dims, d)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        if i < num_dims:
            theta = phase[:, i].numpy()
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x, y, linewidth=1.0)
            ax.scatter([x[0]], [y[0]], s=10, c="cyan", label="start")
            ax.set_title(f"Dim {i}")
            ax.set_aspect("equal", "box")
            ax.set_xlabel("cos(theta)")
            ax.set_ylabel("sin(theta)")
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_recurrence_curve(phase: torch.Tensor) -> np.ndarray:
    """Compute mean recurrence vs temporal distance.

    For each delta in [1, T-1], compute mean_k,t cos(theta_t_k - theta_{t+delta}_k).

    Args:
        phase: Tensor [T, D].

    Returns:
        Array of length T-1 with average recurrence for each delta.
    """
    t, _ = phase.shape
    curve = np.zeros(t - 1, dtype=np.float32)
    for delta in range(1, t):
        diff = phase[:-delta] - phase[delta:]
        curve[delta - 1] = torch.cos(diff).mean().item()
    return curve


def plot_recurrence_curve(curve: np.ndarray, out_path: Path) -> None:
    """Plot recurrence vs temporal distance.

    Args:
        curve: Array [T-1] with average recurrence per delta.
        out_path: Output PNG path.
    """
    x = np.arange(1, len(curve) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, curve, linewidth=1.5)
    plt.title("Temporal Recurrence Curve")
    plt.xlabel("Temporal distance (delta)")
    plt.ylabel("Average recurrence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PhaseMemory recurrence analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="phase_history.pt",
        help="Path to phase_history.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="phase_memory_analysis",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase = load_phase_history(input_path)

    # 1) Recurrence heatmap
    recurrence = compute_recurrence_matrix(phase)
    plot_recurrence_heatmap(recurrence, output_dir / "recurrence_heatmap.png")

    # 2) Phase trajectories
    plot_phase_trajectories(phase, output_dir / "phase_trajectory.png", num_dims=4)

    # 3) Temporal recurrence curve
    curve = compute_recurrence_curve(phase)
    plot_recurrence_curve(curve, output_dir / "recurrence_curve.png")

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
