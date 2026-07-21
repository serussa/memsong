#!/usr/bin/env python3
"""
Phase Temporal Coordinate Analysis
==================================
Empirically tests whether the latent phase variable φ from PhaseMemory
behaves like an emergent temporal coordinate in diffusion dynamics.

Five lines of evidence:
1. Monotonicity — phase consistently increases over diffusion time
2. Time correlation — φ correlates with token index
3. Smoothness — phase velocity is low-variance
4. Ablation — structure degrades when phase is perturbed
5. Behavioral — self-similarity and long-range coherence
6. Predictive — linear probe for temporal boundaries

Usage:
    # From pre-saved phase history:
    python scripts/phase_temporal_coordinate_analysis.py \\
        --phase-path /path/to/phase_history.pt --output-dir ./output

    # From z_r/z_i tensors:
    python scripts/phase_temporal_coordinate_analysis.py \\
        --zr-path z_r.pt --zi-path z_i.pt --output-dir ./output

    # With model inference (collects phase history automatically):
    python scripts/phase_temporal_coordinate_analysis.py \\
        --model-root /path/to/model --pm-dir /path/to/checkpoint --output-dir ./output
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# numpy.trapz → trapezoid compatibility (removed in NumPy 2.x)
if not hasattr(np, "trapezoid"):
    np_trapz = np_trapz
else:
    np_trapz = np.trapezoid

matplotlib_backend_set = False


def _setup_matplotlib():
    global matplotlib_backend_set
    if not matplotlib_backend_set:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib_backend_set = True


import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. PHASE EXTRACTION UTILITY
# ==============================================================================

def compute_phi(z_r: torch.Tensor, z_i: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """
    Compute unwrapped phase φ = unwrap(atan2(z_i, z_r)).

    Args:
        z_r: Real component [..., T, D] or [T, D].
        z_i: Imaginary component, same shape as z_r.
        dim: Time dimension to unwrap over.

    Returns:
        phi: Unwrapped phase, same shape as input, float32 on CPU.
    """
    phi_wrapped = torch.atan2(z_i.float(), z_r.float())
    phi = unwrap_phase(phi_wrapped, dim=dim)
    return phi


def unwrap_phase(phase: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """
    Unwrap phase along specified dimension.

    Detects discontinuities larger than π and corrects by adding/subtracting
    multiples of 2π, mimicking numpy.unwrap.

    Args:
        phase: Tensor of phase angles in radians.
        dim: Dimension along which to unwrap.

    Returns:
        Unwrapped phase tensor.
    """
    if dim != -1:
        phase = phase.transpose(dim, -1)

    diff = phase[..., 1:] - phase[..., :-1]
    dphi = torch.where(
        diff > math.pi, diff - 2 * math.pi,
        torch.where(diff < -math.pi, diff + 2 * math.pi, diff),
    )

    unwrapped = torch.cat([
        phase[..., :1],
        phase[..., :1] + torch.cumsum(dphi, dim=-1),
    ], dim=-1)

    if dim != -1:
        unwrapped = unwrapped.transpose(dim, -1)
    return unwrapped


# ==============================================================================
# 2. EMPIRICAL EVIDENCE METRICS
# ==============================================================================

def compute_monotonicity_ratio(phi: torch.Tensor) -> dict:
    """
    Fraction of time steps where phase increases (Δφ > 0).

    Args:
        phi: [T, D] unwrapped phase.

    Returns:
        dict with monotonicity_ratio and per-dimension values.
    """
    if phi.ndim == 1:
        phi = phi.unsqueeze(-1)

    delta = phi[1:] - phi[:-1]  # [T-1, D]
    mono_per_dim = (delta > 0).float().mean(dim=0).numpy()
    mono_ratio = float(mono_per_dim.mean())

    return {
        "monotonicity_ratio": mono_ratio,
        "monotonicity_per_dim": mono_per_dim.tolist(),
    }


def compute_phi_time_correlation(phi: torch.Tensor) -> dict:
    """
    Pearson correlation between φ and time index for each dimension.

    Args:
        phi: [T, D] unwrapped phase.

    Returns:
        dict with correlation mean, std, and per-dim values.
    """
    T = phi.shape[0]
    t = np.arange(T)

    corrs = []
    ps = []
    for d in range(phi.shape[1]):
        r, p = pearsonr(t, phi[:, d].numpy())
        corrs.append(r)
        ps.append(p)

    corrs = np.array(corrs)
    ps = np.array(ps)

    return {
        "phi_time_corr_mean": float(np.nanmean(corrs)),
        "phi_time_corr_std": float(np.nanstd(corrs)),
        "phi_time_corr_per_dim": corrs.tolist(),
        "phi_time_p_mean": float(np.nanmean(ps)),
    }


def compute_smoothness(phi: torch.Tensor) -> dict:
    """
    Phase velocity statistics and second-order smoothness.

    Args:
        phi: [T, D] unwrapped phase.

    Returns:
        dict with velocity variance, mean absolute acceleration, etc.
    """
    if phi.ndim == 1:
        phi = phi.unsqueeze(-1)

    velocity = phi[1:] - phi[:-1]  # [T-1, D]
    accel = velocity[1:] - velocity[:-1]  # [T-2, D]

    var_v = float(velocity.var(dim=0).mean().item())
    mean_abs_accel = float(accel.abs().mean().item())
    mean_abs_v = float(velocity.abs().mean().item())
    mean_v = float(velocity.mean().item())

    return {
        "phi_velocity_variance": var_v,
        "phi_velocity_mean": mean_v,
        "phi_velocity_abs_mean": mean_abs_v,
        "phi_abs_acceleration_mean": mean_abs_accel,
    }


def compute_all_empirical_metrics(phi: torch.Tensor) -> dict:
    """Compute all empirical evidence metrics (monotonicity, correlation, smoothness)."""
    metrics = {}
    metrics.update(compute_monotonicity_ratio(phi))
    metrics.update(compute_phi_time_correlation(phi))
    metrics.update(compute_smoothness(phi))
    # Composite: if monotonic, correlated, and smooth → strong temporal signal
    phi_val = metrics.get("monotonicity_ratio", 0.0)
    corr_val = abs(metrics.get("phi_time_corr_mean", 0.0))
    smooth_val = 1.0 - min(1.0, metrics.get("phi_abs_acceleration_mean", 1.0) * 10)
    metrics["temporal_coordinate_score"] = float((phi_val + corr_val + smooth_val) / 3.0)
    return metrics


# ==============================================================================
# 3. ABLATION SIMULATION
# ==============================================================================

def make_shuffled_phase(phi: torch.Tensor) -> torch.Tensor:
    """Shuffle phase along time dimension independently per dim."""
    T, D = phi.shape
    shuffled = phi.clone()
    for d in range(D):
        shuffled[:, d] = shuffled[torch.randperm(T), d]
    return shuffled


def make_random_phase(phi: torch.Tensor) -> torch.Tensor:
    """Gaussian noise matched to per-dimension mean/std of phi."""
    means = phi.mean(dim=0, keepdim=True)
    stds = phi.std(dim=0, keepdim=True)
    return torch.randn_like(phi) * stds + means


def make_random_walk_phase(T: int, D: int, scale: float = 0.1) -> torch.Tensor:
    """Random walk baseline: cumsum of Gaussian increments."""
    increments = torch.randn(T - 1, D) * scale
    walk = torch.cat([torch.zeros(1, D), increments.cumsum(dim=0)], dim=0)
    return walk


def compute_temporal_autocorrelation_decay(phi: torch.Tensor) -> np.ndarray:
    """
    Temporal autocorrelation: mean cos(φ[t] - φ[t+d]) for lag d.
    Returns array of length T-1.
    """
    T = phi.shape[0]
    decay = np.zeros(T - 1)
    for d in range(1, T):
        decay[d - 1] = torch.cos(phi[:-d] - phi[d:]).mean().item()
    return decay


def compute_repetition_score(phi: torch.Tensor) -> float:
    """
    Repetition score: mean of maximum self-similarity at non-zero lag.

    A high score means the phase revisits similar states (low temporal
    structure), while a low score means states are distinct over time.
    """
    T = phi.shape[0]
    similarities = torch.zeros(T)
    # Detect when the model revisits similar phase states later in time
    for t in range(T):
        sims = torch.cos(phi[t] - phi).mean(dim=-1)  # [T]
        # Exclude self and very close neighbors (lag < 2)
        mask = torch.ones(T, dtype=torch.bool)
        mask[max(0, t - 1):min(T, t + 2)] = False
        if mask.any():
            similarities[t] = sims[mask].max().item()
    return float(similarities.mean().item())


def run_ablation_analysis(phi: torch.Tensor, seed: int = 42) -> dict:
    """
    Compare temporal structure under original, shuffled, and random phase.

    Args:
        phi: [T, D] unwrapped phase.

    Returns:
        dict with ablation metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = phi.shape[0]

    # Original
    orig_decay = compute_temporal_autocorrelation_decay(phi)
    orig_repetition = compute_repetition_score(phi)
    # Mean squared autocorrelation (robust to sign of decay curve)
    orig_mse = float((orig_decay ** 2).mean())
    # Short-range coherence: average first 5 lags
    orig_short = float(orig_decay[:min(5, len(orig_decay))].mean())
    # Full-range mean absolute coherence (robust to sign flips)
    orig_abs_mean = float(np.abs(orig_decay).mean())

    # Shuffled (temporal order destroyed)
    phi_shuffled = make_shuffled_phase(phi)
    shuf_decay = compute_temporal_autocorrelation_decay(phi_shuffled)
    shuf_repetition = compute_repetition_score(phi_shuffled)
    shuf_mse = float((shuf_decay ** 2).mean())
    shuf_short = float(shuf_decay[:min(5, len(shuf_decay))].mean())
    shuf_abs_mean = float(np.abs(shuf_decay).mean())

    # Random phase (structure destroyed)
    phi_random = make_random_phase(phi)
    rand_decay = compute_temporal_autocorrelation_decay(phi_random)
    rand_repetition = compute_repetition_score(phi_random)
    rand_mse = float((rand_decay ** 2).mean())
    rand_short = float(rand_decay[:min(5, len(rand_decay))].mean())
    rand_abs_mean = float(np.abs(rand_decay).mean())

    # Random walk baseline
    phi_walk = make_random_walk_phase(T, phi.shape[1])
    walk_decay = compute_temporal_autocorrelation_decay(phi_walk)
    walk_repetition = compute_repetition_score(phi_walk)
    walk_mse = float((walk_decay ** 2).mean())
    walk_short = float(walk_decay[:min(5, len(walk_decay))].mean())
    walk_abs_mean = float(np.abs(walk_decay).mean())

    # Relative drop in mean squared coherence (always non-negative)
    drop_shuf = max(0.0, (orig_mse - shuf_mse) / (orig_mse + 1e-8))
    drop_rand = max(0.0, (orig_mse - rand_mse) / (orig_mse + 1e-8))

    return {
        "orig_coherence_mse": orig_mse,
        "shuffled_coherence_mse": shuf_mse,
        "random_coherence_mse": rand_mse,
        "random_walk_coherence_mse": walk_mse,
        "orig_coherence_short": orig_short,
        "shuffled_coherence_short": shuf_short,
        "random_coherence_short": rand_short,
        "random_walk_coherence_short": walk_short,
        "orig_coherence_abs_mean": orig_abs_mean,
        "shuffled_coherence_abs_mean": shuf_abs_mean,
        "random_coherence_abs_mean": rand_abs_mean,
        "random_walk_coherence_abs_mean": walk_abs_mean,
        "ablation_drop_shuffled": float(drop_shuf),
        "ablation_drop_random": float(drop_rand),
        "orig_repetition_score": orig_repetition,
        "shuffled_repetition_score": shuf_repetition,
        "random_repetition_score": rand_repetition,
        "random_walk_repetition_score": walk_repetition,
        "orig_autocorr_decay": orig_decay,
        "shuffled_autocorr_decay": shuf_decay,
        "random_autocorr_decay": rand_decay,
        "random_walk_autocorr_decay": walk_decay,
    }


# ==============================================================================
# 4. BEHAVIORAL EVIDENCE
# ==============================================================================

def compute_self_similarity(phi: torch.Tensor) -> np.ndarray:
    """
    Self-similarity matrix: S[i,j] = mean_k cos(φ_i,k - φ_j,k).

    Args:
        phi: [T, D] phase tensor.

    Returns:
        [T, T] numpy array.
    """
    delta = phi[:, None, :] - phi[None, :, :]  # [T, T, D]
    S = torch.cos(delta).mean(dim=-1).numpy()   # [T, T]
    return S


def compute_coherence_decay_curve(phi: torch.Tensor) -> np.ndarray:
    """
    Mean coherence at each lag: average over cos(φ[t] - φ[t+d]).
    Returns [T-1] array.
    """
    T = phi.shape[0]
    curve = np.zeros(T - 1)
    for d in range(1, T):
        curve[d - 1] = torch.cos(phi[:-d] - phi[d:]).mean().item()
    return curve


def compute_long_range_coherence(phi: torch.Tensor, lag_frac: float = 0.5) -> float:
    """
    Coherence at a specified fraction of total time.

    Args:
        phi: [T, D] phase tensor.
        lag_frac: Fraction of T (e.g. 0.5 = half the sequence).

    Returns:
        Mean cos(Δφ) at that lag.
    """
    T = phi.shape[0]
    lag = max(1, int(T * lag_frac))
    return float(torch.cos(phi[:-lag] - phi[lag:]).mean().item())


def compute_behavioral_evidence(phi: torch.Tensor) -> dict:
    """
    Compute behavioral evidence: self-similarity, coherence, repetition.

    Args:
        phi: [T, D] phase tensor.

    Returns:
        dict of behavioral metrics. Matrix-valued entries (self-similarity,
        coherence decay) are returned separately for saving.
    """
    S = compute_self_similarity(phi)
    decay = compute_coherence_decay_curve(phi)
    long_range_50 = compute_long_range_coherence(phi, lag_frac=0.5)
    long_range_25 = compute_long_range_coherence(phi, lag_frac=0.25)

    # Repetition from self-similarity (max off-diagonal similarity per row)
    T = phi.shape[0]
    off_diag = S.copy()
    for t in range(T):
        off_diag[t, max(0, t - 1):min(T, t + 2)] = -np.inf
    repetition_from_sim = float(off_diag.max(axis=1).mean())

    # PCA on phase: how many dimensions are effectively used?
    phi_np = phi.numpy()
    phi_centered = phi_np - phi_np.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(phi_centered, full_matrices=False)
        var_explained = (s ** 2) / (s ** 2).sum()
        top3_var = float(var_explained[:3].sum())
        effective_rank = int(((s ** 2).cumsum() / (s ** 2).sum() < 0.95).sum().item()) + 1
    except np.linalg.LinAlgError:
        top3_var = float("nan")
        effective_rank = int("nan")

    return {
        "self_similarity_matrix": S,
        "coherence_decay_curve": decay,
        "long_range_coherence_0.5": long_range_50,
        "long_range_coherence_0.25": long_range_25,
        "repetition_score_self_sim": repetition_from_sim,
        "phase_pca_top3_var_explained": top3_var,
        "phase_pca_effective_rank_95": effective_rank,
    }


# ==============================================================================
# 5. PREDICTIVE EVIDENCE (Linear Probe)
# ==============================================================================

def compute_pseudo_boundaries_from_phase(phi: torch.Tensor) -> np.ndarray:
    """
    Detect pseudo-boundaries where phase velocity changes sharply.

    Uses a spectral-flux-like signal: L2 norm of the acceleration
    (second difference) of the phase. Threshold at 75th percentile.

    Args:
        phi: [T, D] phase tensor.

    Returns:
        [T] binary array, 1 at boundary points.
    """
    T = phi.shape[0]
    flux = torch.zeros(T)
    if T < 4:
        return flux.numpy()

    velocity = phi[1:] - phi[:-1]  # [T-1, D]
    accel = velocity[1:] - velocity[:-1]  # [T-2, D]
    flux[1:-1] = accel.norm(dim=-1)

    threshold = torch.quantile(flux, 0.75)
    boundaries = (flux > threshold).float().numpy()
    return boundaries


def compute_pseudo_boundaries_from_embedding(embeddings: np.ndarray) -> np.ndarray:
    """
    Alternative boundary detection using spectral flux of embeddings.

    Args:
        embeddings: [T, D] embedding sequence.

    Returns:
        [T] binary array.
    """
    T = embeddings.shape[0]
    if T < 3:
        return np.zeros(T, dtype=np.float32)

    # L2 norm of frame-to-frame difference
    diff = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=-1)
    flux = np.concatenate([[0], diff])

    threshold = np.percentile(flux, 75)
    return (flux > threshold).astype(np.float32)


def train_boundary_probe(
    phi: torch.Tensor,
    boundaries: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Train a linear probe (logistic regression) to predict boundaries from φ_t.

    Args:
        phi: [T, D] phase tensor.
        boundaries: [T] binary labels.
        test_size: Fraction for test split.
        random_state: Reproducibility seed.

    Returns:
        dict with AUC, F1, and probe metadata.
    """
    X = phi.numpy()
    y = boundaries

    if len(np.unique(y)) < 2:
        return {
            "boundary_auc": float("nan"),
            "boundary_f1": float("nan"),
            "boundary_probe_available": False,
            "boundary_n_classes": len(np.unique(y)),
        }

    # Check enough positive samples
    n_pos = int(y.sum())
    if n_pos < 2 or len(y) - n_pos < 2:
        return {
            "boundary_auc": float("nan"),
            "boundary_f1": float("nan"),
            "boundary_probe_available": False,
            "boundary_n_pos": n_pos,
            "boundary_n_neg": len(y) - n_pos,
        }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        # Chance-level baseline
        base_rate = float(y_test.mean())
        chance_f1 = 2 * base_rate / (1 + base_rate) if base_rate > 0 else 0.0

        return {
            "boundary_auc": float(auc),
            "boundary_f1": float(f1),
            "boundary_chance_f1": float(chance_f1),
            "boundary_probe_available": True,
            "boundary_coef_norm": float(np.linalg.norm(clf.coef_)),
            "boundary_n_test": int(len(y_test)),
            "boundary_n_boundaries": int(y.sum()),
            "boundary_base_rate": float(base_rate),
        }
    except Exception as e:
        return {
            "boundary_auc": float("nan"),
            "boundary_f1": float("nan"),
            "boundary_probe_available": False,
            "boundary_error": str(e),
        }


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_phi_trajectories(phi: torch.Tensor, save_path: str) -> None:
    """Plot sample phase trajectories (first 4 dims) over diffusion steps."""
    _setup_matplotlib()
    T, D = phi.shape
    nplot = min(4, D)
    ncols = min(2, nplot)
    nrows = math.ceil(nplot / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3 * nrows))
    axes = axes.flat if hasattr(axes, "flat") else [axes]
    t = np.arange(T)

    for i in range(nplot):
        ax = axes[i]
        ax.plot(t, phi[:, i].numpy(), linewidth=1.5, color="steelblue")
        ax.set_title(f"Dim {i},  "
                     f"r(t,φ)={pearsonr(t, phi[:, i].numpy())[0]:.3f}", fontsize=10)
        ax.set_xlabel("Diffusion Step")
        ax.set_ylabel("φ (rad)")
        ax.grid(True, alpha=0.3)

    for i in range(nplot, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Phase φ over Diffusion Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_delta_phi_histogram(phi: torch.Tensor, save_path: str) -> None:
    """Histogram of all Δφ values across dims and time."""
    _setup_matplotlib()
    delta = (phi[1:] - phi[:-1]).flatten().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(delta, bins=80, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Δφ=0")
    ax.set_xlabel("Δφ (rad/step)")
    ax.set_ylabel("Count")
    ax.set_title("Phase Velocity Distribution (Δφ across all dims)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_correlation_scatter(phi: torch.Tensor, save_path: str) -> None:
    """Scatter + regression line for φ vs time index, several dims."""
    _setup_matplotlib()
    T, D = phi.shape
    t = np.arange(T)
    nplot = min(6, D)
    ncols = min(3, nplot)
    nrows = math.ceil(nplot / 3)

    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.5 * nrows))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for i in range(nplot):
        ax = axes[i]
        ax.scatter(t, phi[:, i].numpy(), s=8, alpha=0.5, c="steelblue")
        coeffs = np.polyfit(t, phi[:, i].numpy(), 1)
        poly = np.poly1d(coeffs)
        r, p = pearsonr(t, phi[:, i].numpy())
        ax.plot(t, poly(t), "r--", linewidth=1.5,
                label=f"r={r:.3f}" + ("**" if p < 0.01 else "*" if p < 0.05 else ""))
        ax.set_xlabel("t (diffusion step)")
        ax.set_ylabel("φ")
        ax.set_title(f"Dimension {i}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(nplot, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Phase φ vs Time Index", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ablation_comparison(ablation: dict, save_path: str) -> None:
    """Bar chart comparing original vs shuffled vs random vs walk."""
    _setup_matplotlib()
    categories = ["Coherence MSE\n(↑ structured)", "Short-range\nCoherence (↑)"]
    orig = [ablation["orig_coherence_mse"], ablation["orig_coherence_short"]]
    shuffled = [ablation["shuffled_coherence_mse"], ablation["shuffled_coherence_short"]]
    random_ = [ablation["random_coherence_mse"], ablation["random_coherence_short"]]
    walk = [ablation["random_walk_coherence_mse"], ablation["random_walk_coherence_short"]]

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, orig, width, label="Original", color="steelblue", alpha=0.85)
    ax.bar(x - 0.5 * width, shuffled, width, label="Shuffled", color="coral", alpha=0.85)
    ax.bar(x + 0.5 * width, random_, width, label="Random", color="seagreen", alpha=0.85)
    ax.bar(x + 1.5 * width, walk, width, label="Random Walk", color="goldenrod", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: Temporal Structure Degradation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    def _annotate(vals, offset):
        for v in vals:
            ax.text(offset, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    _annotate(orig, x[0] - 1.5 * width)
    _annotate(shuffled, x[0] - 0.5 * width)
    _annotate(random_, x[0] + 0.5 * width)
    _annotate(walk, x[0] + 1.5 * width)
    _annotate(orig, x[1] - 1.5 * width)
    _annotate(shuffled, x[1] - 0.5 * width)
    _annotate(random_, x[1] + 0.5 * width)
    _annotate(walk, x[1] + 1.5 * width)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_coherence_decay_curves(ablation: dict, save_path: str) -> None:
    """Plot all coherence decay curves on one axis."""
    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"orig": "steelblue", "shuffled": "coral",
              "random": "seagreen", "random_walk": "goldenrod"}

    for key, label, ckey in [
        ("orig_autocorr_decay", "Original", "orig"),
        ("shuffled_autocorr_decay", "Shuffled", "shuffled"),
        ("random_autocorr_decay", "Random", "random"),
        ("random_walk_autocorr_decay", "Random Walk", "random_walk"),
    ]:
        if key in ablation:
            curve = ablation[key]
            ax.plot(np.arange(1, len(curve) + 1), curve,
                    label=label, color=colors[ckey],
                    linewidth=1.5)

    ax.set_xlabel("Lag (diffusion steps)")
    ax.set_ylabel("Mean cos(Δφ)")
    ax.set_title("Temporal Coherence Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_self_similarity(S: np.ndarray, save_path: str,
                         title: str = "Phase Self-Similarity") -> None:
    """Self-similarity matrix heatmap."""
    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(S, cmap="viridis", origin="lower", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="cos(Δφ)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Time Step")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_velocity_across_steps(phi: torch.Tensor, save_path: str) -> None:
    """Phase velocity vs diffusion step for sample dims."""
    _setup_matplotlib()
    T, D = phi.shape
    velocity = (phi[1:] - phi[:-1]).numpy()

    nplot = min(4, D)
    ncols = 2
    nrows = math.ceil(nplot / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3 * nrows))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for i in range(nplot):
        ax = axes[i]
        ax.plot(np.arange(1, T), velocity[:, i], linewidth=1.0, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Diffusion Step")
        ax.set_ylabel("Δφ")
        ax.set_title(f"Dim {i}: Phase Velocity")
        ax.grid(True, alpha=0.3)

    for i in range(nplot, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Phase Velocity v = Δφ over Diffusion Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pseudo_boundaries(phi: torch.Tensor, boundaries: np.ndarray,
                           save_path: str) -> None:
    """Mark detected pseudo-boundaries on phase trajectory."""
    _setup_matplotlib()
    T, D = phi.shape
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot first 3 dims as small multiples
    for d in range(min(3, D)):
        offset = d * 5  # vertical offset for visibility
        ax.plot(phi[:, d].numpy() + offset, linewidth=1.0,
                label=f"φ dim {d}", alpha=0.7)

    # Mark boundaries
    b_idx = np.where(boundaries > 0)[0]
    ax.scatter(b_idx, np.full_like(b_idx, -0.5),
               color="red", s=30, label=f"Boundary (n={len(b_idx)})",
               zorder=5, marker="v")

    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("φ (rad, offset)")
    ax.set_title("Pseudo-Boundaries from Phase Acceleration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_correlation_histogram(metrics: dict, save_path: str) -> None:
    """Histogram of per-dimension phi-time correlations."""
    _setup_matplotlib()
    corrs = np.array(metrics.get("phi_time_corr_per_dim", []))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(corrs, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(corrs.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean r={corrs.mean():.3f}")
    ax.set_xlabel("Pearson r (φ vs time)")
    ax.set_ylabel("Count (dimensions)")
    ax.set_title("Distribution of Phase–Time Correlations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_velocity_distribution(ablation: dict, phi: torch.Tensor,
                                save_path: str) -> None:
    """Compare velocity distributions across ablation conditions."""
    _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    conditions = [
        (phi, "Original", "steelblue"),
        (make_shuffled_phase(phi), "Shuffled", "coral"),
        (make_random_phase(phi), "Random", "seagreen"),
    ]

    for ax, (p, label, color) in zip(axes, conditions):
        vel = (p[1:] - p[:-1]).flatten().numpy()
        ax.hist(vel, bins=60, alpha=0.7, color=color, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Δφ")
        ax.set_ylabel("Count")
        ax.set_title(f"Velocity: {label}")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phase Velocity Distribution: Ablation Comparison",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ==============================================================================
# INFERENCE COLLECTION (if model + checkpoint provided)
# ==============================================================================

def collect_phase_history_from_model(
    model,
    pm_module,
    num_steps: int = 50,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """
    Run the REAL generate_audio pipeline and collect wrapped phase snapshots.

    Uses the model's null_condition_emb as conditioning (unconditional generation)
    but goes through the full diffusion loop with proper timestep scheduling,
    ODE solver, and classifier-free guidance.

    Returns:
        phase_history: [T, S, D] wrapped phase angles in [-π, π],
                       where T=diffusion steps, S=tokens, D=mem_dim.
    """
    torch.manual_seed(seed)
    model_dtype = model.dtype
    device_t = torch.device(device)

    # Build generate_audio kwargs matching real pipeline shapes
    B = 1
    audio_acoustic_dim = model.config.audio_acoustic_hidden_dim  # 64
    text_dim = model.config.text_hidden_dim  # 1024
    seq_len = 25 * 30  # 30 seconds at 25 Hz = 750 frames

    # Text conditioning: expand null_condition_emb-style embeddings
    text_len, lyric_len = 50, 100
    # Use small noise for text embeddings (close to null but nonzero)
    text_hidden_states = torch.randn(B, text_len, text_dim, dtype=model_dtype, device=device_t) * 0.02
    text_attention_mask = torch.ones(B, text_len, dtype=model_dtype, device=device_t)
    lyric_hidden_states = torch.randn(B, lyric_len, text_dim, dtype=model_dtype, device=device_t) * 0.02
    lyric_attention_mask = torch.ones(B, lyric_len, dtype=model_dtype, device=device_t)

    # No reference audio for text2music
    refer_audio_acoustic_hidden_states_packed = torch.zeros(1, seq_len, audio_acoustic_dim,
                                                            dtype=model_dtype, device=device_t)
    refer_audio_order_mask = torch.zeros(1, dtype=torch.long, device=device_t)

    # Source latents: random noise
    src_latents = torch.randn(B, seq_len, audio_acoustic_dim, dtype=model_dtype, device=device_t)
    attention_mask = torch.ones(B, seq_len, dtype=model_dtype, device=device_t)
    chunk_masks = torch.ones(B, seq_len, audio_acoustic_dim, dtype=model_dtype, device=device_t)
    silence_latent = torch.randn(B, seq_len, audio_acoustic_dim, dtype=model_dtype, device=device_t)
    is_covers = torch.zeros(B, dtype=torch.long, device=device_t)

    # Hook to capture phase at each diffusion step
    phase_history = []

    def record_phase(mod, inp, out):
        z_r = getattr(mod, "z_r", None)
        z_i = getattr(mod, "z_i", None)
        if z_r is not None and z_i is not None:
            phi = torch.atan2(z_i.float().detach(), z_r.float().detach())
            # Average over batch dimension (handles CFG doubling)
            phi_mean = phi.mean(dim=0, keepdim=True) if phi.dim() == 3 else phi.unsqueeze(0)
            phase_history.append(phi_mean.squeeze(0).cpu())

    handle = pm_module.register_forward_hook(record_phase)

    # Run real generation pipeline
    model.eval()
    print(f"  Running generate_audio: {num_steps} steps, seq_len={seq_len}, "
          f"guidance=7.0, method=ode ...")
    with torch.no_grad():
        outputs = model.generate_audio(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=silence_latent,
            attention_mask=attention_mask,
            seed=seed,
            infer_method="ode",
            use_cache=True,
            infer_steps=num_steps,
            diffusion_guidance_sale=7.0,
            audio_cover_strength=1.0,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            use_progress_bar=False,
            use_adg=False,
            shift=1.0,
        )

    handle.remove()
    phase = torch.stack(phase_history).float()
    print(f"  Collected {len(phase_history)} phase snapshots, shape={phase.shape}")
    return phase  # [T, S, D]


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================

def _per_token_aggregate_empirical(phi: torch.Tensor) -> dict:
    """
    If 3D [T, S, D], compute empirical metrics per token and aggregate.
    Returns dict with _mean and _std suffixed keys.
    """
    if phi.ndim == 2:
        m = compute_all_empirical_metrics(phi)
        return {k: m[k] for k in [
            "monotonicity_ratio", "phi_time_corr_mean", "phi_time_corr_std",
            "phi_velocity_variance", "phi_velocity_mean", "phi_velocity_abs_mean",
            "phi_abs_acceleration_mean", "temporal_coordinate_score",
        ]}

    T, S, D = phi.shape
    keys = ["monotonicity_ratio", "phi_time_corr_mean", "phi_time_corr_std",
            "phi_velocity_variance", "phi_velocity_mean", "phi_velocity_abs_mean",
            "phi_abs_acceleration_mean", "temporal_coordinate_score"]
    buckets = {k: [] for k in keys}
    # phi_time_corr_per_dim is a list, store per-dim separately
    all_corrs = []

    for s in range(S):
        m = compute_all_empirical_metrics(phi[:, s, :])
        for k in keys:
            buckets[k].append(m[k])
        all_corrs.append(m.get("phi_time_corr_per_dim", []))

    aggregated = {}
    for k in keys:
        vals = np.array(buckets[k])
        aggregated[k] = float(vals.mean())
        aggregated[k + "_std"] = float(vals.std())

    # Per-dim correlations averaged over tokens
    all_corrs = np.array(all_corrs)  # [S, D]
    aggregated["phi_time_corr_per_dim"] = all_corrs.mean(axis=0).tolist()
    aggregated["phi_time_corr_per_dim_std"] = all_corrs.std(axis=0).tolist()
    aggregated["phi_time_p_mean"] = float(np.nanmean([
        m.get("phi_time_p_mean", 0.0) for m in [compute_all_empirical_metrics(phi[:, s, :])
                                                  for s in range(min(S, 128))]
    ]))
    return aggregated


def _per_token_ablation(phi: torch.Tensor, max_tokens: int = 256) -> dict:
    """Run ablation on a subset of tokens and aggregate."""
    if phi.ndim == 2:
        return run_ablation_analysis(phi)

    T, S, D = phi.shape
    S_sub = min(S, max_tokens)
    step = max(1, S // S_sub)
    selected = list(range(0, S, step))[:S_sub]

    # Collect scalar keys from ablation
    all_scalars = {k: [] for k in [
        "orig_coherence_mse", "shuffled_coherence_mse", "random_coherence_mse",
        "random_walk_coherence_mse", "orig_coherence_short", "shuffled_coherence_short",
        "random_coherence_short", "random_walk_coherence_short",
        "orig_coherence_abs_mean", "shuffled_coherence_abs_mean",
        "random_coherence_abs_mean", "random_walk_coherence_abs_mean",
        "ablation_drop_shuffled", "ablation_drop_random",
        "orig_repetition_score", "shuffled_repetition_score",
        "random_repetition_score", "random_walk_repetition_score",
    ]}
    all_decays = {k: [] for k in ["orig_autocorr_decay", "shuffled_autocorr_decay",
                                   "random_autocorr_decay", "random_walk_autocorr_decay"]}

    for idx in selected:
        a = run_ablation_analysis(phi[:, idx, :])
        for k in all_scalars:
            if k in a:
                all_scalars[k].append(a[k])
        for k in all_decays:
            if k in a:
                all_decays[k].append(a[k])

    result = {}
    for k, vals in all_scalars.items():
        v = np.array(vals)
        result[k] = float(v.mean())
        result[k + "_std"] = float(v.std())

    for k, vals in all_decays.items():
        result[k] = np.mean(vals, axis=0)  # average decay curve over tokens

    return result


def _per_token_behavioral(phi: torch.Tensor, max_tokens: int = 128) -> dict:
    """Behavioral evidence aggregated over tokens."""
    if phi.ndim == 2:
        return compute_behavioral_evidence(phi)

    T, S, D = phi.shape
    S_sub = min(S, max_tokens)
    step = max(1, S // S_sub)

    scalar_buckets = {k: [] for k in [
        "long_range_coherence_0.5", "long_range_coherence_0.25",
        "repetition_score_self_sim", "phase_pca_top3_var_explained",
        "phase_pca_effective_rank_95",
    ]}

    for idx in range(0, S, step):
        if len(scalar_buckets["long_range_coherence_0.5"]) >= S_sub:
            break
        b = compute_behavioral_evidence(phi[:, idx, :])
        for k in scalar_buckets:
            scalar_buckets[k].append(b[k])

    result = {}
    for k, vals in scalar_buckets.items():
        v = np.array(vals)
        result[k] = float(v.mean())
        result[k + "_std"] = float(v.std())

    # Self-similarity matrix from first token for visualization
    result["self_similarity_matrix"] = compute_self_similarity(phi[:, 0, :])
    # Coherence decay from first token
    result["coherence_decay_curve"] = compute_coherence_decay_curve(phi[:, 0, :])

    return result


def _per_token_probe(phi: torch.Tensor, max_tokens: int = 256) -> dict:
    """Boundary probe aggregated over tokens."""
    if phi.ndim == 2:
        b = compute_pseudo_boundaries_from_phase(phi)
        p = train_boundary_probe(phi, b)
        return p

    T, S, D = phi.shape
    S_sub = min(S, max_tokens)

    aucs, f1s = [], []
    boundaries_0 = None

    for s in range(S_sub):
        p = phi[:, s, :]
        b = compute_pseudo_boundaries_from_phase(p)
        if s == 0:
            boundaries_0 = b
        probe = train_boundary_probe(p, b)
        if probe.get("boundary_probe_available"):
            aucs.append(probe.get("boundary_auc", float("nan")))
            f1s.append(probe.get("boundary_f1", float("nan")))

    aucs = np.array(aucs)
    f1s = np.array(f1s)

    result = {
        "boundary_auc_mean": float(np.nanmean(aucs)),
        "boundary_auc_std": float(np.nanstd(aucs)),
        "boundary_f1_mean": float(np.nanmean(f1s)),
        "boundary_f1_std": float(np.nanstd(f1s)),
        "boundary_n_tokens": len(aucs),
        "boundary_n_boundaries": int(boundaries_0.sum()) if boundaries_0 is not None else 0,
    }

    if len(aucs) > 0 and not all(np.isnan(aucs)):
        result["boundary_auc"] = result["boundary_auc_mean"]
    else:
        result["boundary_auc"] = float("nan")

    return result


def analyze_phase_as_temporal_coordinate(
    phi: torch.Tensor,
    output_dir: str = "./phase_analysis_output",
    run_ablation: bool = True,
    run_probe: bool = True,
    prefix: str = "",
) -> dict:
    """
    Run the full analysis pipeline on an unwrapped phase tensor.

    Args:
        phi: [T, D] or [T, S, D] unwrapped phase (float32, CPU).
        output_dir: Directory for plots + saved data.
        run_ablation: Whether to run ablation simulation.
        run_probe: Whether to run boundary linear probe.
        prefix: Filename prefix for plots.

    Returns:
        metrics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save = lambda name: str(output_dir / (prefix + name))
    is_3d = phi.ndim == 3
    shape_str = f"{phi.shape}" if is_3d else f"({phi.shape[0]}, {phi.shape[1]})"
    print(f"[Phase Analysis] φ shape: {shape_str}, T={phi.shape[0]}")

    # Select a single token for 2D plots (first token, or take 2D input directly)
    phi_2d = phi[:, 0, :] if is_3d else phi

    # ── 1 & 2. Empirical Metrics ──
    print("  [1/5] Computing empirical evidence metrics ...")
    if is_3d:
        metrics = _per_token_aggregate_empirical(phi)
    else:
        metrics = compute_all_empirical_metrics(phi)

    print("  [2/5] Generating plots ...")
    plot_phi_trajectories(phi_2d, _save("01_phi_trajectories.png"))
    plot_delta_phi_histogram(phi_2d, _save("02_delta_phi_histogram.png"))
    plot_correlation_scatter(phi_2d, _save("03_phi_time_correlation.png"))
    plot_velocity_across_steps(phi_2d, _save("04_phase_velocity.png"))
    plot_correlation_histogram(metrics, _save("05_correlation_histogram.png"))

    # ── 3. Ablation ──
    if run_ablation:
        print("  [3/5] Running ablation simulation ...")
        ablation = _per_token_ablation(phi)

        for k, v in ablation.items():
            if isinstance(v, (np.ndarray, list)):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                metrics[k] = float(v)

        plot_ablation_comparison(ablation, _save("06_ablation_comparison.png"))
        plot_coherence_decay_curves(ablation, _save("07_coherence_decay.png"))
        plot_velocity_distribution(ablation, phi_2d, _save("08_velocity_ablation.png"))

        # Self-similarity matrices from first token
        S_orig = compute_self_similarity(phi_2d)
        S_shuf = compute_self_similarity(make_shuffled_phase(phi_2d))
        S_rand = compute_self_similarity(make_random_phase(phi_2d))
        plot_self_similarity(S_orig, _save("09_self_sim_original.png"))
        plot_self_similarity(S_shuf, _save("10_self_sim_shuffled.png"))
        plot_self_similarity(S_rand, _save("11_self_sim_random.png"))

        np.save(_save("orig_autocorr_decay.npy"), ablation.get("orig_autocorr_decay",
                 compute_temporal_autocorrelation_decay(phi_2d)))
        np.save(_save("shuffled_autocorr_decay.npy"), ablation.get("shuffled_autocorr_decay",
                 compute_temporal_autocorrelation_decay(make_shuffled_phase(phi_2d))))
        np.save(_save("random_autocorr_decay.npy"), ablation.get("random_autocorr_decay",
                 compute_temporal_autocorrelation_decay(make_random_phase(phi_2d))))
        np.save(_save("self_sim_original.npy"), S_orig)
        np.save(_save("self_sim_shuffled.npy"), S_shuf)
        np.save(_save("self_sim_random.npy"), S_rand)

    # ── 4. Behavioral Evidence ──
    print("  [4/5] Computing behavioral evidence ...")
    behavioral = _per_token_behavioral(phi)

    if "self_similarity_matrix" in behavioral:
        S = behavioral.pop("self_similarity_matrix")
        plot_self_similarity(S, _save("12_behavioral_self_similarity.png"),
                             title="Phase Self-Similarity (Behavioral)")
        np.save(_save("behavioral_self_sim.npy"), S)

    if "coherence_decay_curve" in behavioral:
        decay = behavioral.pop("coherence_decay_curve")
        np.save(_save("behavioral_coherence_decay.npy"), decay)

    for k, v in behavioral.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            metrics[k] = float(v)

    # ── 5. Predictive Evidence ──
    if run_probe:
        print("  [5/5] Training boundary linear probe ...")
        probe = _per_token_probe(phi)
        for k, v in probe.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                metrics[k] = float(v)

        boundaries_0 = compute_pseudo_boundaries_from_phase(phi_2d)
        plot_pseudo_boundaries(phi_2d, boundaries_0, _save("13_pseudo_boundaries.png"))
        np.save(_save("pseudo_boundaries.npy"), boundaries_0)

    return metrics


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_summary(metrics: dict) -> None:
    """Print a 5-bullet summary of key findings."""
    def _fmt(key, fmt=".4f"):
        v = metrics.get(key)
        return f"{v:{fmt}}" if v is not None else "N/A"

    def _tag(cond, t_val, f_val):
        v = metrics.get(cond, None)
        if v is None:
            return ""
        return t_val if v else f_val

    print("\n" + "=" * 64)
    print("  PHASE TEMPORAL COORDINATE ANALYSIS — SUMMARY")
    print("=" * 64)

    mr = metrics.get("monotonicity_ratio")
    mr_tag = ("monotonic ↑" if mr and mr > 0.85 else
              "non-monotonic" if mr and mr < 0.6 else "mixed")
    cr = metrics.get("phi_time_corr_mean")
    cr_tag = f"r={'positive' if cr and cr > 0 else 'negative'}" if cr and abs(cr) > 0.5 else "weak"
    sv = metrics.get("phi_abs_acceleration_mean")
    sv_tag = "smooth" if sv and sv < 0.01 else "jagged"
    ad = metrics.get("ablation_drop_shuffled")
    ad_tag = "structure informative (drop)" if ad and ad > 0.3 else ("moderate drop" if ad and ad > 0.1 else "structure fragile (low drop)")
    ba = metrics.get("boundary_auc")
    ba_tag = (f"AUC={ba:.2f} — φ predicts boundaries" if ba and ba > 0.65 else
              f"AUC={ba:.2f} — no boundary signal" if ba and ba < 0.55 else
              f"AUC={ba:.2f}" if ba else "N/A")

    bullets = [
        f"Monotonicity: Δφ>0 ratio = {_fmt('monotonicity_ratio')} ({mr_tag})",
        f"Time correlation: r = {_fmt('phi_time_corr_mean')} ± {_fmt('phi_time_corr_std')} ({cr_tag})",
        f"Smoothness: var(v)={_fmt('phi_velocity_variance')}, mean|Δ²φ|={_fmt('phi_abs_acceleration_mean')} ({sv_tag})",
        f"Ablation drop: {_fmt('ablation_drop_shuffled', '.1%')} (shuffled), {_fmt('ablation_drop_random', '.1%')} (random) — {ad_tag}",
        f"Probe: {ba_tag}",
    ]

    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b}")

    print("=" * 64)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase Temporal Coordinate Analysis — "
                    "test whether φ behaves as an emergent temporal coordinate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources (mutually exclusive groups)
    parser.add_argument("--phase-path", type=str, default=None,
                        help="Pre-saved phase_history.pt [T, D] (wrapped or unwrapped)")
    parser.add_argument("--zr-path", type=str, default=None,
                        help="z_r tensor [T, D] (alternative to --phase-path)")
    parser.add_argument("--zi-path", type=str, default=None,
                        help="z_i tensor [T, D] (alternative to --phase-path)")

    # Inference config
    parser.add_argument("--model-root", type=str, default=None,
                        help="DiT model root (for live inference collection)")
    parser.add_argument("--pm-dir", type=str, default=None,
                        help="PhaseMemory checkpoint directory")
    parser.add_argument("--config", type=str, default="acestep-v15-sft",
                        help="ACE-Step config name")
    parser.add_argument("--infer-steps", type=int, default=25,
                        help="Denoising steps for live inference")
    parser.add_argument("--timestep-sampling", type=str, default="linear",
                        choices=["linear", "quadratic", "log"])
    parser.add_argument("--device", type=str, default="cuda")

    # Analysis config
    parser.add_argument("--output-dir", type=str,
                        default="/root/ACE-Step-1.5/output/phase_temporal_coordinate_analysis")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip ablation simulation")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip boundary linear probe")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load phase data ──
    phi_raw = None

    if args.phase_path is not None:
        print(f"[Load] Phase from: {args.phase_path}")
        phi_raw = torch.load(args.phase_path, map_location="cpu").float()
        if phi_raw.ndim == 1:
            phi_raw = phi_raw.unsqueeze(-1)

    elif args.zr_path is not None and args.zi_path is not None:
        print(f"[Load] z_r from: {args.zr_path}")
        print(f"[Load] z_i from: {args.zi_path}")
        z_r = torch.load(args.zr_path, map_location="cpu").float()
        z_i = torch.load(args.zi_path, map_location="cpu").float()
        if z_r.ndim == 1:
            z_r = z_r.unsqueeze(-1)
            z_i = z_i.unsqueeze(-1)
        phi_raw = compute_phi(z_r, z_i, dim=0)

    elif args.model_root is not None and args.pm_dir is not None:
        print(f"[Model] Loading model from {args.model_root} ...")
        sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
        os.environ["ACESTEP_OFFLINE"] = "1"
        os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

        from acestep.handler import AceStepHandler
        from acestep.training.phase_memory_checkpoint import load_phase_memory_weights

        handler = AceStepHandler()
        status, ok = handler.initialize_service(
            project_root=args.model_root,
            config_path=args.config,
            device=args.device,
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"Service init failed: {status}")

        load_phase_memory_weights(handler.model, args.pm_dir)

        pm_module = None
        for name, mod in handler.model.named_modules():
            if getattr(mod, "use_phase_memory", False) and hasattr(mod, "phase_memory"):
                pm_module = mod.phase_memory
                break
        if pm_module is None:
            raise RuntimeError("No PhaseMemory module found!")

        phi_raw = collect_phase_history_from_model(
            handler.model, pm_module,
            num_steps=args.infer_steps,
            device=args.device,
            seed=args.seed,
        )

    else:
        parser.print_help()
        print("\nError: provide --phase-path, --zr-path+--zi-path, or --model-root+--pm-dir")
        sys.exit(1)

    # Unwrap
    phi = unwrap_phase(phi_raw, dim=0)

    # If 3D [T, seq_len, D], analyze per-token and aggregate
    if phi.ndim == 3:
        print(f"  Multi-token φ detected: {phi.shape} → analyzing per-token, reporting mean±std over tokens")
        print(f"    (T={phi.shape[0]}, S={phi.shape[1]}, D={phi.shape[2]})")
        # Don't average — handle 3D throughout

    # Save both wrapped and unwrapped
    torch.save(phi_raw, output_dir / "phi_wrapped.pt")
    torch.save(phi, output_dir / "phi_unwrapped.pt")
    print(f"  φ shape: {phi.shape}  |  Wrapped range: [{phi_raw.min():.3f}, {phi_raw.max():.3f}]"
          f"  |  Unwrapped range: [{phi.min():.3f}, {phi.max():.3f}]")

    # ── Run analysis ──
    print(f"\n[Analysis] Full pipeline → {output_dir}")
    metrics = analyze_phase_as_temporal_coordinate(
        phi=phi,
        output_dir=str(output_dir),
        run_ablation=not args.no_ablation,
        run_probe=not args.no_probe,
    )

    # Save metrics
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            serializable[k] = float(v) if not isinstance(v, (int, np.integer)) else int(v)
        elif isinstance(v, str):
            serializable[k] = v

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    torch.save(metrics, output_dir / "metrics_full.pt")

    # Print summary
    print_summary(metrics)

    print(f"\n[Save] All outputs → {output_dir}")
    print(f"  Data:     phi_wrapped.pt, phi_unwrapped.pt, metrics.json")
    print(f"  Plots:    13 *.png files")
    print(f"  Artifacts: *.npy files")


if __name__ == "__main__":
    main()
