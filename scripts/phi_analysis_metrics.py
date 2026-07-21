"""
phi_analysis_metrics.py
=======================
Standardized metrics for analyzing φ as a temporal coordinate.

Four categories:
  A. Structural metrics — monotonicity, time correlation, smoothness
  B. Temporal structure — short/long-range coherence, self-similarity
  C. Robustness — shuffle ablation, noise baseline
  D. Dimensionality — PCA effective rank

All functions accept φ: [T, D] or [T, S, D] (per-token aggregated).

Functions with _per_token suffix handle 3D by iterating over tokens
and reporting mean±std.
"""

import math
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# numpy.trapz → trapezoid compatibility (removed in NumPy 2.x)
if hasattr(np, "trapezoid"):
    np_trapz = np.trapezoid
else:
    np_trapz = np.trapz


# ==============================================================================
# A. STRUCTURAL METRICS
# ==============================================================================

def monotonicity_ratio(phi: torch.Tensor) -> float:
    """Fraction of time steps where phase increases (Δφ > 0)."""
    delta = phi[1:] - phi[:-1]
    return float((delta > 0).float().mean().item())


def phi_time_correlation(phi: torch.Tensor) -> dict:
    """Pearson r between φ and time index for each dimension.

    Returns: {"corr_mean", "corr_std", "corr_per_dim"}
    """
    T, D = phi.shape
    t = np.arange(T)
    corrs = []
    for d in range(D):
        r, _ = pearsonr(t, phi[:, d].numpy())
        corrs.append(r)
    corrs = np.array(corrs)
    valid = ~np.isnan(corrs)
    return {
        "corr_mean": float(np.mean(corrs[valid])) if valid.any() else float("nan"),
        "corr_std": float(np.std(corrs[valid])) if valid.any() else float("nan"),
        "corr_per_dim": corrs.tolist(),
    }


def phase_smoothness(phi: torch.Tensor) -> dict:
    """Velocity variance and acceleration magnitude.

    Returns: {"velocity_var", "velocity_abs_mean", "accel_abs_mean"}
    """
    if phi.ndim == 1:
        phi = phi.unsqueeze(-1)
    velocity = phi[1:] - phi[:-1]
    accel = velocity[1:] - velocity[:-1]
    return {
        "velocity_var": float(velocity.var(dim=0).mean().item()),
        "velocity_abs_mean": float(velocity.abs().mean().item()),
        "accel_abs_mean": float(accel.abs().mean().item()),
    }


def structural_metrics(phi: torch.Tensor) -> dict:
    """All structural metrics at once."""
    m = {
        "monotonicity_ratio": monotonicity_ratio(phi),
    }
    m.update(phase_smoothness(phi))

    # Correlation (only if T >= 3)
    if phi.shape[0] >= 3:
        corr = phi_time_correlation(phi)
        m["time_corr_mean"] = corr["corr_mean"]
        m["time_corr_std"] = corr["corr_std"]
    else:
        m["time_corr_mean"] = float("nan")
        m["time_corr_std"] = float("nan")

    return m


# ==============================================================================
# B. TEMPORAL STRUCTURE METRICS
# ==============================================================================

def coherence_curve(phi: torch.Tensor, max_lag: int = None) -> np.ndarray:
    """Mean cos(φ[t] - φ[t+d]) for each lag d. Returns [T-1] array."""
    T = phi.shape[0]
    if max_lag is not None:
        max_lag = min(max_lag, T - 1)
    else:
        max_lag = T - 1
    curve = np.zeros(max_lag)
    for d in range(1, max_lag + 1):
        curve[d - 1] = torch.cos(phi[:-d] - phi[d:]).mean().item()
    return curve


def short_range_coherence(phi: torch.Tensor, lags: int = 5) -> float:
    """Mean coherence over first `lags` steps."""
    curve = coherence_curve(phi, max_lag=lags)
    return float(curve.mean()) if len(curve) > 0 else float("nan")


def long_range_coherence(phi: torch.Tensor, start_lag: int = 10, end_lag: int = 25) -> float:
    """Mean coherence over lag range [start_lag, end_lag]."""
    T = phi.shape[0]
    end_lag = min(end_lag, T - 1)
    if start_lag >= T:
        return float("nan")
    curve = coherence_curve(phi, max_lag=end_lag)
    segment = curve[start_lag - 1:] if start_lag > 0 else curve
    return float(segment.mean()) if len(segment) > 0 else float("nan")


def self_similarity_entropy(phi: torch.Tensor) -> float:
    """Entropy of the self-similarity matrix.

    Higher entropy = less temporal structure (more uniform).
    Lower entropy = more temporal structure (blocks/diagonal).
    """
    T, D = phi.shape
    delta = phi[:, None, :] - phi[None, :, :]
    S = torch.cos(delta).mean(dim=-1).numpy()  # [T, T]
    # Normalize to [0, 1] and compute entropy
    S_norm = (S - S.min()) / (S.max() - S.min() + 1e-10)
    # Flatten, avoid log(0)
    p = S_norm.ravel()
    p = p / (p.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = float(-(p * np.log(p)).sum())
    # Normalize by log(T^2) for [0, 1] range
    norm_entropy = entropy / math.log(T * T)
    return norm_entropy


def temporal_structure_metrics(phi: torch.Tensor) -> dict:
    """All temporal structure metrics at once."""
    T = phi.shape[0]
    return {
        "short_range_coherence_l1-5": short_range_coherence(phi, lags=min(5, T - 1)),
        "long_range_coherence_l10-25": long_range_coherence(phi, start_lag=10, end_lag=min(25, T - 1)),
        "self_sim_entropy": self_similarity_entropy(phi),
        "full_range_coherence_auc": float(np_trapz(
            coherence_curve(phi), dx=1.0)) if T > 2 else float("nan"),
    }


# ==============================================================================
# C. ROBUSTNESS METRICS
# ==============================================================================

def shuffle_phase(phi: torch.Tensor) -> torch.Tensor:
    """Shuffle φ along time dim independently per D dim."""
    T, D = phi.shape
    shuffled = phi.clone()
    for d in range(D):
        shuffled[:, d] = shuffled[torch.randperm(T), d]
    return shuffled


def noise_baseline(phi: torch.Tensor) -> torch.Tensor:
    """Gaussian noise matched to per-dim mean/std of φ."""
    means = phi.mean(dim=0, keepdim=True)
    stds = phi.std(dim=0, keepdim=True)
    return torch.randn_like(phi) * stds + means


def random_walk_baseline(T: int, D: int, scale: float = 0.3) -> torch.Tensor:
    """Random walk as null hypothesis for φ dynamics."""
    increments = torch.randn(T - 1, D) * scale
    return torch.cat([torch.zeros(1, D), increments.cumsum(dim=0)], dim=0)


def ablation_metrics(phi: torch.Tensor) -> dict:
    """Compare temporal structure across original / shuffled / noise / walk.

    Uses the current global RNG state for shuffling (ensures per-token
    variability in per-token aggregation). For reproducibility, call
    torch.manual_seed() before this function.

    Returns:
        dict with coherence for each condition and ablation_drop ratios.
    """
    T, D = phi.shape

    orig_short = short_range_coherence(phi, lags=min(5, T - 1))
    orig_curve = coherence_curve(phi)

    # Shuffled
    phi_shuf = shuffle_phase(phi)
    shuf_short = short_range_coherence(phi_shuf, lags=min(5, T - 1))
    shuf_curve = coherence_curve(phi_shuf)
    shuf_drop = max(0.0, (orig_short - shuf_short) / (abs(orig_short) + 1e-8))

    # Noise
    phi_noise = noise_baseline(phi)
    noise_short = short_range_coherence(phi_noise, lags=min(5, T - 1))
    noise_curve = coherence_curve(phi_noise)
    noise_drop = max(0.0, (orig_short - noise_short) / (abs(orig_short) + 1e-8))

    # Random walk
    phi_walk = random_walk_baseline(T, D)
    walk_short = short_range_coherence(phi_walk, lags=min(5, T - 1))
    walk_curve = coherence_curve(phi_walk)

    return {
        "orig_short_coherence": orig_short,
        "shuffled_short_coherence": shuf_short,
        "noise_short_coherence": noise_short,
        "walk_short_coherence": walk_short,
        "ablation_drop_shuffle": float(shuf_drop),
        "ablation_drop_noise": float(noise_drop),
        "orig_coherence_curve": orig_curve,
        "shuffled_coherence_curve": shuf_curve,
        "noise_coherence_curve": noise_curve,
        "walk_coherence_curve": walk_curve,
    }


# ==============================================================================
# D. DIMENSIONALITY
# ==============================================================================

def pca_effective_rank(phi: torch.Tensor, var_threshold: float = 0.95) -> dict:
    """Effective dimensionality of φ via PCA.

    Args:
        phi: [T, D] phase tensor.
        var_threshold: Fraction of variance to retain.

    Returns:
        dict with effective rank, top-3 variance ratio, etc.
    """
    X = phi.numpy()
    X_centered = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        var_explained = (s ** 2) / (s ** 2).sum()
        cumvar = np.cumsum(var_explained)
        effective_rank = int((cumvar < var_threshold).sum()) + 1
        top3_var = float(var_explained[:3].sum()) if len(var_explained) >= 3 else 1.0
        return {
            "pca_effective_rank": effective_rank,
            "pca_top3_var_explained": top3_var,
            "pca_first_var_ratio": float(var_explained[0]),
        }
    except np.linalg.LinAlgError:
        return {
            "pca_effective_rank": int("nan"),
            "pca_top3_var_explained": float("nan"),
            "pca_first_var_ratio": float("nan"),
        }


# ==============================================================================
# PER-TOKEN AGGREGATION (handles [T, S, D] input)
# ==============================================================================

def _per_token(
    metric_fn,
    phi: torch.Tensor,
    max_tokens: int = 256,
    **kwargs,
) -> dict:
    """Run metric_fn on each token of [T, S, D] and aggregate."""
    if phi.ndim == 2:
        m = metric_fn(phi, **kwargs) if kwargs else metric_fn(phi)
        if isinstance(m, dict):
            return m
        return {"value": float(m), "std": 0.0}

    T, S, D = phi.shape
    S_sub = min(S, max_tokens)
    step = max(1, S // S_sub)
    selected = list(range(0, S, step))[:S_sub]
    return _per_token_impl(metric_fn, phi, selected, **kwargs)


def _per_token_impl(metric_fn, phi, selected, **kwargs):
    """Inner aggregation loop."""
    scalar_buckets = {}

    for idx in selected:
        m = metric_fn(phi[:, idx, :], **kwargs) if kwargs else metric_fn(phi[:, idx, :])
        if isinstance(m, dict):
            for k, v in m.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    if k not in scalar_buckets:
                        scalar_buckets[k] = []
                    scalar_buckets[k].append(float(v))
                elif isinstance(v, np.ndarray):
                    pass  # skip array values in aggregation
        else:
            if "_val" not in scalar_buckets:
                scalar_buckets["_val"] = []
            scalar_buckets["_val"].append(float(m))

    result = {}
    for k, vals in scalar_buckets.items():
        v = np.array(vals)
        result[k] = float(v.mean())
        result[k + "_std"] = float(v.std())
    return result


def structural_metrics_per_token(phi: torch.Tensor) -> dict:
    return _per_token(structural_metrics, phi)


def temporal_structure_metrics_per_token(phi: torch.Tensor) -> dict:
    return _per_token(temporal_structure_metrics, phi)


def ablation_metrics_per_token(phi: torch.Tensor) -> dict:
    return _per_token(ablation_metrics, phi)


def pca_metrics_per_token(phi: torch.Tensor) -> dict:
    return _per_token(pca_effective_rank, phi)
