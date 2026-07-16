"""Shared analysis and visualization utilities for RoPE experiments."""
import torch
import math
import numpy as np
from pathlib import Path
from typing import Optional


def rope_frequency_analysis(head_dim: int = 128, rope_theta: float = 1_000_000.0):
    """Compute RoPE frequency characteristics for given config.

    Returns dict with frequency information.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    n_pairs = head_dim // 2
    quarter_cycle = (math.pi / 2) / inv_freq

    return {
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "n_pairs": n_pairs,
        "inv_freq": inv_freq,
        "quarter_cycle_positions": quarter_cycle,
        "freq_range": (inv_freq[0].item(), inv_freq[-1].item()),
        "freq_ratio": (inv_freq[0] / inv_freq[-1]).item(),
    }


def position_encoding_status(
    positions: torch.Tensor,
    head_dim: int = 128,
    rope_theta: float = 1_000_000.0,
):
    """Classify frequency pairs by encoding status at given positions.

    Returns:
        dict with counts per category for each position
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [n_pairs]
    results = {}
    for p in positions:
        cycles = (inv_freq * p) / (2 * math.pi)
        results[int(p)] = {
            "first_cycle": (cycles < 1).sum().item(),
            "second_cycle": ((cycles >= 1) & (cycles < 2)).sum().item(),
            "three_to_five": ((cycles >= 2) & (cycles < 5)).sum().item(),
            "beyond_5": (cycles >= 5).sum().item(),
            "beyond_10": (cycles >= 10).sum().item(),
            "beyond_100": (cycles >= 100).sum().item(),
            "max_cycles": cycles[0].item(),
        }
    return results


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8):
    """Compute per-position attention entropy.

    Args:
        attn_weights: [B, H, T, S] attention weights
    Returns:
        entropy: [B, H, T] entropy per query position
    """
    p = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    return entropy


def compute_receptive_field(attn_weights: torch.Tensor):
    """Compute expected distance from query to key per position.

    Args:
        attn_weights: [B, H, T, S] attention weights
    Returns:
        field: [B, H, T] expected absolute distance
    """
    B, H, T, S = attn_weights.shape
    eps = 1e-8
    p = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + eps)
    key_pos = torch.arange(S, device=attn_weights.device).float().view(1, 1, 1, S)
    query_pos = torch.arange(T, device=attn_weights.device).float().view(1, 1, T, 1)
    expected_key = (p * key_pos).sum(dim=-1)  # [B, H, T]
    field = (expected_key - query_pos.squeeze(-1)).abs()
    return field


def compute_self_similarity(features: np.ndarray):
    """Compute self-similarity matrix from feature sequence.

    Args:
        features: [T, D] feature vectors
    Returns:
        sim: [T, T] cosine similarity matrix
    """
    norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    return sim


def detect_repetition_blocks(sim_matrix: np.ndarray, threshold: float = 0.8, min_block: int = 10):
    """Detect off-diagonal repetition blocks in a self-similarity matrix.

    Args:
        sim_matrix: [T, T] similarity matrix
        threshold: similarity threshold for block detection
        min_block: minimum block size in frames
    Returns:
        blocks: list of (t_start, t_end, offset) for detected repetitions
    """
    T = sim_matrix.shape[0]
    blocks = []
    # Check diagonal bands at various offsets
    for offset in range(min_block, T // 2):
        diagonal = np.diag(sim_matrix, offset)
        # Find contiguous regions above threshold
        in_block = diagonal > threshold
        if not in_block.any():
            continue
        # Find block boundaries
        diffs = np.diff(in_block.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if in_block[0]:
            starts = np.concatenate([[0], starts])
        if in_block[-1]:
            ends = np.concatenate([ends, [len(diagonal)]])
        for s, e in zip(starts, ends):
            if e - s >= min_block:
                blocks.append((s, e, offset))
    return blocks


def compute_novelty_curve(features: np.ndarray, window: int = 1):
    """Compute novelty curve: minimum distance to any prior frame.

    Args:
        features: [T, D] feature vectors
    Returns:
        novelty: [T] array, novelty at each time step
    """
    T = features.shape[0]
    novelty = np.zeros(T)
    for t in range(1, T):
        start = max(0, t - window)
        prior = features[:t]
        dist = np.linalg.norm(features[t] - prior, axis=-1)
        novelty[t] = dist.min()
    return novelty


def summarize_attention_stats(stats: dict) -> dict:
    """Aggregate attention statistics across diffusion steps and layers.

    Args:
        stats: dict with keys (layer_idx, diff_step), values are dicts with
               'entropy': [B, H, T], 'receptive_field': [B, H, T]
    Returns:
        summary: dict with aggregated stats
    """
    summary = {}
    for (layer_idx, step), data in stats.items():
        entropy = data['entropy']  # [B, H, T]
        field = data['receptive_field']  # [B, H, T]
        # Average over batch and heads
        avg_entropy = entropy.mean(dim=(0, 1))  # [T]
        avg_field = field.mean(dim=(0, 1))  # [T]
        summary[(layer_idx, step)] = {
            'avg_entropy': avg_entropy,
            'avg_receptive_field': avg_field,
            'std_entropy': entropy.mean(dim=1).std(dim=0),  # std across heads -> [T]
            'global_entropy': entropy.mean().item(),
            'global_field': field.mean().item(),
        }
    return summary


def generate_rope_report(output_dir: Path):
    """Generate a markdown report summarizing all experiment results."""
    report_path = output_dir / "rope_experiment_report.md"
    lines = [
        "# RoPE Experiment Report",
        "",
        f"Generated at: {torch.randn(1)}",  # placeholder
        "",
        "## Attention Decay Summary",
        "",
    ]
    report_path.write_text("\n".join(lines))
    return report_path
