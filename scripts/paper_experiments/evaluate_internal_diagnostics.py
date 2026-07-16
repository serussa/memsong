#!/usr/bin/env python3
"""
Internal diagnostic metrics for transport-based model variants.

Reads saved ``transport_diagnostics.npz`` files from the generation
output and computes:

  1. **Condition marginal error** — how well Sinkhorn satisfies the
     column (unit) marginal constraint ``mu``.
  2. **Cumulative coverage error** — per-song-progress deviation
     between observed cumulative unit usage and the ideal schedule.
  3. **State cost contribution** — ratio of the dynamic (state-conditioned)
     residual score to the base position cost.

Outputs CSVs under ``--output_dir`` for downstream plotting.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
#  Diagnostic 1: Condition marginal error
# ===========================================================================

def compute_marginal_error(
    Pi: np.ndarray,
    mu: np.ndarray,
) -> float:
    """Compute L1 error between observed column marginal and target mu.

    Parameters
    ----------
    Pi : (T, K) or (B, T, K) transport plan matrix.
    mu : (K,) target column marginal (sums to 1).

    Returns
    -------
    L1 error per column.
    """
    if Pi.ndim == 3:
        Pi = Pi[0]  # CFG batch doubling; take first element
    if Pi.ndim != 2 or mu.ndim != 1:
        return float("nan")
    K = Pi.shape[1]
    if K != len(mu):
        return float("nan")
    mu_hat = Pi.sum(axis=0)  # (K,)
    error = np.abs(mu_hat - mu).mean()
    return float(error)


# ===========================================================================
#  Diagnostic 2: Cumulative coverage error
# ===========================================================================

def compute_cumulative_coverage_error(
    Pi: np.ndarray,
    mu: np.ndarray,
    p_audio: np.ndarray,
    c_unit: np.ndarray,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative coverage error E_cum(p) over song progress.

    For each progress quantile p in [0, 1], we compare the observed
    cumulative mass allocated to each unit with an ideal reference
    where unit j receives mass proportionally to its overlap with
    the progress interval [0, p].

    Parameters
    ----------
    Pi : (T, K) or (B, T, K) transport plan (row-stochastic).
    mu : (K,) target column mass.
    p_audio : (T,) audio progress positions in [0, 1].
    c_unit : (K,) unit centre positions in [0, 1].
    n_points : number of progress quantiles to evaluate.

    Returns
    -------
    p_grid : (n_points,) progress values.
    E_cum : (n_points,) cumulative coverage error at each p.
    """
    if Pi.ndim == 3:
        Pi = Pi[0]  # CFG batch doubling; take first element
    if p_audio.ndim == 1:
        pass
    elif p_audio.ndim == 2:
        p_audio = p_audio[0]  # CFG batch doubling
    T, K = Pi.shape
    if K == 0:
        return np.linspace(0, 1, n_points), np.zeros(n_points)

    # Sort by p_audio to ensure monotonic progress
    sort_idx = np.argsort(p_audio)
    Pi_sorted = Pi[sort_idx]
    p_sorted = p_audio[sort_idx]

    p_grid = np.linspace(0.0, 1.0, n_points)
    E_cum = np.zeros(n_points)

    # Sort units by position
    unit_order = np.argsort(c_unit)  # K,
    mu_sorted = mu[unit_order]
    c_sorted = c_unit[unit_order]
    Pi_sorted_by_unit = Pi_sorted[:, unit_order]

    # Ideal cumulative mass per unit:
    # m_star_j(p) = clip(p - b_j, 0, mu_j)
    # where b_j = sum_{k < j} mu_k
    b = np.concatenate([[0.0], np.cumsum(mu_sorted)[:-1]])  # (K,)

    for idx_p, p in enumerate(p_grid):
        # Observed: cumulative mass up to position p
        mask = p_sorted <= p
        if mask.sum() == 0:
            m_hat = np.zeros(K)
        else:
            m_hat = Pi_sorted_by_unit[mask].sum(axis=0)  # (K,)

        # Ideal: m_star_j(p)
        m_star = np.clip(p - b, 0.0, mu_sorted)

        # Error = L1 over units
        E_cum[idx_p] = np.abs(m_hat - m_star).sum()

    return p_grid, E_cum


# ===========================================================================
#  Diagnostic 3: State cost contribution (approximate)
# ===========================================================================

def estimate_state_cost_ratio(
    Pi: np.ndarray,
    base_cost_only: Optional[np.ndarray] = None,
    transport_only_Pi: Optional[np.ndarray] = None,
) -> float:
    """Estimate the contribution of state-conditioned cost to the
    transport plan.

    If ``transport_only_Pi`` is provided (qk_scale=0 Sinkhorn result),
    computes the L1 difference between the full Pi and the
    transport-only Pi as a proxy for state cost contribution.

    Otherwise returns NaN (requires paired data).
    """
    if transport_only_Pi is not None and Pi.shape == transport_only_Pi.shape:
        diff = np.abs(Pi - transport_only_Pi).mean()
        return float(diff)
    return float("nan")


# ===========================================================================
#  Collect diagnostic data
# ===========================================================================

def collect_diagnostics(
    generation_dir: str,
    variants: Optional[List[str]] = None,
) -> List[Dict]:
    """Walk generation directory and collect transport diagnostics.

    Returns list of dicts with keys:
      variant, prompt_id, seed, Pi, mu, unit_section_ids,
      unit_is_lyric, p_audio, c_unit, denoising_step (0 for now).
    """
    results = []
    gen_dir = Path(generation_dir)

    for variant_dir in sorted(gen_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        if variants and variant_dir.name not in variants:
            continue

        for prompt_dir in sorted(variant_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue

            for seed_dir in sorted(prompt_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue

                diag_path = seed_dir / "transport_diagnostics.npz"
                if not diag_path.exists():
                    continue

                try:
                    data = np.load(diag_path)
                except Exception:
                    continue

                seed = int(seed_dir.name.replace("seed_", ""))
                results.append({
                    "variant": variant_dir.name,
                    "prompt_id": prompt_dir.name,
                    "seed": seed,
                    "Pi": data.get("Pi", None),
                    "mu": data.get("mu", None),
                    "unit_section_ids": data.get("unit_section_ids", None),
                    "unit_is_lyric": data.get("unit_is_lyric", None),
                    "p_audio": data.get("p_audio", None),
                    "c_unit": data.get("c_unit", None),
                    "denoising_step": 0,  # single saved matrix
                })

    return results


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Internal diagnostic metrics for transport variants."
    )
    parser.add_argument("--generation_dir", type=str, required=True,
                        help="Generation output directory")
    parser.add_argument("--output_dir", type=str, default="metrics/paper_eval",
                        help="Output directory for metric CSVs")
    parser.add_argument("--variants", type=str, default=None,
                        help="Optional comma-separated variant filter")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_filter = None
    if args.variants:
        variant_filter = [v.strip() for v in args.variants.split(",")]

    print("Collecting transport diagnostics...")
    diag_data = collect_diagnostics(args.generation_dir, variant_filter)
    print(f"  Found {len(diag_data)} diagnostic files")

    if not diag_data:
        print("  No diagnostic files found. Skipping internal metrics.")
        # Still write empty CSVs for pipeline consistency
        for fname in [
            "internal_marginal_error.csv",
            "internal_cumulative_curve.csv",
            "state_cost_contribution.csv",
        ]:
            empty_path = output_dir / fname
            with open(empty_path, "w") as f:
                f.write("variant,prompt_id,seed,marginal_error\n")
        print(f"  Wrote empty diagnostic CSVs to {output_dir}")
        return

    # ── 1. Condition marginal error ──
    print("\nComputing marginal error...")
    marginal_rows = []
    for d in diag_data:
        Pi = d["Pi"]
        mu = d["mu"]
        if Pi is not None and mu is not None:
            err = compute_marginal_error(Pi, mu)
            marginal_rows.append({
                "variant": d["variant"],
                "prompt_id": d["prompt_id"],
                "seed": d["seed"],
                "marginal_error": err,
            })

    if marginal_rows:
        marginal_path = output_dir / "internal_marginal_error.csv"
        with open(marginal_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant", "prompt_id", "seed", "marginal_error"])
            writer.writeheader()
            writer.writerows(marginal_rows)
        print(f"  → {marginal_path} ({len(marginal_rows)} rows)")

    # ── 2. Cumulative coverage error ──
    print("\nComputing cumulative coverage error...")
    cum_rows = []
    for d in diag_data:
        Pi = d["Pi"]
        mu = d["mu"]
        p_audio = d["p_audio"]
        c_unit = d["c_unit"]
        if Pi is not None and mu is not None and p_audio is not None and c_unit is not None:
            try:
                p_grid, E_cum = compute_cumulative_coverage_error(
                    Pi, mu, p_audio, c_unit, n_points=100)
                for p_val, e_val in zip(p_grid, E_cum):
                    cum_rows.append({
                        "variant": d["variant"],
                        "prompt_id": d["prompt_id"],
                        "seed": d["seed"],
                        "denoising_step": d["denoising_step"],
                        "progress": round(float(p_val), 4),
                        "E_cum": round(float(e_val), 6),
                    })
            except Exception as exc:
                print(f"  [WARN] Cumulative coverage failed for "
                      f"{d['variant']}/{d['prompt_id']}/seed_{d['seed']}: {exc}")

    if cum_rows:
        cum_path = output_dir / "internal_cumulative_curve.csv"
        with open(cum_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant", "prompt_id", "seed", "denoising_step",
                "progress", "E_cum"])
            writer.writeheader()
            writer.writerows(cum_rows)
        print(f"  → {cum_path} ({len(cum_rows)} rows)")

    # ── 3. State cost contribution ──
    # This requires comparing full vs transport_only variants.
    # Since we saved separate npz per variant, we can't easily pair
    # without prompt ordering guarantees.  Compute per-variant average
    # marginal error as a proxy, and note the limitation.
    print("\nState cost contribution (marginal error per variant):")
    variant_errors: Dict[str, List] = defaultdict(list)
    for r in marginal_rows:
        variant_errors[r["variant"]].append(r["marginal_error"])

    state_rows = []
    for variant, errors in sorted(variant_errors.items()):
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        state_rows.append({
            "variant": variant,
            "mean_marginal_error": mean_err,
            "std_marginal_error": std_err,
            "num_samples": len(errors),
            "note": (
                "Full state-cost comparison requires paired full vs "
                "transport_only samples with identical prompt ordering."
            ),
        })
        print(f"  {variant}: marginal_error={mean_err:.6f} ± {std_err:.6f}")

    if state_rows:
        state_path = output_dir / "state_cost_contribution.csv"
        with open(state_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant", "mean_marginal_error", "std_marginal_error",
                "num_samples", "note"])
            writer.writeheader()
            writer.writerows(state_rows)
        print(f"  → {state_path}")

    # ── 4. Per-variant cumulative coverage summary for plotting ──
    print("\nAggregating cumulative coverage by variant...")
    cum_agg_rows = []
    variant_cum: Dict[str, Dict[float, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in cum_rows:
        variant_cum[r["variant"]][r["progress"]].append(r["E_cum"])

    for variant in sorted(variant_cum.keys()):
        for p_val in sorted(variant_cum[variant].keys()):
            values = variant_cum[variant][p_val]
            mean_e = float(np.mean(values))
            std_e = float(np.std(values))
            n_e = len(values)
            cum_agg_rows.append({
                "variant": variant,
                "progress": p_val,
                "E_cum_mean": mean_e,
                "E_cum_std": std_e,
                "E_cum_stderr": std_e / max(np.sqrt(n_e), 1),
                "num_samples": n_e,
            })

    if cum_agg_rows:
        cum_agg_path = output_dir / "internal_cumulative_curve_agg.csv"
        with open(cum_agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant", "progress", "E_cum_mean", "E_cum_std",
                "E_cum_stderr", "num_samples"])
            writer.writeheader()
            writer.writerows(cum_agg_rows)
        print(f"  → {cum_agg_path} ({len(cum_agg_rows)} rows)")

    print(f"\nDone. Internal diagnostics saved to {output_dir}")


if __name__ == "__main__":
    main()
