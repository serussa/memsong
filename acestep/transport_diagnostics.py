"""
Transport Retrieval Diagnostics — QK residual + KL effect analysis.

Captures intermediate tensors from TransportRetrievalAdapter.forward,
computes plan-level metrics, coupling drift, and saves visualizations.
Does NOT change training logic, model structure, or loss.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ===================================================================
#  Capture hook: attach to adapter to store intermediates
# ===================================================================

class DiagnosticsCapture:
    """Context manager that captures adapter intermediates during forward.

    Usage:
        cap = DiagnosticsCapture(adapter)
        with cap:
            output = adapter(...)
        cap.compute_all()
        cap.report()
        cap.save_all(output_dir, sample_id, method)
    """

    def __init__(self, adapter: torch.nn.Module):
        self.adapter = adapter
        self._orig_forward = adapter.forward
        self._stored: Dict[str, Any] = {}

    def __enter__(self):
        self._orig_forward = self.adapter.forward
        self._stored = {}
        self._install_hook()
        return self

    def __exit__(self, *args):
        self._remove_hook()

    def _install_hook(self):
        """Wrap adapter.forward to capture intermediates."""
        stored = self._stored
        orig_fwd = self.adapter.forward

        def _diagnostic_forward(self_obj, *args, **kwargs):
            # Run original forward first
            result = orig_fwd(*args, **kwargs)

            delta_h, Pi_corr, diag = result
            stored["delta_h"] = delta_h.detach().cpu()
            stored["Pi_corr"] = Pi_corr.detach().cpu()
            stored["h"] = args[0].detach().cpu()  # hidden_states is first positional arg

            # Reconstruct L_prior and corrected L from stored intermediates
            # We do this by re-running the internal logic with torch.no_grad
            with torch.no_grad():
                B, T_a = kwargs.get("p_audio", args[3]).shape
                unit_c_pos = kwargs.get("unit_c_pos", args[5])
                unit_mass = kwargs.get("unit_mass", args[6])
                unit_is_lyric = kwargs.get("unit_is_lyric", None)
                device = delta_h.device
                dtype = delta_h.dtype

                K = unit_c_pos.shape[-1]
                if K == 0:
                    stored["Pi_prior"] = torch.zeros(B, T_a, 0)
                    stored["L_prior"] = torch.zeros(B, T_a, 0)
                    stored["L_corr"] = torch.zeros(B, T_a, 0)
                    stored["score_qk"] = torch.zeros(B, T_a, 0)
                    stored["residual_logits"] = torch.zeros(B, T_a, 0)
                    stored["nu"] = torch.full((B, T_a,), 1.0 / T_a)
                    return result

                dist = kwargs.get("p_audio", args[3])[:, :, None] - unit_c_pos[:, None, :]
                C = (dist / self_obj.transport_sigma) ** 2
                L_prior = -C
                nu = torch.full((B, T_a,), 1.0 / T_a, device=device, dtype=dtype)

                # Compute QK residual
                if hasattr(self_obj, "scoring_mode") and self_obj.scoring_mode == "position_only":
                    pm_state = self_obj.pm_state_norm(kwargs.get("pm_state", args[2]))
                    a_feat = torch.stack([kwargs.get("p_audio", args[3]), kwargs.get("p_audio", args[3]) ** 2, 1.0 - kwargs.get("p_audio", args[3])], dim=-1)
                    audio_coord = self_obj.audio_coord_mlp(a_feat)

                    timestep_emb = kwargs.get("timestep_emb", None)
                    if timestep_emb is None:
                        t_emb = torch.zeros(B, T_a, self_obj.time_dim, device=device, dtype=dtype)
                    elif timestep_emb.dim() == 2:
                        t_emb = timestep_emb.unsqueeze(1).expand(-1, T_a, -1)
                    else:
                        t_emb = timestep_emb

                    q_in = torch.cat([pm_state, audio_coord, t_emb], dim=-1)
                    q = torch.nn.functional.normalize(self_obj.q_mlp(q_in), dim=-1, p=2)

                    u_feat = torch.stack([unit_c_pos, unit_c_pos ** 2, 1.0 - unit_c_pos], dim=-1)
                    unit_coord = self_obj.unit_coord_mlp(u_feat)
                    sec_emb = self_obj.section_embedding(kwargs.get("unit_section_id", torch.zeros(B, K, device=device)).long()) \
                        if kwargs.get("unit_section_id", None) is not None else torch.zeros(B, K, self_obj.section_embedding.embedding_dim, device=device, dtype=dtype)

                    k_in = torch.cat([kwargs.get("unit_text_hidden", args[4]), unit_coord, sec_emb], dim=-1)
                    k = torch.nn.functional.normalize(self_obj.k_mlp(k_in), dim=-1, p=2)
                    score_qk = torch.matmul(q, k.transpose(-1, -2))
                    residual_logits = self_obj.residual_scale * score_qk
                    L_corr = L_prior + residual_logits

                    # Pi_prior
                    log_sinkhorn_fn = self_obj.log_sinkhorn if hasattr(self_obj, 'log_sinkhorn') else None
                    if log_sinkhorn_fn is None:
                        from acestep.phase_memory import log_sinkhorn as log_sinkhorn_fn

                    logPi_prior, _ = _log_sinkhorn_safe(L_prior, nu, unit_mass, self_obj.sinkhorn_iters)
                    logPi_corr, _ = _log_sinkhorn_safe(L_corr, nu, unit_mass, self_obj.sinkhorn_iters)
                    Pi_prior = logPi_prior.exp().cpu()
                    logPi_prior_d = logPi_prior.detach().cpu()
                    logPi_corr_d = logPi_corr.detach().cpu()

                    stored["L_prior"] = L_prior.detach().cpu()
                    stored["L_corr"] = L_corr.detach().cpu()
                    stored["score_qk"] = score_qk.detach().cpu()
                    stored["residual_logits"] = residual_logits.detach().cpu()
                    stored["Pi_prior"] = Pi_prior
                    stored["logPi_prior"] = logPi_prior_d
                    stored["logPi_corr"] = logPi_corr_d
                    stored["nu"] = nu.detach().cpu()
                    stored["unit_mass"] = unit_mass.detach().cpu()
                    stored["unit_c_pos"] = unit_c_pos.detach().cpu()
                    stored["p_audio"] = kwargs.get("p_audio", args[3]).detach().cpu()
                else:
                    # Classic mode: still capture basic info
                    stored["L_prior"] = L_prior.detach().cpu()
                    stored["nu"] = nu.detach().cpu()
                    stored["unit_mass"] = unit_mass.detach().cpu()
                    stored["Pi_prior"] = Pi_corr.detach().cpu()
                    stored["L_corr"] = L_prior.detach().cpu()
                    stored["score_qk"] = torch.zeros_like(L_prior).cpu()
                    stored["residual_logits"] = torch.zeros_like(L_prior).cpu()

            return result

        # Monkey-patch
        self.adapter.forward = _diagnostic_forward.__get__(self.adapter, type(self.adapter))

    def _remove_hook(self):
        self.adapter.forward = self._orig_forward


def _log_sinkhorn_safe(log_P, row_mass, col_mass, iters=5):
    """Minimal log-domain Sinkhorn for diagnostic use. Returns (logPi, info)."""
    B, T, K = log_P.shape
    log_nu = torch.log(row_mass.clamp(min=1e-30))
    log_mu = torch.log(col_mass.clamp(min=1e-30))

    for _ in range(iters):
        log_P = log_P - (torch.logsumexp(log_P, dim=-1, keepdim=True) - log_nu.unsqueeze(-1))
        log_P = log_P - (torch.logsumexp(log_P, dim=-2, keepdim=True) - log_mu.unsqueeze(-2))

    Pi = torch.exp(log_P)
    Pi = torch.nan_to_num(Pi, nan=0.0).clamp(min=0.0, max=1.0)

    with torch.no_grad():
        row_err = (Pi.sum(dim=-1) - row_mass).abs().mean().item()
        col_err = (Pi.sum(dim=-2) - col_mass).abs().mean().item()

    return log_P, {"row_error": row_err, "col_error": col_err}


# ===================================================================
#  Metric computation
# ===================================================================

class TransportDiagnostics:
    """Compute all diagnostic metrics from captured tensors."""

    def __init__(self, stored: Dict[str, torch.Tensor]):
        self.s = {k: v for k, v in stored.items()}
        self.metrics: Dict[str, float] = {}

    def compute_all(self) -> Dict[str, float]:
        s = self.s
        eps = 1e-8

        Pi_corr = s.get("Pi_corr")
        Pi_prior = s.get("Pi_prior")
        logPi_prior = s.get("logPi_prior")
        logPi_corr = s.get("logPi_corr")
        L_prior = s.get("L_prior")
        L_corr = s.get("L_corr")
        score_qk = s.get("score_qk")
        residual_logits = s.get("residual_logits")
        nu = s.get("nu")
        unit_mass = s.get("unit_mass")
        h = s.get("h")
        delta_h = s.get("delta_h")

        m: Dict[str, float] = {}

        # 3. plan_delta
        if Pi_corr is not None and Pi_prior is not None and Pi_prior.numel() > 0:
            m["plan_delta"] = (Pi_corr - Pi_prior).abs().sum(dim=(-2, -1)).mean().item()
        else:
            m["plan_delta"] = 0.0

        # 4. kl_to_prior
        if logPi_corr is not None and logPi_prior is not None and Pi_corr is not None and Pi_corr.numel() > 0:
            Pi_safe = Pi_corr.clamp(min=1e-30)
            m["kl_to_prior"] = (Pi_safe * (logPi_corr - logPi_prior)).sum(dim=(-2, -1)).mean().item()
        elif Pi_corr is not None and Pi_prior is not None and Pi_corr.numel() > 0:
            # Fallback: compute KL from Pi directly
            Pi_safe = Pi_corr.clamp(min=1e-30)
            Pi_prior_safe = Pi_prior.clamp(min=1e-30)
            m["kl_to_prior"] = (Pi_safe * (Pi_safe.log() - Pi_prior_safe.log())).sum(dim=(-2, -1)).mean().item()
        else:
            m["kl_to_prior"] = 0.0

        # 5. logit_ratio
        if residual_logits is not None and L_prior is not None and L_prior.numel() > 0:
            m["logit_ratio"] = (residual_logits.std().item() / (L_prior.std().item() + eps))
        else:
            m["logit_ratio"] = 0.0

        # 6. entropy
        if Pi_prior is not None and Pi_prior.numel() > 0:
            Pi_p_safe = Pi_prior.clamp(min=1e-30)
            m["entropy_prior"] = (-Pi_p_safe * Pi_p_safe.log()).sum(dim=(-2, -1)).mean().item()
        else:
            m["entropy_prior"] = 0.0

        if Pi_corr is not None and Pi_corr.numel() > 0:
            Pi_c_safe = Pi_corr.clamp(min=1e-30)
            m["entropy_corr"] = (-Pi_c_safe * Pi_c_safe.log()).sum(dim=(-2, -1)).mean().item()
        else:
            m["entropy_corr"] = 0.0

        # 7. marginal error
        if Pi_corr is not None and Pi_corr.numel() > 0 and nu is not None:
            row_sums = Pi_corr.sum(dim=-1)  # [B, T]
            col_sums = Pi_corr.sum(dim=-2)  # [B, K]
            target_row = nu.to(Pi_corr.device)
            target_col = unit_mass.to(Pi_corr.device) if unit_mass is not None else None
            m["row_error"] = (row_sums - target_row).abs().mean().item()
            m["col_error"] = (col_sums - target_col).abs().mean().item() if target_col is not None else 0.0
        else:
            m["row_error"] = 0.0
            m["col_error"] = 0.0

        # 8. res_ratio
        if delta_h is not None and h is not None and h.numel() > 0:
            m["res_ratio"] = (delta_h.pow(2).mean().sqrt().item() / (h.pow(2).mean().sqrt().item() + eps))
            m["h_norm"] = h.pow(2).mean().sqrt().item()
            m["delta_h_norm"] = delta_h.pow(2).mean().sqrt().item()
        else:
            m["res_ratio"] = 0.0
            m["h_norm"] = 0.0
            m["delta_h_norm"] = 0.0

        # transported context norm z_norm ≈ delta_h_norm / write_alpha
        write_alpha = getattr(self._get_adapter(), 'write_alpha', None)
        if write_alpha is not None and m["delta_h_norm"] > 0:
            wa = write_alpha.item() if hasattr(write_alpha, 'item') else float(write_alpha)
            m["z_norm"] = m["delta_h_norm"] / max(wa, eps)
        else:
            m["z_norm"] = 0.0

        # Additional: QK stats
        if score_qk is not None and score_qk.numel() > 0:
            m["qk_std"] = score_qk.std().item()
            m["qk_mean"] = score_qk.mean().item()
        else:
            m["qk_std"] = 0.0
            m["qk_mean"] = 0.0

        self.metrics = m
        return m

    def _get_adapter(self):
        """Try to recover adapter reference from stored tensors."""
        return None

    def get_check_messages(self) -> List[str]:
        m = self.metrics
        msgs = []

        if m.get("plan_delta", 0) < 1e-3 or m.get("logit_ratio", 0) < 0.02:
            msgs.append("[DIAG] QK residual may be too weak or suppressed by prior/KL.")
        if m.get("kl_to_prior", 1) < 1e-6 and abs(m.get("entropy_corr", 0) - m.get("entropy_prior", 0)) < 0.01:
            msgs.append("[DIAG] Corrected coupling is almost identical to prior coupling.")
        if m.get("res_ratio", 0) < 0.005:
            msgs.append("[DIAG] Residual injection is likely too weak to affect hidden states.")
        if m.get("row_error", 0) > 1e-3 or m.get("col_error", 0) > 1e-3:
            msgs.append("[DIAG] Sinkhorn marginal constraints may not be satisfied.")

        return msgs


# ===================================================================
#  Long-form drift diagnostics
# ===================================================================

def compute_drift_diagnostics(
    Pi_corr: torch.Tensor,
    Pi_prior: torch.Tensor,
    nu: torch.Tensor,
    mu: torch.Tensor,
) -> Dict[str, float]:
    """Compute prefix usage drift and late drift.

    Args:
        Pi_corr: [B, T, K] corrected coupling.
        Pi_prior: [B, T, K] prior coupling.
        nu: [B, T] row marginal (uniform 1/T).
        mu: [B, K] column marginal.

    Returns:
        dict of drift metrics.
    """
    eps = 1e-8
    dd: Dict[str, float] = {}
    T = Pi_corr.shape[-2] if Pi_corr.dim() >= 2 else 0
    K = Pi_corr.shape[-1] if Pi_corr.dim() >= 2 else 0
    if T == 0:
        return dd

    # Global marginal error (already in main metrics)
    # Prefix usage drift
    prefix_drifts_corr = []
    prefix_drifts_prior = []
    target_prefix = mu[0].cumsum(dim=0) if mu is not None else torch.zeros(K)

    for t in range(1, T + 1):
        prefix_usage_corr = Pi_corr[0, :t].sum(dim=0)  # [K]
        prefix_usage_prior = Pi_prior[0, :t].sum(dim=0)

        # Expected prefix coverage: how much mass should have arrived by position t/T
        frac = t / T
        target_at_t = (mu[0] * frac).cumsum(dim=0) if mu is not None else torch.zeros(K)
        # Actually simpler: the target is just cumsum(mu) clamped at frac
        target_at_t = (mu[0] * T * frac) if mu is not None else torch.ones(K) * frac

        # Drift = || prefix_usage - target_at_t ||_1
        drift_c = (prefix_usage_corr - target_at_t).abs().mean().item()
        drift_p = (prefix_usage_prior - target_at_t).abs().mean().item()
        prefix_drifts_corr.append(drift_c)
        prefix_drifts_prior.append(drift_p)

    dd["prefix_drift_mean_corr"] = float(np.mean(prefix_drifts_corr)) if prefix_drifts_corr else 0.0
    dd["prefix_drift_mean_prior"] = float(np.mean(prefix_drifts_prior)) if prefix_drifts_prior else 0.0

    # Late drift (last 40%)
    late_start = int(0.6 * T)
    if late_start < T:
        late_drifts_c = prefix_drifts_corr[late_start:]
        late_drifts_p = prefix_drifts_prior[late_start:]
        dd["late_drift_corr"] = float(np.mean(late_drifts_c)) if late_drifts_c else 0.0
        dd["late_drift_prior"] = float(np.mean(late_drifts_p)) if late_drifts_p else 0.0
    else:
        dd["late_drift_corr"] = 0.0
        dd["late_drift_prior"] = 0.0

    return dd


# ===================================================================
#  Save helpers
# ===================================================================

def save_diagnostics_json(metrics: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[DIAG] Saved: {path}")


def save_diagnostics_csv(metrics_list: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not metrics_list:
        return
    fieldnames = list(metrics_list[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in metrics_list:
            w.writerow(row)
    print(f"[DIAG] Saved: {path}")


# ===================================================================
#  Visualization
# ===================================================================

def save_coupling_heatmaps(
    Pi_prior: torch.Tensor,
    Pi_corr: torch.Tensor,
    sample_id: str,
    output_dir: str,
    unit_labels: Optional[List[str]] = None,
):
    """Save prior, corrected, and difference heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[DIAG] matplotlib not available, skipping heatmaps")
        return

    os.makedirs(output_dir, exist_ok=True)
    Pi_p = Pi_prior[0].numpy() if Pi_prior.dim() == 3 else Pi_prior.numpy()
    Pi_c = Pi_corr[0].numpy() if Pi_corr.dim() == 3 else Pi_corr.numpy()

    T, K = Pi_p.shape
    tags = unit_labels if unit_labels else [f"U{k}" for k in range(K)]

    # Subsample T for display
    step = max(1, T // 512)
    if step > 1:
        Td = T // step
        Pi_p = Pi_p[:Td * step].reshape(Td, step, K).mean(axis=1)
        Pi_c = Pi_c[:Td * step].reshape(Td, step, K).mean(axis=1)

    pairs = [
        ("prior_coupling", Pi_p, "Prior Coupling Π_prior"),
        ("corrected_coupling", Pi_c, "Corrected Coupling Π_corr"),
        ("coupling_difference", Pi_c - Pi_p, "Difference Π_corr − Π_prior"),
    ]

    for fname, data, title in pairs:
        fig, ax = plt.subplots(figsize=(max(5, K * 0.4), 5))
        im = ax.imshow(data.T, aspect="auto", cmap="viridis" if "difference" not in fname else "RdBu",
                       origin="lower", interpolation="nearest")
        ax.set_xlabel("Audio position")
        ax.set_ylabel("Condition unit")
        ax.set_title(f"{title} [{sample_id}]")
        if len(tags) <= 40:
            ax.set_yticks(range(K))
            ax.set_yticklabels(tags, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[DIAG] Heatmaps saved to {output_dir}")


def save_condition_usage_bar(
    mu: torch.Tensor,
    Pi_prior: torch.Tensor,
    Pi_corr: torch.Tensor,
    sample_id: str,
    output_dir: str,
    unit_labels: Optional[List[str]] = None,
):
    """Save condition usage bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    K = Pi_prior.shape[-1]
    tags = unit_labels if unit_labels else [f"U{k}" for k in range(K)]

    usage_prior = Pi_prior[0].sum(dim=0).numpy() if Pi_prior.dim() == 3 else Pi_prior.sum(dim=0).numpy()
    usage_corr = Pi_corr[0].sum(dim=0).numpy() if Pi_corr.dim() == 3 else Pi_corr.sum(dim=0).numpy()
    mu_np = mu[0].numpy() if mu.dim() == 2 else mu.numpy() if mu.dim() == 1 else mu.numpy()

    x = np.arange(K)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(5, K * 0.3), 4))
    ax.bar(x - w, mu_np, w, label="Target μ", color="gray", alpha=0.5)
    ax.bar(x, usage_prior, w, label="Prior usage", color="steelblue", alpha=0.7)
    ax.bar(x + w, usage_corr, w, label="Corrected usage", color="coral", alpha=0.7)
    ax.set_xlabel("Condition unit")
    ax.set_ylabel("Total mass")
    ax.set_title(f"Condition Usage [{sample_id}]")
    if len(tags) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=7, rotation=45)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "condition_usage.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prefix_drift_curve(
    Pi_prior: torch.Tensor,
    Pi_corr: torch.Tensor,
    mu: torch.Tensor,
    sample_id: str,
    output_dir: str,
):
    """Save prefix drift curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    T, K = Pi_prior.shape[-2], Pi_prior.shape[-1]
    mu_0 = mu[0] if mu.dim() == 2 else mu

    prefix_prior = []
    prefix_corr = []
    prefix_target = []

    for t in range(1, T + 1):
        frac = t / T
        target_at_t = (mu_0.cumsum(dim=0).clamp(max=frac).to(Pi_prior.device) if mu_0 is not None
                       else torch.full((K,), frac))
        # Actually: target prefix usage = min(cumsum(mu), frac) is not right
        # Better: target = frac (since each unit should have received frac of its total mu)
        target_at_t = mu_0 * frac

        usage_p = Pi_prior[0, :t].sum(dim=0).cpu()
        usage_c = Pi_corr[0, :t].sum(dim=0).cpu()
        tgt = target_at_t.cpu()

        prefix_prior.append((usage_p - tgt).abs().mean().item())
        prefix_corr.append((usage_c - tgt).abs().mean().item())
        prefix_target.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, T + 1), prefix_prior, label="Prior drift", color="steelblue", alpha=0.7, linewidth=1)
    ax.plot(range(1, T + 1), prefix_corr, label="Corrected drift", color="coral", alpha=0.7, linewidth=1)
    ax.axvline(x=0.6 * T, color="gray", linestyle="--", alpha=0.4, label="60% threshold")
    ax.set_xlabel("Prefix length t")
    ax.set_ylabel("Mean absolute drift")
    ax.set_title(f"Prefix Usage Drift [{sample_id}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prefix_drift_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
#  Full diagnostic pipeline
# ===================================================================

def run_diagnostics(
    adapter: torch.nn.Module,
    sample_id: str = "sample_0",
    method: str = "qk_kl",
    output_dir: str = "./diagnostics",
    unit_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full diagnostic capture and compute all metrics.

    This function is intended to be called during evaluation forward pass,
    after the adapter has been run with intermediates captured.
    """
    cap = DiagnosticsCapture(adapter)
    # The caller must have stored intermediates in cap._stored via the context manager
    diag = TransportDiagnostics(cap._stored)
    metrics = diag.compute_all()
    msgs = diag.get_check_messages()

    # Drift diagnostics
    Pi_corr = cap._stored.get("Pi_corr")
    Pi_prior = cap._stored.get("Pi_prior")
    nu = cap._stored.get("nu")
    mu = cap._stored.get("unit_mass")

    drift = {}
    if Pi_corr is not None and Pi_prior is not None:
        drift = compute_drift_diagnostics(Pi_corr, Pi_prior, nu, mu)
    metrics.update(drift)

    # Save
    sample_out = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_out, exist_ok=True)

    save_diagnostics_json({"sample_id": sample_id, "method": method, **metrics},
                          os.path.join(sample_out, "diagnostics.json"))

    # Save csv for all metrics
    metrics_row = {"sample_id": sample_id, "method": method, **metrics}
    from pathlib import Path as P
    csv_path = os.path.join(output_dir, "diagnostics_qk_kl.csv")
    existing = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
        if not existing:
            w.writeheader()
        w.writerow(metrics_row)

    # Visualizations
    if Pi_corr is not None and Pi_prior is not None:
        save_coupling_heatmaps(Pi_prior, Pi_corr, sample_id, sample_out, unit_labels)
        if mu is not None:
            save_condition_usage_bar(mu, Pi_prior, Pi_corr, sample_id, sample_out, unit_labels)
            save_prefix_drift_curve(Pi_prior, Pi_corr, mu, sample_id, sample_out)

    return {"metrics": metrics, "messages": msgs, "output_dir": sample_out}
