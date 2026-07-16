#!/usr/bin/env python3
"""
Stage 0 + Stage 1: State-adaptive Sinkhorn scoring diagnosis.

Runs 6 configs through forward-only diagnosis, selects best, does short training.
"""

from __future__ import annotations
import os, sys, json, math, time, copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.phase_memory import (
    TransportRetrievalAdapter, PMRetrievalPhaseMemory,
    parse_lyrics_to_units, build_duration_scaffold,
)
from acestep.training_v2.timestep_sampling import sample_timesteps

SAMPLE_ID = "b704dd0b245aa2ebaf6229399ca942a13bc907cb_1754"
DATASET_DIR = Path("/root/autodl-tmp/musicdata/dataset")
TENSOR_DIR = Path("/root/autodl-tmp/musicdata/train_tensors")

OUTPUT = Path("/root/ACE-Step-1.5/diag2_output")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── 6 configs ──────────────────────────────────────────────────────────
CONFIGS = [
    dict(label="A", beta_pos=1.0, gamma_state=0.5, temperature=1.0, qk_lr_mult=3.0),
    dict(label="B", beta_pos=1.0, gamma_state=1.0, temperature=1.0, qk_lr_mult=3.0),
    dict(label="C", beta_pos=0.5, gamma_state=0.5, temperature=1.0, qk_lr_mult=3.0),
    dict(label="D", beta_pos=1.0, gamma_state=0.5, temperature=1.5, qk_lr_mult=3.0),
    dict(label="E", beta_pos=1.0, gamma_state=1.0, temperature=1.5, qk_lr_mult=5.0),
    dict(label="F", beta_pos=2.0, gamma_state=1.0, temperature=1.0, qk_lr_mult=5.0),
]

REJECTION_THRESHOLDS = dict(
    min_state_score_ratio=0.2,
    max_state_score_ratio=2.0,
)

# =========================================================================
# Helpers
# =========================================================================

def _grad_norm(mod_or_params):
    """Compute L2 gradient norm."""
    total = 0.0
    if isinstance(mod_or_params, torch.nn.Module):
        params = mod_or_params.parameters()
    else:
        params = mod_or_params
    for p in params:
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return math.sqrt(total) if total > 0 else 0.0


def compute_coupling_diff(Pi_a: torch.Tensor, Pi_b: torch.Tensor) -> float:
    """Mean abs difference between two transport plans."""
    return (Pi_a - Pi_b).abs().mean().item()


def compute_expected_idx(Pi: torch.Tensor) -> np.ndarray:
    """E[unit_index] per audio position: sum_k Pi[t,k] * k."""
    K = Pi.shape[-1]
    idx = torch.arange(K, device=Pi.device, dtype=torch.float32)
    expected = (Pi * idx).sum(dim=-1)  # [B, T]
    return expected.squeeze(0).cpu().numpy()


def monotonicity_spearman(expected_idx: np.ndarray) -> float:
    """Spearman rho between expected unit index and linear audio time."""
    from scipy.stats import spearmanr
    T = len(expected_idx)
    rho, _ = spearmanr(expected_idx, np.linspace(0, 1, T))
    return float(rho)


def monotonicity_slope(expected_idx: np.ndarray) -> float:
    """Linear regression slope of expected unit index vs time."""
    T = len(expected_idx)
    x = np.linspace(0, 1, T)
    A = np.vstack([x, np.ones(T)]).T
    slope, _ = np.linalg.lstsq(A, expected_idx, rcond=None)[0]
    return float(slope)


def non_lyric_leakage(Pi: torch.Tensor, unit_is_lyric: torch.Tensor) -> float:
    """Fraction of total transport mass on non-lyric (silence) units."""
    if unit_is_lyric is None:
        return 0.0
    K = Pi.shape[-1]
    silence_mask = (~unit_is_lyric).float().squeeze(0)  # [K]
    total = Pi.sum()
    leakage = (Pi * silence_mask.unsqueeze(0).unsqueeze(0)).sum() / (total + 1e-10)
    return float(leakage.cpu())


def build_lyric_data(
    lyrics_text: str, L: int, device: torch.device,
) -> Dict[str, Any]:
    """Parse lyrics, build scaffold (shared across experiments)."""
    parser = LyricsStructureParser()
    parsed = parser.parse(lyrics_text, num_chunks=L)
    section_ids = parsed.section_type_ids
    units, _, debug = parse_lyrics_to_units(
        lyrics_text, section_ids, auto_transition_ratios={},
    )
    scaffold = build_duration_scaffold(
        units, text_len=L, tag_control_mask=debug.get("tag_control_mask"),
    )
    scaffold = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}
    return {
        "section_ids": section_ids.to(device),
        "scaffold": scaffold,
        "units": units,
        "U": len(scaffold["unit_boundaries"]) - 1,
        "c_all": ((scaffold["unit_boundaries"][:-1] + scaffold["unit_boundaries"][1:]) / 2),
        "mu_all": (scaffold["unit_duration"] / scaffold["unit_duration"].sum()),
        "unit_section_id": scaffold["unit_section_ids"],
        "unit_is_lyric": scaffold["lyric_unit_mask"],
    }


def build_adapter(
    D: int, cfg: dict, device: torch.device, dtype: torch.dtype,
    sinkhorn_iters: int = 50,
) -> TransportRetrievalAdapter:
    """Build adapter with state_adaptive scoring."""
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        transport_mode="sinkhorn",
        sinkhorn_iters=sinkhorn_iters,
        transport_sigma=0.18,
        transport_qk_scale=1.0,  # unused in state_adaptive mode
        scoring_mode="state_adaptive",
        beta_pos=cfg["beta_pos"],
        gamma_state=cfg["gamma_state"],
        temperature=cfg["temperature"],
        use_double_center=True,
        use_score_standardize=True,
        standardize_pos_score=True,  # both scores get std≈1, gamma/beta is meaningful
        write_alpha_init=0.001,
        write_alpha_max=0.01,
        out_proj_init_std=0.01,
    ).to(device).float()
    return adapt


def build_pm(D: int, device: torch.device) -> PMRetrievalPhaseMemory:
    pm = PMRetrievalPhaseMemory(
        dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True,
    ).to(device).float()
    return pm


def unit_text_hidden_from_scaffold(
    eh: torch.Tensor, scaffold: dict, lyric_data: dict, B: int, D: int,
) -> torch.Tensor:
    """Pool encoder_hidden per lyric unit."""
    token_to_unit = scaffold["token_to_unit"]
    lyric_mask = scaffold["lyric_mask"]
    unit_is_lyric = lyric_data["unit_is_lyric"]
    U = lyric_data["U"]
    eh_f = eh.float()
    pooled_list = []
    for uid in range(U):
        is_l = unit_is_lyric[uid].item()
        tmask = (token_to_unit == uid) & lyric_mask if is_l else (token_to_unit == uid)
        tmask_b = tmask.unsqueeze(0).expand(B, -1)
        if tmask_b.any():
            pooled_list.append(eh_f[tmask_b].view(B, -1, D).mean(dim=1))
        else:
            pooled_list.append(torch.zeros(B, D, device=eh.device, dtype=torch.float32))
    return torch.stack(pooled_list, dim=1)


def warmup_H(
    model, xt, t, x0, x1, eh, eam, am, ctx,
) -> torch.Tensor:
    """Run decoder forward, return layer 12 hidden states (no grad)."""
    hs_list = []
    def _hook(m, i, o): hs_list.append(o[0])
    handle = model.decoder.layers[12].register_forward_hook(_hook)
    with torch.no_grad():
        model.decoder(
            hidden_states=xt, timestep=t, timestep_r=t,
            attention_mask=am, encoder_hidden_states=eh,
            encoder_attention_mask=eam, context_latents=ctx,
            use_cache=False, output_attentions=False,
        )
    handle.remove()
    return hs_list[0].detach().float()


def run_adapter_forward(
    adapt, pm, H, t, eh_f, lyric_data, p_audio, unit_text_hidden,
    gamma_state_override: Optional[float] = None,
    pm_state_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Run adapter forward, collecting detailed diagnostics.
    Optionally override gamma_state or pm_state for ablation.
    Always uses S_pos normalization to match the adapter's
    standardize_pos_score=True config.
    """
    B, T_a = p_audio.shape
    device = p_audio.device
    K = lyric_data["U"]
    eps_score = 1e-6

    pm_state = pm(H, t) if pm_state_override is None else pm_state_override
    c_unit = lyric_data["c_all"].unsqueeze(0)
    mu = lyric_data["mu_all"].unsqueeze(0)

    # Manually trace the scoring computation for diagnosis
    # Step 1: Position cost
    dist = p_audio[:, :, None] - c_unit[:, None, :]
    C = (dist / adapt.transport_sigma) ** 2
    S_pos = -C  # [B, T, K]

    # Step 2: Q/K
    pm_state_n = adapt.pm_state_norm(pm_state)
    a_feat = torch.stack([p_audio, p_audio ** 2, 1.0 - p_audio], dim=-1)
    audio_coord = adapt.audio_coord_mlp(a_feat)

    u_feat = torch.stack([c_unit, c_unit ** 2, 1.0 - c_unit], dim=-1)
    unit_coord = adapt.unit_coord_mlp(u_feat)

    usid = lyric_data["unit_section_id"].unsqueeze(0)
    sec_emb = adapt.section_embedding(usid.long())

    t_emb = torch.zeros(B, T_a, adapt.time_dim, device=device, dtype=p_audio.dtype)

    q_in = torch.cat([pm_state_n, audio_coord, t_emb], dim=-1)
    q = F.normalize(adapt.q_mlp(q_in), dim=-1, p=2)

    k_in = torch.cat([unit_text_hidden, unit_coord, sec_emb], dim=-1)
    k = F.normalize(adapt.k_mlp(k_in), dim=-1, p=2)

    R = torch.matmul(q, k.transpose(-1, -2))  # [B, T, K]

    # Step 3: State-adaptive scoring (matches adapter forward exactly)
    B_state = R
    B_mean_row = B_state.mean(dim=-1, keepdim=True)
    B_mean_col = B_state.mean(dim=-2, keepdim=True)
    B_mean_all = B_state.mean(dim=(-1, -2), keepdim=True)
    B_centered = B_state - B_mean_row - B_mean_col + B_mean_all
    B_std = B_centered.std(dim=(-1, -2), keepdim=True) + eps_score
    B_standardized = B_centered / B_std

    # Also standardize S_pos (standardize_pos_score=True in adapter)
    S_pos_norm = S_pos / (S_pos.std(dim=(-1, -2), keepdim=True) + eps_score)

    gamma = adapt.gamma_state if gamma_state_override is None else gamma_state_override
    S = adapt.beta_pos * S_pos_norm + gamma * B_standardized
    L = S / adapt.temperature

    # Step 4: Sinkhorn
    nu = torch.full((B, T_a,), 1.0 / T_a, device=device, dtype=p_audio.dtype)
    Pi, sinkhorn_info = log_sinkhorn_from_score(
        L, nu, mu, iters=adapt.sinkhorn_iters,
    )

    # Step 5: Value + writer
    v = adapt.v_mlp(unit_text_hidden)
    ctx = torch.matmul(Pi, v)
    ctx = ctx / nu.unsqueeze(-1).clamp(min=1e-10)
    raw_res = adapt.out_proj(ctx)
    raw_rms = torch.sqrt(raw_res.pow(2).mean(dim=-1, keepdim=True) + adapt.writer_eps)
    unit_res = raw_res / raw_rms
    h_rms = torch.sqrt(H.pow(2).mean(dim=-1, keepdim=True)).detach()
    delta_h = adapt.write_alpha * h_rms * unit_res

    # -- Diagnostics --
    diag = dict(sinkhorn_info)
    with torch.no_grad():
        diag["S_pos_std"] = S_pos.std().item()
        diag["B_state_raw_std"] = R.std().item()
        diag["B_centered_std"] = B_centered.std().item()
        diag["B_standardized_std"] = B_standardized.std().item()
        diag["gamma_state"] = gamma
        diag["beta_pos"] = adapt.beta_pos
        diag["temperature"] = adapt.temperature
        diag["S_std"] = S.std().item()
        diag["L_std"] = L.std().item()

        scaled_state = gamma * B_standardized
        scaled_pos = adapt.beta_pos * S_pos_norm
        state_ratio = scaled_state.std() / (scaled_pos.std() + eps_score)
        diag["state_score_ratio"] = state_ratio.item()
        diag["scaled_state_std"] = scaled_state.std().item()
        diag["scaled_pos_std"] = scaled_pos.std().item()
        diag["S_pos_raw_std"] = S_pos.std().item()

        # Expected index
        exp_idx = (Pi * torch.arange(K, device=Pi.device, dtype=torch.float32)).sum(dim=-1)
        diag["expected_idx_mean"] = exp_idx.mean().item()
        diag["expected_idx_range"] = (exp_idx.max() - exp_idx.min()).item()

        # Delta_h diagnostics
        diag["delta_h_rms"] = delta_h.pow(2).mean().sqrt().item()
        diag["delta_h_ratio"] = diag["delta_h_rms"] / (H.pow(2).mean().sqrt().item() + 1e-10)

    return {
        "Pi": Pi,
        "delta_h": delta_h,
        "diag": diag,
        "B_state": B_standardized,
        "S_pos": S_pos,
        "S": S,
    }


def log_sinkhorn_from_score(
    log_P: torch.Tensor, row_mass: torch.Tensor, col_mass: torch.Tensor,
    iters: int = 50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Log-domain Sinkhorn with NaN-safe masking."""
    B, T, K = log_P.shape
    device = log_P.device
    log_nu = torch.log(row_mass.clamp(min=1e-30))
    log_mu = torch.log(col_mass.clamp(min=1e-30))

    for _ in range(iters):
        log_P = log_P - (torch.logsumexp(log_P, dim=-1, keepdim=True) - log_nu.unsqueeze(-1))
        log_P = log_P - (torch.logsumexp(log_P, dim=-2, keepdim=True) - log_mu.unsqueeze(-2))

    Pi = torch.exp(log_P)
    Pi = torch.nan_to_num(Pi, nan=0.0)
    Pi = Pi.clamp(min=0.0, max=1.0)

    with torch.no_grad():
        row_error = (Pi.sum(dim=-1) - row_mass).abs().mean().item()
        col_error = (Pi.sum(dim=-2) - col_mass).abs().mean().item()
        entropy = (-Pi * (Pi + 1e-10).log()).sum(dim=-1).mean().item()
        has_nan = float(not torch.isfinite(Pi).all())

    info = {"row_error": row_error, "col_error": col_error,
            "entropy": entropy, "sinkhorn_has_nan": has_nan}
    return Pi, info


# =========================================================================
# STAGE 0 — Forward-only diagnosis
# =========================================================================

def stage0(
    model, pt_data, lyrics_text: str, device, dtype,
) -> Dict[str, Any]:
    """Run forward-only diagnosis for all configs."""
    D = model.config.hidden_size
    B = 1
    L = pt_data["encoder_hidden_states"].shape[0]

    lyric_data = build_lyric_data(lyrics_text, L, device)
    U = lyric_data["U"]

    eh = pt_data["encoder_hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
    unit_text_hidden = unit_text_hidden_from_scaffold(
        eh, lyric_data["scaffold"], lyric_data, B, D,
    )

    # ── Prepare batch ─────────────────────────────────────────────────
    t = torch.full((B,), 0.5, device=device, dtype=dtype)
    x1 = torch.randn_like(pt_data["target_latents"]).unsqueeze(0).to(device=device, dtype=dtype)
    x0 = pt_data["target_latents"].unsqueeze(0).to(device=device, dtype=dtype)
    xt = t.view(-1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1)) * x0

    eh_in = eh; eam = pt_data["encoder_attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
    am = pt_data["attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
    ctx = pt_data["context_latents"].unsqueeze(0).to(device=device, dtype=dtype)

    # ── Warmup ────────────────────────────────────────────────────────
    H = warmup_H(model, xt, t, x0, x1, eh_in, eam, am, ctx)
    T_a = H.shape[1]
    p_audio = torch.linspace(0, 1, T_a, device=device, dtype=torch.float32).unsqueeze(0)

    def pm_with_H(pm, H, t):
        return pm(H, t)

    results = {}
    for cfg in CONFIGS:
        label = cfg["label"]
        print(f"\n  ── Config {label} ──")

        pm = build_pm(D, device)
        adapt = build_adapter(D, cfg, device, dtype)

        # ── (A) gamma=0 baseline ──────────────────────────────────────
        pm_state = pm_with_H(pm, H, t)
        out0 = run_adapter_forward(
            adapt, pm, H, t, eh, lyric_data, p_audio, unit_text_hidden,
            gamma_state_override=0.0, pm_state_override=pm_state,
        )
        Pi_gamma0 = out0["Pi"]

        # ── (B) Full forward ──────────────────────────────────────────
        out = run_adapter_forward(
            adapt, pm, H, t, eh, lyric_data, p_audio, unit_text_hidden,
            gamma_state_override=None, pm_state_override=None,
        )
        Pi = out["Pi"]
        d = out["diag"]

        # ── (C) PM-off forward ───────────────────────────────────────
        pm_state_zero = torch.zeros_like(pm_state)
        out_pmoff = run_adapter_forward(
            adapt, pm, H, t, eh, lyric_data, p_audio, unit_text_hidden,
            gamma_state_override=None, pm_state_override=pm_state_zero,
        )
        Pi_pmoff = out_pmoff["Pi"]

        # ── Compute metrics ───────────────────────────────────────────
        coupling_gamma0 = compute_coupling_diff(Pi, Pi_gamma0)
        coupling_pm = compute_coupling_diff(Pi, Pi_pmoff)
        state_ratio = d["state_score_ratio"]

        exp_idx = compute_expected_idx(Pi)
        mono_rho = monotonicity_spearman(exp_idx)
        mono_slope = monotonicity_slope(exp_idx)

        leakage = non_lyric_leakage(Pi, lyric_data["unit_is_lyric"])

        has_nan = d.get("sinkhorn_has_nan", 0.0) > 0 or d.get("has_nan", 0.0) > 0
        row_ok = d["row_error"] < 1e-3
        col_ok = d["col_error"] < 1e-3

        entry = dict(
            label=label,
            beta_pos=cfg["beta_pos"], gamma_state=cfg["gamma_state"],
            temperature=cfg["temperature"], qk_lr_mult=cfg["qk_lr_mult"],
            coupling_gamma0=coupling_gamma0,
            coupling_pm=coupling_pm,
            state_score_ratio=state_ratio,
            Pi_entropy=d["entropy"],
            row_error=d["row_error"],
            col_error=d["col_error"],
            leakage=leakage,
            expected_idx_rho=mono_rho,
            expected_idx_slope=mono_slope,
            expected_idx_range=d["expected_idx_range"],
            has_nan=has_nan,
            S_pos_std=d["S_pos_std"],
            scaled_pos_std=d["scaled_pos_std"],
            scaled_state_std=d["scaled_state_std"],
            delta_h_ratio=d["delta_h_ratio"],
            B_raw_std=d["B_state_raw_std"],
            B_std_after=d["B_standardized_std"],
            S_std=d["S_std"],
            L_std=d["L_std"],
        )
        results[label] = entry

        ok = not has_nan and row_ok and col_ok
        ratio_ok = REJECTION_THRESHOLDS["min_state_score_ratio"] <= state_ratio <= REJECTION_THRESHOLDS["max_state_score_ratio"]

        status = "✓" if (ok and ratio_ok) else "✗"
        reject_reasons = []
        if has_nan: reject_reasons.append("NaN")
        if not row_ok: reject_reasons.append(f"row_err={d['row_error']:.2e}")
        if not col_ok: reject_reasons.append(f"col_err={d['col_error']:.2e}")
        if not ratio_ok: reject_reasons.append(f"state_ratio={state_ratio:.4f} ∉ [0.2,2]")

        print(f"    coupling_gamma0={coupling_gamma0:.2e} coupling_pm={coupling_pm:.2e}")
        print(f"    state_score_ratio={state_ratio:.4f} entropy={d['entropy']:.4f}")
        print(f"    expected_idx_rho={mono_rho:.4f} slope={mono_slope:.4f}")
        print(f"    leakage={leakage:.4f} delta_h_ratio={d['delta_h_ratio']:.6f}")
        print(f"    S_pos_std={d['S_pos_std']:.4f} scaled_state_std={d['scaled_state_std']:.4f}")
        print(f"    row_err={d['row_error']:.2e} col_err={d['col_error']:.2e}")
        print(f"    {status} {' | '.join(reject_reasons) if reject_reasons else 'PASS'}")

        # Save Pi for later analysis
        torch.save(Pi.cpu(), OUTPUT / f"stage0_Pi_{label}.pt")

    return results


# =========================================================================
# STAGE 1 — Short training
# =========================================================================

def stage1(
    model, pt_data, lyrics_text: str, device, dtype,
    selected_labels: List[str],
    num_steps: int = 300,
):
    """Train selected configs, log detailed metrics."""
    D = model.config.hidden_size
    B = 1
    L = pt_data["encoder_hidden_states"].shape[0]
    lyric_data = build_lyric_data(lyrics_text, L, device)

    eh = pt_data["encoder_hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
    unit_text_hidden = unit_text_hidden_from_scaffold(
        eh, lyric_data["scaffold"], lyric_data, B, D,
    )
    eh_in = eh
    eam = pt_data["encoder_attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
    am = pt_data["attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
    ctx = pt_data["context_latents"].unsqueeze(0).to(device=device, dtype=dtype)

    history = {}

    for cfg in CONFIGS:
        if cfg["label"] not in selected_labels:
            continue
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"Training config {label} for {num_steps} steps")
        print(f"  beta_pos={cfg['beta_pos']} gamma_state={cfg['gamma_state']} "
              f"temperature={cfg['temperature']} qk_lr_mult={cfg['qk_lr_mult']}")

        pm = build_pm(D, device)
        adapt = build_adapter(D, cfg, device, dtype)

        # Setup optimizer: q/k/PM get qk_lr_mult multiplier, no weight decay on q/k/PM/write_logit
        no_wd_names = {"q_mlp", "k_mlp", "out_proj", "write_logit", "pm_state_norm"}
        # PM has its own params
        adapt_no_wd = []
        adapt_wd = []
        for name, p in adapt.named_parameters():
            base = name.split(".")[0]
            if base in no_wd_names:
                adapt_no_wd.append(p)
            else:
                adapt_wd.append(p)

        lr = 1e-4
        qk_lr_mult = cfg["qk_lr_mult"]
        opt = torch.optim.AdamW([
            {"params": pm.parameters(), "lr": lr * qk_lr_mult, "weight_decay": 0.0},
            {"params": adapt_no_wd, "lr": lr * qk_lr_mult, "weight_decay": 0.0},
            {"params": adapt_wd, "lr": lr, "weight_decay": 0.01},
        ], lr=lr, weight_decay=0.01)

        step_log = []

        for step in range(num_steps):
            # ── Sample batch ──────────────────────────────────────────
            t, r = sample_timesteps(B, device, dtype, data_proportion=0.5,
                                    timestep_mu=-0.4, timestep_sigma=1.0, use_meanflow=False)
            x1 = torch.randn_like(pt_data["target_latents"]).unsqueeze(0).to(device=device, dtype=dtype)
            x0 = pt_data["target_latents"].unsqueeze(0).to(device=device, dtype=dtype)
            xt = t.view(-1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1)) * x0

            # ── Warmup → H ────────────────────────────────────────────
            H = warmup_H(model, xt, t, x0, x1, eh_in, eam, am, ctx)
            T_a = H.shape[1]
            p_audio = torch.linspace(0, 1, T_a, device=device, dtype=torch.float32).unsqueeze(0)
            pm_state = pm(H, t)

            # ── Adapter forward (train mode) ──────────────────────────
            adapt.train()
            pm.train()

            # We need gradients for this — redo within grad context
            with torch.enable_grad():
                # Clone H for gradient tracking
                H_grad = H.detach().clone().requires_grad_(True)
                pm_state_grad = pm(H_grad, t)

                out = run_adapter_forward(
                    adapt, pm, H_grad, t, eh, lyric_data, p_audio, unit_text_hidden,
                    gamma_state_override=None, pm_state_override=pm_state_grad,
                )
                Pi = out["Pi"]
                delta_h = out["delta_h"]
                final_h = H_grad + delta_h

                # Inject and decode
                def _inject(m, i, o):
                    return (final_h.to(dtype=o[0].dtype, device=o[0].device), *o[1:])
                inj_handle = model.decoder.layers[12].register_forward_hook(_inject)
                dec_out = model.decoder(
                    hidden_states=xt, timestep=t, timestep_r=t,
                    attention_mask=am, encoder_hidden_states=eh_in,
                    encoder_attention_mask=eam, context_latents=ctx,
                    use_cache=False, output_attentions=False,
                )
                inj_handle.remove()

                flow = x1 - x0
                loss = F.mse_loss(dec_out[0], flow)

            # ── Backward ──────────────────────────────────────────────
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pm.parameters()) + list(adapt.parameters()), 1.0,
            )

            # ── Capture grad norms BEFORE optimizer step ──────────────
            with torch.no_grad():
                entry = dict(
                    step=step, loss=loss.item(),
                    write_logit=adapt.write_logit.item(),
                    write_alpha=adapt.write_alpha.item(),
                    delta_h_ratio=out["diag"]["delta_h_ratio"],
                    Pi_entropy=out["diag"]["entropy"],
                    state_score_ratio=out["diag"]["state_score_ratio"],
                    row_error=out["diag"]["row_error"],
                    col_error=out["diag"]["col_error"],
                    scaled_state_std=out["diag"]["scaled_state_std"],
                    scaled_pos_std=out["diag"]["scaled_pos_std"],
                    coupling_gamma0=0.0,  # compute separately
                )
                for name, mod in [
                    ("pm", pm), ("q_mlp", adapt.q_mlp),
                    ("k_mlp", adapt.k_mlp), ("v_mlp", adapt.v_mlp),
                    ("out_proj", adapt.out_proj),
                    ("audio_coord", adapt.audio_coord_mlp),
                    ("unit_coord", adapt.unit_coord_mlp),
                ]:
                    entry[f"{name}_grad"] = _grad_norm(mod)

                if adapt.write_logit.grad is not None:
                    entry["write_logit_grad"] = adapt.write_logit.grad.norm().item()

                step_log.append(entry)

            # ── Optimizer step ────────────────────────────────────────
            opt.step()

            if step % 50 == 0:
                print(f"    step {step:4d}: loss={entry['loss']:.4f} "
                      f"state_ratio={entry['state_score_ratio']:.4f} "
                      f"q_grad={entry['q_mlp_grad']:.6f} "
                      f"v_grad={entry['v_mlp_grad']:.6f} "
                      f"row_err={entry['row_error']:.2e}")

        history[label] = step_log

    return history


# =========================================================================
# MAIN
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated config labels for stage 1 (empty = auto-select)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    pt_data = torch.load(str(TENSOR_DIR / f"{SAMPLE_ID}.pt"), map_location="cpu", weights_only=True)
    lyrics_text = (DATASET_DIR / f"{SAMPLE_ID}.lyrics.txt").read_text(encoding="utf-8").strip()
    print(f"Data: T={pt_data['target_latents'].shape[0]}, L={pt_data['encoder_hidden_states'].shape[0]}")

    # ── Load model ────────────────────────────────────────────────────
    print("Loading SFT model...")
    handler = AceStepHandler()
    handler.initialize_service(
        project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
        device="cuda", use_flash_attention=False, compile_model=False,
    )
    model = handler.model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    model.config.use_section_rope_offset = False
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_section_rope", False): layer_mod.use_section_rope = False
        if getattr(layer_mod, "use_phase_memory", False): layer_mod.use_phase_memory = False

    if args.stage == 0:
        print("\n" + "="*60)
        print("STAGE 0: Forward-only diagnosis")
        print("="*60)

        results = stage0(model, pt_data, lyrics_text, device, dtype)

        # ── Summary table ─────────────────────────────────────────────
        print("\n" + "="*60)
        print("STAGE 0 SUMMARY")
        print("="*60)
        header = f"{'Cfg':>4} | {'b_pos':>5} {'g_state':>7} {'temp':>4} {'lr×':>4} | {'couple_γ0':>9} {'couple_pm':>9} {'sratio':>6} {'entropy':>7} {'mono_ρ':>6} | {'leak':>5} {'row':>8} {'col':>8} | {'status':>4}"
        print(header)
        print("-" * len(header))
        for label, r in sorted(results.items()):
            ok = not r["has_nan"] and r["row_error"] < 1e-3 and r["col_error"] < 1e-3
            ratio_ok = REJECTION_THRESHOLDS["min_state_score_ratio"] <= r["state_score_ratio"] <= REJECTION_THRESHOLDS["max_state_score_ratio"]
            coupling_ok = r["coupling_gamma0"] >= REJECTION_THRESHOLDS["min_coupling_diff"]
            status = "PASS" if (ok and ratio_ok and coupling_ok) else "FAIL"
            print(f"{r['label']:>4} | {r['beta_pos']:5.1f} {r['gamma_state']:7.2f} {r['temperature']:4.1f} {r['qk_lr_mult']:4.0f} | "
                  f"{r['coupling_gamma0']:9.2e} {r['coupling_pm']:9.2e} {r['state_score_ratio']:6.3f} {r['Pi_entropy']:7.4f} {r['expected_idx_rho']:6.3f} | "
                  f"{r['leakage']:5.3f} {r['row_error']:8.2e} {r['col_error']:8.2e} | {status:>4}")

        # Save results
        with open(OUTPUT / "stage0_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nStage 0 results -> {OUTPUT / 'stage0_results.json'}")

        # Auto-select for Stage 1
        passed = [r for r in results.values() if not r["has_nan"] and r["row_error"] < 1e-3 and r["col_error"] < 1e-3
                  and REJECTION_THRESHOLDS["min_state_score_ratio"] <= r["state_score_ratio"] <= REJECTION_THRESHOLDS["max_state_score_ratio"]]
        passed.sort(key=lambda r: r["expected_idx_rho"] + r["state_score_ratio"], reverse=True)
        print(f"\nConfigs passing all criteria (ranked by coupling_gamma0):")
        for p in passed:
            print(f"  {p['label']}: coupling={p['coupling_gamma0']:.2e} ratio={p['state_score_ratio']:.3f} "
                  f"mono_rho={p['expected_idx_rho']:.3f}")
        top = passed[:6] if len(passed) >= 4 else passed
        print(f"Recommended for Stage 1: {[p['label'] for p in top]}")

    elif args.stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: Short training")
        print("="*60)

        # Determine which configs to train
        if args.configs:
            selected = [c.strip() for c in args.configs.split(",")]
        else:
            # Auto-select from Stage 0 results
            s0_path = OUTPUT / "stage0_results.json"
            if s0_path.exists():
                s0 = json.loads(s0_path.read_text())
                passed = [r for r in s0.values()
                          if not r["has_nan"] and r["row_error"] < 1e-3 and r["col_error"] < 1e-3
                          and 0.2 <= r["state_score_ratio"] <= 2.0]
                passed.sort(key=lambda r: -r["state_score_ratio"] * r["expected_idx_rho"])
                selected = [p["label"] for p in passed[:6]] if passed else ["A", "B", "C"]
            else:
                selected = ["A", "B", "C", "D"]
        print(f"Training configs: {selected}")

        history = stage1(model, pt_data, lyrics_text, device, dtype, selected, args.steps)

        # ── Report ────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("STAGE 1 RESULTS")
        print("="*60)

        report = {}
        for label, log in history.items():
            first, last = log[0], log[-1]
            # Gradient statistics
            q_grads = [s["q_mlp_grad"] for s in log]
            k_grads = [s["k_mlp_grad"] for s in log]
            v_grads = [s["v_mlp_grad"] for s in log]
            o_grads = [s["out_proj_grad"] for s in log]
            report[label] = dict(
                loss_first=first["loss"], loss_last=last["loss"],
                write_alpha_first=first["write_alpha"], write_alpha_last=last["write_alpha"],
                q_grad_mean=float(np.mean(q_grads)), q_grad_last=q_grads[-1],
                k_grad_mean=float(np.mean(k_grads)), k_grad_last=k_grads[-1],
                v_grad_mean=float(np.mean(v_grads)),
                out_proj_grad_mean=float(np.mean(o_grads)),
                state_ratio_first=first["state_score_ratio"],
                state_ratio_last=last["state_score_ratio"],
                Pi_entropy_mean=float(np.mean([s["Pi_entropy"] for s in log])),
                col_error_mean=float(np.mean([s["col_error"] for s in log])),
                delta_h_ratio_last=last["delta_h_ratio"],
            )
            print(f"  {label}: loss={first['loss']:.4f}→{last['loss']:.4f} "
                  f"q_grad={q_grads[-1]:.6f}(μ={float(np.mean(q_grads)):.6f}) "
                  f"state_ratio={first['state_score_ratio']:.3f}→{last['state_score_ratio']:.3f}")

        with open(OUTPUT / "stage1_results.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nStage 1 results -> {OUTPUT / 'stage1_results.json'}")

        # ── Recommendation ────────────────────────────────────────────
        print("\n=== RECOMMENDATIONS ===")
        rankings = sorted(report.items(), key=lambda kv: (
            -kv[1]["q_grad_mean"], -kv[1]["state_ratio_last"], -kv[1]["loss_last"]
        ), reverse=False)
        # Re-rank by coupling proxy: higher q_grad_mean = more coupling signal
        rankings = sorted(report.items(), key=lambda kv: -kv[1]["q_grad_mean"])
        for rank, (label, r) in enumerate(rankings, 1):
            print(f"  #{rank}: {label} | q_grad_μ={r['q_grad_mean']:.6f} ratio={r['state_ratio_last']:.3f} "
                  f"loss={r['loss_last']:.4f} entropy={r['Pi_entropy_mean']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
