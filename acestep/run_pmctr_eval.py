#!/usr/bin/env python3
"""
ACE-Step 1.5 PM-CTR 统一推理入口

支持以下方法:
  baseline          原始 SFT，不加载 adapter
  lyric_only_transport  只使用 lyric units，无 control/silence units
  full_pmctr        lyric + control units + PM qk residual + Sinkhorn + RMS writer
  softmax_retrieval  用 row-softmax 替代 Sinkhorn (其余配置一致)
  qk_scale_0        Sinkhorn 但 transport_qk_scale=0 (无 PM residual)
  write_alpha_0     加载 full checkpoint 但 writer 强度 = 0

保存:
  - generated.flac
  - inference_config.json
  - transport_units.csv   (transport 方法)
  - transport_summary.json (transport 方法)
  - transport_heatmap.png
  - condition_usage_histogram.png
  - lyric_center_curve.png
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.phase_memory import (
    PMRetrievalPhaseMemory, TransportRetrievalAdapter, LyricRetrievalAdapter,
    parse_lyrics_to_units, build_duration_scaffold,
)
import acestep.phase_memory as pm_module

# Capture original log_sinkhorn for restore after monkey-patching
_ORIGINAL_LOG_SINKHORN = pm_module.log_sinkhorn


# ===================================================================
#  Softmax retrieval — replace Sinkhorn with row-softmax
# ===================================================================

def _softmax_retrieval(log_P, row_mass, col_mass, iters=5, mask=None):
    """Row-softmax replacement for log_sinkhorn.

    Args:
        log_P: [B, T, K] log-scale transport scores.
        row_mass: [B, T] target row sums (ignored for softmax).
        col_mass: [B, K] target column sums (ignored for softmax).
        mask: [B, K] bool, True for valid columns.

    Returns:
        Pi: [B, T, K] row-softmax weights (each row sums to 1).
        info: dict with diagnostics.
    """
    if mask is not None:
        log_P = log_P.masked_fill(~mask.unsqueeze(1), float("-inf"))
    Pi = F.softmax(log_P, dim=-1)
    Pi = torch.nan_to_num(Pi, nan=0.0).clamp(min=0.0, max=1.0)

    with torch.no_grad():
        row_error = (Pi.sum(dim=-1).mean() - row_mass.sum(dim=-1).mean()).abs().item()
        col_error = float("nan")  # softmax doesn't conserve column mass
        entropy_val = (-Pi * (Pi + 1e-10).log()).sum(dim=-1).mean().item()
        has_nan = float(not torch.isfinite(Pi).all())

    info = {
        "row_error": float(row_error),
        "col_error": col_error,
        "entropy": float(entropy_val),
        "sinkhorn_has_nan": has_nan,
        "softmax_mode": 1.0,
    }
    return Pi, info


# ===================================================================
#  Helper: count trainable adapter params
# ===================================================================

def count_trainable_adapter_params(model: torch.nn.Module) -> int:
    """Count trainable parameters in adapter modules (transport_pm, transport_adapter)."""
    total = 0
    for name, mod in model.named_modules():
        if any(k in name for k in ("transport_pm", "transport_adapter", "retrieval_pm", "retrieval_adapter")):
            for p in mod.parameters():
                if p.requires_grad:
                    total += p.numel()
    return total


# ===================================================================
#  Adapter loading
# ===================================================================

def load_adapter(
    model: torch.nn.Module,
    method: str,
    ckpt_path: str,
    device: torch.device,
    transport_qk_scale: Optional[float] = None,
    write_alpha_multiplier: Optional[float] = None,
    force_write_scale: Optional[float] = None,
) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Dict[str, Any]]:
    """Create and load PM + adapter for the given method.

    Returns:
        (pm, adapter, info_dict)
    """
    if method == "baseline":
        return None, None, {"adapter_loaded": False}

    D = model.config.hidden_size
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pm_sd = saved.get("phase_memory", saved)
    adapt_sd = saved.get("retrieval_adapter", saved)

    # ---- PMRetrievalPhaseMemory ----
    pm = PMRetrievalPhaseMemory(
        dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True,
    )
    pm.load_state_dict(pm_sd, strict=False)
    pm = pm.to(device).float().eval()

    # ---- Adapter ----
    if method == "softmax_retrieval":
        # Load TransportRetrievalAdapter but monkey-patch log_sinkhorn later
        qk_scale = transport_qk_scale if transport_qk_scale is not None else 1.0
        adapt = _build_transport_adapter(D, qk_scale=qk_scale)
        adapt.load_state_dict(adapt_sd, strict=False)
        adapt.to(device).float().eval()
    else:
        qk_scale = 0.0 if method == "qk_scale_0" else (transport_qk_scale or 1.0)
        adapt = _build_transport_adapter(D, qk_scale=qk_scale)
        adapt.load_state_dict(adapt_sd, strict=False)

        # Override write_alpha for write_alpha_0
        if method == "write_alpha_0" or (write_alpha_multiplier is not None and write_alpha_multiplier == 0):
            adapt.write_logit.data.fill_(-100.0)

        adapt.to(device).float().eval()

    # ---- Force write scale (diagnostic: amplify adapter output) ----
    if force_write_scale is not None and force_write_scale > 0:
        with torch.no_grad():
            # Scale out_proj to make value pathway visible
            adapt.out_proj.weight.data *= force_write_scale
            adapt.out_proj.bias.data *= force_write_scale
            # Force write_alpha to a meaningful level
            adapt.write_logit.data.fill_(10.0)  # sigmoid≈1 → write_alpha≈1e-3
            adapt.null_value.data *= force_write_scale * 10
        print(f"[PM-CTR] FORCE WRITE: out_proj scaled ×{force_write_scale}, "
              f"write_alpha forced to {adapt.write_alpha.item():.6f}")

    info = {
        "adapter_loaded": True,
        "pm_tensors": len(pm_sd),
        "adapt_tensors": len(adapt_sd),
        "write_alpha": adapt.write_alpha.item(),
    }
    return pm, adapt, info


def _build_transport_adapter(D: int, qk_scale: float = 1.0) -> TransportRetrievalAdapter:
    # Must match training config: write_alpha_max=0.01, out_proj_init_std=0.01
    return TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        sinkhorn_iters=5, transport_sigma=0.18,
        transport_qk_scale=qk_scale,
        write_alpha_init=1e-4, write_alpha_max=0.01,
        out_proj_init_std=0.01,
    )


# ===================================================================
#  Scaffold building
# ===================================================================

def build_scaffold_from_lyrics(
    lyrics_text: str,
    text_hidden: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor],
    lyric_only: bool = False,
) -> Dict[str, Any]:
    """Build unit scaffold from lyrics and text encoder output.

    Args:
        lyric_only: If True, only include lyric units (filter out control/silence).

    Returns:
        dict with scaffold data.
    """
    L = text_hidden.shape[1]
    D = text_hidden.shape[-1]
    device = text_hidden.device
    B = text_hidden.shape[0]

    fake_section_ids = torch.zeros(L, dtype=torch.long, device="cpu")
    units, _, debug = parse_lyrics_to_units(lyrics_text, fake_section_ids)
    tcm = debug.get("tag_control_mask", None)
    sc = build_duration_scaffold(units, text_len=L, tag_control_mask=tcm)

    U = len(sc["unit_boundaries"]) - 1
    all_uids = torch.arange(U)
    lyric_unit_mask = sc["lyric_unit_mask"]

    if lyric_only:
        # Select only lyric units
        keep_mask = lyric_unit_mask
        keep_ids = all_uids[keep_mask]
        if keep_ids.numel() == 0:
            raise RuntimeError("lyric_only mode: no lyric units found!")
        selected = keep_ids
    else:
        selected = all_uids

    K = selected.numel()
    c_all = (sc["unit_boundaries"][:-1] + sc["unit_boundaries"][1:]) / 2  # [U]
    mu_all = sc["unit_duration"]  # [U]
    usid_all = sc["unit_section_ids"]  # [U]

    # Select
    c_sel = c_all[selected]
    mu_sel = mu_all[selected]
    usid_sel = usid_all[selected]
    unit_is_lyric_sel = lyric_unit_mask[selected]

    # Renormalize mu to sum to 1.0 for selected units
    mu_sel = mu_sel / (mu_sel.sum() + 1e-10)

    # Pool text_hidden per unit
    eh_f = text_hidden.float()
    token_to_unit = sc["token_to_unit"].to(device)
    lyric_mask_t = sc["lyric_mask"].to(device)
    unit_h_list = []
    for uid in selected.tolist():
        uid_int = int(uid)
        is_lyric_u = lyric_unit_mask[uid_int].item()
        if is_lyric_u:
            token_mask = (token_to_unit == uid_int) & lyric_mask_t
        else:
            token_mask = token_to_unit == uid_int
        if token_mask.any():
            pooled = eh_f[:, token_mask].mean(dim=1)
        else:
            pooled = torch.zeros(B, D, device=eh_f.device, dtype=torch.float32)
        unit_h_list.append(pooled)

    unit_text_hidden = torch.stack(unit_h_list, dim=1)  # [B, K, D]

    # Determine unit types
    unit_types = []
    for uid in selected.tolist():
        uid_int = int(uid)
        u = units[uid_int]
        if u.is_silence:
            unit_types.append("silence")
        elif u.is_control:
            unit_types.append("control")
        else:
            unit_types.append("lyric")

    # Section labels for display
    unit_section_labels = []
    for uid in selected.tolist():
        uid_int = int(uid)
        u = units[uid_int]
        sec = u.section.upper() if u.section else "UNKNOWN"
        unit_section_labels.append(sec)

    # Text preview
    unit_text_preview = []
    for uid in selected.tolist():
        u = units[int(uid)]
        txt = u.text.strip() if u.text else ""
        if len(txt) > 30:
            txt = txt[:27] + "..."
        unit_text_preview.append(txt)

    result = {
        "K": K,
        "U": U,
        "unit_text_hidden": unit_text_hidden,
        "c_unit": c_sel.to(device),
        "mu": mu_sel.to(device),
        "usid": usid_sel.to(device),
        "unit_is_lyric": unit_is_lyric_sel.to(device),
        "unit_types": unit_types,
        "unit_section_labels": unit_section_labels,
        "unit_text_preview": unit_text_preview,
        "selected_indices": selected.tolist(),
        "all_unit_boundaries": sc["unit_boundaries"],
        "all_unit_duration": mu_all,
        "all_lyric_unit_mask": lyric_unit_mask,
        "units": units,
    }
    return result


# ===================================================================
#  Hook installation
# ===================================================================

def install_hooks(
    model: torch.nn.Module,
    pm: Optional[torch.nn.Module],
    adapt: Optional[torch.nn.Module],
    method: str,
    scaffold_cache: Dict[str, Any],
) -> List[torch.utils.hooks.RemovableHandle]:
    """Install pre-hook and inject hook.

    Returns:
        List of hook handles (for cleanup).
    """
    handles = []

    if method == "baseline":
        return handles

    # ---- Pre-hook: capture encoder_hidden_states + build scaffold ----
    lyric_only = (method == "lyric_only_transport")

    def capture_enc_hook(module, inputs, kwargs):
        eh = kwargs.get("encoder_hidden_states", None)
        if eh is None:
            return
        L = eh.shape[1]

        if "scaffold_ready" not in scaffold_cache:
            meta = getattr(model, "_gen_metadata", {})
            lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""
            if lyrics_text and L > 0:
                sc_data = build_scaffold_from_lyrics(
                    lyrics_text, eh.float(),
                    kwargs.get("encoder_attention_mask", None),
                    lyric_only=lyric_only,
                )
                # Move non-tensor fields first
                for k in list(sc_data.keys()):
                    if isinstance(sc_data[k], torch.Tensor):
                        sc_data[k] = sc_data[k].to(eh.device)
                scaffold_cache.update(sc_data)
                scaffold_cache["scaffold_ready"] = True
                scaffold_cache["text_hidden"] = eh.float()

                is_l = sc_data["unit_is_lyric"]
                print(f"[PM-CTR] Scaffold built: K={sc_data['K']} units ({is_l.sum().item()} lyric, "
                      f"{sc_data['K'] - is_l.sum().item()} control/silence) from L={L} tokens")

    pre_handle = model.decoder.register_forward_pre_hook(capture_enc_hook, with_kwargs=True)
    handles.append(pre_handle)

    # ---- Inject hook: transport retrieval at layer 12 ----
    def inject_hook(module, inputs, output):
        h = output[0].float()
        B, T = h.shape[:2]
        dev = h.device

        K = scaffold_cache.get("K", 0)
        if K == 0 or not scaffold_cache.get("scaffold_ready", False):
            return (h.to(dtype=output[0].dtype), *output[1:])

        # Gather inputs
        unit_text_hidden = scaffold_cache["unit_text_hidden"].to(dev)
        c_unit_b = scaffold_cache["c_unit"].unsqueeze(0).expand(B, -1)
        mu_b = scaffold_cache["mu"].unsqueeze(0).expand(B, -1)
        usid_b = scaffold_cache["usid"].unsqueeze(0).expand(B, -1)
        unit_is_lyric_b = scaffold_cache["unit_is_lyric"].unsqueeze(0).expand(B, -1)
        p_audio = torch.linspace(0, 1, T, device=dev, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            pm_state = pm(h, None)
            delta_h, Pi, diag = adapt(
                hidden_states=h,
                text_hidden=unit_text_hidden,
                pm_state=pm_state,
                p_audio=p_audio,
                unit_text_hidden=unit_text_hidden,
                unit_c_pos=c_unit_b,
                unit_mass=mu_b,
                unit_section_id=usid_b,
                unit_is_lyric=unit_is_lyric_b,
            )

            # Capture diagnostics
            scaffold_cache["Pi"] = Pi.detach().cpu()
            scaffold_cache["pm_state"] = pm_state.detach().cpu()
            scaffold_cache["p_audio"] = p_audio[0].detach().cpu()
            scaffold_cache["batch_diag"] = diag
            scaffold_cache["R_std"] = diag.get("qk_std", 0.0)
            scaffold_cache["base_logit_std"] = diag.get("base_logit_std", 0.0)
            scaffold_cache["transport_qk_ratio"] = diag.get("transport_qk_ratio", 0.0)
            scaffold_cache["write_ratio"] = diag.get("write_ratio", 0.0)

            h_new = h + delta_h

        return (h_new.to(dtype=output[0].dtype), *output[1:])

    hook_handle = model.decoder.layers[12].register_forward_hook(inject_hook)
    handles.append(hook_handle)
    print(f"[PM-CTR] Layer 12 {method} hook installed")
    return handles


# ===================================================================
#  Diagnostics saving
# ===================================================================

def save_transport_units_csv(
    scaffold_cache: Dict[str, Any],
    sample_id: int,
    method: str,
    seed: int,
    output_dir: str,
):
    """Save per-unit transport diagnostics to CSV."""
    K = scaffold_cache.get("K", 0)
    if K == 0:
        return

    Pi = scaffold_cache.get("Pi")  # [B, T, K]
    if Pi is None:
        return

    Pi_0 = Pi[0]  # [T, K]
    T = Pi_0.shape[0]
    nu = 1.0 / T
    mu_target = scaffold_cache["mu"].cpu()  # [K]
    p_audio = scaffold_cache["p_audio"].cpu()  # [T]

    actual_usage = Pi_0.sum(dim=0)  # [K] = sum_t Pi_{t,k}
    expected_usage = mu_target * T * nu  # each column should get mu mass

    center_time = []
    peak_time = []
    for k in range(K):
        col = Pi_0[:, k]
        if col.sum() > 1e-10:
            c = (col * p_audio).sum() / col.sum()
        else:
            c = 0.0
        center_time.append(c.item())
        peak_time.append(p_audio[col.argmax()].item())

    unit_types = scaffold_cache.get("unit_types", ["lyric"] * K)
    unit_section_labels = scaffold_cache.get("unit_section_labels", [""] * K)
    unit_text_preview = scaffold_cache.get("unit_text_preview", [""] * K)
    unit_is_lyric = scaffold_cache.get("unit_is_lyric", torch.ones(K, dtype=torch.bool))
    c_unit = scaffold_cache.get("c_unit", torch.zeros(K))

    csv_path = os.path.join(output_dir, "transport_units.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "method", "seed",
            "unit_idx", "unit_type", "section", "text_preview",
            "target_mu", "actual_usage", "usage_ratio",
            "center_time", "peak_time", "unit_coordinate",
            "is_lyric", "is_control", "is_silence",
        ])
        for k in range(K):
            ut = unit_types[k] if k < len(unit_types) else "lyric"
            sec = unit_section_labels[k] if k < len(unit_section_labels) else ""
            txt = unit_text_preview[k] if k < len(unit_text_preview) else ""
            is_l = bool(unit_is_lyric[k]) if k < len(unit_is_lyric) else False
            is_sil = ut == "silence"
            is_ctrl = ut == "control"
            ratio = (actual_usage[k] / (expected_usage[k] + 1e-10)).item()
            writer.writerow([
                sample_id, method, seed,
                k, ut, sec, txt,
                f"{expected_usage[k].item():.6f}",
                f"{actual_usage[k].item():.6f}",
                f"{ratio:.4f}",
                f"{center_time[k]:.6f}",
                f"{peak_time[k]:.6f}",
                f"{c_unit[k].item() if k < len(c_unit) else 0:.6f}",
                int(is_l), int(is_ctrl), int(is_sil),
            ])

    print(f"[PM-CTR] Transport units CSV saved: {csv_path}")


def save_transport_summary_json(
    scaffold_cache: Dict[str, Any],
    sample_id: int,
    method: str,
    seed: int,
    output_dir: str,
):
    """Save transport summary JSON."""
    K = scaffold_cache.get("K", 0)
    Pi = scaffold_cache.get("Pi")
    if Pi is None or K == 0:
        return

    Pi_0 = Pi[0]
    T = Pi_0.shape[0]
    nu = 1.0 / T
    mu_target = scaffold_cache["mu"].cpu()
    unit_is_lyric = scaffold_cache.get("unit_is_lyric", torch.ones(K, dtype=torch.bool))
    unit_types = scaffold_cache.get("unit_types", ["lyric"] * K)
    unit_section_labels = scaffold_cache.get("unit_section_labels", [""] * K)

    actual_usage = Pi_0.sum(dim=0)  # [K] total mass received per column
    expected_usage = mu_target * T * nu  # [K] target mass = mu_target (since nu=1/T)
    usage_error = (actual_usage - expected_usage).abs()
    usage_ratio = actual_usage / (expected_usage + 1e-10)

    # Fractions (normalized by T) — comparable across Sinkhorn and softmax
    # For Sinkhorn: actual_usage ≈ mu_target, so frac ≈ mu_target/T
    # For softmax:  actual_usage ≈ T*avg_attn, so frac ≈ avg_attn
    actual_frac = actual_usage / T

    # By category
    lyric_mask_t = torch.tensor([ut == "lyric" for ut in unit_types], dtype=torch.bool)
    control_mask_t = torch.tensor([ut == "control" for ut in unit_types], dtype=torch.bool)
    silence_mask_t = torch.tensor([ut == "silence" for ut in unit_types], dtype=torch.bool)

    total_lyric_usage = actual_usage[lyric_mask_t].sum().item() if lyric_mask_t.any() else 0.0
    total_control_usage = actual_usage[control_mask_t].sum().item() if control_mask_t.any() else 0.0
    total_silence_usage = actual_usage[silence_mask_t].sum().item() if silence_mask_t.any() else 0.0

    # Normalized fractions (= average per-time-step mass)
    lyric_frac = actual_frac[lyric_mask_t].sum().item() if lyric_mask_t.any() else 0.0
    control_frac = actual_frac[control_mask_t].sum().item() if control_mask_t.any() else 0.0
    silence_frac = actual_frac[silence_mask_t].sum().item() if silence_mask_t.any() else 0.0

    # By section (raw and frac)
    intro_usage = torch.tensor(
        [actual_usage[k].item() for k in range(K) if k < len(unit_section_labels) and unit_section_labels[k] == "INTRO"],
        dtype=torch.float32
    ).sum().item()
    inst_usage = torch.tensor(
        [actual_usage[k].item() for k in range(K) if k < len(unit_section_labels) and unit_section_labels[k] == "INSTRUMENTAL"],
        dtype=torch.float32
    ).sum().item()
    outro_usage = torch.tensor(
        [actual_usage[k].item() for k in range(K) if k < len(unit_section_labels) and unit_section_labels[k] == "OUTRO"],
        dtype=torch.float32
    ).sum().item()

    skipped = (usage_ratio < 0.3).sum().item()
    overused = (usage_ratio > 2.0).sum().item()

    # Monotonic center violations
    p_audio = scaffold_cache["p_audio"].cpu()
    b_i = torch.zeros(T)
    for k in range(K):
        col = Pi_0[:, k]
        if col.sum() > 1e-10:
            normalized = col / col.sum()
            b_i += normalized * c_unit_k(k, scaffold_cache)

    # Actually compute b_i more efficiently
    nu_t = torch.full((T,), nu)
    normalized_plan = Pi_0 / nu_t.unsqueeze(-1).clamp(min=1e-10)  # [T, K]
    c_unit_t = scaffold_cache.get("c_unit", torch.zeros(K)).cpu()
    b_i = (normalized_plan * c_unit_t.unsqueeze(0)).sum(dim=1)  # [T]
    monotonic_violations = (b_i[1:] < b_i[:-1] - 1e-6).sum().item()

    # Row/col errors
    row_sums = Pi_0.sum(dim=1)
    row_error = (row_sums - nu).abs().mean().item()
    col_sums = Pi_0.sum(dim=0)
    col_error = (col_sums - expected_usage).abs().mean().item()

    # Diagnostics
    diag = scaffold_cache.get("batch_diag", {})
    qk_std = diag.get("qk_std", 0.0)
    tqr = diag.get("transport_qk_ratio", 0.0)
    wr = diag.get("write_ratio", 0.0)

    summary = {
        "sample_id": sample_id,
        "method": method,
        "seed": seed,
        "total_lyric_usage": round(total_lyric_usage, 6),
        "total_control_usage": round(total_control_usage, 6),
        "total_silence_usage": round(total_silence_usage, 6),
        "intro_usage": round(intro_usage, 6),
        "inst_usage": round(inst_usage, 6),
        "outro_usage": round(outro_usage, 6),
        "lyric_frac": round(lyric_frac, 6),
        "control_frac": round(control_frac, 6),
        "silence_frac": round(silence_frac, 6),
        "skipped_units": int(skipped),
        "overused_units": int(overused),
        "monotonic_center_violations": int(monotonic_violations),
        "mean_abs_usage_error": round(usage_error.mean().item(), 6),
        "max_abs_usage_error": round(usage_error.max().item(), 6),
        "row_error": round(row_error, 6),
        "col_error": round(col_error, 6),
        "qk_std": round(qk_std, 6),
        "transport_qk_ratio": round(tqr, 6),
        "write_ratio": round(wr, 6),
    }

    json_path = os.path.join(output_dir, "transport_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[PM-CTR] Transport summary saved: {json_path}")


def c_unit_k(k, scaffold_cache):
    cu = scaffold_cache.get("c_unit")
    if cu is not None and k < len(cu):
        return cu[k].item()
    return 0.0


# ===================================================================
#  Visualization
# ===================================================================

def save_transport_heatmap(
    scaffold_cache: Dict[str, Any],
    output_dir: str,
):
    """Save transport plan heatmap."""
    Pi = scaffold_cache.get("Pi")
    if Pi is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping heatmap")
        return

    Pi_0 = Pi[0].numpy()  # [T, K]
    K = Pi_0.shape[1]

    # Build unit tags
    unit_types = scaffold_cache.get("unit_types", ["?"] * K)
    unit_section_labels = scaffold_cache.get("unit_section_labels", [""] * K)

    tags = []
    for k in range(K):
        sec = unit_section_labels[k] if k < len(unit_section_labels) else ""
        ut = unit_types[k] if k < len(unit_types) else ""
        if ut == "lyric":
            tags.append(f"L{k}")
        elif ut == "silence":
            if sec == "INTRO":
                tags.append("Intro")
            elif sec == "INSTRUMENTAL":
                tags.append("Inst")
            elif sec == "OUTRO":
                tags.append("Outro")
            else:
                tags.append(f"S{k}")
        elif ut == "control":
            tags.append(f"C{k}")
        else:
            tags.append(f"U{k}")

    T = Pi_0.shape[0]
    # Subsample T for display if too large
    step = max(1, T // 512)
    if step > 1:
        # Average pooling
        T_disp = T // step
        Pi_disp = Pi_0[:T_disp * step].reshape(T_disp, step, K).mean(axis=1)
    else:
        Pi_disp = Pi_0
        T_disp = T

    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 6))
    im = ax.imshow(Pi_disp.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Audio position bin")
    ax.set_ylabel("Condition unit")
    ax.set_yticks(range(K))
    ax.set_yticklabels(tags, fontsize=8)
    ax.set_title("Transport Plan Π (T × K)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    path = os.path.join(output_dir, "transport_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PM-CTR] Heatmap saved: {path}")


def save_condition_usage_histogram(
    scaffold_cache: Dict[str, Any],
    output_dir: str,
):
    """Save condition usage histogram."""
    Pi = scaffold_cache.get("Pi")
    if Pi is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping histogram")
        return

    Pi_0 = Pi[0].numpy()
    K = Pi_0.shape[1]
    nu = 1.0 / Pi_0.shape[0]
    mu_target = scaffold_cache["mu"].cpu().numpy()

    actual_usage = Pi_0.sum(axis=0)  # [K]
    expected_usage = mu_target * Pi_0.shape[0] * nu

    fig, ax = plt.subplots(figsize=(max(6, K * 0.4), 5))
    x = np.arange(K)
    width = 0.35
    bars1 = ax.bar(x - width / 2, actual_usage, width, label="Actual usage", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, expected_usage, width, label="Target (mu)", color="coral", alpha=0.6)

    ax.set_xlabel("Unit index")
    ax.set_ylabel("Mass")
    ax.set_title("Condition Usage: Actual vs Target")
    ax.legend()
    ax.set_xticks(x)

    path = os.path.join(output_dir, "condition_usage_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PM-CTR] Usage histogram saved: {path}")


def save_lyric_center_curve(
    scaffold_cache: Dict[str, Any],
    output_dir: str,
):
    """Save lyric center curve b_i = sum_j normalized_plan_ij * c_j."""
    Pi = scaffold_cache.get("Pi")
    if Pi is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping center curve")
        return

    Pi_0 = Pi[0].numpy()
    T, K = Pi_0.shape
    nu = 1.0 / T
    c_unit = scaffold_cache.get("c_unit", torch.zeros(K)).cpu().numpy()

    # b_i = sum_j (Pi_ij / nu_i) * c_j
    normalized_plan = Pi_0 / nu  # [T, K]
    b_i = (normalized_plan * c_unit[np.newaxis, :]).sum(axis=1)  # [T]

    p_audio = scaffold_cache["p_audio"].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p_audio, b_i, "b-", linewidth=1, alpha=0.8, label="b_i (expected coordinate)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Identity")
    ax.set_xlabel("Audio position p_i")
    ax.set_ylabel("Expected condition coordinate b_i")
    ax.set_title("Lyric Center Curve: b_i vs p_i")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    path = os.path.join(output_dir, "lyric_center_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PM-CTR] Center curve saved: {path}")


# ===================================================================
#  Main entry point
# ===================================================================

def run_inference(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    params: GenerationParams,
    config: GenerationConfig,
    method: str,
    ckpt_path: str,
    output_dir: str,
    sample_id: int = 0,
    seed: int = 42,
    save_transport_diagnostics: bool = True,
    save_heatmap: bool = True,
    transport_qk_scale: Optional[float] = None,
    write_alpha_multiplier: Optional[float] = None,
    force_write_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Run inference with the given method and save results."""
    model = dit_handler.model.eval()

    # Store lyrics for hook access
    model._gen_metadata = {"lyrics": params.lyrics}

    # Load adapter
    pm, adapt, adapt_info = load_adapter(
        model, method, ckpt_path, model.device,
        transport_qk_scale=transport_qk_scale,
        write_alpha_multiplier=write_alpha_multiplier,
        force_write_scale=force_write_scale,
    )

    # Check adapter loading
    if method != "baseline" and not adapt_info.get("adapter_loaded", False):
        raise RuntimeError(f"Adapter loading failed for method={method}, checkpoint={ckpt_path}")

    # Register adapter modules
    if pm is not None:
        model.add_module("transport_pm", pm)
    if adapt is not None:
        model.add_module("transport_adapter", adapt)

    # Print config
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Adapter loaded: {adapt_info.get('adapter_loaded', False)}")
    print(f"Use transport retrieval: {method not in ('baseline',)}")
    print(f"Use control units: {method not in ('baseline', 'lyric_only_transport')}")
    print(f"Retrieval adapter dim: 256")
    adapt_qk = getattr(adapt, "transport_qk_scale", None) if adapt is not None else None
    print(f"Transport qk scale: {adapt_qk if adapt_qk is not None else 'N/A'}")
    adapt_wa = getattr(adapt, "write_alpha", None) if adapt is not None else None
    print(f"Write alpha: {adapt_wa.item() if adapt_wa is not None else 'N/A'}")
    print(f"Write alpha multiplier: {write_alpha_multiplier}")
    print(f"Sinkhorn iters: {getattr(adapt, 'sinkhorn_iters', 'N/A') if adapt is not None else 'N/A'}")
    print("=" * 60)

    # Install hooks
    scaffold_cache: Dict[str, Any] = {}
    handles = install_hooks(model, pm, adapt, method, scaffold_cache)

    # Softmax retrieval: monkey-patch log_sinkhorn before generation
    if method == "softmax_retrieval":
        pm_module.log_sinkhorn = _softmax_retrieval

    # Generate
    os.makedirs(output_dir, exist_ok=True)
    try:
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=output_dir,
        )
    finally:
        # Always restore original log_sinkhorn
        pm_module.log_sinkhorn = _ORIGINAL_LOG_SINKHORN

    # Clean hooks
    for h in handles:
        h.remove()
    # Remove adapter modules
    if pm is not None and hasattr(model, "transport_pm"):
        del model.transport_pm
    if adapt is not None and hasattr(model, "transport_adapter"):
        del model.transport_adapter

    if not result.success:
        return {"success": False, "error": result.error}

    # ---- Save config ----
    inference_config = {
        "method": method,
        "checkpoint_path": ckpt_path,
        "adapter_loaded": adapt_info.get("adapter_loaded", False),
        "seed": seed,
        "sample_id": sample_id,
        "caption": params.caption,
        "duration": params.duration,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "inference_steps": params.inference_steps,
        "guidance_scale": params.guidance_scale,
        "transport_qk_scale_override": transport_qk_scale,
        "write_alpha_multiplier": write_alpha_multiplier,
        "force_write_scale": force_write_scale,
    }
    config_path = os.path.join(output_dir, "inference_config.json")
    with open(config_path, "w") as f:
        json.dump(inference_config, f, indent=2, ensure_ascii=False)

    # ---- Save transport diagnostics ----
    if save_transport_diagnostics and method in (
        "full_pmctr", "lyric_only_transport", "softmax_retrieval", "qk_scale_0"
    ):
        save_transport_units_csv(scaffold_cache, sample_id, method, seed, output_dir)
        save_transport_summary_json(scaffold_cache, sample_id, method, seed, output_dir)

    # ---- Save visualizations ----
    if save_heatmap and method in (
        "full_pmctr", "lyric_only_transport", "softmax_retrieval", "qk_scale_0"
    ):
        save_transport_heatmap(scaffold_cache, output_dir)
        save_condition_usage_histogram(scaffold_cache, output_dir)
        save_lyric_center_curve(scaffold_cache, output_dir)

    return {
        "success": True,
        "audio_path": result.audios[0]["path"] if result.audios else None,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="ACE-Step PM-CTR 统一推理入口")
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "lyric_only_transport", "full_pmctr",
                                 "softmax_retrieval", "qk_scale_0", "write_alpha_0"])
    parser.add_argument("--checkpoint-path", type=str,
                        default="/tmp/transport_1epoch_v5_control/final/pm_retrieval.pt")
    parser.add_argument("--lyrics-file", type=str, default=None,
                        help="Path to lyrics text file. If not given, uses default lyrics.")
    parser.add_argument("--output-dir", type=str, default="./eval_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-off", action="store_true", default=True)
    parser.add_argument("--transport-qk-scale", type=float, default=None,
                        help="Override transport qk scale")
    parser.add_argument("--write-alpha-multiplier", type=float, default=None,
                        help="Override writer alpha multiplier")
    parser.add_argument("--save-transport-diagnostics", action="store_true", default=True)
    parser.add_argument("--save-heatmap", action="store_true", default=True)
    parser.add_argument("--force-write-scale", type=float, default=None,
                        help="Diagnostic: amplify adapter output by this factor")
    parser.add_argument("--sample-id", type=int, default=0)
    args = parser.parse_args()

    # Default lyrics
    default_lyrics = """[INTRO]

[VERSE]
爱总忽然退潮 心慌乱触礁
沉没在深海里 看海面闪耀
但回忆像水草 紧紧的缠绕
梦才温热眼角 就冰冷掉
努力越过风暴 向着未来飘
我们才会遇到 感动的拥抱
你总是能知道 我的坚强剩多少

[PRECHORUS]
给我最刚好的依靠

[CHORUS]
你手心的太阳 只轻放在我背上
委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享 而不是漫长

[INSTRUMENTAL]

[VERSE]
让眼睛看不到 嫉妒的燃烧
让耳朵听不到 谎言的吵闹
再没有人相信 爱能永恒那一秒
我们正坚定的微笑

[CHORUS]
你手心的太阳 有种安定的力量
就算世界再乱我也 不心慌
我手心的太阳 或许只像个月亮
却用所有爱 为你投射我最暖的光芒
你手心的太阳 只轻放在我背上

[INSTRUMENTAL]

[BRIDGE]
委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享
你手心的太阳 有种安定的力量

[CHORUS]
就算世界再乱我也 不心慌
我手心的太阳 或许只像个月亮
却用所有爱 为你投射我 最暖的光芒

[OUTRO]
"""

    if args.lyrics_file and os.path.exists(args.lyrics_file):
        with open(args.lyrics_file, "r") as f:
            lyrics = f.read()
    else:
        lyrics = default_lyrics

    # ========== 1. Initialize handlers ==========
    print("\n[1/5] Initializing handlers...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # ========== 2. Initialize DiT ==========
    print("\n[2/5] Initializing DiT model...")
    dit_status, dit_success = dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not dit_success:
        print(f"[FAIL] DiT init failed: {dit_status}")
        sys.exit(1)

    # ========== 3. Initialize 5Hz LM ==========
    print("\n[3/5] Initializing 5Hz language model...")
    if not args.llm_off:
        llm_success = llm_handler.initialize(
            checkpoint_dir=str(MODEL_ROOT),
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="pt",
            device="cuda",
        )
        if not llm_success:
            print("[FAIL] 5Hz LM init failed")
            sys.exit(1)
    else:
        print("[SKIP] LLM disabled (--llm-off)")

    # ========== 4. Configure generation ==========
    print("\n[4/5] Configuring generation...")

    params = GenerationParams(
        task_type="text2music",
        caption="ballad, pop, female vocal, piano, strings, drums, romantic, melancholic, 137 bpm, E major",
        lyrics=lyrics,
        instrumental=False,
        bpm=137,
        keyscale="E major",
        timesignature="4",
        vocal_language="zh",
        duration=268,
        inference_steps=50,
        guidance_scale=7.0,
        seed=args.seed,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        lm_temperature=0.0,
    )

    gen_config = GenerationConfig(
        batch_size=1,
        audio_format="flac",
        use_random_seed=False,
    )

    # ========== 5. Run inference ==========
    print(f"\n[5/5] Running inference method={args.method} seed={args.seed}")
    print("=" * 70)

    output_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(output_dir, exist_ok=True)

    result = run_inference(
        dit_handler=dit_handler,
        llm_handler=llm_handler if not args.llm_off else None,
        params=params,
        config=gen_config,
        method=args.method,
        ckpt_path=args.checkpoint_path,
        output_dir=output_dir,
        sample_id=args.sample_id,
        seed=args.seed,
        save_transport_diagnostics=args.save_transport_diagnostics,
        save_heatmap=args.save_heatmap,
        transport_qk_scale=args.transport_qk_scale,
        write_alpha_multiplier=args.write_alpha_multiplier,
        force_write_scale=args.force_write_scale,
    )

    if result["success"]:
        print(f"\n[OK] Generation success!")
        print(f"Audio: {result.get('audio_path', 'N/A')}")
    else:
        print(f"\n[FAIL] Generation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
