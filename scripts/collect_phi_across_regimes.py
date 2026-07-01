"""
collect_phi_across_regimes.py
=============================
Hook system + regime-specific φ collection for PhaseMemory analysis.

Three regimes:
  A: Simplified decoder rollout (no CFG, no ODE solver)
  B: ODE-only pipeline (full diffusion steps, no CFG)
  C: Full generation pipeline (with CFG + conditioning)

Each regime collects φ = atan2(z_i, z_r) at every diffusion step,
producing a tensor of shape [T_steps, S_tokens, D_mem].

Usage:
    from collect_phi_across_regimes import PhaseMemoryHook, collect_*

    hook = PhaseMemoryHook(pm_module)
    with hook:
        phi = collect_regime_a(model, hook, ...)
    # phi.shape = [T, S, D]
"""

import math
from typing import Optional, Union

import torch
import numpy as np
from pathlib import Path


# ==============================================================================
# HOOK SYSTEM
# ==============================================================================

class PhaseMemoryHook:
    """Context-managed forward hook to capture φ = atan2(z_i, z_r).

    Usage:
        hook = PhaseMemoryHook(pm_module)
        with hook:
            model.generate_audio(...)
        phi = hook.get_phi(unwrap=True)
    """

    def __init__(self, pm_module: torch.nn.Module):
        self.pm_module = pm_module
        self.phi_history: list[torch.Tensor] = []
        self._handle = None

    def _hook_fn(self, mod, inp, out):
        z_r = getattr(mod, "z_r", None)
        z_i = getattr(mod, "z_i", None)
        if z_r is None or z_i is None:
            return
        phi = torch.atan2(z_i.float().detach(), z_r.float().detach())
        # Average over batch dim (B=2 when CFG doubles the batch)
        if phi.dim() == 3:
            phi = phi.mean(dim=0)
        self.phi_history.append(phi.cpu())

    def __enter__(self):
        self.phi_history.clear()
        self._handle = self.pm_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_phi(self, unwrap: bool = True) -> torch.Tensor:
        """Return φ tensor [T, S, D] (float32, CPU)."""
        if len(self.phi_history) == 0:
            raise RuntimeError("No phase snapshots collected — was the model run?")
        phi = torch.stack(self.phi_history).float()
        if unwrap:
            phi = _unwrap_phase(phi, dim=0)
        return phi

    def reset(self):
        self.phi_history.clear()


def _unwrap_phase(phase: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Unwrap phase along specified dimension (2π jump correction)."""
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
# REGIME A: Simplified decoder rollout
# ==============================================================================

def _make_conditioning(
    model: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> dict:
    """Build minimal conditioning tensors for decoder forward pass."""
    torch.manual_seed(seed)
    B = 1
    audio_dim = model.config.audio_acoustic_hidden_dim  # 64
    text_dim = model.config.text_hidden_dim               # 1024

    # Text conditioning (random noise near null embedding)
    text_len, lyric_len = 50, 100
    text_hs = torch.randn(B, text_len, text_dim, dtype=dtype, device=device) * 0.02
    text_am = torch.ones(B, text_len, dtype=dtype, device=device)
    lyric_hs = torch.randn(B, lyric_len, text_dim, dtype=dtype, device=device) * 0.02
    lyric_am = torch.ones(B, lyric_len, dtype=dtype, device=device)

    # No reference audio (zeros)
    ref_audio = torch.zeros(1, seq_len, audio_dim, dtype=dtype, device=device)
    ref_mask = torch.zeros(1, dtype=torch.long, device=device)

    # Source latents
    src = torch.randn(B, seq_len, audio_dim, dtype=dtype, device=device)
    attn_mask = torch.ones(B, seq_len, dtype=dtype, device=device)
    chunk = torch.ones(B, seq_len, audio_dim, dtype=dtype, device=device)
    silence = torch.randn(B, seq_len, audio_dim, dtype=dtype, device=device)
    is_covers = torch.zeros(B, dtype=torch.long, device=device)

    return {
        "text_hidden_states": text_hs,
        "text_attention_mask": text_am,
        "lyric_hidden_states": lyric_hs,
        "lyric_attention_mask": lyric_am,
        "refer_audio_acoustic_hidden_states_packed": ref_audio,
        "refer_audio_order_mask": ref_mask,
        "src_latents": src,
        "attention_mask": attn_mask,
        "chunk_masks": chunk,
        "silence_latent": silence,
        "is_covers": is_covers,
    }


def collect_regime_a(
    model: torch.nn.Module,
    hook: PhaseMemoryHook,
    num_steps: int = 50,
    seq_len: int = 750,
    seed: int = 42,
) -> torch.Tensor:
    """
    Regime A: Simplified decoder rollout.

    Runs a pure decoder denoising loop WITHOUT:
    - Full generate_audio pipeline
    - CFG (classifier-free guidance)
    - ODE solver trajectory coupling

    This isolates PhaseMemory's intrinsic dynamics from pipeline artifacts.
    """
    from acestep.phase_memory import reset_phase_memory

    device = next(model.parameters()).device
    model_dtype = model.dtype

    # Reset PhaseMemory
    reset_phase_memory(model)

    # Build conditioning
    cond = _make_conditioning(model, seq_len, device, model_dtype, seed)

    # Prepare encoder hidden states + context latents
    with torch.no_grad():
        enc_hs, enc_am, context_latents = model.prepare_condition(
            text_hidden_states=cond["text_hidden_states"],
            text_attention_mask=cond["text_attention_mask"],
            lyric_hidden_states=cond["lyric_hidden_states"],
            lyric_attention_mask=cond["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=cond["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=cond["refer_audio_order_mask"],
            hidden_states=cond["src_latents"],
            attention_mask=cond["attention_mask"],
            silence_latent=cond["silence_latent"],
            src_latents=cond["src_latents"],
            chunk_masks=cond["chunk_masks"],
            is_covers=cond["is_covers"],
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )

    # Build timesteps (linear schedule)
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=model_dtype)

    # Initial noise
    noise = model.prepare_noise(context_latents, seed)
    xt = noise.clone()

    # Run decoder loop
    B = context_latents.shape[0]
    model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t_curr = t[i]
            t_next = t[i + 1]

            t_tensor = t_curr.expand(B)
            t_next_tensor = t_next.expand(B)

            decoder_outputs = model.decoder(
                hidden_states=xt,
                timestep=t_tensor,
                timestep_r=t_next_tensor,
                attention_mask=cond["attention_mask"],
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_am,
                context_latents=context_latents,
                use_cache=True,
                beat_phase=None,
            )

            vt = decoder_outputs[0]
            # Euler step: x_{t+1} = x_t - v_t * dt
            dt = (t_curr - t_next).unsqueeze(-1).unsqueeze(-1)
            xt = xt - vt * dt

    phi = hook.get_phi(unwrap=True)
    return phi


# ==============================================================================
# REGIME B: ODE-only pipeline (no CFG)
# ==============================================================================

def collect_regime_b(
    model: torch.nn.Module,
    hook: PhaseMemoryHook,
    num_steps: int = 50,
    seq_len: int = 750,
    seed: int = 42,
) -> torch.Tensor:
    """
    Regime B: ODE-only pipeline.

    Calls model.generate_audio() with guidance_scale=1.0 (disables CFG).
    Includes full diffusion schedule, ODE solver, but no CFG branching.
    """
    from acestep.phase_memory import reset_phase_memory

    device = next(model.parameters()).device
    model_dtype = model.dtype

    reset_phase_memory(model)

    cond = _make_conditioning(model, seq_len, device, model_dtype, seed)

    model.eval()
    with torch.no_grad():
        _ = model.generate_audio(
            text_hidden_states=cond["text_hidden_states"],
            text_attention_mask=cond["text_attention_mask"],
            lyric_hidden_states=cond["lyric_hidden_states"],
            lyric_attention_mask=cond["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=cond["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=cond["refer_audio_order_mask"],
            src_latents=cond["src_latents"],
            chunk_masks=cond["chunk_masks"],
            is_covers=cond["is_covers"],
            silence_latent=cond["silence_latent"],
            attention_mask=cond["attention_mask"],
            seed=seed,
            infer_method="ode",
            use_cache=True,
            infer_steps=num_steps,
            diffusion_guidance_sale=1.0,   # <-- disables CFG
            audio_cover_strength=1.0,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            use_progress_bar=False,
            use_adg=False,
            shift=1.0,
        )

    phi = hook.get_phi(unwrap=True)
    return phi


# ==============================================================================
# REGIME C: Full generation pipeline (with CFG)
# ==============================================================================

def collect_regime_c(
    model: torch.nn.Module,
    hook: PhaseMemoryHook,
    num_steps: int = 50,
    seq_len: int = 750,
    guidance_scale: float = 7.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Regime C: Full generation pipeline.

    Calls model.generate_audio() with CFG enabled (guidance_scale > 1).
    Batch dimension is doubled internally (cond + uncond).
    Hook averages over batch to maintain [T, S, D] output.
    """
    from acestep.phase_memory import reset_phase_memory

    device = next(model.parameters()).device
    model_dtype = model.dtype

    reset_phase_memory(model)

    cond = _make_conditioning(model, seq_len, device, model_dtype, seed)

    model.eval()
    with torch.no_grad():
        _ = model.generate_audio(
            text_hidden_states=cond["text_hidden_states"],
            text_attention_mask=cond["text_attention_mask"],
            lyric_hidden_states=cond["lyric_hidden_states"],
            lyric_attention_mask=cond["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=cond["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=cond["refer_audio_order_mask"],
            src_latents=cond["src_latents"],
            chunk_masks=cond["chunk_masks"],
            is_covers=cond["is_covers"],
            silence_latent=cond["silence_latent"],
            attention_mask=cond["attention_mask"],
            seed=seed,
            infer_method="ode",
            use_cache=True,
            infer_steps=num_steps,
            diffusion_guidance_sale=guidance_scale,  # > 1.0 enables CFG
            audio_cover_strength=1.0,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            use_progress_bar=False,
            use_adg=False,
            shift=1.0,
        )

    phi = hook.get_phi(unwrap=True)
    return phi


# ==============================================================================
# BATCH COLLECTION (multiple seeds for statistics)
# ==============================================================================

def collect_regime_with_stats(
    collect_fn,
    model: torch.nn.Module,
    pm_module: torch.nn.Module,
    num_steps: int = 50,
    seq_len: int = 750,
    n_seeds: int = 5,
    guidance_scale: float = 7.0,
    label: str = "",
) -> dict:
    """
    Run a collection function across multiple seeds and aggregate.

    Returns:
        dict with:
          - "phis": list of [T, S, D] tensors (one per seed)
          - "phi_mean": [T, S, D] mean over seeds
          - "phi_std":  [T, S, D] std over seeds
          - "label": regime label
    """
    phis = []
    for seed in range(1, n_seeds + 1):
        hook = PhaseMemoryHook(pm_module)
        with hook:
            if "guidance_scale" in collect_fn.__name__ or collect_fn.__name__ == "collect_regime_c":
                phi = collect_fn(model, hook, num_steps=num_steps,
                                 seq_len=seq_len, seed=seed,
                                 guidance_scale=guidance_scale)
            elif collect_fn.__name__ == "collect_regime_b":
                phi = collect_fn(model, hook, num_steps=num_steps,
                                 seq_len=seq_len, seed=seed)
            else:
                phi = collect_fn(model, hook, num_steps=num_steps,
                                 seq_len=seq_len, seed=seed)
        phis.append(phi)

    stack = torch.stack(phis)  # [N, T, S, D]
    return {
        "label": label,
        "phis": phis,
        "phi_mean": stack.mean(dim=0),
        "phi_std": stack.std(dim=0),
        "n_seeds": n_seeds,
        "shape": list(phis[0].shape),
    }


# ==============================================================================
# PATH HELPER
# ==============================================================================

from pathlib import Path
