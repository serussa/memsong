"""Recurrent Complex Phase State for DiT layers.

Minimal design — one recurrent complex oscillator per sample.

    z_k  = z_{k-1} * exp(i * omega(text))          complex rotation
    h    = h + proj_out(real(z_k))                  inject into hidden

Training: z starts from learnable z_init; differentiable, no cross-batch buffer.
Inference: z persists across denoising steps via buffer with .detach().

Zero-init: proj_out is zero-initialised -> Step 0 injection = 0 -> model output
identical to frozen backbone (no catastrophic forgetting).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class PhaseMemory(nn.Module):
    """Recurrent complex state -- single oscillator per sample.

    Args:
        dim: Hidden dimension (hidden_size).
        mem_dim: Internal complex dimension.  Default = dim.
        init_scale: Std of learnable initial phase (default 0.01).
    """

    def __init__(self, dim: int, mem_dim: int = None, init_scale: float = 0.01):
        super().__init__()
        mem_dim = mem_dim or dim

        self.pm_scale = 1.0
        self.last_signal: torch.Tensor | None = None

        # text -> rotation frequency  omega > 0
        self.proj_omega = nn.Linear(dim, mem_dim)

        # complex -> hidden injection
        self.proj_out = nn.Linear(mem_dim, dim)
        # Backward-compatible alias used by older init code paths.
        self.out = self.proj_out
        # ---- ZERO-INIT (safe Step 0) ----
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        # Flag: prevent HF _init_weights from over-writing zeros on children.
        self.proj_out._pm_safe_output = True

        # ---- learnable canonical initial phase ----
        self.z_init_real = nn.Parameter(torch.randn(1, mem_dim) * init_scale)
        self.z_init_imag = nn.Parameter(torch.randn(1, mem_dim) * init_scale)

        # ---- inference buffer (step-to-step recurrence) ----
        self.register_buffer("z_real", None, persistent=False)
        self.register_buffer("z_imag", None, persistent=False)
        self.register_buffer("state_ready", torch.tensor(False), persistent=False)

    # ----------------------------------------------------------------
    def forward(self, h: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """z <- z * exp(i*omega)  ;  h <- h + proj_out(real(z))."""
        b = h.shape[0]

        if text_emb.ndim == 0:
            text_emb = text_emb.view(1)
        if text_emb.ndim == 1:
            text_emb = text_emb.view(b, 1)
        if text_emb.shape[-1] != self.proj_omega.in_features:
            text_emb = text_emb.to(h.dtype).expand(b, self.proj_omega.in_features)

        # omega = softplus(proj(text)) -- positive frequency
        omega = F.softplus(self.proj_omega(text_emb))          # [B, M]
        cos_w = torch.cos(omega)
        sin_w = torch.sin(omega)

        # ---- state source ----
        if self.training:
            zr = self.z_init_real.expand(b, -1)                # no buffer
            zi = self.z_init_imag.expand(b, -1)
        else:
            if not self.state_ready or self.z_real is None or self.z_real.shape[0] != b:
                self.z_real = self.z_init_real.expand(b, -1)
                self.z_imag = self.z_init_imag.expand(b, -1)
                self.state_ready.fill_(True)
            zr = self.z_real
            zi = self.z_imag

        # ---- complex rotation  z' = z * e^{i*omega} ----
        zr_new = zr * cos_w - zi * sin_w
        zi_new = zr * sin_w + zi * cos_w

        if not self.training:
            self.z_real = zr_new.detach()                      # persist (no grad)
            self.z_imag = zi_new.detach()

        # ---- inject real part ----
        injected = self.proj_out(zr_new) * self.pm_scale       # [B, D]
        self.last_signal = zr_new.detach()
        return h + injected.unsqueeze(1)

    def reset(self) -> None:
        """Reset persistent diffusion-phase state for a fresh inference run."""
        self.z_real = None
        self.z_imag = None
        self.state_ready.fill_(False)


# ==================================================================
# Utilities -- unchanged API for training pipeline
# ==================================================================

def freeze_except_phase_memory(model: nn.Module) -> tuple[int, int]:
    """Freeze all params except those in PhaseMemory sub-modules."""
    trainable = total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "phase_memory" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    logger.info(
        "PhaseMemory: %s / %s params trainable (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1),
    )
    return trainable, total


def unfreeze_all(model: nn.Module) -> None:
    """Restore all parameters to requires_grad=True."""
    for p in model.parameters():
        p.requires_grad = True
    logger.info("All params unfrozen.")


def reset_phase_memory(model: nn.Module) -> None:
    """Reset all PhaseMemory modules inside a model."""
    for module in model.modules():
        if isinstance(module, PhaseMemory):
            module.reset()


def set_phase_memory_scale(model: nn.Module, scale: float) -> None:
    """Set the PhaseMemory injection scale across all modules."""
    for module in model.modules():
        if isinstance(module, PhaseMemory):
            module.pm_scale = float(scale)
