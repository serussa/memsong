"""
FixedLoRAModule -- Corrected adapter training step for ACE-Step V2.

This module contains the ``FixedLoRAModule`` (nn.Module) responsible for
the per-step training logic: CFG dropout, logit-normal timestep sampling,
flow-matching interpolation, and the decoder forward pass.

Also includes small device/dtype/precision helpers used by both the
Fabric and basic training loops.
"""

from __future__ import annotations

import math
import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ACE-Step utilities
from acestep.phase_memory import PhaseMemory
from acestep.training.lora_injection import inject_lora_into_dit
from acestep.training.lora_utils import check_peft_available
from acestep.training.lokr_utils import (
    check_lycoris_available,
    inject_lokr_into_dit,
)

# V2 modules
from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, PhaseMemoryConfigV2, TrainingConfigV2
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training_v2.ui import TrainingUpdate

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2, PhaseMemoryConfigV2]


class _LastLossAccessor:
    """Lightweight wrapper that provides ``[-1]`` and bool access.

    Avoids storing an unbounded list of floats while keeping backward
    compatibility with code that reads ``module.training_losses[-1]``
    or checks ``if module.training_losses:``.
    """

    def __init__(self, module: "FixedLoRAModule") -> None:
        self._module = module
        self._has_value = False

    def append(self, value: float) -> None:
        self._module.last_training_loss = value
        self._has_value = True

    def __getitem__(self, idx: int) -> float:
        if idx == -1 or idx == 0:
            return self._module.last_training_loss
        raise IndexError("only index -1 or 0 is supported")

    def __bool__(self) -> bool:
        return self._has_value

    def __len__(self) -> int:
        return 1 if self._has_value else 0


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        return "16-mixed"
    return "32-true"


# ===========================================================================
# FixedLoRAModule -- corrected training step
# ===========================================================================


class FixedLoRAModule(nn.Module):
    """Adapter training module with corrected timestep sampling and CFG dropout.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.  The training
    step is identical for both -- only the injection and weight format differ.

    Training flow (per step):
        1. Load pre-computed tensors (from ``PreprocessedDataModule``).
        2. Apply **CFG dropout** on ``encoder_hidden_states``.
        3. Sample noise ``x1`` and continuous timestep ``t`` via
           ``sample_timesteps()`` (logit-normal).
        4. Interpolate ``x_t = t * x1 + (1 - t) * x0``.
        5. Forward through decoder, compute flow matching loss.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.adapter_config = adapter_config
        self.adapter_type = training_config.adapter_type
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")

        # LyCORIS network reference (only set for LoKR)
        self.lycoris_net: Any = None
        self.adapter_info: Dict[str, Any] = {}

        # -- Adapter injection -----------------------------------------------
        if self.adapter_type == "lokr":
            self._inject_lokr(model, adapter_config)  # type: ignore[arg-type]
        elif self.adapter_type == "phase_memory":
            self._inject_phase_memory(model)
        else:
            self._inject_lora(model, adapter_config)  # type: ignore[arg-type]

        # Backward-compat alias
        self.lora_info = self.adapter_info

        # Model config (for timestep params read at runtime)
        self.config = model.config

        # -- Null condition embedding for CFG dropout ------------------------
        # ``model.null_condition_emb`` is a Parameter on the top-level model
        # (not the decoder).
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None
            logger.warning(
                "[WARN] model.null_condition_emb not found -- CFG dropout disabled"
            )

        # -- Timestep sampling params ----------------------------------------
        self._timestep_mu = training_config.timestep_mu
        self._timestep_sigma = training_config.timestep_sigma
        self._data_proportion = training_config.data_proportion
        self._cfg_ratio = training_config.cfg_ratio

        # When gradient checkpointing is enabled via wrapper layers that don't
        # expose enable_input_require_grads(), force at least one forward input
        # to require grad so checkpointed segments keep a valid autograd graph.
        self.force_input_grads_for_checkpointing: bool = False

        # Book-keeping -- store only the most recent loss to avoid
        # unbounded memory growth over long training runs.
        self.last_training_loss: float = 0.0

        # Backward-compat: property provides list-like [-1] access
        # for callers that read ``training_losses[-1]``.
        self.training_losses = _LastLossAccessor(self)

    # -----------------------------------------------------------------------
    # Adapter injection helpers
    # -----------------------------------------------------------------------

    def _inject_lora(self, model: nn.Module, cfg: LoRAConfigV2) -> None:
        """Inject LoRA adapters via PEFT.

        Raises:
            RuntimeError: If PEFT is not installed.
        """
        if not check_peft_available():
            raise RuntimeError(
                "PEFT is required for LoRA training but is not installed.\n"
                "Install it with:  uv pip install peft"
            )
        self.model, self.adapter_info = inject_lora_into_dit(model, cfg)
        logger.info(
            "[OK] LoRA injected: %s trainable params",
            f"{self.adapter_info['trainable_params']:,}",
        )

    def _inject_lokr(self, model: nn.Module, cfg: LoKRConfigV2) -> None:
        """Inject LoKR adapters via LyCORIS.

        After injection, explicitly moves the model to the target device
        so that newly created LoKR parameters (which LyCORIS creates on
        CPU) end up on GPU before Fabric wraps the model.

        Raises:
            RuntimeError: If LyCORIS is not installed.
        """
        if not check_lycoris_available():
            raise RuntimeError(
                "LyCORIS is required for LoKR training but is not installed.\n"
                "Install it with:  uv pip install lycoris-lora"
            )
        self.model, self.lycoris_net, self.adapter_info = inject_lokr_into_dit(
            model,
            cfg,
        )
        # LyCORIS creates adapter parameters on CPU.  Move the entire
        # model to the target device so all parameters (including the
        # new LoKR ones) are co-located before Fabric setup.
        self.model = self.model.to(self.device)
        logger.info(
            "[OK] LoKR injected: %s trainable params (moved to %s)",
            f"{self.adapter_info['trainable_params']:,}",
            self.device,
        )

    def _inject_phase_memory(self, model: nn.Module) -> None:
        """Activate PhaseMemory training: freeze all params except PhaseMemory.

        PhaseMemory is already embedded in the DiT layers (no injection needed).
        This method calls ``model.freeze_except_phase_memory()`` to freeze
        everything except the PhaseMemory sub-modules.

        Raises:
            AttributeError: If the model does not support freeze_except_phase_memory.
        """
        if not hasattr(model, "freeze_except_phase_memory"):
            raise AttributeError(
                "Model does not support PhaseMemory training. "
                "Ensure the model has PhaseMemory layers injected in DiT layers."
            )
        trainable, total = model.freeze_except_phase_memory()
        self.model = model
        self.adapter_info = {
            "trainable_params": trainable,
            "total_params": total,
        }
        logger.info(
            "[OK] PhaseMemory training activated: %s trainable / %s total "
            "params (%.2f%%)",
            f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1),
        )

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with corrected timestep sampling + CFG dropout.

        Args:
            batch: Dict with keys ``target_latents``, ``attention_mask``,
                ``encoder_hidden_states``, ``encoder_attention_mask``,
                ``context_latents``.

        Returns:
            Scalar loss tensor (``float32`` for stable backward).
        """
        # Mixed-precision context
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            nb = self.transfer_non_blocking

            target_latents = batch["target_latents"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            attention_mask = batch["attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            encoder_hidden_states = batch["encoder_hidden_states"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            encoder_attention_mask = batch["encoder_attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            context_latents = batch["context_latents"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )

            bsz = target_latents.shape[0]

            # ---- CFG dropout (CORRECTED -- missing in original trainer) ----
            if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    self._null_cond_emb,
                    cfg_ratio=self._cfg_ratio,
                )

            # ---- Flow matching noise ----------------------------------------
            x1 = torch.randn_like(target_latents)  # noise
            x0 = target_latents  # data

            # ---- Continuous timestep sampling (CORRECTED) -------------------
            t, r = sample_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self._data_proportion,
                timestep_mu=self._timestep_mu,
                timestep_sigma=self._timestep_sigma,
                use_meanflow=False,  # r = t for all ACE-Step variants
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # ---- Interpolate x_t -------------------------------------------
            xt = t_ * x1 + (1.0 - t_) * x0
            if self.force_input_grads_for_checkpointing:
                xt = xt.requires_grad_(True)

            # ---- Decoder forward -------------------------------------------
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,  # r = t
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # ---- Flow matching loss ----------------------------------------
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

            # ---- Beat alignment loss (PhaseMemory only) --------------------
            beat_align_lambda = getattr(
                self.training_config, "beat_align_lambda", 0.0
            )
            if beat_align_lambda > 0.0:
                has_bp = batch.get("has_beat_phase")
                if has_bp is not None and has_bp.any().item():
                    beat_phase = batch["beat_phase"].to(
                        self.device, dtype=self.dtype, non_blocking=nb
                    )
                    has_bp = has_bp.to(self.device, non_blocking=nb)
                    phase_loss = self._compute_beat_align_loss(
                        beat_phase, attention_mask, has_bp,
                    )
                    if phase_loss is not None:
                        diffusion_loss = (
                            diffusion_loss
                            + beat_align_lambda * phase_loss
                        )

        # fp32 for stable backward
        diffusion_loss = diffusion_loss.float()
        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss

    # -----------------------------------------------------------------------
    # Beat alignment loss
    # -----------------------------------------------------------------------

    def _compute_beat_align_loss(
        self,
        beat_phase: torch.Tensor,
        attention_mask: torch.Tensor,
        has_beat_phase: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Huber loss between PhaseMemory delta_phi and beat-phase target.

        Uses per-frame complex states stored by PhaseMemory during the
        decoder forward.  beat_phase (at 25 Hz) is down-sampled to the
        patched frame rate to match the internal DiT frame count.

        Args:
            beat_phase: [B, T_latent] beat phase in [0, 1).
            attention_mask: [B, T_latent] padding mask (1=valid).
            has_beat_phase: [B] bool mask per sample.

        Returns:
            Scalar loss tensor, or None if PhaseMemory / z states are absent.
        """
        # locate the PhaseMemory module inside the decoder
        pm: Optional[PhaseMemory] = None
        for module in self.model.modules():
            if isinstance(module, PhaseMemory):
                pm = module
                break
        if pm is None:
            return None

        zr_all = getattr(pm, "last_zr_all", None)
        zi_all = getattr(pm, "last_zi_all", None)
        if zr_all is None or zi_all is None:
            return None

        b = zr_all.shape[0]

        # ---- delta_phi from adjacent frame pairs -------------------------
        # zr_all / zi_all: [B, T_patched, M]
        dot = (
            zr_all[:, 1:, :] * zr_all[:, :-1, :]
            + zi_all[:, 1:, :] * zi_all[:, :-1, :]
        )  # [B, T-1, M]
        cross = (
            zr_all[:, 1:, :] * zi_all[:, :-1, :]
            - zi_all[:, 1:, :] * zr_all[:, :-1, :]
        )  # [B, T-1, M]
        delta_phi = torch.atan2(cross, dot)  # [B, T-1, M]

        # average over latent dimensions -> scalar per frame
        delta_phi_scalar = delta_phi.mean(dim=-1)  # [B, T-1]

        # ---- omega_target from beat_phase ---------------------------------
        # beat_phase [B, T_latent] at 25 Hz  ->  patched frame rate
        patch_size = getattr(self.config, "patch_size", 2)
        T_latent = beat_phase.shape[1]

        # match the padding that the decoder applies before Conv1d
        pad_len = 0
        if T_latent % patch_size != 0:
            pad_len = patch_size - (T_latent % patch_size)
        bp = F.pad(beat_phase, (0, pad_len), mode="constant", value=0.0)

        # average over each patch window  [B, T_patched_eff]
        bp = bp.view(b, -1, patch_size).mean(dim=-1)

        # ---- target angular velocity --------------------------------------
        delta = bp[:, 1:] - bp[:, :-1]  # [B, T_patched_eff - 1]
        # wrap-around correction for beat-phase in [0, 1)
        delta = torch.where(delta < -0.5, delta + 1.0, delta)
        omega_target = 2.0 * math.pi * delta

        # ---- align lengths (both start at frame 1) ------------------------
        min_len = min(delta_phi_scalar.shape[1], omega_target.shape[1])
        delta_phi_scalar = delta_phi_scalar[:, :min_len]
        omega_target = omega_target[:, :min_len]

        # ---- mask: only valid (non-padded) frames -------------------------
        am = F.pad(attention_mask, (0, pad_len), mode="constant", value=0.0)
        am = am.view(b, -1, patch_size).any(dim=-1).float()  # [B, T_patched]
        # each loss term corresponds to the later frame of a pair
        am = am[:, 1:1 + min_len].contiguous()

        # per-sample availability mask
        sample_mask = has_beat_phase.float().unsqueeze(-1)  # [B, 1]

        # ---- Huber loss (delta=0.1, robust to beat-detection errors) ------
        diff = delta_phi_scalar - omega_target
        huber_delta = 0.1
        phase_loss = torch.where(
            diff.abs() < huber_delta,
            0.5 * diff ** 2,
            huber_delta * (diff.abs() - 0.5 * huber_delta),
        )
        phase_loss = (phase_loss * am * sample_mask).sum() / (
            (am * sample_mask).sum().clamp(min=1)
        )
        return phase_loss
