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
from acestep.phase_memory import (
    PhaseMemoryDurationClock, PMRetrievalPhaseMemory, PMDCResidualClock,
    PhaseControlledLyricScaffoldPM,
    LyricRetrievalAdapter, TransportRetrievalAdapter, log_sinkhorn,
    reset_phase_memory, build_duration_scaffold,
    parse_lyrics_to_units, scaffold_progress,
)
from acestep.training.lora_injection import inject_lora_into_dit
from acestep.training.lora_utils import check_peft_available
from acestep.training.lokr_utils import (
    check_lycoris_available,
    inject_lokr_into_dit,
)

# V2 modules
from acestep.training_v2.configs import (
    LoRAConfigV2,
    LoKRConfigV2,
    PhaseMemoryConfigV2,
    PMDCConfigV2,
    TrainingConfigV2,
)
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training_v2.ui import TrainingUpdate

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2, PhaseMemoryConfigV2, PMDCConfigV2]


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


def _grad_norm(module: nn.Module) -> float:
    """Compute L2 gradient norm of all parameters in *module*."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return math.sqrt(total)


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
        elif self.adapter_type == "section_rope":
            self._inject_section_rope(model)
        elif self.adapter_type == "pmdc_clock":
            self._inject_pmdc_clock(model, training_config)
        elif self.adapter_type == "pm_retrieval":
            self._inject_pm_retrieval(model, training_config)
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

        # PhaseMemory diagnostics (updated each step)
        self.pm_diag: Dict[str, float] = {}
        # Section-RoPE diagnostics (updated each step)
        self.section_rope_diag: Dict[str, float] = {}

        # PMDC / Phase-Controlled PM fields (set by _inject_pmdc_clock)
        self.pmdc_clock: Optional[PMDCResidualClock] = None
        self.pmdc_gate_logit: Optional[nn.Parameter] = None
        self.pmdc_scaffold: Optional[dict] = None
        self.pmdc_bias_cache: Optional[torch.Tensor] = None
        self.pmdc_config: Optional[Any] = None
        self.pmdc_diag: Dict[str, float] = {}
        self._pmdc_patch_call_count: int = 0

        # Phase-Controlled Lyric Scaffold PM fields
        self.controlled_pm: Optional[PhaseControlledLyricScaffoldPM] = None
        self.pm_gate_pm: Optional[nn.Parameter] = None
        self.pm_gate_attn: Optional[nn.Parameter] = None
        self._residual_handle: Optional[Any] = None

        # PM + LyricRetrievalAdapter fields (pm_retrieval mode)
        # NOTE: `_pmr` is set in _inject_pm_retrieval on model, not on self.
        self._pmr_step: int = 0
        self._pmr_tracked: Optional[dict] = None
        self.pm_retrieval_scaffold: Optional[dict] = None
        self._pm_retrieval_inject_handle: Optional[Any] = None

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
    # Section-RoPE Offset injection
    # -----------------------------------------------------------------------

    def _inject_section_rope(self, model: nn.Module) -> None:
        """Activate Section-RoPE Offset training: freeze backbone, train only section_rope."""
        if hasattr(model, "config"):
            model.config.use_section_rope_offset = True
            model.config.use_token_weights = True

        # Inject module into layer 12 if not already present (config wasn't set at model init)
        from acestep.tgca.section_rope import SectionRoPEOffset
        layer12 = model.decoder.layers[12]
        if not getattr(layer12, "use_section_rope", False):
            layer12.use_section_rope = True
            layer12.section_rope_offset_module = SectionRoPEOffset(
                num_heads=model.config.num_key_value_heads,
                num_section_types=8, rope_pair_dim=16,
                max_offset=0.03, init_log_scale=-3.5, strength=1.0,
            ).to(model.dtype).to(layer12.self_attn_norm.weight.device)
            layer12.section_rope_time_dim = 32
            layer12.use_phase_memory = False  # must disable PhaseMemory to avoid forward conflict

        trainable_ids: set[int] = set()
        for name, module in model.named_modules():
            if "section_rope_offset_module" in name.split(".") and hasattr(module, "parameters"):
                for param in module.parameters():
                    trainable_ids.add(id(param))

        total_params = 0
        trainable_params = 0
        for param in model.parameters():
            total_params += param.numel()
            if id(param) in trainable_ids:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        self.model = model
        self.adapter_info = {
            "trainable_params": trainable_params,
            "total_params": total_params,
        }
        logger.info(
            "[OK] Section-RoPE Offset training activated: %s trainable / %s total "
            "params (%.2f%%)",
            f"{trainable_params:,}", f"{total_params:,}",
            100 * trainable_params / max(total_params, 1),
        )
        logger.info("[Section-RoPE] use_pm=false  use_pm_kv=false  use_traj=false  use_anchor=false  use_entropy=false  use_kl=false")
        logger.info("[Section-RoPE] section_type_vocab: UNKNOWN=0 INTRO=1 VERSE=2 PRECHORUS=3 CHORUS=4 BRIDGE=5 OUTRO=6 INSTRUMENTAL=7")
        logger.info("[Section-RoPE] rope_pair_dim=16  time_dim=32  max_offset=0.2  init_scale=0.01")

    # -----------------------------------------------------------------------
    # PMDC Residual Clock injection
    # -----------------------------------------------------------------------

    def _inject_pmdc_clock(self, model: nn.Module, cfg: Any) -> None:
        """Create PMDCResidualClock and freeze backbone.

        PMDC is NOT embedded in DiT layers.  It reads layer 12 hidden
        states via forward hook during the warmup pass.  Only the clock
        and gate parameters are trainable.

        ⚠️ Forces ``_attn_implementation='eager'`` so that the PMDC
        attention patch (which replaces ``eager_attention_forward``) is
        actually called.  The default ``sdpa`` implementation bypasses
        the patched function entirely.
        """
        import math

        # CRITICAL: force eager attention so our patch runs
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                old = m.config._attn_implementation
                if old != "eager":
                    m.config._attn_implementation = "eager"
                    logger.info("[PMDC] Forced %s._attn_implementation: %s → eager", type(m).__name__, old)
        if hasattr(model.config, "_attn_implementation_compiled"):
            model.config._attn_implementation_compiled = None
        logger.info("[PMDC] Forced _attn_implementation='eager' (required for attention patch)")

        use_controlled = getattr(cfg, "use_controlled_pm", False)

        if use_controlled:
            # PhaseControlledLyricScaffoldPM (new: hidden residual + p_dyn)
            import math as _math
            clock = PhaseControlledLyricScaffoldPM(
                dim=model.config.hidden_size,
                mem_dim=128,
                coord_dim=32,
                hidden_dim=256,
                beta_max=0.15,
            ).to(self.device).to(self.dtype)
            self.controlled_pm = clock

            # Gate for hidden residual
            self.pm_gate_pm = nn.Parameter(torch.tensor(-3.0, device=self.device, dtype=self.dtype))

            # Gate for attention bias
            attn_logit = _math.log(max(cfg.pmdc_gate_init / (1.0 - cfg.pmdc_gate_init + 1e-6), 1e-6))
            self.pm_gate_attn = nn.Parameter(torch.tensor(attn_logit, device=self.device, dtype=self.dtype))

            trainable_count = sum(p.numel() for p in clock.parameters()) \
                              + self.pm_gate_pm.numel() + self.pm_gate_attn.numel()
            logger.info("[ControlledPM] Created PhaseControlledLyricScaffoldPM: %s params", f"{trainable_count:,}")

        # Freeze backbone FIRST (before adding trainable modules)
        for param in model.parameters():
            param.requires_grad = False

        # THEN register modules (they'll start with grad=True by default)
        if use_controlled:
            model.add_module("controlled_pm", clock)
            model.register_parameter("pm_gate_pm", self.pm_gate_pm)
            model.register_parameter("pm_gate_attn", self.pm_gate_attn)
        else:
            model.add_module("pmdc_clock", clock)
            model.register_parameter("pmdc_gate_logit", gate_logit)

            trainable_count = sum(p.numel() for p in clock.parameters()) + gate_logit.numel()

        frozen_fixed = sum(p.numel() for p in model.parameters())
        total_count = frozen_fixed + trainable_count

        self.model = model
        self.adapter_info = {
            "trainable_params": trainable_count,
            "total_params": total_count,
        }
        self.pmdc_config = cfg

        logger.info(
            "[OK] PMDC Residual Clock injected: %s trainable / %s total "
            "params (%.2f%%) — backbone frozen",
            f"{trainable_count:,}", f"{total_count:,}",
            100 * trainable_count / max(total_count, 1),
        )

    # -----------------------------------------------------------------------
    # PM + LyricRetrievalAdapter injection
    # -----------------------------------------------------------------------

    def _inject_pm_retrieval(self, model: nn.Module, cfg: Any) -> None:
        """Inject PhaseMemory + LyricRetrievalAdapter.

        PhaseMemory provides recurrent pm_state (no hidden residual).
        LyricRetrievalAdapter reads pm_state + scaffold and produces
        a gated hidden residual.  The original cross-attention is untouched.
        """
        import math

        # Force eager attention for consistency
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                old = m.config._attn_implementation
                if old != "eager":
                    m.config._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation_compiled"):
            model.config._attn_implementation_compiled = None

        D = model.config.hidden_size

        # PhaseMemory (recurrent state provider, no residual injection)
        pm_ret = PMRetrievalPhaseMemory(
            dim=D,
            mem_dim=128,
            hidden_dim=256,
            normalize_internal_state=True,
        ).to(self.device).float()

        # ---- Transport vs standard retrieval ----------------------------------
        use_transport = getattr(cfg, 'use_transport_retrieval', False)

        if use_transport:
            # TransportRetrievalAdapter (v5: unit-level Sinkhorn transport)
            adapt = TransportRetrievalAdapter(
                hidden_dim=D,
                text_dim=D,
                pm_dim=256,
                d_r=getattr(cfg, 'retrieval_adapter_dim', 256),
                sinkhorn_iters=getattr(cfg, 'sinkhorn_iters', 5),
                transport_sigma=getattr(cfg, 'transport_sigma', 0.18),
                transport_qk_scale=getattr(cfg, 'transport_qk_scale', 1.0),
                scoring_mode=getattr(cfg, 'scoring_mode', 'classic'),
                use_pm_gate=getattr(cfg, 'use_pm_gate', True),
                gate_hidden_dim=getattr(cfg, 'gate_hidden_dim', 128),
                write_alpha_init=getattr(cfg, 'write_alpha_init', 1e-4),
                write_alpha_max=getattr(cfg, 'write_alpha_max', 1e-3),
                out_proj_init_std=getattr(cfg, 'out_proj_init_std', 0.01),
                residual_scale=getattr(cfg, 'residual_scale', 0.01),
                kl_weight=getattr(cfg, 'kl_weight', 0.001),
            ).to(self.device).float()
            logger.info("[Transport] TransportRetrievalAdapter created: K_dim=%d sinkhorn_iters=%d sigma=%.3f qk_scale=%.2f",
                        64, getattr(cfg, 'sinkhorn_iters', 5),
                        getattr(cfg, 'transport_sigma', 0.18),
                        getattr(cfg, 'transport_qk_scale', 1.0))
        else:
            # LyricRetrievalAdapter (v4 with phase bias + RMS writer)
            adapt = LyricRetrievalAdapter(
                hidden_dim=D,
                text_dim=D,
                pm_dim=256,
                d_r=getattr(cfg, 'retrieval_adapter_dim', 256),
                time_dim=getattr(cfg, 'retrieval_adapter_time_dim', 128),
                # v3 params (kept for compat)
                residual_scale=0.1,
                gamma_init=0.1,
                use_adapter_scaffold_prior=getattr(cfg, 'use_adapter_scaffold_prior', True),
                adapter_prior_sigma=0.18,
                adapter_prior_lambda=0.2,
                adapter_prior_clamp_min=-2.0,
                adapter_prior_dropout=0.3,
                # v4 phase bias
                use_phase_bias=getattr(cfg, 'use_phase_bias', True),
                phase_bias_lambda=getattr(cfg, 'phase_bias_lambda', 0.03),
                phase_offset_max=getattr(cfg, 'phase_offset_max', 0.05),
                phase_num_freqs=getattr(cfg, 'phase_num_freqs', 4),
                phase_bias_sigma=getattr(cfg, 'phase_bias_sigma', 0.18),
                phase_bias_dropout=getattr(cfg, 'phase_bias_dropout', 0.3),
                phase_bias_clamp_min=getattr(cfg, 'phase_bias_clamp_min', -2.0),
                # v4 RMS writer
                use_rms_writer=getattr(cfg, 'use_rms_writer', True),
                write_alpha_init=getattr(cfg, 'write_alpha_init', 1e-4),
                write_alpha_max=getattr(cfg, 'write_alpha_max', 1e-3),
            ).to(self.device).float()

        # Store references on the model object (not self) to avoid nn.Module.__setattr__ interference
        model._pmr = {"pm": pm_ret, "adapter": adapt}

        # Snapshot initial params for delta tracking
        model._pmr_init_params = {}
        for name, p in pm_ret.named_parameters():
            model._pmr_init_params[f"pm.{name}"] = p.data.cpu().clone()
        for name, p in adapt.named_parameters():
            model._pmr_init_params[f"adapter.{name}"] = p.data.cpu().clone()

        trainable_count = sum(p.numel() for p in pm_ret.parameters()) + \
                          sum(p.numel() for p in adapt.parameters())
        logger.info("[PM_Retrieval] PhaseMemory: %s params", f"{sum(p.numel() for p in pm_ret.parameters()):,}")

        if use_transport:
            logger.info("[Transport] TransportRetrievalAdapter: %s params (d_r=%d sinkhorn_iters=%d)",
                        f"{sum(p.numel() for p in adapt.parameters()):,}",
                        getattr(adapt, 'd_r', 64), getattr(adapt, 'sinkhorn_iters', 5))
            logger.info("[Transport] write_alpha_init=%.6f write_alpha_max=%.6f current_alpha=%.8f",
                        getattr(cfg, 'write_alpha_init', 1e-4),
                        getattr(cfg, 'write_alpha_max', 1e-3),
                        adapt.write_alpha.item())
        else:
            logger.info("[PM_Retrieval] qk_score_scale init = %.4f (logit=%.4f, exp=%.4f)",
                        2.0, adapt.logit_score_scale.item(), adapt.qk_score_scale.item())
            logger.info("[PM_Retrieval] LyricRetrievalAdapter: %s params", f"{sum(p.numel() for p in adapt.parameters()):,}")
            logger.info("[PM-R V4] use_phase_bias=%s phase_bias_lambda=%.4f phase_offset_max=%.4f phase_num_freqs=%d",
                        adapt.use_phase_bias, adapt.phase_bias_lambda, adapt.use_rms_writer and adapt.phase_scaffold_warp.phase_offset_max or 0,
                        adapt.phase_scaffold_warp.phase_num_freqs if hasattr(adapt, 'phase_scaffold_warp') else 0)
            logger.info("[PM-R V4] use_rms_writer=%s write_alpha_init=%.6f write_alpha_max=%.6f current_alpha=%.8f",
                        adapt.use_rms_writer, getattr(cfg, 'write_alpha_init', 1e-4),
                        getattr(cfg, 'write_alpha_max', 1e-3),
                        adapt.write_alpha.item())
            # Verify PhaseScaffoldWarp final layer zero-init
            if hasattr(adapt, 'phase_scaffold_warp'):
                last_w = adapt.phase_scaffold_warp.coef_head[-1].weight.norm().item()
                last_b = adapt.phase_scaffold_warp.coef_head[-1].bias.norm().item()
                logger.info("[PM-R V4] PhaseScaffoldWarp coef_head[-1] zero-init check: weight_norm=%.10f bias_norm=%.10f",
                            last_w, last_b)

        # Freeze backbone FIRST
        for param in model.parameters():
            param.requires_grad = False

        # THEN register modules onto model so their params appear in model.parameters()
        model.add_module("pm_retrieval_pm", pm_ret)
        model.add_module("pm_retrieval_adapter", adapt)

        frozen_fixed = sum(p.numel() for p in model.parameters())
        total_count = frozen_fixed + trainable_count

        self.model = model
        self.pmdc_config = cfg
        self.adapter_info = {
            "trainable_params": trainable_count,
            "total_params": total_count,
        }
        logger.info(
            "[OK] PM + Retrieval Adapter injected: %s trainable / %s total "
            "params (%.2f%%) — backbone frozen",
            f"{trainable_count:,}", f"{total_count:,}",
            100 * trainable_count / max(total_count, 1),
        )

    # -----------------------------------------------------------------------
    # PM-Retrieval training step
    # -----------------------------------------------------------------------

    def _pm_retrieval_training_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        section_ids: Optional[torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Training step for PhaseMemory + LyricRetrievalAdapter.

        Flow:
          1. Warmup forward → collect layer 12 hidden state H.
          2. PMRetrievalPhaseMemory(H) → pm_state (differentiable, no detach).
          3. Build scaffold from batch metadata.
          4. LyricRetrievalAdapter(H, pm_state, scaffold, text) → ret_res, diag.
          5. Inject ret_res as hidden residual via hook on layer 12.
          6. Decoder forward with modified hidden → flow_loss.
          7. Backward updates PM + Adapter params only.
        """
        from acestep.phase_memory import build_duration_scaffold, parse_lyrics_to_units, scaffold_progress
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        ADAPTER_LAYER = 12
        bsz = xt.shape[0]
        device = self.device

        # ---- Step 1: Warmup forward → collect H from layer 12 ------------
        hs_list = []
        def _whook(m, i, o):
            hs_list.append(o[0])
        handle = self.model.decoder.layers[ADAPTER_LAYER].register_forward_hook(_whook)
        try:
            _ = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            handle.remove()

        T_h = hs_list[0].shape[1] if hs_list else xt.shape[1]
        H = hs_list[0].float() if hs_list else torch.zeros(bsz, T_h, self.model.config.hidden_size, device=device, dtype=torch.float32)

        # ---- Step 2: PMRetrievalPhaseMemory forward → pm_state ------------
        # Returns [B, T_h, 256] — differentiable, no detach, no residual
        pm_state = self.model._pmr["pm"](H, t)

        # ---- Step 3: Build/load scaffold from batch metadata -------------
        if self.pm_retrieval_scaffold is None:
            meta_list = batch.get("metadata", batch.get("metadatas", []))
            meta = meta_list[0] if isinstance(meta_list, list) and len(meta_list) > 0 else {}
            lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""

            # Generate section_ids if not provided in batch (all-UNKNOWN=0)
            if section_ids is None and lyrics_text:
                L_eff = encoder_hidden_states.shape[1]
                section_ids = torch.zeros(bsz, L_eff, dtype=torch.long, device=device)
            elif section_ids is not None:
                section_ids = section_ids.to(device, non_blocking=self.transfer_non_blocking)

            if lyrics_text and section_ids is not None:
                L_eff = section_ids.shape[-1]
                units, _, debug = parse_lyrics_to_units(
                    lyrics_text, section_ids[0].cpu(),
                    auto_transition_ratios={
                        "intro": 0.0, "outro": 0.0,
                        "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
                        "bridge_to_chorus": 0.0,
                    },
                )
                tcm = debug.get("tag_control_mask", None)
                self.pm_retrieval_scaffold = build_duration_scaffold(
                    units, text_len=L_eff, tag_control_mask=tcm,
                )

        # ---- Step 4: LyricRetrievalAdapter forward ------------------------
        adapt = self.model._pmr["adapter"]
        adapt.train()
        adapter_diag = {}

        if self.pm_retrieval_scaffold is not None:
            sc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.pm_retrieval_scaffold.items()}
            p_audio, c_text, ttid = scaffold_progress(sc, T_h, device=device, batch_size=bsz)
            section_id = torch.zeros(sc["lyric_mask"].shape[-1], device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            ad_time_dim = getattr(adapt, 'time_dim', 128)
            t_emb = torch.zeros(bsz, ad_time_dim, device=device, dtype=torch.float32)
            amask_text = encoder_attention_mask.bool() if encoder_attention_mask is not None else None

            ret_res, attn_r, adapter_diag = adapt(
                hidden_states=H,
                text_hidden=encoder_hidden_states.float(),
                pm_state=pm_state,
                p_audio=p_audio,
                c_text=c_text,
                section_id=section_id,
                token_type_id=ttid,
                timestep_emb=t_emb,
                attention_mask=amask_text,
                use_scaffold_prior=False,
                use_phase_bias=getattr(self.model._pmr["adapter"], 'use_phase_bias', False),
            )
        else:
            ret_res = torch.zeros_like(H)
            attn_r = None

        # v4 RMS writer returns full delta_h; v3 old writer needs gamma_r scaling.
        if getattr(adapt, 'use_rms_writer', False):
            final_h = H + ret_res
        else:
            final_h = H + adapt.gamma_r * ret_res

        # Track residual stats
        with torch.no_grad():
            retrieval_residual_norm = ret_res.norm(dim=-1).mean().item()
            hidden_delta_norm = (final_h - H).norm(dim=-1).mean().item()

        # ---- Step 5: Hook to inject modified hidden at layer 12 -----------
        # Remove any stale handle first
        if hasattr(self, '_pm_retrieval_inject_handle') and self._pm_retrieval_inject_handle is not None:
            self._pm_retrieval_inject_handle.remove()
            self._pm_retrieval_inject_handle = None

        def _inject_hook(m, i, o):
            fh = final_h.to(dtype=o[0].dtype, device=o[0].device)
            return (fh, *o[1:])
        self._pm_retrieval_inject_handle = self.model.decoder.layers[ADAPTER_LAYER].register_forward_hook(_inject_hook)

        # ---- Step 6: Decoder forward with injected hidden → flow_loss ----
        try:
            decoder_outputs = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            if self._pm_retrieval_inject_handle is not None:
                self._pm_retrieval_inject_handle.remove()
                self._pm_retrieval_inject_handle = None

        flow = x1 - x0
        flow_loss = F.mse_loss(decoder_outputs[0], flow)
        loss = flow_loss

        # ---- KL constraint (if adapter computed one) ---------------------------
        if hasattr(adapt, '_last_kl_loss') and adapt._last_kl_loss is not None:
            kl_val = adapt._last_kl_loss
            kl_w = getattr(adapt, 'kl_weight', 0.001)
            loss = loss + kl_w * kl_val
            adapter_diag['kl_loss'] = kl_val.detach().item()

        # ---- Step 7: Parameter tracking (before step 0 clone, after step 20 delta) ---
        pm = self.model._pmr["pm"]
        if self._pmr_step == 0 and self._pmr_tracked is None:
            self._pmr_tracked = {
                "pm_proj_r": pm.proj_r.weight.detach().clone().cpu(),
                "pm_omega": pm.omega.weight.detach().clone().cpu(),
                "q_mlp": adapt.q_mlp[0].weight.detach().clone().cpu(),
                "k_mlp": adapt.k_mlp[0].weight.detach().clone().cpu(),
                "v_mlp": adapt.v_mlp[0].weight.detach().clone().cpu(),
                "out_proj": adapt.out_proj.weight.detach().clone().cpu(),
                "gamma_r": adapt.gamma_r.detach().clone().cpu(),
            }
            if hasattr(adapt, 'write_logit'):
                self._pmr_tracked["write_logit"] = adapt.write_logit.detach().clone().cpu()
            if hasattr(adapt, 'phase_scaffold_warp'):
                psw = adapt.phase_scaffold_warp
                self._pmr_tracked["coef_head_w"] = psw.coef_head[-1].weight.detach().clone().cpu()
                self._pmr_tracked["coef_head_b"] = psw.coef_head[-1].bias.detach().clone().cpu()
            # Re-init pmr_init_params to include v4 params
            self.model._pmr_init_params = {}
            for name, p in pm.named_parameters():
                self.model._pmr_init_params[f"pm.{name}"] = p.data.cpu().clone()
            for name, p in adapt.named_parameters():
                self.model._pmr_init_params[f"adapter.{name}"] = p.data.cpu().clone()
            print("[PM-R-TRACK] params snapshotted at step 0", flush=True)

        if self._pmr_step == 20 and self._pmr_tracked is not None:
            t = self._pmr_tracked
            def delta(name, before):
                after_map = {"pm_proj_r": pm.proj_r.weight, "pm_omega": pm.omega.weight,
                             "q_mlp": adapt.q_mlp[0].weight, "k_mlp": adapt.k_mlp[0].weight,
                             "v_mlp": adapt.v_mlp[0].weight, "out_proj": adapt.out_proj.weight,
                             "gamma_r": adapt.gamma_r}
                if name == "write_logit" and hasattr(adapt, 'write_logit'):
                    after = adapt.write_logit
                elif name == "coef_head_w" and hasattr(adapt, 'phase_scaffold_warp'):
                    after = adapt.phase_scaffold_warp.coef_head[-1].weight
                elif name == "coef_head_b" and hasattr(adapt, 'phase_scaffold_warp'):
                    after = adapt.phase_scaffold_warp.coef_head[-1].bias
                else:
                    after = after_map.get(name)
                    if after is None:
                        return -1.0
                return (after.detach().cpu() - before).abs().mean().item()
            print("\n[PM-R-DELTA] ========== Parameter deltas after 20 steps ==========", flush=True)
            for k in t:
                d = delta(k, t[k])
                if d >= 0:
                    print(f"  {k}_delta = {d:.12e}", flush=True)
            print("[PM-R-DELTA] ===================================================\n", flush=True)

        self._pmr_step += 1

        # ---- Step 8: Diagnostics ------------------------------------------
        with torch.no_grad():
            diag = dict(adapter_diag)
            diag["flow_loss"] = flow_loss.item()
            diag["gamma_r"] = adapt.gamma_r.item()
            diag["retrieval_residual_norm"] = retrieval_residual_norm
            diag["hidden_delta_norm"] = hidden_delta_norm

            # Register backward hook to populate real grad norms in diag.
            _pmr_gmods = [
                ("PM", self.model._pmr["pm"]),
                ("q_mlp", adapt.q_mlp),
                ("k_mlp", adapt.k_mlp),
                ("v_mlp", adapt.v_mlp),
                ("out_proj", adapt.out_proj),
                ("audio_coord_mlp", adapt.audio_coord_mlp),
                ("text_coord_mlp", adapt.text_coord_mlp),
            ]
            def _capture_grads_hook(grad):
                for name, mod in _pmr_gmods:
                    if mod is not None:
                        diag[f"{name}_grad_norm"] = _grad_norm(mod)
                # Param deltas (relative to init snapshot)
                init_ps = getattr(self.model, '_pmr_init_params', None)
                if init_ps is not None:
                    delta_sum = 0.0; count = 0
                    for pm_name, p in self.model._pmr["pm"].named_parameters():
                        key = f"pm.{pm_name}"
                        if key in init_ps:
                            diff = (p.data.cpu() - init_ps[key]).norm().item()
                            init_n = init_ps[key].norm().item()
                            delta_sum += diff / max(init_n, 1e-8); count += 1
                    for ad_name, p in adapt.named_parameters():
                        key = f"adapter.{ad_name}"
                        if key in init_ps:
                            diff = (p.data.cpu() - init_ps[key]).norm().item()
                            init_n = init_ps[key].norm().item()
                            delta_sum += diff / max(init_n, 1e-8); count += 1
                    diag["param_delta_mean"] = delta_sum / max(count, 1)
                    gamma_init = init_ps.get("adapter.gamma_r")
                    if gamma_init is not None:
                        diag["gamma_r_delta"] = (adapt.gamma_r.data.cpu() - gamma_init).item()
                has_adapter = diag.get("score_qk_std", 0) > 1e-8
                print(
                    f"  [PM-R] flow={diag['flow_loss']:.4f} "
                    f"γ={diag['gamma_r']:.5f} "
                    f"δγ={diag.get('gamma_r_delta', 0):.5f} "
                    + (f"δp={diag.get('param_delta_mean', 0):.2e} "
                       f"ret|h|={diag.get('retrieval_residual_norm', 0):.2e} "
                       f"Δh={diag.get('hidden_delta_norm', 0):.2e} "
                       f"PM|g|={diag.get('PM_grad_norm', -1):.2e} "
                       f"q|g|={diag.get('q_mlp_grad_norm', -1):.2e} "
                       f"out|g|={diag.get('out_proj_grad_norm', -1):.2e} "
                       f"qkσ={diag.get('score_qk_std', 0):.4f} "
                       + (f"phσ={diag.get('phase_bias_std', 0):.4f} "
                          f"δp|={diag.get('delta_p_abs_mean', 0):.4f} "
                          f"α={diag.get('write_alpha', 0):.2e} "
                          f"w|r={diag.get('write_ratio', 0):.2e}"
                          if "phase_bias_std" in diag else
                          f"pσ={diag.get('score_prior_std', 0):.4f}")
                       if has_adapter else "adapter=SKIPPED"),
                    flush=True,
                )
            loss.register_hook(_capture_grads_hook)
            self.pm_diag = diag

            # NaN/Inf check
            has_nan = not torch.isfinite(loss).item()
            diag["has_nan"] = float(has_nan)
            if has_nan:
                print("\n[DEBUG pm_retrieval] NaN loss detected!")
                for k, v in diag.items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        print(f"  {k} = {v}")

            self.pm_diag = diag

        return loss

    # -----------------------------------------------------------------------
    # Transport retrieval training step
    # -----------------------------------------------------------------------

    def _pm_transport_training_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        section_ids: Optional[torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Training step for TransportRetrievalAdapter (unit-level Sinkhorn).

        Flow:
          1. Warmup forward → collect layer 12 hidden state H.
          2. PMRetrievalPhaseMemory(H) → pm_state.
          3. Build/load scaffold from batch metadata; compute unit-level tensors.
          4. TransportRetrievalAdapter → delta_h (RMS-calibrated).
          5. Inject delta_h via hook → decoder forward → flow_loss.
          6. Backward updates transport adapter + PM params.
          7. Diagnostics + param delta tracking.
        """
        from acestep.phase_memory import build_duration_scaffold, parse_lyrics_to_units, scaffold_progress

        ADAPTER_LAYER = 12
        bsz = xt.shape[0]
        device = self.device

        # ---- Step 1: Warmup forward → collect H from layer 12 ------------
        hs_list = []
        def _whook(m, i, o):
            hs_list.append(o[0])
        handle = self.model.decoder.layers[ADAPTER_LAYER].register_forward_hook(_whook)
        try:
            _ = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            handle.remove()

        T_h = hs_list[0].shape[1] if hs_list else xt.shape[1]
        H = hs_list[0].float() if hs_list else torch.zeros(bsz, T_h, self.model.config.hidden_size, device=device, dtype=torch.float32)

        # ---- Step 2: PMRetrievalPhaseMemory forward → pm_state ------------
        pm_state = self.model._pmr["pm"](H, t)

        # ---- Step 3: Build/load scaffold from batch metadata -------------
        adapt = self.model._pmr["adapter"]
        adapt.train()
        transport_diag: Dict[str, float] = {}

        if self.pm_retrieval_scaffold is None:
            meta_list = batch.get("metadata", batch.get("metadatas", []))
            meta = meta_list[0] if isinstance(meta_list, list) and len(meta_list) > 0 else {}
            lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""

            if section_ids is None and lyrics_text:
                L_eff = encoder_hidden_states.shape[1]
                section_ids = torch.zeros(bsz, L_eff, dtype=torch.long, device=device)
            elif section_ids is not None:
                section_ids = section_ids.to(device, non_blocking=self.transfer_non_blocking)

            if lyrics_text and section_ids is not None:
                L_eff = section_ids.shape[-1]
                units, _, debug = parse_lyrics_to_units(
                    lyrics_text, section_ids[0].cpu(),
                    auto_transition_ratios={
                        "intro": 0.0, "outro": 0.0,
                        "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
                        "bridge_to_chorus": 0.0,
                    },
                )
                tcm = debug.get("tag_control_mask", None)
                self.pm_retrieval_scaffold = build_duration_scaffold(
                    units, text_len=L_eff, tag_control_mask=tcm,
                )
                # Cache units for transport unit-level info
                self._transport_units = units

        sc = self.pm_retrieval_scaffold
        units = getattr(self, '_transport_units', None)

        if sc is not None and units is not None:
            # ---- Step 3b: Compute unit-level tensors for transport --------
            sc_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
            token_to_unit = sc_dev["token_to_unit"]  # [L]
            lyric_mask = sc_dev["lyric_mask"]        # [L]
            lyric_unit_mask = sc_dev.get("lyric_unit_mask")  # [U] bool
            unit_boundaries = sc_dev["unit_boundaries"]      # [U+1]
            unit_duration = sc_dev["unit_duration"]          # [U]
            unit_section_ids_sc = sc_dev.get("unit_section_ids")  # [U]

            if lyric_unit_mask is not None:
                lyric_uids = torch.where(lyric_unit_mask)[0]  # [K]
                K = len(lyric_uids)
            else:
                K = 0

            if K == 0:
                # No lyric units → zero residual
                delta_h = torch.zeros_like(H)
                Pi = torch.zeros(bsz, T_h, 0, device=device)
                transport_diag["transport_skipped"] = 1.0
            else:
                # Unit centres (from boundaries)
                c_all = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2  # [U]
                c_unit = c_all[lyric_uids]  # [K]
                mu_all = unit_duration
                mu = mu_all[lyric_uids]  # [K]
                mu = mu / mu.sum()  # normalise to sum 1

                # Unit section IDs
                usid = unit_section_ids_sc[lyric_uids] if unit_section_ids_sc is not None \
                       else torch.zeros(K, dtype=torch.long, device=device)

                # Lyric unit IDs for token masking
                lyric_uids_set = set(lyric_uids.tolist())
                B, L_eff, D = encoder_hidden_states.shape

                # Pool text_hidden per lyric unit
                unit_text_hidden_list = []
                eh_f = encoder_hidden_states.float()
                for uid in lyric_uids.tolist():
                    token_mask = (token_to_unit == uid) & lyric_mask  # [L]
                    token_mask_b = token_mask.unsqueeze(0).expand(B, -1)  # [B, L]
                    if token_mask_b.any():
                        pooled = eh_f[token_mask_b].view(B, -1, D).mean(dim=1)  # [B, D]
                    else:
                        pooled = torch.zeros(B, D, device=device, dtype=torch.float32)
                    unit_text_hidden_list.append(pooled)
                unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)  # [B, K, D]

                # Expand for batch
                c_unit_b = c_unit.unsqueeze(0).expand(B, -1)  # [B, K]
                mu_b = mu.unsqueeze(0).expand(B, -1)          # [B, K]
                usid_b = usid.unsqueeze(0).expand(B, -1)      # [B, K]

                # p_audio (linear progress)
                p_audio = torch.linspace(0, 1, T_h, device=device, dtype=torch.float32)
                p_audio = p_audio.unsqueeze(0).expand(B, -1)

                # ---- Step 4: TransportRetrievalAdapter forward ------------
                delta_h, Pi, adapter_diag = adapt(
                    hidden_states=H,
                    text_hidden=eh_f,
                    pm_state=pm_state,
                    p_audio=p_audio,
                    unit_text_hidden=unit_text_hidden,
                    unit_c_pos=c_unit_b,
                    unit_mass=mu_b,
                    unit_section_id=usid_b,
                )
                transport_diag.update(adapter_diag)

        else:
            # No scaffold → zero residual
            delta_h = torch.zeros_like(H)
            Pi = None
            transport_diag["transport_skipped"] = 1.0

        final_h = H + delta_h

        # Track residual stats
        with torch.no_grad():
            retrieval_residual_norm = delta_h.norm(dim=-1).mean().item()
            hidden_delta_norm = (final_h - H).norm(dim=-1).mean().item()
            transport_diag["retrieval_residual_norm"] = retrieval_residual_norm
            transport_diag["hidden_delta_norm"] = hidden_delta_norm

        # ---- Step 5: Hook to inject modified hidden at layer 12 -----------
        if hasattr(self, '_pm_retrieval_inject_handle') and self._pm_retrieval_inject_handle is not None:
            self._pm_retrieval_inject_handle.remove()
            self._pm_retrieval_inject_handle = None

        def _inject_hook(m, i, o):
            fh = final_h.to(dtype=o[0].dtype, device=o[0].device)
            return (fh, *o[1:])
        self._pm_retrieval_inject_handle = self.model.decoder.layers[ADAPTER_LAYER].register_forward_hook(_inject_hook)

        # ---- Step 6: Decoder forward with injected hidden → flow_loss ----
        try:
            decoder_outputs = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            if self._pm_retrieval_inject_handle is not None:
                self._pm_retrieval_inject_handle.remove()
                self._pm_retrieval_inject_handle = None

        flow = x1 - x0
        flow_loss = F.mse_loss(decoder_outputs[0], flow)
        loss = flow_loss

        # ---- Step 7: Parameter tracking (step 0 clone, step 20 delta) ----
        pm = self.model._pmr["pm"]
        if self._pmr_step == 0 and self._pmr_tracked is None:
            self._pmr_tracked = {
                "pm_proj_r": pm.proj_r.weight.detach().clone().cpu(),
                "pm_omega": pm.omega.weight.detach().clone().cpu(),
                "q_mlp": adapt.q_mlp[0].weight.detach().clone().cpu(),
                "k_mlp": adapt.k_mlp[0].weight.detach().clone().cpu(),
                "v_mlp": adapt.v_mlp[0].weight.detach().clone().cpu(),
                "out_proj": adapt.out_proj.weight.detach().clone().cpu(),
                "write_logit": adapt.write_logit.detach().clone().cpu(),
            }
            # Re-init pmr_init_params
            self.model._pmr_init_params = {}
            for name, p in pm.named_parameters():
                self.model._pmr_init_params[f"pm.{name}"] = p.data.cpu().clone()
            for name, p in adapt.named_parameters():
                self.model._pmr_init_params[f"adapter.{name}"] = p.data.cpu().clone()
            print("[Transport-TRACK] params snapshotted at step 0", flush=True)

        if self._pmr_step == 20 and self._pmr_tracked is not None:
            t = self._pmr_tracked
            def delta(name, before):
                after_map = {"pm_proj_r": pm.proj_r.weight, "pm_omega": pm.omega.weight,
                             "q_mlp": adapt.q_mlp[0].weight, "k_mlp": adapt.k_mlp[0].weight,
                             "v_mlp": adapt.v_mlp[0].weight, "out_proj": adapt.out_proj.weight}
                if name == "write_logit" and hasattr(adapt, 'write_logit'):
                    after = adapt.write_logit
                else:
                    after = after_map.get(name)
                    if after is None:
                        return -1.0
                return (after.detach().cpu() - before).abs().mean().item()
            print("\n[Transport-DELTA] ========== Parameter deltas after 20 steps ==========", flush=True)
            for k in t:
                d = delta(k, t[k])
                if d >= 0:
                    print(f"  {k}_delta = {d:.12e}", flush=True)
            print("[Transport-DELTA] ===================================================\n", flush=True)

        self._pmr_step += 1

        # ---- Step 8: Diagnostics (no hooks — grads captured post-backward) ---
        with torch.no_grad():
            diag = dict(transport_diag)
            diag["flow_loss"] = flow_loss.item()
            diag["write_alpha"] = adapt.write_alpha.item() if hasattr(adapt, 'write_alpha') else 0.0
            diag["retrieval_residual_norm"] = retrieval_residual_norm
            diag["hidden_delta_norm"] = hidden_delta_norm

            # Param deltas (relative to init snapshot) — computed from .data,
            # safe to compute here (no grad needed).
            init_ps = getattr(self.model, '_pmr_init_params', None)
            if init_ps is not None:
                delta_sum = 0.0; count = 0
                for pm_name, p in self.model._pmr["pm"].named_parameters():
                    key = f"pm.{pm_name}"
                    if key in init_ps:
                        diff = (p.data.cpu() - init_ps[key]).norm().item()
                        init_n = init_ps[key].norm().item()
                        delta_sum += diff / max(init_n, 1e-8); count += 1
                for ad_name, p in adapt.named_parameters():
                    key = f"adapter.{ad_name}"
                    if key in init_ps:
                        diff = (p.data.cpu() - init_ps[key]).norm().item()
                        init_n = init_ps[key].norm().item()
                        delta_sum += diff / max(init_n, 1e-8); count += 1
                diag["param_delta_mean"] = delta_sum / max(count, 1)

            has_pi = diag.get("transport_max", 0) > 0
            print(
                f"  [Transport] flow={diag['flow_loss']:.4f} "
                f"rowε={diag.get('row_error', -1):.2e} "
                f"colε={diag.get('col_error', -1):.2e} "
                f"H|ε={diag.get('entropy', -1):.2f} "
                f"α={diag.get('write_alpha', 0):.2e} "
                f"w|r={diag.get('write_ratio', -1):.2e} "
                f"qkσ={diag.get('qk_std', 0):.4f} "
                + (f"skipped" if not has_pi else
                   f"δp={diag.get('param_delta_mean', 0):.2e} "
                   f"Δh={diag.get('hidden_delta_norm', 0):.2e}"),
                flush=True,
            )

            # Store grad-capture modules so the training loop reads grads
            # after loss.backward() completes (NOT via register_hook, which
            # fires before parameter grads are populated).
            if not hasattr(self, '_pmr_grad_mods'):
                self._pmr_grad_mods = [
                    ("PM", self.model._pmr["pm"]),
                    ("q_mlp", adapt.q_mlp),
                    ("k_mlp", adapt.k_mlp),
                    ("v_mlp", adapt.v_mlp),
                    ("out_proj", adapt.out_proj),
                ]

            self.pm_diag = diag

        return loss

    # -----------------------------------------------------------------------
    # Post-backward grad capture (called after loss.backward(), before
    # optimizer.step()).  Reads p.grad directly to avoid register_hook
    # timing issues (hook fires before param grads are populated).
    # -----------------------------------------------------------------------

    def _pmr_capture_grads(self) -> None:
        """Read grad norms from stored grad modules into ``pm_diag``.

        Must be called after ``loss.backward()`` completes but before
        ``optimizer.step()`` zeros gradients.
        """
        mods = getattr(self, '_pmr_grad_mods', None)
        if mods is None:
            return
        diag = getattr(self, 'pm_diag', None)
        if diag is None:
            return
        for name, mod in mods:
            if mod is not None:
                g = _grad_norm(mod)
                diag[f"{name}_grad_norm"] = g

    # -----------------------------------------------------------------------
    # PhaseMemory diagnostics
    # -----------------------------------------------------------------------

    def _collect_pm_diag(self) -> None:
        """Read PhaseMemory internal state after decoder forward.

        Handles old ``PhaseMemoryDurationClock`` (with persistent z_r/z_i buffers).
        PMRetrievalPhaseMemory diagnostics are collected in ``_pm_retrieval_training_step``.
        """
        diag: Dict[str, float] = {}
        for module in self.model.modules():
            if not isinstance(module, PhaseMemoryDurationClock):
                continue
            pm = module
            # Old PhaseMemory: use z_r / z_i buffers
            if hasattr(pm, "z_r") and pm.z_r is not None:
                zr, zi = pm.z_r, pm.z_i
                diag["z_norm_mean"] = torch.sqrt(zr**2 + zi**2).mean().item()
                phi = torch.atan2(zi, zr)
                diag["phi_std"] = phi.std().item()
                diag["phi_mean"] = phi.mean().item()
                diag["anchor_norm"] = pm.anchor.norm().item()
                if pm.traj is not None:
                    diag["traj_norm"] = pm.traj.norm().item()
                diag["out_weight_norm"] = pm.out.weight.norm().item()
            break  # only one PM module
        self.pm_diag = diag

    # -----------------------------------------------------------------------
    # Section-RoPE Offset diagnostics
    # -----------------------------------------------------------------------

    def _collect_section_rope_diag(self) -> None:
        """Collect Section-RoPE V2 internal state after decoder forward."""

        diag: Dict[str, float] = {}
        for module in self.model.modules():
            if not hasattr(module, "section_rope_offset_module"):
                continue
            sec_mod = module.section_rope_offset_module
            w = sec_mod.section_phase          # [H, N, D]
            ls = sec_mod.log_scale             # [H, 1, 1]

            # Overall embedding norm
            diag["phase_norm"] = w.norm().item()
            diag["log_scale_mean"] = ls.mean().item()
            diag["log_scale_std"] = ls.std().item()

            # Per-type embedding norm (averaged over heads)
            for sid in range(w.shape[1]):
                per_type_norm = w[:, sid, :].norm().item()
                diag[f"norm_type_{sid}"] = per_type_norm

            # Per-head embedding norm (averaged over types)
            for hid in range(w.shape[0]):
                per_head_norm = w[hid].norm().item()
                diag[f"norm_head_{hid}"] = per_head_norm

            # Gradient norms
            if w.grad is not None:
                diag["grad_norm"] = w.grad.norm().item()
            if ls.grad is not None:
                diag["log_scale_grad_norm"] = ls.grad.norm().item()

            # Delta stats via forward
            dummy = torch.arange(w.shape[1], dtype=torch.long, device=w.device)
            delta = sec_mod(dummy.unsqueeze(0))  # [1, N] -> [B, H, N, D]
            diag["delta_mean_abs"] = delta.abs().mean().item()
            diag["delta_max_abs"] = delta.abs().max().item()
            # Per-head delta stats
            diag["delta_per_head_mean_abs"] = delta.abs().mean(dim=(0, 2, 3)).mean().item()
            diag["delta_per_head_std"] = delta.abs().mean(dim=(0, 2, 3)).std().item()

            # Token weight stats (if available from recent forward)
            if hasattr(sec_mod, "_last_token_weights") and sec_mod._last_token_weights is not None:
                tw = sec_mod._last_token_weights
                diag["token_weight_mean"] = tw.mean().item()
                diag["token_weight_min"] = tw.min().item()
                diag["token_weight_max"] = tw.max().item()

            break
        self.section_rope_diag = diag

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
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            attention_mask = batch["attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_hidden_states = batch["encoder_hidden_states"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_attention_mask = batch["encoder_attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            context_latents = batch["context_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )

            bsz = target_latents.shape[0]

            # Reset PhaseMemory so pm_kv doesn't carry stale T from prior batch
            reset_phase_memory(self.model)

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

            section_ids = batch.get("section_ids")
            if section_ids is not None:
                section_ids = section_ids.to(self.device, non_blocking=self.transfer_non_blocking)

            # ---- PMDC dual-forward path -----------------------------------
            if self.adapter_type == "pmdc_clock" and self.pmdc_clock is not None:
                loss = self._pmdc_training_step(
                    xt, t, x0, x1,
                    attention_mask, encoder_hidden_states,
                    encoder_attention_mask, context_latents,
                    section_ids, batch,
                )
                self.training_losses.append(loss.item())
                return loss.float()

            # ---- PM + Retrieval Adapter forward ---------------------------
            if self.adapter_type == "pm_retrieval" and self.model._pmr is not None:
                if getattr(self.training_config, 'use_transport_retrieval', False):
                    loss = self._pm_transport_training_step(
                        xt, t, x0, x1,
                        attention_mask, encoder_hidden_states,
                        encoder_attention_mask, context_latents,
                        section_ids, batch,
                    )
                else:
                    loss = self._pm_retrieval_training_step(
                        xt, t, x0, x1,
                        attention_mask, encoder_hidden_states,
                        encoder_attention_mask, context_latents,
                        section_ids, batch,
                    )
                self.training_losses.append(loss.item())
                return loss.float()

            # ---- Standard decoder forward ----------------------------------
            decoder_kwargs = dict(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,  # r = t
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            if section_ids is not None:
                decoder_kwargs["section_ids"] = section_ids
            decoder_outputs = self.model.decoder(**decoder_kwargs)

            # ---- Flow matching loss ----------------------------------------
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

            # ---- PhaseMemory diagnostics (no-grad, no overhead) ------------
            self._collect_pm_diag()
            self._collect_section_rope_diag()

        # fp32 for stable backward
        diffusion_loss = diffusion_loss.float()
        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss

    # -----------------------------------------------------------------------
    # PMDC training step
    # -----------------------------------------------------------------------

    def _pmdc_training_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        section_ids: Optional[torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """PMDC dual-forward training step.

        1. Warmup forward (no PMDC bias) → collect layer 12 H.
        2. PMDCResidualClock(H) → p_final.
        3. Build duration-interval bias from p_final + scaffold.
        4. Patch eager_attention_forward → main decoder forward with bias.
        5. Return flow_loss + PMDC regularization losses.
        """
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        import torch.nn.functional as F
        from acestep.phase_memory import (
            parse_lyrics_to_units, build_duration_scaffold,
        )

        bsz = xt.shape[0]
        device = self.device
        cfg = self.pmdc_config

        # ---- Step 1: Warmup forward (no PMDC bias) → collect H ------------
        hs_list = []
        def _warmup_hook(m, i, o):
            hs_list.append(o[0])
        handle = self.model.decoder.layers[12].register_forward_hook(_warmup_hook)
        try:
            _ = self.model.decoder(
                hidden_states=xt,
                timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            handle.remove()

        T_pmdc = hs_list[0].shape[1] if hs_list else xt.shape[1]
        H_warm = hs_list[0].detach().float() if hs_list else \
                 torch.zeros(bsz, T_pmdc, 2048, device=device, dtype=torch.float32)

        # ---- Build scaffold from batch metadata ---------------------------
        if self.pmdc_scaffold is None:
            meta_list = batch.get("metadata", batch.get("metadatas", []))
            if isinstance(meta_list, list) and len(meta_list) > 0:
                meta = meta_list[0] if isinstance(meta_list[0], dict) else {}
            else:
                meta = {}
            lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""
            if lyrics_text and section_ids is not None:
                L_eff = section_ids.shape[-1]
                # Explicitly disable all silence/transition scaffolding
                units, _, debug = parse_lyrics_to_units(
                    lyrics_text, section_ids[0].cpu(),
                    auto_transition_ratios={
                        "intro": 0.0, "outro": 0.0,
                        "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
                        "bridge_to_chorus": 0.0,
                    },
                )
                tcm = debug.get("tag_control_mask", None)
                self.pmdc_scaffold = build_duration_scaffold(
                    units, text_len=L_eff, tag_control_mask=tcm,
                )

        # ---- Detect PM variant -------------------------------------------------
        use_controlled = getattr(cfg, "use_controlled_pm", False) and self.controlled_pm is not None
        use_standard = not use_controlled and self.pmdc_clock is not None

        # ---- Step 2: PM forward ------------------------------------------------
        p_base = torch.linspace(0, 1, T_pmdc, device=device, dtype=torch.float32)
        p_base = p_base.unsqueeze(0).expand(bsz, -1)

        pm_residual = None
        speed_residual = None
        p_final = p_base
        log_v = None

        if use_controlled:
            p_dyn, sr, pmr = self.controlled_pm(H_warm, p_base, output_residual=True)
            speed_residual = sr
            pm_residual = pmr
            p_final = p_dyn
            gate_pm = torch.sigmoid(self.pm_gate_pm)
            gate_attn = torch.sigmoid(self.pm_gate_attn)
            gate_val = gate_attn
            logger_type = "ControlledPM"
        elif use_standard:
            speed_residual, p_final, log_v = self.pmdc_clock(H_warm, p_base)
            gate_val = torch.sigmoid(self.pmdc_gate_logit)
            gate_pm = torch.tensor(0.0, device=device)
            logger_type = "PMDC"
        else:
            raise RuntimeError("No PM variant available")

        # ---- Step 3: Build duration bias from p_final ---------------------------
        bias = None
        if self.pmdc_scaffold is not None:
            from acestep.phase_memory import build_duration_interval_bias
            bias = build_duration_interval_bias(
                p_final=p_final,
                unit_boundaries=self.pmdc_scaffold["unit_boundaries"],
                token_to_unit=self.pmdc_scaffold["token_to_unit"],
                attendable_mask=self.pmdc_scaffold.get("attendable_mask",
                                                         self.pmdc_scaffold["lyric_mask"]),
                sigma=cfg.pmdc_sigma, lambda_=cfg.pmdc_lambda,
                max_bias=cfg.pmdc_max_bias,
            )
            if next(self.model.parameters()).dtype == torch.bfloat16:
                bias = bias.to(torch.bfloat16)
            self.pmdc_bias_cache = bias

        # ---- Step 3.5: Install residual hook (controlled PM only) -------------
        residual_handle = None
        if use_controlled and pm_residual is not None:
            def _residual_hook(module, inpt, output):
                # output[0] is [B, T, D] hidden states from layer 12
                # Add pm_residual with gate
                gate = torch.sigmoid(self.pm_gate_pm)
                return (output[0] + gate * pm_residual, *output[1:])

            residual_handle = self.model.decoder.layers[12].register_forward_hook(_residual_hook)

        # ---- Step 4: Patch attention → main forward ----------------------------
        # Patch eager_attention_forward
        ca_module = self.model.decoder.layers[12].cross_attn
        import sys
        cls = type(ca_module)
        attn_mod = sys.modules[cls.__module__]
        orig_eaf = getattr(attn_mod, "eager_attention_forward", None)

        def _pmdc_patched_forward(*args, **kwargs):
            self._pmdc_patch_call_count += 1
            mod = args[0]; q = args[1]; k = args[2]; v = args[3]
            am = args[4] if len(args) > 4 else kwargs.get("attention_mask")
            sc = kwargs.get("scaling", args[5] if len(args) > 5 else None)
            dr = kwargs.get("dropout", 0.0)
            ks = repeat_kv(k, mod.num_key_value_groups)
            vs = repeat_kv(v, mod.num_key_value_groups)
            aw = torch.matmul(q, ks.transpose(2, 3)) * sc
            if am is not None and isinstance(am, torch.Tensor):
                aw = aw + am[:, :, :, :ks.shape[-2]]

            if self.pmdc_bias_cache is not None:
                bias_t = self.pmdc_bias_cache
                if bias_t.shape[0] != aw.shape[0]:
                    bf = aw.shape[0] // max(bias_t.shape[0], 1)
                    bias_t = bias_t.repeat(bf, 1, 1, 1) if bf > 1 else bias_t
                bias_exp = bias_t.unsqueeze(1)

                amask = self.pmdc_scaffold.get("attendable_mask",
                                                self.pmdc_scaffold["lyric_mask"])
                if amask.dim() == 1:
                    amask = amask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                if amask.shape[0] != aw.shape[0]:
                    amask = amask.expand(aw.shape[0], -1, -1, -1)
                amask = amask.bool()

                text_float = amask.float()
                attn_base = F.softmax(aw, dim=-1, dtype=torch.float32)
                text_mass_base = (attn_base * text_float).sum(dim=-1)

                text_logits = (aw + gate_val * bias_exp).masked_fill(~amask, float("-inf"))
                attn_text = F.softmax(text_logits, dim=-1, dtype=torch.float32)
                attn_text = attn_text.masked_fill(~amask, 0.0)

                non_text_logits = aw.masked_fill(amask, float("-inf"))
                attn_non_text = F.softmax(non_text_logits, dim=-1, dtype=torch.float32)
                attn_non_text = attn_non_text.masked_fill(amask, 0.0)

                aw = (attn_text * text_mass_base.unsqueeze(-1) +
                      attn_non_text * (1.0 - text_mass_base).unsqueeze(-1))
                aw = aw / (aw.sum(dim=-1, keepdim=True) + 1e-10)
                aw = aw.to(q.dtype)
            else:
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)

            aw = F.dropout(aw, p=dr, training=mod.training)
            return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw

        setattr(attn_mod, "eager_attention_forward", _pmdc_patched_forward)
        try:
            decoder_outputs = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            if orig_eaf is not None:
                setattr(attn_mod, "eager_attention_forward", orig_eaf)
            self.pmdc_bias_cache = None
            if residual_handle is not None:
                residual_handle.remove()

        # ---- Step 5: Losses ----------------------------------------------------
        flow = x1 - x0
        flow_loss = F.mse_loss(decoder_outputs[0], flow)

        pbase_loss = F.smooth_l1_loss(p_final, p_base)

        loss = flow_loss + cfg.pmdc_w_pbase * pbase_loss

        if cfg.pmdc_w_smooth > 0 and log_v is not None:
            smooth_loss = F.mse_loss(log_v[:, 1:], log_v[:, :-1])
            loss = loss + cfg.pmdc_w_smooth * smooth_loss

        if cfg.pmdc_w_res > 0:
            res_loss = speed_residual.pow(2).mean()
            loss = loss + cfg.pmdc_w_res * res_loss

        # Store diagnostics
        with torch.no_grad():
            diag = {
                "flow_loss": flow_loss.item(),
                "pbase_loss": pbase_loss.item(),
                "total_loss": loss.item(),
                "gate_attn": gate_val.item(),
                "mean_abs_p_delta": (p_final - p_base).abs().mean().item(),
                "p_final_min": p_final.min().item(),
                "p_final_max": p_final.max().item(),
            }
            if use_controlled:
                diag["gate_pm"] = torch.sigmoid(self.pm_gate_pm).item()
                diag["pm_residual_norm"] = pm_residual.norm().item() if pm_residual is not None else 0.0
                diag["beta"] = self.controlled_pm.beta.item() if hasattr(self.controlled_pm, "beta") else 0.0
                diag["speed_residual_std"] = speed_residual.std().item() if speed_residual is not None else 0.0
            else:
                diag["beta"] = self.pmdc_clock.beta.item() if hasattr(self.pmdc_clock, "beta") else 0.0
                diag["speed_residual_std"] = speed_residual.std().item() if speed_residual is not None else 0.0
            self.pmdc_diag = diag

        return loss
