"""
Extended Training Configuration for ACE-Step Training V2

Uses base configs from ``acestep.training.configs``.  Extends them with
corrected-training-specific fields (CFG dropout,
continuous timestep sampling parameters, estimation, TensorBoard, sample
generation, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# Vendored base configs -- no base ACE-Step installation required
from acestep.training.configs import (  # noqa: F401
    LoRAConfig,
    LoKRConfig,
    PMDCConfig,
    PhaseMemoryConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Extended LoRA config (unchanged for now, but available for future extension)
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfigV2(LoRAConfig):
    """Extended LoRA configuration.

    Inherits all fields from the original LoRAConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        return base

    # --- Data loading (declared here for compatibility with base packages
    #     that may not include these fields in TrainingConfig) -----------------
    num_workers: int = 4
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs."""

    pin_memory_device: str = ""
    """Device for pinned memory ("" = default CUDA device)."""


# ---------------------------------------------------------------------------
# Extended LoKR config
# ---------------------------------------------------------------------------

@dataclass
class LoKRConfigV2(LoKRConfig):
    """Extended LoKR configuration.

    Inherits all fields from the original LoKRConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        return base


# ---------------------------------------------------------------------------
# Extended PhaseMemory config
# ---------------------------------------------------------------------------

@dataclass
class PhaseMemoryConfigV2(PhaseMemoryConfig):
    """Extended PhaseMemory configuration.

    Inherits all fields from the original PhaseMemoryConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        return base


# ---------------------------------------------------------------------------
# PMDC Clock config
# ---------------------------------------------------------------------------

@dataclass
class PMDCConfigV2(PMDCConfig):
    """Extended PMDC configuration."""

    def to_dict(self) -> dict:
        return super().to_dict()


# ---------------------------------------------------------------------------
@dataclass
class TrainingConfigV2(TrainingConfig):
    """Extended training configuration with corrected-training fields.

    New fields compared to the original TrainingConfig:
    - CFG dropout (cfg_ratio)
    - Continuous timestep sampling parameters (timestep_mu, timestep_sigma,
      data_proportion)
    - Model variant selection
    - Device / precision auto-detection
    - Estimation parameters
    - Extended TensorBoard logging
    - Sample generation during training
    - Checkpoint resume
    - Preprocessing flags
    """

    # --- Data loading (declared here for compatibility with base packages
    #     that may not include these fields in TrainingConfig) -----------------
    num_workers: int = 4
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs."""

    pin_memory_device: str = ""
    """Device for pinned memory ("" = default CUDA device)."""

    # --- Optimizer / Scheduler ------------------------------------------------
    optimizer_type: str = "adamw"
    """Optimizer: 'adamw', 'adamw8bit', 'adafactor', 'prodigy'."""

    scheduler_type: str = "cosine"
    """LR scheduler: 'cosine', 'cosine_restarts', 'linear', 'constant', 'constant_with_warmup'."""

    # --- VRAM management ------------------------------------------------------
    gradient_checkpointing: bool = True
    """Trade compute for memory by recomputing activations during backward.
    Enabled by default to match ACE-Step's behaviour and save ~40-60%
    activation VRAM.  Adds ~10-30% training time overhead."""

    offload_encoder: bool = False
    """Move encoder/VAE to CPU after setup to free ~2-4 GB VRAM."""

    vram_profile: str = "auto"
    """VRAM preset: 'auto', 'comfortable', 'standard', 'tight', 'minimal'."""

    # --- Corrected training params ------------------------------------------
    cfg_ratio: float = 0.15
    """Classifier-free guidance dropout probability."""

    timestep_mu: float = -0.4
    """Mean for logit-normal timestep sampling (from model config)."""

    timestep_sigma: float = 1.0
    """Std for logit-normal timestep sampling (from model config)."""

    data_proportion: float = 0.5
    """Data proportion for sample_t_r (from model config)."""

    # --- Adapter selection ----------------------------------------------------
    adapter_type: str = "lora"
    """Adapter type: 'lora' (PEFT), 'lokr' (LyCORIS), 'phase_memory' (targeted),
    'section_rope' (Section-RoPE Offset), 'pmdc_clock' (PMDC Residual Clock),
    or 'pm_retrieval' (PM + LyricRetrievalAdapter)."""

    # --- PhaseMemory-specific params ------------------------------------------
    phase_memory_lr_multiplier: float = 1.0
    """Learning rate multiplier for PhaseMemory parameters (relative to base LR).

    Set > 1.0 to train PhaseMemory faster when doing joint training with other
    adapter types.  Ignored when adapter_type is 'lora' or 'lokr' only.
    """

    # --- PMDC Clock-specific params -------------------------------------------
    pmdc_hidden_dim: int = 128
    pmdc_beta_init: float = 0.05
    pmdc_beta_max: float = 0.15
    pmdc_use_delta_h: bool = True
    pmdc_gate_init: float = 0.35
    pmdc_sigma: float = 0.03
    pmdc_lambda: float = 0.5
    pmdc_max_bias: float = 1.0
    pmdc_w_pbase: float = 0.005
    pmdc_w_res: float = 0.0
    pmdc_w_smooth: float = 0.0
    use_controlled_pm: bool = False

    # --- Transport retrieval-specific params -----------------------------------
    use_transport_retrieval: bool = False
    """Enable unit-level Sinkhorn transport retrieval (replaces token softmax)."""
    sinkhorn_iters: int = 5
    """Number of log-domain Sinkhorn iterations."""
    transport_sigma: float = 0.18
    """Gaussian width for position-based transport cost."""
    transport_qk_scale: float = 1.0
    """Scale factor for dynamic QK residual in transport logit."""

    # --- PM-Retrieval-specific params -----------------------------------------
    retrieval_adapter_layers: str = "12"
    """Comma-separated layer indices for retrieval adapter injection."""
    use_pm_residual: bool = False
    """Whether to inject PhaseMemory hidden residual (off by default)."""
    pm_retrieval_mem_dim: int = 128
    """PhaseMemory internal complex-state dimension (→ pm_dim = 2*mem_dim)."""
    retrieval_adapter_dim: int = 256
    """Retrieval adapter query/key/value inner dimension."""
    normalize_pm_state: bool = True
    """Apply LayerNorm to pm_state before q_mlp."""
    qk_score_scale_init: float = 2.0
    """Initial qk score scale (learnable exp param, so init=2.0 → exp(log(2)))."""
    use_adapter_scaffold_prior: bool = True
    """Enable weak scaffold prior in retrieval attention."""
    adapter_prior_sigma: float = 0.18
    """Scaffold prior Gaussian width."""
    adapter_prior_lambda: float = 0.2
    """Scaffold prior strength."""
    adapter_prior_clamp_min: float = -2.0
    """Scaffold prior minimum clamp value."""
    adapter_prior_dropout: float = 0.3
    """Scaffold prior dropout probability during training."""
    gamma_r_init: float = 0.1
    """Initial value for the learnable gamma_r gate parameter."""
    residual_scale: float = 0.1
    """Tanh-bounded residual scale factor."""
    retrieval_adapter_time_dim: int = 128
    """Timestep embedding dimension fed into the retrieval query MLP."""

    # --- V4: Phase bias config -------------------------------------------------
    use_phase_bias: bool = True
    """Enable PM-conditioned dynamic phase bias (v4)."""
    phase_bias_lambda: float = 0.03
    """Phase bias strength multiplier."""
    phase_offset_max: float = 0.05
    """Maximum phase offset (clamp range for delta_p)."""
    phase_num_freqs: int = 4
    """Number of Fourier frequencies for phase warp."""
    phase_bias_sigma: float = 0.18
    """Phase bias Gaussian width."""
    phase_bias_dropout: float = 0.3
    """Phase bias dropout probability."""
    phase_bias_clamp_min: float = -2.0
    """Phase bias minimum clamp value."""

    # --- V4: RMS writer config -------------------------------------------------
    use_rms_writer: bool = True
    """Enable RMS-calibrated residual writer (v4)."""
    write_alpha_init: float = 1e-4
    """Initial write_alpha (fraction of h_rms to write)."""
    write_alpha_max: float = 1e-3
    """Maximum write_alpha (upper bound via sigmoid)."""

    # --- Model / paths ------------------------------------------------------
    model_variant: str = "turbo"
    """Model variant: 'turbo', 'base', or 'sft'."""

    checkpoint_dir: str = "./checkpoints"
    """Path to checkpoints root directory."""

    dataset_dir: str = ""
    """Directory containing preprocessed .pt tensor files."""

    # --- Device / precision -------------------------------------------------
    device: str = "auto"
    """Device selection: 'auto', 'cuda', 'cuda:0', 'mps', 'xpu', 'cpu'."""

    precision: str = "auto"
    """Precision: 'auto', 'bf16', 'fp16', 'fp32'."""

    # --- Checkpointing ------------------------------------------------------
    resume_from: Optional[str] = None
    """Path to checkpoint directory to resume training from."""

    # --- Extended TensorBoard logging ---------------------------------------
    log_dir: Optional[str] = None
    """TensorBoard log directory.  Defaults to {output_dir}/runs."""

    log_every: int = 10
    """Log basic metrics (loss, LR) every N optimiser steps."""

    log_heavy_every: int = 50
    """Log per-layer gradient norms every N optimiser steps."""

    # --- Sample generation --------------------------------------------------
    sample_every_n_epochs: int = 0
    """Generate an audio sample every N epochs (0 = disabled)."""

    # --- Estimation params --------------------------------------------------
    estimate_batches: Optional[int] = None
    """Number of batches for gradient estimation (None = auto from GPU)."""

    top_k: int = 16
    """Number of top modules to select during estimation."""

    granularity: str = "module"
    """Estimation granularity: 'layer' or 'module'."""

    module_config: Optional[str] = None
    """Path to JSON module config produced by the estimate subcommand."""

    auto_estimate: bool = False
    """Run estimation inline before training."""

    estimate_output: Optional[str] = None
    """Path to write module config JSON (estimate subcommand only)."""

    # --- Preprocessing flags ------------------------------------------------
    preprocess: bool = False
    """Run preprocessing before training."""

    audio_dir: Optional[str] = None
    """Source audio directory for preprocessing."""

    dataset_json: Optional[str] = None
    """Labeled dataset JSON for preprocessing."""

    tensor_output: Optional[str] = None
    """Output directory for preprocessed .pt tensor files."""

    max_duration: float = 240.0
    """Maximum audio duration in seconds (preprocessing)."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def effective_log_dir(self) -> Path:
        """Return the resolved TensorBoard log directory."""
        if self.log_dir is not None:
            return Path(self.log_dir)
        return Path(self.output_dir) / "runs"

    def to_dict(self) -> dict:
        """Serialize every field, including new ones."""
        base = super().to_dict()
        base.update(
            {
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "prefetch_factor": self.prefetch_factor,
                "persistent_workers": self.persistent_workers,
                "pin_memory_device": self.pin_memory_device,
                "optimizer_type": self.optimizer_type,
                "scheduler_type": self.scheduler_type,
                "gradient_checkpointing": self.gradient_checkpointing,
                "offload_encoder": self.offload_encoder,
                "vram_profile": self.vram_profile,
                "adapter_type": self.adapter_type,
                "cfg_ratio": self.cfg_ratio,
                "timestep_mu": self.timestep_mu,
                "timestep_sigma": self.timestep_sigma,
                "data_proportion": self.data_proportion,
                "model_variant": self.model_variant,
                "checkpoint_dir": self.checkpoint_dir,
                "dataset_dir": self.dataset_dir,
                "device": self.device,
                "precision": self.precision,
                "resume_from": self.resume_from,
                "log_dir": self.log_dir,
                "log_every": self.log_every,
                "log_heavy_every": self.log_heavy_every,
                "sample_every_n_epochs": self.sample_every_n_epochs,
                "estimate_batches": self.estimate_batches,
                "top_k": self.top_k,
                "granularity": self.granularity,
                "module_config": self.module_config,
                "auto_estimate": self.auto_estimate,
                "estimate_output": self.estimate_output,
                "preprocess": self.preprocess,
                "audio_dir": self.audio_dir,
                "dataset_json": self.dataset_json,
                "tensor_output": self.tensor_output,
                "max_duration": self.max_duration,
                # Transport retrieval params
                "use_transport_retrieval": self.use_transport_retrieval,
                "sinkhorn_iters": self.sinkhorn_iters,
                "transport_sigma": self.transport_sigma,
                "transport_qk_scale": self.transport_qk_scale,
                # PM-Retrieval params
                "retrieval_adapter_layers": self.retrieval_adapter_layers,
                "use_pm_residual": self.use_pm_residual,
                "pm_retrieval_mem_dim": self.pm_retrieval_mem_dim,
                "retrieval_adapter_dim": self.retrieval_adapter_dim,
                "normalize_pm_state": self.normalize_pm_state,
                "qk_score_scale_init": self.qk_score_scale_init,
                "use_adapter_scaffold_prior": self.use_adapter_scaffold_prior,
                "adapter_prior_sigma": self.adapter_prior_sigma,
                "adapter_prior_lambda": self.adapter_prior_lambda,
                "adapter_prior_clamp_min": self.adapter_prior_clamp_min,
                "adapter_prior_dropout": self.adapter_prior_dropout,
                "gamma_r_init": self.gamma_r_init,
                "residual_scale": self.residual_scale,
                "retrieval_adapter_time_dim": self.retrieval_adapter_time_dim,
                # V4
                "use_phase_bias": self.use_phase_bias,
                "phase_bias_lambda": self.phase_bias_lambda,
                "phase_offset_max": self.phase_offset_max,
                "phase_num_freqs": self.phase_num_freqs,
                "phase_bias_sigma": self.phase_bias_sigma,
                "phase_bias_dropout": self.phase_bias_dropout,
                "phase_bias_clamp_min": self.phase_bias_clamp_min,
                "use_rms_writer": self.use_rms_writer,
                "write_alpha_init": self.write_alpha_init,
                "write_alpha_max": self.write_alpha_max,
            }
        )
        return base

    def save_json(self, path: Path) -> None:
        """Persist the full config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfigV2":
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
