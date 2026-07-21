#!/usr/bin/env python3
"""
manifold_progress_probe — raw-audio mode.

Scans real audio files from a music dataset, finds lyrics/metadata, runs the
baseline ACE-Step 1.5 SFT model with teacher‐forcing of real VAE latents,
captures layer‑12 hidden states and cross-attention, and diagnoses whether
the low-dimensional manifold tangent contains lyric coverage velocity info.

Usage:
    python scripts/manifold_progress_probe.py \
        --mode raw_audio \
        --audio-dir /root/autodl-tmp/musicdata/audios \
        --tensor-dir /root/autodl-tmp/musicdata/train_tensors \
        --num-audio-samples 20 \
        --output-dir /root/autodl-tmp/manifold_progress_probe

Output layout:
    {output_dir}/
        selected_audio_files.json
        raw_audio_metadata_summary.json
        raw_audio_hooks/{sample_id}.pt
        raw_audio_figures/{sample_id}/*.png
        raw_audio_results.csv
        raw_audio_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
TENSOR_DIR_DEFAULT = Path("/root/autodl-tmp/musicdata/train_tensors")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
from acestep.training_v2.preprocess_vae import TARGET_SR, tiled_vae_encode

# ---------------------------------------------------------------------------
# Section vocabulary
# ---------------------------------------------------------------------------
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Manifold Progress Probe — raw audio")

# Mode
parser.add_argument("--mode", type=str, default="raw_audio", choices=["raw_audio"],
                    help="Diagnostic mode (default: raw_audio)")

# Data scanning
parser.add_argument("--audio-dir", type=str, default="/root/autodl-tmp/musicdata/audios",
                    help="Directory containing raw audio files")
parser.add_argument("--tensor-dir", type=str, default=str(TENSOR_DIR_DEFAULT),
                    help="Directory containing preprocessed .pt files")
parser.add_argument("--dataset-dir", type=str, default="/root/autodl-tmp/musicdata/dataset",
                    help="Directory containing .caption.txt / .lyrics.txt sidecar files")
parser.add_argument("--num-audio-samples", type=int, default=20,
                    help="Number of audio samples to random-select (default: 20)")
parser.add_argument("--min-duration", type=float, default=180,
                    help="Minimum audio duration in seconds (default: 180)")
parser.add_argument("--max-duration", type=float, default=300,
                    help="Maximum audio duration in seconds (default: 300)")

# Model
parser.add_argument("--model-variant", type=str, default="sft",
                    help="Model variant (default: sft)")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device (default: cuda)")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    help="Model dtype (default: bfloat16)")

# Teacher-forcing noise level
parser.add_argument("--teacher-noise", type=float, default=0.0,
                    help="Noise level for teacher-forcing t (default: 0.0 = clean)")

# Output
parser.add_argument("--output-dir", type=str,
                    default="/root/autodl-tmp/manifold_progress_probe",
                    help="Output directory")

# ---------------------------------------------------------------------------
# Fixed Progress Bias — intervention experiment
# ---------------------------------------------------------------------------
parser.add_argument("--use-fixed-progress-bias", action="store_true",
                    help="Enable fixed linear progress bias on cross-attn logits")
parser.add_argument("--compare-fixed-progress-bias", action="store_true",
                    help="Compare baseline vs fixed progress bias on same samples")
parser.add_argument("--progress-bias-layer", type=int, default=12,
                    help="Layer index for progress bias (default: 12)")
parser.add_argument("--progress-sigma", type=float, default=0.15,
                    help="Lyric position window width (default: 0.15)")
parser.add_argument("--progress-lambda", type=float, default=1.0,
                    help="Bias strength multiplier (default: 1.0)")
parser.add_argument("--progress-max-bias", type=float, default=3.0,
                    help="Max |bias| clamp value (default: 3.0)")
parser.add_argument("--progress-gate", type=float, default=1.0,
                    help="Bias gate multiplier at inference (default: 1.0)")
parser.add_argument("--progress-lyric-mask-mode", type=str, default="section_positive",
                    choices=["section_positive"],
                    help="Lyric mask mode (default: section_positive)")

# Analysis
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for sample selection")
parser.add_argument("--pca-dims", type=str, default="2,4,8,16",
                    help="PCA dimensions for manifold analysis (comma-separated)")

args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
PCA_DIMS = [int(x) for x in args.pca_dims.split(",")]

# Shortcut for progress bias params
PB = {
    "layer": args.progress_bias_layer,
    "sigma": args.progress_sigma,
    "lambda_": args.progress_lambda,
    "max_bias": args.progress_max_bias,
    "gate": args.progress_gate,
    "mask_mode": args.progress_lyric_mask_mode,
}


# ===================================================================
#  PART 1 — Audio scanning & metadata matching
# ===================================================================

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def scan_audio_files(audio_dir: Path) -> List[Path]:
    """Recursively scan for audio files, return sorted list."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def get_audio_duration_ffprobe(path: Path) -> Optional[float]:
    """Get audio duration in seconds via ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def get_audio_duration_librosa(path: Path) -> Optional[float]:
    """Fallback duration via librosa (fast, no decode)."""
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True, duration=5)
        # Quick estimate from header — just get metadata
        return None
    except Exception:
        return None


def find_matching_pt(audio_stem: str, tensor_dir: Path) -> Optional[Path]:
    """Find matching preprocessed .pt by hash prefix (part before last '_')."""
    # audio stems:  hash_NNNN.mp3 -> check hash_NNNN.pt
    pt_path = tensor_dir / f"{audio_stem}.pt"
    if pt_path.is_file():
        return pt_path
    # Try without numeric suffix
    base = audio_stem.rsplit("_", 1)[0]
    for f in tensor_dir.glob(f"{base}_*.pt"):
        return f
    return None


def find_metadata_files(audio_stem: str, audio_dir: Path, dataset_dir: Path) -> dict:
    """Find sidecar metadata for an audio file.

    Priority: .json > .lrc > .txt (general), then look in dataset_dir for
    .caption.txt and .lyrics.txt files.
    """
    result = {"json": None, "lrc": None, "txt": None,
              "caption_txt": None, "lyrics_txt": None}

    # Same-directory sidecars
    for f in audio_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if f.stem == audio_stem:
            if f.suffix == ".json":
                result["json"] = f
            elif f.suffix == ".lrc":
                result["lrc"] = f
            elif f.suffix == ".txt":
                result["txt"] = f
        # Handle double extensions: foo.lyrics.txt, foo.caption.txt
        elif name == f"{audio_stem}.lyrics.txt":
            result["lyrics_txt"] = f
        elif name == f"{audio_stem}.caption.txt":
            result["caption_txt"] = f

    # Dataset-directory sidecars
    if dataset_dir is not None:
        cap = dataset_dir / f"{audio_stem}.caption.txt"
        if cap.is_file():
            result["caption_txt"] = cap
        lyr = dataset_dir / f"{audio_stem}.lyrics.txt"
        if lyr.is_file():
            result["lyrics_txt"] = lyr

    return result


def read_lyrics(meta_files: dict) -> Optional[str]:
    """Try metadata files in priority order, return lyrics text or None."""
    # 1. .json: try keys "lyrics", "lyric", "text"
    if meta_files.get("json"):
        try:
            with open(meta_files["json"]) as f:
                data = json.load(f)
            for key in ("lyrics", "lyric", "text"):
                if key in data and data[key]:
                    return str(data[key])
        except Exception:
            pass

    # 2. .lrc
    if meta_files.get("lrc"):
        try:
            lines = meta_files["lrc"].read_text(encoding="utf-8")
            if lines.strip():
                return lines
        except Exception:
            pass

    # 3. .txt
    if meta_files.get("txt"):
        try:
            lines = meta_files["txt"].read_text(encoding="utf-8")
            if lines.strip():
                return lines
        except Exception:
            pass

    # 4. .lyrics.txt (from dataset/)
    if meta_files.get("lyrics_txt"):
        try:
            lines = meta_files["lyrics_txt"].read_text(encoding="utf-8")
            if lines.strip():
                return lines
        except Exception:
            pass

    return None


def read_caption(meta_files: dict) -> Optional[str]:
    """Read caption prompt if available."""
    if meta_files.get("caption_txt"):
        try:
            return meta_files["caption_txt"].read_text(encoding="utf-8").strip()
        except Exception:
            pass
    if meta_files.get("json"):
        try:
            with open(meta_files["json"]) as f:
                data = json.load(f)
            for key in ("caption", "prompt", "tags"):
                if key in data and data[key]:
                    return str(data[key])
        except Exception:
            pass
    return None


def load_preprocessed_data(pt_path: Path) -> dict:
    """Load all tensors from a preprocessed .pt file (teacher-forcing source).

    Tensors are stored squeezed (no batch dim) — see preprocess.py lines 439-441.
    We add batch dim back before feeding to the decoder.
    """
    data = torch.load(str(pt_path), weights_only=True, map_location="cpu")
    return {
        "target_latents": data["target_latents"],                   # [T, 64]
        "attention_mask": data["attention_mask"],                   # [T]
        "encoder_hidden_states": data["encoder_hidden_states"],     # [L, D]
        "encoder_attention_mask": data["encoder_attention_mask"],   # [L]
        "context_latents": data["context_latents"],                 # [T', 128]
        "metadata": data.get("metadata", {}),
    }


def generate_sample_id(audio_path: Path) -> str:
    """Short unique sample ID derived from audio filename."""
    return audio_path.stem


# ===================================================================
#  PART 2 — Model & hooks
# ===================================================================

class HiddenCollector:
    """Forward hook: collect layer12 hidden state (mean over heads)."""
    def __init__(self):
        self.hidden = []  # list of [B, T, D]

    def __call__(self, module, input, output):
        hs = output[0]  # [B, T, D]
        self.hidden.append(hs.detach().cpu())

    def get(self):
        if not self.hidden:
            return None
        return torch.cat(self.hidden, dim=0)  # [1, T, D] (single forward)


class CrossAttnCollector:
    """Forward hook on cross-attention module: capture attention weights."""
    def __init__(self):
        self.weights = []  # list of [B, H, T, L]

    def __call__(self, module, input, output):
        # output is (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.weights.append(output[1].detach().cpu())

    def get(self):
        if not self.weights:
            return None
        # [N_forward_calls, B, H, T, L] -> squeeze to [H, T, L]
        stacked = torch.cat(self.weights, dim=0)  # [1, 1, H, T, L] typically
        return stacked.squeeze()  # [H, T, L]


def setup_model(device: str = "cuda") -> AceStepHandler:
    """Load baseline SFT model, disable all adapters."""
    dit_handler = AceStepHandler()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}

    dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit_handler.model.eval()

    # Disable Section-RoPE everywhere
    model.config.use_section_rope_offset = False
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_section_rope", False):
            layer_mod.use_section_rope = False
        if getattr(layer_mod, "use_phase_memory", False):
            layer_mod.use_phase_memory = False

    print(f"  Model: {type(model).__name__}, hidden_size={model.config.hidden_size}")
    print(f"  All adapters disabled (clean baseline)")
    return dit_handler


def register_hooks(model) -> Tuple[HiddenCollector, CrossAttnCollector, List]:
    """Register forward hooks on layer 12's decoder layer and cross-attn."""
    hidden_collector = HiddenCollector()
    attn_collector = CrossAttnCollector()

    handles = []
    # Hidden state hook on the full decoder layer
    h = model.decoder.layers[12].register_forward_hook(hidden_collector)
    handles.append(h)

    # Cross-attention hook
    h = model.decoder.layers[12].cross_attn.register_forward_hook(attn_collector)
    handles.append(h)

    # Force output_attentions=True by patching cross-attn's forward
    ca_module = model.decoder.layers[12].cross_attn
    orig_forward = ca_module.forward

    def _patched_forward(*fargs, **fkwargs):
        fkwargs["output_attentions"] = True
        return orig_forward(*fargs, **fkwargs)

    ca_module.forward = _patched_forward
    handles.append(lambda: setattr(ca_module, "forward", orig_forward))

    return hidden_collector, attn_collector, handles


def remove_hooks(handles: List):
    for h in handles:
        try:
            if callable(h):
                h()
            else:
                h.remove()
        except Exception:
            pass


# ===================================================================
#  PART 3 — Teacher-forcing forward pass
# ===================================================================

def run_teacher_forward(
    model,
    target_latents: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    t_noise: float = 0.0,
    device: str = "cuda",
) -> None:
    """Run a single decoder forward pass with (near-)clean latents.

    At t=0 (clean) the model predicts near-zero velocity; the forward pass
    still processes through all layers and produces meaningful hidden states
    and cross-attention.

    Args:
        t_noise: If > 0, add this much Gaussian noise for a
                 slightly-noisy teacher-forcing diagnostic.
    """
    T = target_latents.shape[0]
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # Preprocessed .pt stores all tensors squeezed (no batch dim).
    # See preprocess.py lines 437-441: .squeeze(0).cpu()
    hs = target_latents.unsqueeze(0).to(device=model_device, dtype=model_dtype)       # [1, T, 64]
    am = attention_mask.unsqueeze(0).to(device=model_device, dtype=model_dtype)        # [1, T]
    enc_hs = encoder_hidden_states.unsqueeze(0).to(device=model_device, dtype=model_dtype)  # [1, L, D]
    enc_am = encoder_attention_mask.unsqueeze(0).to(device=model_device, dtype=model_dtype)  # [1, L]
    ctx = context_latents.unsqueeze(0).to(device=model_device, dtype=model_dtype)      # [1, T', 128]

    # Optional noise injection
    if t_noise > 0:
        noise = torch.randn_like(hs)
        hs = (1 - t_noise) * hs + t_noise * noise

    # Timestep: use a scalar close to 0 for clean / lightly-noised latents
    t_tensor = torch.full((1,), t_noise, device=model_device, dtype=model_dtype)

    with torch.no_grad():
        _ = model.decoder(
            hidden_states=hs,
            timestep=t_tensor,
            timestep_r=t_tensor,
            attention_mask=am,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_am,
            context_latents=ctx,
            use_cache=False,
            output_attentions=True,
        )


# ===================================================================
#  PART 4 — Lyric position & attention centroid
# ===================================================================

def compute_lyric_pos(section_ids: torch.Tensor) -> np.ndarray:
    """Compute normalized lyric position for each non-UNKNOWN token.

    lyric_pos[j] = (position of token j among lyric tokens) / (count - 1)
    lyric tokens: section_ids in [1, 7].
    Returns float array of shape [L], with non-lyric tokens set to -1.
    """
    L = len(section_ids)
    pos = np.full(L, -1.0, dtype=np.float32)
    ids_np = section_ids.numpy() if isinstance(section_ids, torch.Tensor) else section_ids
    mask = (ids_np >= 1) & (ids_np <= 7)
    indices = np.where(mask)[0]
    if len(indices) > 1:
        for k, idx in enumerate(indices):
            pos[idx] = k / (len(indices) - 1)
    elif len(indices) == 1:
        pos[indices[0]] = 0.5
    return pos


# ===================================================================
#  Fixed Progress Bias — intervention construction
# ===================================================================

def compute_lyric_pos_and_mask(
    section_ids: torch.Tensor, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute lyric_pos and lyric_mask for progress bias construction.

    lyric_mask = (section_ids >= 1) & (section_ids <= 7)
    For valid lyric tokens, r_j is linearly interpolated in [0, 1].
    Non-lyric tokens get r_j = 0.0 (masked to 0 in bias).

    Returns:
        lyric_pos: [L] float32 tensor, r_j in [0, 1] for lyric tokens else 0
        lyric_mask: [L] bool tensor, True for lyric tokens
    """
    section_ids_t = section_ids.to(device) if isinstance(section_ids, torch.Tensor) else torch.tensor(section_ids, device=device)
    lyric_mask = (section_ids_t >= 1) & (section_ids_t <= 7)
    L = len(section_ids_t)
    lyric_pos = torch.zeros(L, dtype=torch.float32, device=device)

    valid_indices = torch.where(lyric_mask)[0]
    n_valid = len(valid_indices)
    if n_valid > 1:
        positions = torch.arange(n_valid, dtype=torch.float32, device=device) / (n_valid - 1)
        lyric_pos[valid_indices] = positions
    elif n_valid == 1:
        lyric_pos[valid_indices] = 0.5

    return lyric_pos, lyric_mask


def build_progress_bias(
    T_audio: int,
    lyric_pos: torch.Tensor,
    lyric_mask: torch.Tensor,
    sigma: float = 0.15,
    lambda_: float = 1.0,
    max_bias: float = 3.0,
    gate: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Build fixed linear progress bias for cross-attention logits.

    bias_ij = -lambda * ((r_j - p_i) / sigma)^2
    clamped to [-max_bias, 0], masked for non-lyric tokens.

    Args:
        T_audio: Number of audio time steps.
        lyric_pos: [L] float32 tensor, r_j in [0, 1].
        lyric_mask: [L] bool tensor, True for valid lyric tokens.
        sigma: Gaussian width.
        lambda_: Strength.
        max_bias: Max absolute clamp.
        gate: Inference-time multiplier.
        device: Torch device.

    Returns:
        bias: [1, 1, T, L] float32 tensor, values in [-max_bias * gate, 0].
    """
    L = lyric_pos.shape[0]

    # Audio progress p_i in [0, 1]
    audio_idx = torch.arange(T_audio, device=device)
    progress = audio_idx.float() / max(T_audio - 1, 1)  # [T]

    # Distance r_j - p_i
    # lyric_pos: [L] → [1, L]; progress: [T] → [T, 1]
    dist = lyric_pos.unsqueeze(0) - progress.unsqueeze(1)  # [T, L]

    # Gaussian bias
    bias = -lambda_ * (dist / sigma) ** 2
    bias = bias.clamp(min=-max_bias, max=0.0)
    bias = bias * lyric_mask.unsqueeze(0).float()
    bias = bias * gate

    # Add batch and head dims
    bias = bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, L]

    return bias.contiguous()


def _make_biased_attention_forward():
    """Return a modified ``eager_attention_forward`` that injects
    ``module._progress_bias`` into attention logits **before** softmax.

    Used by ``ProgressBiasContext`` — only modules with ``_progress_bias``
    attribute set are affected; all others behave identically to the original.
    """
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    def biased_forward(module, query, key, value, attention_mask,
                       scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # ═══════════════════════════════════════════════════════════════
        # Inject progress bias **before** softmax
        # ═══════════════════════════════════════════════════════════════
        progress_bias = getattr(module, "_progress_bias", None)
        if progress_bias is not None:
            if attn_weights.shape[-2:] == progress_bias.shape[-2:]:
                attn_weights = attn_weights + progress_bias.to(
                    dtype=attn_weights.dtype, device=attn_weights.device,
                )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                    dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout,
                                                    training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return biased_forward


class ProgressBiasContext:
    """Context manager that patches ``eager_attention_forward`` globally
    to inject a fixed progress bias into a **specific layer**'s cross-attention.

    Usage::

        bias_t = build_progress_bias(T, lyric_pos, lyric_mask, ...)
        with ProgressBiasContext(model, bias_t, layer=12):
            run_teacher_forward(model, ...)

    On exit the original ``eager_attention_forward`` is restored and the
    per-module bias attribute is cleared.
    """

    def __init__(self, model, bias_tensor: torch.Tensor, layer: int = 12):
        self.model = model
        self.bias_tensor = bias_tensor
        self.layer = layer
        self.ca_module = model.decoder.layers[layer].cross_attn
        self._orig_eaf = None

    def __enter__(self):
        # Store bias on the cross-attn module (read by biased_forward)
        ref = next(self.ca_module.parameters())
        self.ca_module._progress_bias = self.bias_tensor.to(
            device=ref.device, dtype=ref.dtype,
        )

        # CRITICAL: patch eager_attention_forward in the SAME module that
        # the model's AceStepAttention.forward resolves against.
        # The model may be loaded from a different path (e.g. transformers_modules/)
        # than the source checkout, so we must get the module from the instance.
        import sys
        cls = type(self.ca_module)
        attn_mod = sys.modules[cls.__module__]
        self._orig_eaf = attn_mod.eager_attention_forward
        attn_mod.eager_attention_forward = _make_biased_attention_forward()
        self._patched_mod = attn_mod
        return self

    def __exit__(self, *args):
        if hasattr(self, "_patched_mod") and self._orig_eaf is not None:
            self._patched_mod.eager_attention_forward = self._orig_eaf
        if hasattr(self.ca_module, "_progress_bias"):
            del self.ca_module._progress_bias


def run_teacher_with_hooks(
    model,
    pt_data: dict,
    t_noise: float = 0.0,
    device: str = "cuda",
    use_progress_bias: bool = False,
    progress_bias_params: Optional[dict] = None,
) -> Optional[dict]:
    """Run a single teacher-forcing forward pass, collect hooks, compute metrics.

    Args:
        use_progress_bias: If True, apply fixed progress bias.
        progress_bias_params: Dict with keys ``bias_tensor``, ``layer``.

    Returns:
        dict with ``H_np``, ``A_np``, ``H_tensor``, ``A_tensor``,
        ``T_eff``, ``L_eff``, ``T_raw``, ``L_raw``, or None on error.
    """
    target_latents = pt_data["target_latents"]
    attention_mask = pt_data["attention_mask"]
    encoder_hidden_states = pt_data["encoder_hidden_states"]
    encoder_attention_mask = pt_data["encoder_attention_mask"]
    context_latents = pt_data["context_latents"]

    T_raw = target_latents.shape[0]
    L_raw = encoder_hidden_states.shape[0]

    hidden_collector, attn_collector, handles = register_hooks(model)

    try:
        if use_progress_bias and progress_bias_params is not None:
            bias_t = progress_bias_params["bias_tensor"]
            layer = progress_bias_params.get("layer", 12)
            with ProgressBiasContext(model, bias_t, layer=layer):
                run_teacher_forward(
                    model, target_latents, attention_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    context_latents, t_noise=t_noise, device=device,
                )
        else:
            run_teacher_forward(
                model, target_latents, attention_mask,
                encoder_hidden_states, encoder_attention_mask,
                context_latents, t_noise=t_noise, device=device,
            )
    except Exception:
        remove_hooks(handles)
        raise

    H_tensor = hidden_collector.get()
    A_tensor = attn_collector.get()
    remove_hooks(handles)

    if H_tensor is None or A_tensor is None:
        return None

    H_np = H_tensor.squeeze(0).float().numpy()    # [T_eff, D]
    A_np = A_tensor.float().numpy()                # [H, T_eff, L_eff]
    T_eff, L_eff = H_np.shape[0], A_np.shape[2]

    return {
        "H_np": H_np,
        "A_np": A_np,
        "H_tensor": H_tensor,
        "A_tensor": A_tensor,
        "T_eff": T_eff,
        "L_eff": L_eff,
        "T_raw": T_raw,
        "L_raw": L_raw,
    }


def compute_attention_centroid(
    attn: np.ndarray,
    lyric_pos: np.ndarray,
) -> np.ndarray:
    """Compute centroid of attention over lyric positions.

    Args:
        attn: [H, T, L] — attention weights
        lyric_pos: [L] — normalized lyric positions (-1 for UNKNOWN)

    Returns:
        c: [H, T] — centroid per head and audio time step
    """
    H, T, L = attn.shape
    c = np.zeros((H, T), dtype=np.float32)
    valid = lyric_pos >= 0
    if not valid.any():
        return c
    valid_pos = lyric_pos[valid]
    for h in range(H):
        for t in range(T):
            w = attn[h, t, valid]
            total = w.sum()
            if total > 1e-10:
                c[h, t] = (w * valid_pos).sum() / total
    return c  # [H, T]


# ===================================================================
#  PART 5 — Coverage & centroid metrics
# ===================================================================

def compute_coverage_metrics(
    attn: np.ndarray, lyric_pos: np.ndarray
) -> dict:
    """Compute coverage and centroid dynamics metrics.

    Args:
        attn: [H, T, L]
        lyric_pos: [L]
    """
    H, T, L = attn.shape

    # Mean over heads
    attn_mean = attn.mean(axis=0)  # [T, L]

    # Centroid
    c = compute_attention_centroid(attn, lyric_pos)  # [H, T]
    c_mean = c.mean(axis=0)  # [T]

    # Centroid dynamics
    delta_c = np.diff(c_mean)  # [T-1]

    # Reversal: centroid goes backward significantly
    reversal_rate = float(np.mean(delta_c < -0.005))
    # Jump: centroid jumps forward > 5% of lyric range in one step
    jump_rate = float(np.mean(delta_c > 0.05))
    # Stagnant: centroid barely moves
    stagnant_rate = float(np.mean(np.abs(delta_c) < 0.001))

    # Spearman correlation of centroid with linear time
    from scipy.stats import spearmanr
    time_lin = np.linspace(0, 1, T)
    corr, _ = spearmanr(c_mean, time_lin)
    centroid_spearman_time = float(corr)

    # Centroid range
    centroid_range = float(c_mean.max() - c_mean.min())

    # Coverage distribution
    coverage_j = attn_mean.sum(axis=0)  # [L]
    coverage_j = coverage_j / (coverage_j.sum() + 1e-10)

    # Coverage entropy
    eps = 1e-10
    coverage_entropy = float(-np.sum(coverage_j * np.log(coverage_j + eps)) / np.log(L))

    # Gini coefficient
    sorted_cov = np.sort(coverage_j)
    cumsum = np.cumsum(sorted_cov)
    gini = float(1 - 2 * np.sum(cumsum) / (L * cumsum[-1] + eps))

    # Coverage hole ratio: fraction of lyrics tokens with near-zero coverage
    hole_ratio = float(np.mean(coverage_j < 0.01 / L))

    # Over-coverage ratio: fraction of total weight on top 20% tokens
    top_k = max(1, L // 5)
    top_idx = np.argsort(coverage_j)[-top_k:]
    over_coverage_ratio = float(coverage_j[top_idx].sum())

    # Effective rank of attention
    s = np.linalg.svd(attn_mean, compute_uv=False)
    p = s / (s.sum() + eps)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + eps))))

    return {
        "reversal_rate": reversal_rate,
        "jump_rate": jump_rate,
        "stagnant_rate": stagnant_rate,
        "centroid_spearman_time": centroid_spearman_time,
        "centroid_range": centroid_range,
        "coverage_entropy": coverage_entropy,
        "coverage_gini": gini,
        "coverage_hole_ratio": hole_ratio,
        "over_coverage_ratio": over_coverage_ratio,
        "attention_effective_rank": eff_rank,
    }


# ===================================================================
#  PART 6 — Low-dimensional manifold / tangent analysis
# ===================================================================

def run_manifold_analysis(
    H: np.ndarray,  # [T, D]
    c: np.ndarray,  # [T] — mean centroid
    delta_c: np.ndarray,  # [T-1] — centroid delta
    pc_dims: List[int],
) -> dict:
    """Run PCA + linear probes on hidden states.

    Controls:
        - Random projection (same dims as PCA)
        - Shuffled tangent directions
        - Shuffled lyric_pos / centroid

    Returns dict of probe metrics with train/val split.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    T = H.shape[0]
    results = {}

    # ---- Time features (baseline) ----
    audio_time = np.linspace(0, 1, T).reshape(-1, 1)  # [T, 1]

    # ---- Split indices for all probes ----
    idx = np.arange(T)
    if T >= 10:
        train_idx, val_idx = train_test_split(idx, test_size=0.3, random_state=42)
    else:
        # Too small for split, use all for train
        train_idx, val_idx = idx, idx

    # c and delta_c targets
    c_target = c  # [T]
    dc_target = delta_c  # [T-1]

    # ---- Helper: probe function ----
    def _probe(X: np.ndarray, y_target: np.ndarray,
               tmask: np.ndarray, vmask: np.ndarray) -> dict:
        """Linear regression from X to y_target, return metrics on val split.

        Returns dict with ``r2``, ``spearman``, ``mae``.  Returns all -999
        when the regression is degenerate (too few samples, constant target,
        NaN features).
        """
        if X.shape[0] != y_target.shape[0]:
            # Mismatch — likely delta_c vs time
            return {"r2": -999.0, "spearman": -999.0, "mae": -999.0}

        if len(tmask) < 2 or len(vmask) < 1 or X.shape[0] < 2:
            return {"r2": -999.0, "spearman": -999.0, "mae": -999.0}

        # Check target variance — degenerate targets produce meaningless R²
        y_var = np.var(y_target)
        if y_var < 1e-10:
            return {"r2": 0.0, "spearman": 0.0, "mae": float(np.mean(np.abs(y_target)))}

        X_train, y_train = X[tmask], y_target[tmask]
        X_val, y_val = X[vmask], y_target[vmask]

        try:
            reg = LinearRegression().fit(X_train, y_train)
            y_pred = reg.predict(X_val)

            # Protect against NaN predictions
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return {"r2": -999.0, "spearman": -999.0, "mae": -999.0}

            r2 = float(r2_score(y_val, y_pred))
            mae = float(np.mean(np.abs(y_pred - y_val)))

            if np.std(y_pred) > 1e-10 and np.std(y_val) > 1e-10:
                sp, _ = spearmanr(y_pred.flatten(), y_val.flatten())
            else:
                sp = 0.0
            return {"r2": r2, "spearman": float(sp), "mae": mae}
        except Exception:
            return {"r2": -999.0, "spearman": -999.0, "mae": -999.0}

    # ---- 1. Baseline: audio_time -> c ----
    tmask = np.isin(idx, train_idx)
    vmask = np.isin(idx, val_idx)
    probe_time_c = _probe(audio_time, c_target, tmask, vmask)
    results["probe_audio_time_R2"] = probe_time_c["r2"]
    results["probe_audio_time_spearman"] = probe_time_c["spearman"]
    results["probe_audio_time_MAE"] = probe_time_c["mae"]

    # ---- 2. Baseline: audio_time -> delta_c ----
    # Align: delta_c has T-1 points, audio_time has T
    if T > 1:
        dc_time = audio_time[:-1]
        dc_tmask = np.isin(np.arange(T - 1), train_idx[train_idx < T - 1])
        dc_vmask = np.isin(np.arange(T - 1), val_idx[val_idx < T - 1])
        probe_dt = _probe(dc_time, dc_target, dc_tmask, dc_vmask)
        results["probe_delta_time_R2"] = probe_dt["r2"]
    else:
        results["probe_delta_time_R2"] = -999.0

    # ---- 3. PCA on H ----
    H_centered = H - H.mean(axis=0, keepdims=True)

    for k in pc_dims:
        k_actual = min(k, T, H.shape[1])
        if k_actual < 1:
            continue

        pca = PCA(n_components=k_actual)
        Z = pca.fit_transform(H_centered)  # [T, k]
        ev_ratio = pca.explained_variance_ratio_.sum()
        results[f"pca_explained_var_top{k}"] = float(ev_ratio)
        results[f"pca_components_{k}"] = k_actual

        if k == pc_dims[0]:
            # Store PC1 for visualization
            results["pc1"] = Z[:, 0].tolist() if T <= 5000 else Z[:5000, 0].tolist()
            results["pc1_var"] = float(pca.explained_variance_ratio_[0])

        # Probe: Z -> c
        tmask_zk = np.isin(idx, train_idx)
        vmask_zk = np.isin(idx, val_idx)
        probe_z = _probe(Z, c_target, tmask_zk, vmask_zk)
        results[f"probe_z{k}_R2"] = probe_z["r2"]
        results[f"probe_z{k}_spearman"] = probe_z["spearman"]
        results[f"probe_z{k}_MAE"] = probe_z["mae"]

        # Probe: tangent [Z_t, Z_t - Z_{t-1}] -> delta_c
        if T >= 3:
            Z_prev = Z[:-1]  # [T-1, k]
            Z_curr = Z[1:]   # [T-1, k]
            tangent = Z_curr - Z_prev  # [T-1, k]
            Z_tangent = np.concatenate([Z_prev, tangent], axis=1)  # [T-1, 2k]
            dc_arr = dc_target[:len(Z_prev)]

            tmask_tg = np.isin(np.arange(T - 1), train_idx[train_idx < T - 1])
            vmask_tg = np.isin(np.arange(T - 1), val_idx[val_idx < T - 1])
            probe_tg = _probe(Z_tangent, dc_arr, tmask_tg, vmask_tg)
            results[f"probe_z{k}_tangent_R2"] = probe_tg["r2"]
            results[f"probe_z{k}_tangent_spearman"] = probe_tg["spearman"]
            results[f"probe_z{k}_tangent_MAE"] = probe_tg["mae"]
        else:
            results[f"probe_z{k}_tangent_R2"] = -999.0

    # ---- 4. Controls: random projection ----
    for k in pc_dims:
        k_actual = min(k, H.shape[1])
        if k_actual < 1:
            continue
        W = np.random.randn(H.shape[1], k_actual).astype(np.float32)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        Z_rand = H_centered @ W  # [T, k]

        tmask_r = np.isin(idx, train_idx)
        vmask_r = np.isin(idx, val_idx)
        probe_rand = _probe(Z_rand, c_target, tmask_r, vmask_r)
        results[f"random_projection_{k}_R2"] = probe_rand["r2"]
        results[f"random_projection_{k}_spearman"] = probe_rand["spearman"]

    # ---- 5. Control: shuffled tangent ----
    if T >= 3:
        for k in pc_dims:
            key_z = f"probe_z{k}_tangent_R2"
            if key_z not in results:
                continue
            # Re-fit PCA
            k_actual = min(k, T, H.shape[1])
            pca = PCA(n_components=k_actual)
            Z = pca.fit_transform(H_centered)
            Z_prev = Z[:-1]
            Z_curr = Z[1:]
            tangent = Z_curr - Z_prev
            # Shuffle tangent rows
            np.random.shuffle(tangent)
            Z_tangent_shuf = np.concatenate([Z_prev, tangent], axis=1)
            dc_arr = dc_target[:len(Z_prev)]
            tmask_sh = np.isin(np.arange(T - 1), train_idx[train_idx < T - 1])
            vmask_sh = np.isin(np.arange(T - 1), val_idx[val_idx < T - 1])
            probe_sh = _probe(Z_tangent_shuf, dc_arr, tmask_sh, vmask_sh)
            results[f"shuffled_tangent_{k}_R2"] = probe_sh["r2"]
            results[f"shuffled_tangent_{k}_spearman"] = probe_sh["spearman"]
    else:
        for k in pc_dims:
            results[f"shuffled_tangent_{k}_R2"] = -999.0

    # ---- 6. Effective rank ----
    s = np.linalg.svd(H_centered, compute_uv=False)
    p = s / (s.sum() + 1e-10)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    results["effective_rank"] = eff_rank

    return results


# ===================================================================
#  PART 7 — Visualization
# ===================================================================

def visualize_sample(
    sample_id: str,
    H: np.ndarray,
    attn: np.ndarray,
    c: np.ndarray,
    lyric_pos: np.ndarray,
    section_ids: torch.Tensor,
    probe_results: dict,
    output_dir: Path,
):
    """Generate 4 diagnostic plots per sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "raw_audio_figures" / sample_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    T = attn.shape[1]
    L = attn.shape[2]
    time_axis = np.arange(T) / T  # normalized time

    c_mean = c.mean(axis=0)  # [T]

    # ---- 1. Attention centroid vs time ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, c_mean, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Attention centroid (lyric pos)")
    ax.set_title(f"Attention Centroid — {sample_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "attention_centroid.png", dpi=150)
    plt.close(fig)

    # ---- 2. PC1 vs time (first PC of hidden states) ----
    H_centered = H - H.mean(axis=0, keepdims=True)
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=min(2, T, H.shape[1]))
    Z = pca_2d.fit_transform(H_centered)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, Z[:, 0], color="coral", linewidth=1.2)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    ax.set_title(f"PC1 of Hidden States vs Time — {sample_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pc1_vs_time.png", dpi=150)
    plt.close(fig)

    # ---- 3. PC1 vs centroid overlay ----
    fig, ax1 = plt.subplots(figsize=(10, 4))
    # Normalize both to [0, 1]
    z_norm = (Z[:, 0] - Z[:, 0].min()) / (Z[:, 0].max() - Z[:, 0].min() + 1e-10)
    c_norm = (c_mean - c_mean.min()) / (c_mean.max() - c_mean.min() + 1e-10)

    ax1.plot(time_axis, z_norm, color="coral", linewidth=1.2, label="PC1 (norm)")
    ax1.plot(time_axis, c_norm, color="steelblue", linewidth=1.2, label="Centroid (norm)")
    ax1.set_xlabel("Normalized audio time")
    ax1.set_ylabel("Normalized value")
    ax1.set_title(f"PC1 vs Centroid — {sample_id}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pc1_vs_centroid.png", dpi=150)
    plt.close(fig)

    # ---- 4. Tangent vs delta centroid ----
    if T >= 3:
        delta_c = np.diff(c_mean)
        # Predict delta_c from [z, m] using top-4 PCA
        k = min(4, Z.shape[1])
        Z_k = Z[:, :k]
        Z_prev = Z_k[:-1]
        Z_curr = Z_k[1:]
        tangent = Z_curr - Z_prev
        Z_tangent = np.concatenate([Z_prev, tangent], axis=1)

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(Z_tangent, delta_c)
        delta_pred = reg.predict(Z_tangent)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_axis[1:], delta_c, color="gray", linewidth=1.0, alpha=0.7, label="delta centroid")
        ax.plot(time_axis[1:], delta_pred, color="darkgreen", linewidth=1.2, label="predicted from [z, m]")
        ax.set_xlabel("Normalized audio time")
        ax.set_ylabel("Delta centroid / prediction")
        ax.set_title(f"Tangent vs Delta Centroid — {sample_id}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "tangent_vs_delta_centroid.png", dpi=150)
        plt.close(fig)

    print(f"  Figures -> {fig_dir}")


# ===================================================================
#  PART 7B — Comparison visualization (baseline vs intervention)
# ===================================================================

def visualize_comparison(
    sample_id: str,
    c_base: np.ndarray,
    c_intv: np.ndarray,
    attn_base: np.ndarray,
    attn_intv: np.ndarray,
    lyric_pos: np.ndarray,
    output_dir: Path,
):
    """Generate 6 comparison plots: centroid overlay, centroid individual,
    and coverage distribution overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures" / sample_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    T = c_base.shape[0]
    time_axis = np.arange(T) / T

    c_base_mean = c_base.mean(axis=0) if c_base.ndim > 1 else c_base
    c_intv_mean = c_intv.mean(axis=0) if c_intv.ndim > 1 else c_intv

    # ---- 1. Baseline centroid ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, c_base_mean, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Attention centroid (lyric pos)")
    ax.set_title(f"Attention Centroid (Baseline) — {sample_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "centroid_baseline.png", dpi=150)
    plt.close(fig)

    # ---- 2. Intervention centroid ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, c_intv_mean, color="darkgreen", linewidth=1.2)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Attention centroid (lyric pos)")
    ax.set_title(f"Attention Centroid (Fixed Bias) — {sample_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "centroid_intervention.png", dpi=150)
    plt.close(fig)

    # ---- 3. Centroid overlay ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, c_base_mean, color="steelblue", linewidth=1.2,
            alpha=0.8, label="Baseline")
    ax.plot(time_axis, c_intv_mean, color="darkgreen", linewidth=1.2,
            alpha=0.8, label="Fixed bias")
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Attention centroid (lyric pos)")
    ax.set_title(f"Attention Centroid Overlay — {sample_id}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "centroid_overlay.png", dpi=150)
    plt.close(fig)

    # ---- 4. Coverage distribution (baseline) ----
    H_base, L_base = attn_base.shape[1], attn_base.shape[2]
    valid = lyric_pos >= 0
    if valid.any():
        attn_mean_base = attn_base.mean(axis=0)  # [T, L]
        coverage_base = attn_mean_base.sum(axis=0)
        coverage_base = coverage_base / (coverage_base.sum() + 1e-10)

        attn_mean_intv = attn_intv.mean(axis=0)
        coverage_intv = attn_mean_intv.sum(axis=0)
        coverage_intv = coverage_intv / (coverage_intv.sum() + 1e-10)

        lyric_idx = np.where(valid)[0]
        sorted_base = np.sort(coverage_base[lyric_idx])
        sorted_intv = np.sort(coverage_intv[lyric_idx])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(sorted_base)), sorted_base, width=1.0,
               color="steelblue", alpha=0.6, label="Baseline")
        ax.set_xlabel("Lyric token (sorted by coverage)")
        ax.set_ylabel("Total attention weight")
        ax.set_title(f"Coverage Distribution (Baseline) — {sample_id}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "coverage_baseline.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(sorted_intv)), sorted_intv, width=1.0,
               color="darkgreen", alpha=0.6, label="Fixed bias")
        ax.set_xlabel("Lyric token (sorted by coverage)")
        ax.set_ylabel("Total attention weight")
        ax.set_title(f"Coverage Distribution (Fixed Bias) — {sample_id}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "coverage_intervention.png", dpi=150)
        plt.close(fig)

        # ---- 5. Coverage overlay ----
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(len(sorted_base)) - 0.15, sorted_base, width=0.3,
               color="steelblue", alpha=0.6, label="Baseline")
        ax.bar(np.arange(len(sorted_intv)) + 0.15, sorted_intv, width=0.3,
               color="darkgreen", alpha=0.6, label="Fixed bias")
        ax.set_xlabel("Lyric token (sorted by coverage)")
        ax.set_ylabel("Total attention weight")
        ax.set_title(f"Coverage Distribution Overlay — {sample_id}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "coverage_overlay.png", dpi=150)
        plt.close(fig)

    print(f"  Comparison figures -> {fig_dir}")


# ===================================================================
#  PART 8 — Summary printing
# ===================================================================

def print_summary(
    scan_results: dict,
    sample_results: List[dict],
):
    """Print data availability and conclusion statements."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Data availability
    print("\n--- Data Availability ---")
    print(f"  Scanned audio files:       {scan_results.get('scanned', 0)}")
    print(f"  Valid (duration filter):   {scan_results.get('valid_long', 0)}")
    print(f"  With lyrics:               {scan_results.get('with_lyrics', 0)}")
    print(f"  Skipped (no lyrics):       {scan_results.get('skipped_no_lyrics', 0)}")
    print(f"  Skipped (too short):       {scan_results.get('skipped_too_short', 0)}")
    print(f"  Actually processed:        {len(sample_results)}")

    if not sample_results:
        print("\n⚠  No samples processed — nothing to conclude.")
        return

    # Aggregate metrics
    reversal_rates = [r.get("reversal_rate", 0) for r in sample_results]
    jump_rates = [r.get("jump_rate", 0) for r in sample_results]
    hole_ratios = [r.get("coverage_hole_ratio", 0) for r in sample_results]
    r2_tangent = [r.get("probe_z8_tangent_R2", -999) for r in sample_results]
    r2_time = [r.get("probe_audio_time_R2", -999) for r in sample_results]
    r2_random = [r.get("random_projection_8_R2", -999) for r in sample_results]
    r2_shuffled = [r.get("shuffled_tangent_8_R2", -999) for r in sample_results]

    # 1. Non-monotonic retrieval?
    mean_reversal = float(np.mean(reversal_rates))
    mean_jump = float(np.mean(jump_rates))
    mean_hole = float(np.mean(hole_ratios))
    print("\n--- Coverage-Free Retrieval ---")
    print(f"  Mean reversal_rate:  {mean_reversal:.4f}")
    print(f"  Mean jump_rate:      {mean_jump:.4f}")
    print(f"  Mean hole_ratio:     {mean_hole:.4f}")
    if mean_reversal > 0.01 or mean_jump > 0.05 or mean_hole > 0.1:
        print("  ➜ Baseline shows NON-MONOTONIC lyric retrieval behavior.")
    else:
        print("  ➜ Baseline appears roughly monotonic in lyric retrieval.")

    # 2. Manifold progress valid?
    valid_tangent = [v for v in r2_tangent if v > -998]
    valid_time = [v for v in r2_time if v > -998]
    valid_random = [v for v in r2_random if v > -998]
    valid_shuffled = [v for v in r2_shuffled if v > -998]

    print("\n--- Manifold Progress Evidence ---")
    if valid_tangent and valid_time and valid_random and valid_shuffled:
        mean_tangent = float(np.mean(valid_tangent))
        mean_time = float(np.mean(valid_time))
        mean_random = float(np.mean(valid_random))
        mean_shuffled = float(np.mean(valid_shuffled))
        print(f"  Mean R² [z,tangent] -> delta_c: {mean_tangent:.4f}")
        print(f"  Mean R² audio time  -> c:       {mean_time:.4f}")
        print(f"  Mean R² random proj -> c:       {mean_random:.4f}")
        print(f"  Mean R² shuffled tg -> delta_c: {mean_shuffled:.4f}")

        if mean_tangent > mean_time + 0.02 and mean_tangent > mean_random + 0.02:
            print("  ➜ Evidence supports that manifold tangent contains lyric coverage velocity")
            print("    information beyond linear audio time.")
        elif mean_tangent > mean_random + 0.01:
            print("  ➜ Weak evidence: manifold tangent slightly outperforms controls.")
        else:
            print("  ➜ No evidence that manifold tangent provides additional lyric coverage")
            print("    information beyond linear audio time.")
    else:
        print("  ➜ Insufficient data for manifold progress evaluation.")

    print("\n" + "=" * 70)


# ===================================================================
#  PART 8B — Comparison summary (baseline vs fixed progress bias)
# ===================================================================

COMPARISON_METRICS = [
    ("reversal_rate",            "reversal_rate",           True),
    ("jump_rate",                "jump_rate",               True),
    ("stagnant_rate",            "stagnant_rate",           False),
    ("coverage_hole_ratio",      "hole_ratio",              True),
    ("coverage_gini",            "coverage_gini",           True),
    ("over_coverage_ratio",      "over_coverage_ratio",     True),
    ("centroid_spearman_time",   "centroid_spearman_time",  False),
    ("centroid_range",           "centroid_range",          True),
]


def print_comparison_summary(
    baseline_results: list,
    intervention_results: list,
):
    """Print formatted comparison table with Delta and auto-conclusion."""
    print("\n" + "=" * 70)
    print("BASELINE vs FIXED PROGRESS BIAS — COMPARISON")
    print("=" * 70)

    if not baseline_results or not intervention_results:
        print("  Insufficient data for comparison.")
        return

    import numpy as np

    print(f"\n  Samples compared: {len(baseline_results)}")
    print(f"  Bias params: sigma={PB['sigma']}, lambda={PB['lambda_']}, "
          f"max_bias={PB['max_bias']}, gate={PB['gate']}")

    # Aggregate
    agg_rows = []
    for key, label, lower_better in COMPARISON_METRICS:
        b_vals = [r.get(key, float("nan")) for r in baseline_results]
        i_vals = [r.get(key, float("nan")) for r in intervention_results]
        b_mean = float(np.nanmean(b_vals))
        i_mean = float(np.nanmean(i_vals))
        delta = i_mean - b_mean
        agg_rows.append((label, b_mean, i_mean, delta, lower_better))

    # Print table
    print(f"\n  {'Metric':<30} {'Baseline':<12} {'FixedBias':<12} {'Delta':<12}  Dir")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}  {'-'*4}")
    for label, b, i, d, lower_better in agg_rows:
        arrow = "<" if (lower_better and d < 0) or (not lower_better and d > 0) else ">"
        print(f"  {label:<30} {b:<12.4f} {i:<12.4f} {d:<+12.4f}  {arrow}")

    # Collect verdicts
    improved = []
    worsened = []
    for label, b, i, d, lower_better in agg_rows:
        if not np.isfinite(d):
            continue
        if (lower_better and d < 0) or (not lower_better and d > 0):
            improved.append(label)
        elif (lower_better and d > 0) or (not lower_better and d < 0):
            worsened.append(label)

    # Auto-conclusion
    print("\n--- Auto-Conclusion ---")

    reversal_delta = None
    hole_delta = None
    spearman_delta = None
    for label, _, _, d, _ in agg_rows:
        if label == "reversal_rate":
            reversal_delta = d
        elif label == "hole_ratio":
            hole_delta = d
        elif label == "centroid_spearman_time":
            spearman_delta = d

    reversal_improved = reversal_delta is not None and reversal_delta < 0
    hole_improved = hole_delta is not None and hole_delta < 0

    if reversal_delta is not None and hole_delta is not None:
        if reversal_improved and hole_improved:
            print("  Case A - Fixed monotonic progress bias improves lyric retrieval coverage.")
            print("  reversal_rate down, hole_ratio down. Proceed to learnable tangent progress head.")
        elif reversal_improved and not hole_improved:
            print("  Case B - Bias improves monotonicity but may over-constrain coverage.")
            print("  reversal_rate down, but hole_ratio up. Try larger sigma or smaller gate.")
        elif not reversal_improved and not hole_improved:
            print("  Case C - Fixed progress bias has weak effect or metrics unchanged.")
            delta_thresh = 0.01
            if abs(reversal_delta) < delta_thresh and abs(hole_delta) < delta_thresh:
                print("  Metrics essentially unchanged. Need check insertion point / logits / mask.")
            else:
                print("  Metrics in unexpected direction. Verify bias construction.")
        else:
            print("  Case D - Fixed progress bias hurts retrieval.")
            print("  reversal_rate up or hole_ratio up. Reduce gate/lambda or verify lyric_pos.")

    print(f"\n  Improved metrics ({len(improved)}/{len(agg_rows)}): {', '.join(improved)}")
    if worsened:
        print(f"  Worsened metrics: {', '.join(worsened)}")

    print("\n" + "=" * 70)


# ===================================================================
# MAIN
# ===================================================================

def _run_compare_mode(model, samples, scan_results):
    """Run baseline vs fixed progress bias comparison on all samples.

    Each sample is processed twice (baseline then intervention), producing
    separate result lists and comparison figures.
    """
    import csv

    OUT = OUTPUT_DIR
    baseline_dir = OUT / "baseline"
    intervention_dir = OUT / "fixed_progress_bias"
    fig_dir_parent = OUT / "figures"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    intervention_dir.mkdir(parents=True, exist_ok=True)
    fig_dir_parent.mkdir(parents=True, exist_ok=True)

    baseline_results = []
    intervention_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'─' * 70}")
        print(f"[{idx+1}/{len(samples)}] {sample['sample_id']}")
        print(f"  Duration: {sample['duration']:.1f}s")

        sid = sample["sample_id"]
        lyrics = sample["lyrics"]
        pt_path_str = sample.get("pt_path")

        if pt_path_str is None or not Path(pt_path_str).is_file():
            print(f"  No preprocessed .pt found — skipping")
            continue

        try:
            pt_data = load_preprocessed_data(Path(pt_path_str))
            T_raw = pt_data["target_latents"].shape[0]
            L_raw = pt_data["encoder_hidden_states"].shape[0]
            print(f"  T_raw={T_raw}, L_raw={L_raw}")

            # ── [A] Baseline ──────────────────────────────────────────
            print(f"  [A] Baseline forward...")
            result_b = run_teacher_with_hooks(
                model, pt_data, t_noise=args.teacher_noise,
                device=args.device, use_progress_bias=False,
            )
            if result_b is None:
                print(f"  Baseline hook collection failed — skipping")
                continue

            T_eff = result_b["T_eff"]
            L_eff = result_b["L_eff"]

            # Parse section_ids (shared between runs)
            parser = LyricsStructureParser()
            parsed = parser.parse(lyrics, num_chunks=L_eff)
            section_ids = parsed.section_type_ids

            # ── [B] Build progress bias ───────────────────────────────
            lyric_pos_t, lyric_mask_t = compute_lyric_pos_and_mask(section_ids, device=args.device)
            bias_t = build_progress_bias(
                T_eff, lyric_pos_t, lyric_mask_t,
                sigma=PB["sigma"], lambda_=PB["lambda_"],
                max_bias=PB["max_bias"], gate=PB["gate"],
                device=args.device,
            )

            print(f"  [B] Intervention forward (sigma={PB['sigma']}, gate={PB['gate']})...")
            bias_params = {"bias_tensor": bias_t, "layer": PB["layer"]}
            result_i = run_teacher_with_hooks(
                model, pt_data, t_noise=args.teacher_noise,
                device=args.device, use_progress_bias=True,
                progress_bias_params=bias_params,
            )
            if result_i is None:
                print(f"  Intervention hook collection failed — skipping")
                continue

            # ── Metrics for baseline ──────────────────────────────────
            lyric_pos_np = compute_lyric_pos(section_ids)
            coverage_b = compute_coverage_metrics(result_b["A_np"], lyric_pos_np)
            c_h_b = [compute_attention_centroid(result_b["A_np"][np.newaxis, h], lyric_pos_np).squeeze(0) for h in range(result_b["A_np"].shape[0])]
            c_b = np.mean(c_h_b, axis=0)
            dc_b = np.diff(c_b) if T_eff > 1 else np.array([0.0])
            manifold_b = run_manifold_analysis(result_b["H_np"], c_b, dc_b, PCA_DIMS)

            entry_b = {
                "sample_id": sid, "audio_path": sample["audio_path"],
                "duration": sample["duration"],
                "T_raw": result_b["T_raw"], "L_raw": result_b["L_raw"],
                "T_eff": result_b["T_eff"], "L_eff": result_b["L_eff"],
                **coverage_b,
                **{k: v for k, v in manifold_b.items() if not k.startswith("pc1")},
            }
            baseline_results.append(entry_b)

            # ── Metrics for intervention ──────────────────────────────
            coverage_i = compute_coverage_metrics(result_i["A_np"], lyric_pos_np)
            c_h_i = [compute_attention_centroid(result_i["A_np"][np.newaxis, h], lyric_pos_np).squeeze(0) for h in range(result_i["A_np"].shape[0])]
            c_i = np.mean(c_h_i, axis=0)
            dc_i = np.diff(c_i) if T_eff > 1 else np.array([0.0])
            manifold_i = run_manifold_analysis(result_i["H_np"], c_i, dc_i, PCA_DIMS)

            entry_i = {
                "sample_id": sid, "audio_path": sample["audio_path"],
                "duration": sample["duration"],
                "T_raw": result_i["T_raw"], "L_raw": result_i["L_raw"],
                "T_eff": result_i["T_eff"], "L_eff": result_i["L_eff"],
                **coverage_i,
                **{k: v for k, v in manifold_i.items() if not k.startswith("pc1")},
            }
            intervention_results.append(entry_i)

            # ── Comparison figures ────────────────────────────────────
            visualize_comparison(sid, c_b, c_i, result_b["A_np"], result_i["A_np"], lyric_pos_np, OUT)

            # ── Quick per-sample stats ────────────────────────────────
            print(f"  reversal:        base={coverage_b['reversal_rate']:.4f}  bias={coverage_i['reversal_rate']:.4f}")
            print(f"  hole_ratio:      base={coverage_b['coverage_hole_ratio']:.4f}  bias={coverage_i['coverage_hole_ratio']:.4f}")
            print(f"  centroid_spear:  base={coverage_b['centroid_spearman_time']:.4f}  bias={coverage_i['centroid_spearman_time']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            continue

    # Save baseline results
    if baseline_results:
        fnames = list(baseline_results[0].keys())
        with open(baseline_dir / "raw_audio_results.csv", "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fnames)
            writer.writeheader()
            writer.writerows(baseline_results)
        with open(baseline_dir / "raw_audio_results.json", "w") as fp:
            json.dump(baseline_results, fp, indent=2)
        print(f"\n  Baseline results -> {baseline_dir}")

    if intervention_results:
        fnames = list(intervention_results[0].keys())
        with open(intervention_dir / "raw_audio_results.csv", "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fnames)
            writer.writeheader()
            writer.writerows(intervention_results)
        with open(intervention_dir / "raw_audio_results.json", "w") as fp:
            json.dump(intervention_results, fp, indent=2)
        print(f"  Intervention results -> {intervention_dir}")

    # Comparison summary with auto-conclusion
    print_comparison_summary(baseline_results, intervention_results)


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hook_dir = OUTPUT_DIR / "raw_audio_hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    fig_dir_parent = OUTPUT_DIR / "raw_audio_figures"
    fig_dir_parent.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir)
    tensor_dir = Path(args.tensor_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    print("=" * 70)
    print("MANIFOLD PROGRESS PROBE — raw-audio mode")
    print("=" * 70)

    # =================================================================
    # Step 1: Scan audio files
    # =================================================================
    print("\n[1/8] Scanning audio files...")
    all_audio = scan_audio_files(audio_dir)
    print(f"  Found {len(all_audio)} audio files")

    # Get durations via ffprobe
    audio_durations = {}
    for f in all_audio:
        dur = get_audio_duration_ffprobe(f)
        if dur is not None:
            audio_durations[f.name] = dur

    # Filter by duration
    valid_audio = []
    for f in all_audio:
        dur = audio_durations.get(f.name)
        if dur is None:
            continue
        if args.min_duration <= dur <= args.max_duration:
            valid_audio.append((f, dur))

    print(f"  {len(valid_audio)} files within [{args.min_duration}, {args.max_duration}]s")

    if len(valid_audio) == 0:
        print("  ❌ No valid audio files found. Aborting.")
        sys.exit(1)

    # Random select
    selected = random.sample(valid_audio, min(args.num_audio_samples, len(valid_audio)))
    print(f"  Selected {len(selected)} samples randomly")

    # Save selection list
    selection_list = []
    for f, dur in selected:
        selection_list.append({
            "audio_path": str(f),
            "duration": dur,
            "stem": f.stem,
        })
    with open(OUTPUT_DIR / "selected_audio_files.json", "w") as fp:
        json.dump(selection_list, fp, indent=2)
    print(f"  Saved selection list -> {OUTPUT_DIR / 'selected_audio_files.json'}")

    # =================================================================
    # Step 2: Find lyrics/metadata
    # =================================================================
    print("\n[2/8] Finding lyrics and metadata...")

    scanned_count = len(all_audio)
    valid_long_count = len(valid_audio)
    skipped_no_lyrics = 0
    skipped_too_short = len(all_audio) - len(valid_audio)
    with_lyrics_count = 0

    samples = []
    for f, dur in selected:
        sid = generate_sample_id(f)
        meta_files = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta_files)
        caption = read_caption(meta_files)

        if lyrics is None:
            print(f"  ⚠  {f.name}: no lyrics found (skipped)")
            skipped_no_lyrics += 1
            continue

        # Find matching preprocessed .pt
        pt_path = find_matching_pt(f.stem, tensor_dir)

        samples.append({
            "sample_id": sid,
            "audio_path": str(f),
            "duration": dur,
            "lyrics": lyrics,
            "caption": caption or "",
            "meta_files": {k: str(v) for k, v in meta_files.items() if v is not None},
            "pt_path": str(pt_path) if pt_path else None,
        })
        with_lyrics_count += 1

    print(f"  Samples with lyrics: {len(samples)}")
    print(f"  Skipped (no lyrics): {skipped_no_lyrics}")

    # Save metadata summary
    meta_summary = []
    for s in samples:
        # Parse section_ids for summary
        parser = LyricsStructureParser()
        L_sections = 128  # rough estimate for summary
        parsed = parser.parse(s["lyrics"], num_chunks=L_sections)
        section_dist = {}
        for sid_val in range(8):
            cnt = (parsed.section_type_ids == sid_val).sum().item()
            if cnt > 0:
                section_dist[SECTION_NAMES[sid_val]] = cnt

        meta_summary.append({
            "sample_id": s["sample_id"],
            "audio_path": s["audio_path"],
            "duration": s["duration"],
            "has_pt": s["pt_path"] is not None,
            "section_distribution": section_dist,
        })

    with open(OUTPUT_DIR / "raw_audio_metadata_summary.json", "w") as fp:
        json.dump(meta_summary, fp, indent=2)
    print(f"  Saved metadata summary -> {OUTPUT_DIR / 'raw_audio_metadata_summary.json'}")

    scan_results = {
        "scanned": scanned_count,
        "valid_long": valid_long_count,
        "with_lyrics": with_lyrics_count,
        "skipped_no_lyrics": skipped_no_lyrics,
        "skipped_too_short": skipped_too_short,
    }

    # =================================================================
    # Step 3: Load model
    # =================================================================
    print("\n[3/8] Loading baseline model (all adapters off)...")
    dit_handler = setup_model(device=args.device)
    model = dit_handler.model

    # ── Compare mode branch ──────────────────────────────────────────
    if args.compare_fixed_progress_bias:
        print("  ** Compare mode: running baseline vs fixed progress bias **")
        _run_compare_mode(model, samples, scan_results)
        print(f"\nDone. All outputs in {OUTPUT_DIR}")
        return

    # =================================================================
    # Step 4-6: Process each sample
    # =================================================================
    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'─' * 70}")
        print(f"[{idx+1}/{len(samples)}] {sample['sample_id']}")
        print(f"  Duration: {sample['duration']:.1f}s")
        print(f"  Audio: {Path(sample['audio_path']).name}")

        sid = sample["sample_id"]
        lyrics = sample["lyrics"]
        pt_path_str = sample.get("pt_path")

        if pt_path_str is None or not Path(pt_path_str).is_file():
            print(f"  ⚠  No preprocessed .pt found — skipping (re-encode not supported yet)")
            continue

        try:
            # ---- Load preprocessed data ----
            pt_data = load_preprocessed_data(Path(pt_path_str))
            target_latents = pt_data["target_latents"]          # [T, 64]
            attention_mask = pt_data["attention_mask"]           # [T]
            encoder_hidden_states = pt_data["encoder_hidden_states"]  # [1, L, D]
            encoder_attention_mask = pt_data["encoder_attention_mask"]  # [1, L]
            context_latents = pt_data["context_latents"]         # [T', 128]

            T_raw = target_latents.shape[0]
            L_raw = encoder_hidden_states.shape[0]  # squeezed [L, D]
            print(f"  Raw T_audio: {T_raw} frames ({T_raw/25:.1f}s at 25Hz)")
            print(f"  Raw L_text:  {L_raw} tokens (full)")
            print(f"  Valid L (from mask): {int(encoder_attention_mask.sum().item())} tokens")

            # ---- Teacher-forcing forward + hooks ----
            if args.use_fixed_progress_bias:
                # Pass 1: baseline forward to get T_eff / L_eff
                result_tmp = run_teacher_with_hooks(
                    model, pt_data, t_noise=args.teacher_noise,
                    device=args.device, use_progress_bias=False,
                )
                if result_tmp is None:
                    print(f"  ⚠  Baseline forward failed — skipping")
                    continue
                T_eff_tmp = result_tmp["T_eff"]
                L_eff_tmp = result_tmp["L_eff"]

                # Build bias using T_eff from baseline
                parser_tmp = LyricsStructureParser()
                parsed_tmp = parser_tmp.parse(lyrics, num_chunks=L_eff_tmp)
                lyric_pos_t_tmp, lyric_mask_t_tmp = compute_lyric_pos_and_mask(
                    parsed_tmp.section_type_ids, device=args.device,
                )
                bias_t = build_progress_bias(
                    T_eff_tmp, lyric_pos_t_tmp, lyric_mask_t_tmp,
                    sigma=PB["sigma"], lambda_=PB["lambda_"],
                    max_bias=PB["max_bias"], gate=PB["gate"],
                    device=args.device,
                )

                # Pass 2: intervention forward with bias
                result = run_teacher_with_hooks(
                    model, pt_data, t_noise=args.teacher_noise,
                    device=args.device, use_progress_bias=True,
                    progress_bias_params={"bias_tensor": bias_t, "layer": PB["layer"]},
                )
                if result is None:
                    print(f"  ⚠  Intervention forward failed — skipping")
                    continue
                H_np = result["H_np"]
                A_np = result["A_np"]
                H_tensor = result["H_tensor"]
                A_tensor = result["A_tensor"]
                T_eff, L_eff = result["T_eff"], result["L_eff"]
            else:
                # Original flow: single forward pass, manual hooks
                hidden_collector, attn_collector, handles = register_hooks(model)
                run_teacher_forward(
                    model, target_latents, attention_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    context_latents, t_noise=args.teacher_noise,
                    device=args.device,
                )
                H_tensor = hidden_collector.get()
                A_tensor = attn_collector.get()
                remove_hooks(handles)
                if H_tensor is None or A_tensor is None:
                    print(f"  ⚠  No hook data collected — skipping")
                    continue
                H_np = H_tensor.squeeze(0).float().numpy()
                A_np = A_tensor.float().numpy()
                T_eff, L_eff = H_np.shape[0], A_np.shape[2]

            print(f"  H: {H_np.shape}, A: {A_np.shape}")
            print(f"  T_eff={T_eff} (~{T_eff/50:.1f}s at 50Hz internal rate)")
            print(f"  L_eff={L_eff} (non-padded encoder tokens)")

            # ---- Parse section_ids with the actual L from attention ----
            parser = LyricsStructureParser()
            parsed = parser.parse(lyrics, num_chunks=L_eff)
            section_ids = parsed.section_type_ids  # [L_eff]
            section_dist = {}
            for sid_val in range(8):
                cnt = (section_ids == sid_val).sum().item()
                if cnt > 0:
                    section_dist[SECTION_NAMES[sid_val]] = cnt

            # ---- Compute lyric_pos ----
            lyric_pos = compute_lyric_pos(section_ids)

            # =========================================================
            # Compute coverage metrics
            # =========================================================
            print(f"  Computing coverage metrics...")
            coverage_metrics = compute_coverage_metrics(A_np, lyric_pos)

            # Centroid for manifold analysis
            c_h = []  # collect [T] per head
            for h in range(A_np.shape[0]):
                centroid_h = compute_attention_centroid(
                    A_np[np.newaxis, h], lyric_pos
                )  # returns [1, T]
                c_h.append(centroid_h.squeeze(0))  # [T]
            c = np.mean(c_h, axis=0)  # [T]
            delta_c = np.diff(c) if T_eff > 1 else np.array([0.0])

            # =========================================================
            # Manifold / tangent analysis
            # =========================================================
            print(f"  Running manifold analysis...")
            manifold_results = run_manifold_analysis(
                H_np, c, delta_c, PCA_DIMS
            )

            # =========================================================
            # Visualization
            # =========================================================
            visualize_sample(
                sid, H_np, A_np, c.reshape(1, -1), lyric_pos, section_ids,
                manifold_results, OUTPUT_DIR,
            )

            # =========================================================
            # Save hook data
            # =========================================================
            hook_save = {
                "H": H_tensor.squeeze(0).float().cpu(),  # [T_eff, D]
                "A": A_tensor.float().cpu(),              # [H, T_eff, L_eff]
                "section_ids": section_ids.cpu(),
                "lyric_pos": torch.from_numpy(lyric_pos),
                "audio_path": sample["audio_path"],
                "lyrics": lyrics,
                "duration": sample["duration"],
                "T_raw": T_raw,
                "L_raw": L_raw,
                "T_eff": T_eff,
                "L_eff": L_eff,
            }
            torch.save(hook_save, hook_dir / f"{sid}.pt")
            print(f"  Hooks -> {hook_dir / f'{sid}.pt'}")

            # =========================================================
            # Collect results
            # =========================================================
            result_entry = {
                "sample_id": sid,
                "audio_path": sample["audio_path"],
                "duration": sample["duration"],
                "T_raw": T_raw,
                "L_raw": L_raw,
                "T_eff": T_eff,
                "L_eff": L_eff,
                "section_distribution": section_dist,
                **coverage_metrics,
                **{k: v for k, v in manifold_results.items()
                   if not k.startswith("pc1")},  # exclude array data
            }
            all_results.append(result_entry)

            # Print quick stats
            print(f"  centroid_spearman_time: {coverage_metrics['centroid_spearman_time']:.4f}")
            print(f"  reversal_rate:          {coverage_metrics['reversal_rate']:.4f}")
            print(f"  effective_rank:         {coverage_metrics['attention_effective_rank']:.2f}")
            print(f"  probe_z16_tangent_R2:   {manifold_results.get('probe_z16_tangent_R2', -999):.4f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
            remove_hooks(handles) if 'handles' in dir() else None
            continue

    # =================================================================
    # Step 7: Save aggregated results
    # =================================================================
    print(f"\n{'=' * 70}")
    print("[7/8] Saving aggregated results...")

    # CSV
    if all_results:
        import csv
        fieldnames = list(all_results[0].keys())
        csv_path = OUTPUT_DIR / "raw_audio_results.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  CSV -> {csv_path}")

        # JSON
        json_path = OUTPUT_DIR / "raw_audio_results.json"
        with open(json_path, "w") as fp:
            json.dump(all_results, fp, indent=2)
        print(f"  JSON -> {json_path}")

        # Aggregated stats
        agg = {}
        for key in fieldnames:
            vals = [r[key] for r in all_results if isinstance(r.get(key), (int, float))]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_median"] = float(np.median(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        agg_path = OUTPUT_DIR / "raw_audio_results_aggregated.json"
        with open(agg_path, "w") as fp:
            json.dump(agg, fp, indent=2)
        print(f"  Aggregated -> {agg_path}")
    else:
        print("  ⚠  No results to save.")

    # =================================================================
    # Step 8: Print summary
    # =================================================================
    print("\n[8/8] Summary:")
    print_summary(scan_results, all_results)

    print(f"\nDone. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
