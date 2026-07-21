#!/usr/bin/env python3
"""
pmdc_static_probe — Stage 2: PMDC Static Probe.

Validates that PhaseMemoryDurationClock (PMDC) components improve
cross-attention lyric-retrieval geometry under teacher-forcing.

Modes
-----
baseline                        — raw cross-attention, no bias
fixed_linear                    — token-level linear progress bias
static_duration                 — unit-level duration interval bias + plain softmax
static_duration_mass_preserve   — unit-level bias + mass-preserving split-softmax
pmdc_zero_init                  — PMDC forward (zero-init) + mp attention

Usage
-----
    python scripts/pmdc_static_probe.py \\
        --num-samples 50 --mode all \\
        --output-dir /root/autodl-tmp/pmdc_probe

Smoke test:
    python scripts/pmdc_static_probe.py \\
        --num-samples 5 --mode all \\
        --output-dir /root/autodl-tmp/pmdc_probe_smoke \\
        --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
TENSOR_DIR_DEFAULT = Path("/root/autodl-tmp/musicdata/train_tensors")
TENSOR_TAR_PATH = TENSOR_DIR_DEFAULT.parent / "train_tensors.tar"

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser

from acestep.phase_memory import (
    LyricUnit,
    PhaseMemoryDurationClock,
    build_duration_scaffold,
    build_duration_interval_bias,
    mass_preserving_attention,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}

SECTION_NAME_LOOKUP = {
    "INTRO": "INTRO", "VERSE": "VERSE", "PRECHORUS": "PRECHORUS",
    "PRE-CHORUS": "PRECHORUS", "CHORUS": "CHORUS", "BRIDGE": "BRIDGE",
    "OUTRO": "OUTRO", "INSTRUMENTAL": "INSTRUMENTAL", "INSTR": "INSTRUMENTAL",
    "UNKNOWN": "UNKNOWN",
}

ALL_MODES = [
    "baseline",
    "fixed_linear",
    "static_duration",
    "static_duration_mass_preserve",
    "pmdc_zero_init",
]


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


# ===================================================================
#  PART 1 — Audio scanning & metadata matching
# ===================================================================

def scan_audio_files(audio_dir: Path) -> List[Path]:
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def get_audio_duration_ffprobe(path: Path) -> Optional[float]:
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


def find_matching_pt(audio_stem: str, tensor_dir: Path) -> Optional[Path]:
    pt_path = tensor_dir / f"{audio_stem}.pt"
    if pt_path.is_file():
        return pt_path
    base = audio_stem.rsplit("_", 1)[0]
    for f in tensor_dir.glob(f"{base}_*.pt"):
        return f
    if TENSOR_TAR_PATH.is_file():
        pt_name = f"{audio_stem}.pt"
        base_pt_name = f"{base}.pt"
        try:
            import tarfile
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                for name in tar.getnames():
                    leaf = name.split("/")[-1]
                    if leaf == pt_name or leaf == base_pt_name:
                        return Path(tensor_dir) / leaf
        except Exception:
            pass
    return None


def find_metadata_files(audio_stem: str, audio_dir: Path, dataset_dir: Path) -> dict:
    result = {"json": None, "lrc": None, "txt": None,
              "caption_txt": None, "lyrics_txt": None}
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
        elif name == f"{audio_stem}.lyrics.txt":
            result["lyrics_txt"] = f
        elif name == f"{audio_stem}.caption.txt":
            result["caption_txt"] = f
    if dataset_dir is not None:
        cap = dataset_dir / f"{audio_stem}.caption.txt"
        if cap.is_file():
            result["caption_txt"] = cap
        lyr = dataset_dir / f"{audio_stem}.lyrics.txt"
        if lyr.is_file():
            result["lyrics_txt"] = lyr
    return result


def read_lyrics(meta_files: dict) -> Optional[str]:
    if meta_files.get("json"):
        try:
            with open(meta_files["json"]) as f:
                data = json.load(f)
            for key in ("lyrics", "lyric", "text"):
                if key in data and data[key]:
                    return str(data[key])
        except Exception:
            pass
    if meta_files.get("lrc"):
        try:
            lines = meta_files["lrc"].read_text(encoding="utf-8")
            if lines.strip():
                return lines
        except Exception:
            pass
    if meta_files.get("txt"):
        try:
            lines = meta_files["txt"].read_text(encoding="utf-8")
            if lines.strip():
                return lines
        except Exception:
            pass
    if meta_files.get("lyrics_txt"):
        try:
            lines = meta_files["lyrics_txt"].read_text(encoding="utf-8")
            if lines.strip():
                return lines
        except Exception:
            pass
    return None


def load_preprocessed_data(pt_path: Path) -> Optional[dict]:
    """Load tensors from .pt file, supports tar extraction."""
    pt_name = pt_path.name
    if pt_path.is_file():
        try:
            data = torch.load(str(pt_path), weights_only=True, map_location="cpu")
            return {
                "target_latents": data["target_latents"],
                "attention_mask": data["attention_mask"],
                "encoder_hidden_states": data["encoder_hidden_states"],
                "encoder_attention_mask": data["encoder_attention_mask"],
                "context_latents": data["context_latents"],
                "metadata": data.get("metadata", {}),
            }
        except Exception:
            return None
    if TENSOR_TAR_PATH.is_file():
        import tarfile
        import tempfile
        try:
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                tar_path = None
                for name in tar.getnames():
                    leaf = name.split("/")[-1]
                    if leaf == pt_name:
                        tar_path = name
                        break
                if tar_path is None:
                    return None
                member = tar.getmember(tar_path)
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    tmp_path = tmp.name
                    with tar.extractfile(member) as src:
                        tmp.write(src.read())
                data = torch.load(tmp_path, weights_only=True, map_location="cpu")
                os.unlink(tmp_path)
                return {
                    "target_latents": data["target_latents"],
                    "attention_mask": data["attention_mask"],
                    "encoder_hidden_states": data["encoder_hidden_states"],
                    "encoder_attention_mask": data["encoder_attention_mask"],
                    "context_latents": data["context_latents"],
                    "metadata": data.get("metadata", {}),
                }
        except Exception:
            return None
    return None


def generate_sample_id(audio_path: Path) -> str:
    return audio_path.stem


# ===================================================================
#  PART 2 — Model loading and hooking
# ===================================================================

class HiddenCollector:
    def __init__(self):
        self.hidden = []

    def __call__(self, module, input, output):
        hs = output[0]
        self.hidden.append(hs.detach().cpu())

    def get(self):
        return torch.cat(self.hidden, dim=0) if self.hidden else None


class LogitValueCollector:
    """Patches eager_attention_forward to capture pre-softmax logits and V."""

    def __init__(self):
        self.logits = None   # [B, H, T, L]
        self.value = None    # [B, H, L, Dh]
        self._orig_fn = None
        self._patched_mod = None

    def _make_patched_fn(self):
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        collector = self

        def patched_forward(module, query, key, value, attention_mask,
                            scaling, dropout=0.0, **kwargs):
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)

            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Capture pre-softmax logits and value
            collector.logits = attn_weights.detach().cpu()
            collector.value = value_states.detach().cpu()

            attn_weights = F.softmax(attn_weights, dim=-1,
                                     dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout,
                                     training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        return patched_forward

    def install(self, model, layer: int = 12):
        ca_module = model.decoder.layers[layer].cross_attn
        import sys
        cls = type(ca_module)
        attn_mod = sys.modules[cls.__module__]
        self._orig_fn = getattr(attn_mod, "eager_attention_forward", None)
        self._patched_mod = attn_mod
        setattr(attn_mod, "eager_attention_forward", self._make_patched_fn())

    def uninstall(self):
        if self._patched_mod is not None and self._orig_fn is not None:
            setattr(self._patched_mod, "eager_attention_forward", self._orig_fn)
        self._patched_mod = None
        self._orig_fn = None


def setup_model(device: str = "cuda") -> AceStepHandler:
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit_handler.model.eval()
    model.config.use_section_rope_offset = False
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_section_rope", False):
            layer_mod.use_section_rope = False
        if getattr(layer_mod, "use_phase_memory", False):
            layer_mod.use_phase_memory = False
    print(f"  Model: {type(model).__name__}, hidden_size={model.config.hidden_size}")
    print(f"  All adapters disabled (clean baseline)")
    return dit_handler


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
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    hs = target_latents.unsqueeze(0).to(device=model_device, dtype=model_dtype)
    am = attention_mask.unsqueeze(0).to(device=model_device, dtype=model_dtype)
    enc_hs = encoder_hidden_states.unsqueeze(0).to(device=model_device, dtype=model_dtype)
    enc_am = encoder_attention_mask.unsqueeze(0).to(device=model_device, dtype=model_dtype)
    ctx = context_latents.unsqueeze(0).to(device=model_device, dtype=model_dtype)

    if t_noise > 0:
        noise = torch.randn_like(hs)
        hs = (1 - t_noise) * hs + t_noise * noise

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
#  PART 3 — Lyric structure to LyricUnit conversion
# ===================================================================

def parse_lyrics_to_units(
    lyrics_text: str,
    section_ids: torch.Tensor,
    num_lyric_chunks: int = 128,
) -> Tuple[List[LyricUnit], torch.Tensor]:
    """Parse lyrics text + section_ids into a structured list of LyricUnits.

    Strategy:
      1. Split lyrics text into lines (each line is a unit).
      2. Map each line's section type via the section_ids tensor (chunked
         to match the text encoder's L token positions).
      3. Count characters and assign token indices.

    Returns:
        units: list of LyricUnit
        lyric_pos: [L] normalized lyric position (-1 for non-lyric)
    """
    lines = [l for l in lyrics_text.strip().split("\n") if l.strip()]

    # Map each lyric line to a section by chunking section_ids
    L = len(section_ids)
    ids_np = section_ids.cpu().numpy() if isinstance(section_ids, torch.Tensor) else section_ids

    # Determine section per line: take the majority section among its tokens
    # For now, distribute section_ids uniformly over lines
    chunk_edges = np.linspace(0, L, len(lines) + 1, dtype=int)
    line_sections = []
    for i in range(len(lines)):
        start, end = chunk_edges[i], chunk_edges[i + 1]
        chunk = ids_np[start:end]
        if len(chunk) == 0:
            line_sections.append("UNKNOWN")
        else:
            mode_id = np.bincount(chunk[chunk >= 0]).argmax() if (chunk >= 0).any() else 0
            line_sections.append(SECTION_NAMES.get(mode_id, "UNKNOWN"))

    # Build units
    units = []
    next_unit_id = 0
    token_pos = np.full(L, -1, dtype=np.float32)
    token_unit_id = np.full(L, -1, dtype=np.int32)

    # Determine lyric token positions (section 1..7)
    lyric_mask = (ids_np >= 1) & (ids_np <= 7)
    lyric_indices = np.where(lyric_mask)[0]

    if len(lyric_indices) == 0:
        return [], torch.from_numpy(token_pos)

    # Distribute lyric_indices among non-silence lines
    non_silence_lines = [(i, lines[i], sec) for i, sec in enumerate(line_sections)
                         if sec not in ("UNKNOWN", "INSTRUMENTAL", "INSTR")]

    if len(non_silence_lines) == 0:
        # All lines are silence / unknown — create one silence unit
        units.append(LyricUnit(
            unit_id=next_unit_id, section="UNKNOWN", text=lyrics_text,
            char_count=len(lyrics_text), token_indices=list(lyric_indices),
            occurrence_id=0, is_silence=True,
        ))
        next_unit_id += 1
        return units, torch.from_numpy(token_pos)

    # Distribute lyric indices across non-silence lines
    token_splits = np.array_split(lyric_indices, len(non_silence_lines))

    # Build units with proper token indices
    for (orig_idx, line, sec), token_idxs in zip(non_silence_lines, token_splits):
        if len(token_idxs) == 0:
            continue
        char_count = len(line.replace(" ", ""))
        units.append(LyricUnit(
            unit_id=next_unit_id, section=sec, text=line,
            char_count=char_count,
            token_indices=[int(t) for t in token_idxs],
            occurrence_id=0, is_silence=False,
        ))
        token_unit_id[token_idxs] = next_unit_id
        next_unit_id += 1

    # Add silence units for UNKNOWN/INSTR lines that have no lyrics
    for i, line in enumerate(lines):
        sec = line_sections[i]
        if sec in ("UNKNOWN", "INSTRUMENTAL", "INSTR"):
            # Check if this line overlaps with any lyric tokens
            start, end = chunk_edges[i], chunk_edges[i + 1]
            chunk_mask = lyric_mask[start:end]
            if not chunk_mask.any():
                units.append(LyricUnit(
                    unit_id=next_unit_id, section=sec, text=line,
                    char_count=0, token_indices=[],
                    occurrence_id=0, is_silence=True,
                ))
                next_unit_id += 1

    # Compute lyric_pos for metric purposes
    valid_token_indices = np.where(lyric_mask)[0]
    if len(valid_token_indices) > 1:
        for k, idx in enumerate(valid_token_indices):
            token_pos[idx] = k / (len(valid_token_indices) - 1)
    elif len(valid_token_indices) == 1:
        token_pos[valid_token_indices[0]] = 0.5

    return units, torch.from_numpy(token_pos)


# ===================================================================
#  PART 4 — Fixed linear bias (reused from manifold_progress_probe.py)
# ===================================================================

def build_fixed_linear_bias(
    T_audio: int,
    L_text: int,
    lyric_pos: torch.Tensor,
    lyric_mask: torch.Tensor,
    sigma: float = 0.15,
    lambda_: float = 1.0,
    max_bias: float = 3.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Build fixed linear progress bias: bias_ij = -lambda * ((r_j - p_i) / sigma)^2.

    Returns:
        bias: [1, 1, T, L] float32.
    """
    audio_idx = torch.arange(T_audio, device=device)
    progress = audio_idx.float() / max(T_audio - 1, 1)

    dist = lyric_pos.unsqueeze(0) - progress.unsqueeze(1)  # [T, L]
    bias = -lambda_ * (dist / sigma) ** 2
    bias = bias.clamp(min=-max_bias, max=0.0)
    bias = bias * lyric_mask.unsqueeze(0).float()
    bias = bias.unsqueeze(0).unsqueeze(0)
    return bias.contiguous()


# ===================================================================
#  PART 5 — Attention computation per mode
# ===================================================================

def compute_attention_baseline(
    logits: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard softmax attention."""
    H, T, L = logits.shape[1], logits.shape[2], logits.shape[3]
    attn = F.softmax(logits, dim=-1, dtype=torch.float32)
    output = torch.matmul(attn.to(value.dtype), value)
    return output, attn


def compute_attention_fixed_linear(
    logits: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    gate: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed linear bias + plain softmax."""
    biased = logits + gate * bias.to(device=logits.device, dtype=logits.dtype)
    attn = F.softmax(biased, dim=-1, dtype=torch.float32)
    output = torch.matmul(attn.to(value.dtype), value)
    return output, attn


def compute_attention_static_duration(
    logits: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    gate: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Duration interval bias + plain softmax."""
    return compute_attention_fixed_linear(logits, value, bias, gate=gate)


def compute_attention_mass_preserving(
    logits: torch.Tensor,
    value: torch.Tensor,
    lyric_mask: torch.Tensor,
    bias: torch.Tensor,
    gate: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Mass-preserving split-softmax using PMDC's mass_preserving_attention."""
    output, attn, stats = mass_preserving_attention(
        logits=logits.to(torch.float32),
        value=value.to(torch.float32),
        lyric_mask=lyric_mask,
        bias=bias.to(torch.float32),
        gate=gate,
    )
    return output, attn, stats


# ===================================================================
#  PART 6 — Metrics
# ===================================================================

def compute_coverage_metrics(
    attn: np.ndarray,
    lyric_mask: np.ndarray,
    token_to_unit: np.ndarray,
    unit_boundaries: np.ndarray,
    unit_duration: np.ndarray,
    lyric_pos: np.ndarray,
) -> dict:
    """Compute coverage and attention metrics.

    Args:
        attn: [H, T, L] attention weights
        lyric_mask: [L] bool
        token_to_unit: [L] unit id (-1 = non-lyric)
        unit_boundaries: [U + 1]
        unit_duration: [U]
        lyric_pos: [L] normalized lyric position (-1 = non-lyric)

    Returns:
        dict of scalar metrics
    """
    H, T, L = attn.shape
    attn_mean = attn.mean(axis=0)  # [T, L]

    # Lyric-masked attention
    lyric_mask_b = lyric_mask.astype(bool)
    n_lyric = lyric_mask_b.sum()

    # Centroid over lyric tokens using duration-scaffold positions
    # lyric_pos_j = centre of unit u
    unit_centers = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
    scaffold_pos = np.full(L, -1.0, dtype=np.float32)
    for j in range(L):
        if lyric_mask_b[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            scaffold_pos[j] = unit_centers[min(u, len(unit_centers) - 1)]

    # Centroid (only lyric tokens)
    c = np.zeros(T, dtype=np.float32)
    for t in range(T):
        w = attn_mean[t, lyric_mask_b]
        total = w.sum()
        if total > 1e-10:
            c[t] = (w * scaffold_pos[lyric_mask_b]).sum() / total

    # Centroid dynamics
    delta_c = np.diff(c) if T > 1 else np.array([0.0])
    reversal_rate = float(np.mean(delta_c < -0.005)) if len(delta_c) > 0 else 0.0
    jump_rate = float(np.mean(delta_c > 0.05)) if len(delta_c) > 0 else 0.0
    stagnant_rate = float(np.mean(np.abs(delta_c) < 0.001)) if len(delta_c) > 0 else 0.0

    # Centroid vs time Spearman
    from scipy.stats import spearmanr
    time_lin = np.linspace(0, 1, T)
    corr, _ = spearmanr(c, time_lin)
    centroid_spearman_time = float(corr)

    centroid_range = float(c.max() - c.min()) if T > 0 else 0.0

    # Coverage distribution (total weight per token)
    coverage = attn_mean.sum(axis=0)  # [L]
    coverage = coverage / (coverage.sum() + 1e-10)

    # Coverage entropy (normalized)
    eps = 1e-10
    coverage_entropy = float(-np.sum(coverage * np.log(coverage + eps)) / np.log(L))

    # Gini
    sorted_cov = np.sort(coverage)
    cumsum = np.cumsum(sorted_cov)
    gini = float(1 - 2 * np.sum(cumsum) / (L * cumsum[-1] + eps))

    # Hole ratio: fraction of lyric tokens with near-zero coverage
    hole_ratio = float(np.mean(coverage[lyric_mask_b] < 0.01 / n_lyric)) if n_lyric > 0 else 1.0

    # Coverage error to duration scaffold
    # Aggregate mass per unit (only non-silence units)
    U = len(unit_boundaries) - 1
    coverage_u = np.zeros(U, dtype=np.float32)
    unit_has_lyric = np.zeros(U, dtype=bool)
    for j in range(L):
        if lyric_mask_b[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            coverage_u[min(u, U - 1)] += coverage[j]
            unit_has_lyric[min(u, U - 1)] = True

    if unit_has_lyric.any():
        coverage_u_lyric = coverage_u[unit_has_lyric]
        target_u = unit_duration[unit_has_lyric]
        coverage_u_lyric = coverage_u_lyric / (coverage_u_lyric.sum() + 1e-10)
        target_u = target_u / (target_u.sum() + 1e-10)
        coverage_error_to_duration = float(np.mean(np.abs(coverage_u_lyric - target_u)))
    else:
        coverage_error_to_duration = float("nan")

    return {
        "reversal_rate": reversal_rate,
        "jump_rate": jump_rate,
        "stagnant_rate": stagnant_rate,
        "centroid_spearman_time": centroid_spearman_time,
        "centroid_range": centroid_range,
        "coverage_entropy": coverage_entropy,
        "coverage_gini": gini,
        "coverage_hole_ratio": hole_ratio,
        "coverage_error_to_duration": coverage_error_to_duration,
    }


# ===================================================================
#  PART 7 — Visualization
# ===================================================================

def visualize_sample(
    sample_id: str,
    fig_dir: Path,
    mode_attns: Dict[str, np.ndarray],
    mode_metrics: Dict[str, dict],
    mode_centroids: Dict[str, np.ndarray],
    lyric_units_str: str,
    scaffold: dict,
    p_base: np.ndarray,
    p_final: Optional[np.ndarray],
    lyric_pos: np.ndarray,
    section_ids: torch.Tensor,
    T_eff: int,
    L_eff: int,
    duration: float,
):
    """Generate diagnostic plots per sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Lyric units debug file ----
    if lyric_units_str:
        (fig_dir / "lyric_units.txt").write_text(lyric_units_str)

    # ---- Colors ----
    mode_colors = {
        "baseline": "gray",
        "fixed_linear": "coral",
        "static_duration": "steelblue",
        "static_duration_mass_preserve": "darkgreen",
        "pmdc_zero_init": "darkviolet",
    }

    norm_time = np.arange(T_eff) / T_eff

    # ---- Duration scaffold ----
    boundaries_np = scaffold.get("unit_boundaries")
    if boundaries_np is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for b in boundaries_np:
            ax.axvline(x=float(b), color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Normalized audio time")
        ax.set_ylabel("Progress")
        ax.set_title(f"Duration Scaffold — {sample_id}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "duration_scaffold.png", dpi=150)
        plt.close(fig)

    # ---- p_base vs p_final ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(norm_time, p_base, color="gray", linewidth=1.0, linestyle=":", label="p_base (linear)")
    if p_final is not None:
        ax.plot(norm_time, p_final, color="darkviolet", linewidth=1.2, label="p_final (PMDC zero-init)")
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Progress")
    ax.set_title(f"p_base vs p_final — {sample_id}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "p_base_vs_p_final.png", dpi=150)
    plt.close(fig)

    # ---- Attention centroid per mode ----
    for mode in ALL_MODES:
        if mode in mode_centroids:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(norm_time, mode_centroids[mode],
                    color=mode_colors.get(mode, "blue"), linewidth=1.2)
            ax.set_xlabel("Normalized audio time")
            ax.set_ylabel("Attention centroid (scaffold pos)")
            ax.set_title(f"Attention Centroid ({mode}) — {sample_id}")
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(fig_dir / f"attention_centroid_{mode}.png", dpi=150)
            plt.close(fig)

    # ---- Coverage by unit ----
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(boundaries_np) - 1) if boundaries_np is not None else np.arange(1)
    width = 0.15
    for i, mode in enumerate(ALL_MODES):
        if mode in mode_metrics and "coverage_error_to_duration" in mode_metrics[mode]:
            err = mode_metrics[mode]["coverage_error_to_duration"]
            ax.bar(x + i * width - 2 * width, err if not isinstance(err, list) else 0,
                   width, label=mode, color=mode_colors.get(mode, "blue"), alpha=0.7)
    ax.set_xlabel("Unit index")
    ax.set_ylabel("Coverage error (vs scaffold)")
    ax.set_title(f"Coverage Error by Unit — {sample_id}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_by_unit.png", dpi=150)
    plt.close(fig)

    # ---- Lyric mass curve ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for mode in ALL_MODES:
        if mode in mode_attns:
            A = mode_attns[mode]
            lyric_mass = A[:, :, lyric_pos >= 0].sum(axis=-1).mean(axis=0) if (lyric_pos >= 0).any() else np.zeros(T_eff)
            ax.plot(norm_time, lyric_mass, color=mode_colors.get(mode, "blue"),
                    linewidth=1.0, alpha=0.7, label=mode)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Lyric attention mass")
    ax.set_title(f"Lyric Attention Mass — {sample_id}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "lyric_mass_curve.png", dpi=150)
    plt.close(fig)

    print(f"  Figures -> {fig_dir}")


# ===================================================================
#  PART 8 — Mode runner
# ===================================================================

def compute_lyric_pos_from_scaffold(
    token_to_unit: np.ndarray,
    unit_boundaries: np.ndarray,
    L: int,
) -> np.ndarray:
    """Compute position for each lyric token from its unit's centre."""
    pos = np.full(L, -1.0, dtype=np.float32)
    centers = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
    for j in range(L):
        if token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            pos[j] = centers[min(u, len(centers) - 1)]
    return pos


def run_single_sample(
    model,
    args,
    sample: dict,
    device: str,
) -> Optional[dict]:
    """Teacher-forcing forward + all attention modes on one sample.

    Returns a dict with:
      - sample_id, T_eff, L_eff, duration
      - units, scaffold dict
      - mode_attns: {mode: np.ndarray [H, T, L]}
      - mode_metrics: {mode: dict}
      - mode_centroids: {mode: np.ndarray [T]}
      - p_base, p_final arrays
      - hidden_classifier... (not used here)
    """
    sid = sample["sample_id"]
    lyrics = sample["lyrics"]
    pt_path_str = sample.get("pt_path")

    if pt_path_str is None:
        return None

    pt_data = load_preprocessed_data(Path(pt_path_str))
    if pt_data is None:
        return None

    # ---- Teacher-forcing forward + capture logits and hidden ----
    hidden_collector = HiddenCollector()
    logit_collector = LogitValueCollector()

    h_handle = model.decoder.layers[args.layer].register_forward_hook(hidden_collector)
    logit_collector.install(model, layer=args.layer)

    try:
        run_teacher_forward(
            model,
            pt_data["target_latents"],
            pt_data["attention_mask"],
            pt_data["encoder_hidden_states"],
            pt_data["encoder_attention_mask"],
            pt_data["context_latents"],
            device=device,
        )
    except Exception as e:
        h_handle.remove()
        logit_collector.uninstall()
        print(f"    Forward error: {e}")
        return None

    h_handle.remove()
    logit_collector.uninstall()

    H_tensor = hidden_collector.get()
    logits_tensor = logit_collector.logits
    value_tensor = logit_collector.value

    if H_tensor is None or logits_tensor is None or value_tensor is None:
        return None

    # Convert to float32 on CPU for safety
    H = H_tensor.squeeze(0).float().numpy()  # [T_eff, D]
    logits_cpu = logits_tensor.float().cpu()   # [B, H, T, L]
    values_cpu = value_tensor.float().cpu()    # [B, H, L, Dh]

    T_eff, D = H.shape
    L_eff = logits_cpu.shape[-1]

    # ---- Parse section_ids ----
    parser = LyricsStructureParser()
    parsed = parser.parse(lyrics, num_chunks=L_eff)
    section_ids = parsed.section_type_ids

    # ---- Build lyric units ----
    units, raw_lyric_pos = parse_lyrics_to_units(lyrics, section_ids)

    if len(units) == 0:
        print(f"    Skip: no valid lyric units")
        return None

    # Build scaffold
    scaffold = build_duration_scaffold(
        units=units,
        text_len=L_eff,
        device="cpu",
    )
    unit_boundaries = scaffold["unit_boundaries"]
    token_to_unit = scaffold["token_to_unit"]
    lyric_mask = scaffold["lyric_mask"]

    # Check that we have lyric tokens
    if not lyric_mask.any():
        print(f"    Skip: no lyric tokens")
        return None

    # Lyric position from scaffold for metrics
    scaffold_lyric_pos = compute_lyric_pos_from_scaffold(
        token_to_unit.numpy(), unit_boundaries.numpy(), L_eff,
    )

    # ---- Build p_base (linear) ----
    p_base_np = np.linspace(0, 1, T_eff, dtype=np.float32)

    # ---- Build fixed linear bias for fixed_linear mode ----
    raw_lyric_pos_t = raw_lyric_pos.float().to(device)
    lyric_mask_t = lyric_mask.to(device)
    fixed_bias = build_fixed_linear_bias(
        T_eff, L_eff, raw_lyric_pos_t, lyric_mask_t,
        sigma=0.15, lambda_=1.0, max_bias=3.0, device=device,
    )  # [1, 1, T, L]

    # ---- Build duration interval bias ----
    p_base_t = torch.from_numpy(p_base_np).float().unsqueeze(0).to(device)  # [1, T]
    duration_bias_t = build_duration_interval_bias(
        p_final=p_base_t,
        unit_boundaries=unit_boundaries.to(device),
        token_to_unit=token_to_unit.to(device),
        lyric_mask=lyric_mask.to(device),
        sigma=args.sigma,
        lambda_=args.lambda_,
        max_bias=args.max_bias,
    )  # [B, T, L]

    # ---- PMDC zero-init ----
    pmdc_clock = PhaseMemoryDurationClock(
        dim=D,
        mem_dim=args.mem_dim,
        hidden_dim=args.probe_dim,
        beta=args.beta,
        use_delta_h=True,
    ).to(device)
    pmdc_clock.eval()

    H_t = torch.from_numpy(H).float().unsqueeze(0).to(device)  # [1, T, D]
    with torch.no_grad():
        s_pm_t, p_final_t = pmdc_clock(H_t, p_base_t)

    p_final_np = p_final_t[0].cpu().numpy()

    # Build PMDC zero-init bias
    pmdc_bias_t = build_duration_interval_bias(
        p_final=p_final_t,
        unit_boundaries=unit_boundaries.to(device),
        token_to_unit=token_to_unit.to(device),
        lyric_mask=lyric_mask.to(device),
        sigma=args.sigma,
        lambda_=args.lambda_,
        max_bias=args.max_bias,
    )

    # PMDC p_mae check
    p_mae = (p_final_t - p_base_t).abs().mean().item()

    # ---- Apply each mode on CPU with float32 ----
    logits_cpu_device = logits_cpu  # [B, H, T, L] float32
    values_cpu_device = values_cpu   # [B, H, L, Dh] float32

    # Pre-expand biases on CPU
    fl_bias_exp = fixed_bias.expand(-1, logits_cpu_device.shape[1], -1, -1).float().cpu()
    sd_bias_exp = duration_bias_t.unsqueeze(1).float().cpu()  # [B, 1, T, L]
    pmdc_bias_exp = pmdc_bias_t.unsqueeze(1).float().cpu()    # [B, 1, T, L]

    # Pre-prepare masks on CPU
    lyric_mask_cpu = lyric_mask.cpu()
    duration_bias_cpu = duration_bias_t.float().cpu()
    pmdc_bias_cpu = pmdc_bias_t.float().cpu()

    # Helper: wrap a mode's result into (output, attn, stats) tuple
    def _wrap_baseline(l, v):
        out, attn = compute_attention_baseline(l, v)
        return out, attn, {}

    modes_to_run: List[Tuple[str, callable]] = [
        ("baseline", functools.partial(_wrap_baseline,
                                       logits_cpu_device.clone(),
                                       values_cpu_device.clone())),
        ("fixed_linear", functools.partial(
            lambda l, v, b, g: (*compute_attention_fixed_linear(l, v, b, g), {}),
            logits_cpu_device.clone(), values_cpu_device.clone(),
            fl_bias_exp, 0.3,
        )),
        ("static_duration", functools.partial(
            lambda l, v, b, g: (*compute_attention_static_duration(l, v, b, g), {}),
            logits_cpu_device.clone(), values_cpu_device.clone(),
            sd_bias_exp, args.gate,
        )),
        ("static_duration_mass_preserve", functools.partial(
            compute_attention_mass_preserving,
            logits_cpu_device.clone(), values_cpu_device.clone(),
            lyric_mask_cpu, duration_bias_cpu, args.gate,
        )),
        ("pmdc_zero_init", functools.partial(
            compute_attention_mass_preserving,
            logits_cpu_device.clone(), values_cpu_device.clone(),
            lyric_mask_cpu, pmdc_bias_cpu, args.gate,
        )),
    ]

    # Run
    mode_attns = {}
    mode_metrics = {}
    mode_centroids = {}
    mode_stats_extra = {}

    for mname, fn in modes_to_run:
        if args.mode != "all" and mname not in [args.mode] + ["baseline"]:
            continue
        try:
            result = fn()
            if len(result) == 3:
                out, attn, stats = result
                mode_stats_extra[mname] = stats
            else:
                out, attn = result
                mode_stats_extra[mname] = None

            attn_np = attn[0].float().numpy() if attn.ndim == 4 else attn.float().numpy()

            # Compute metrics
            metrics = compute_coverage_metrics(
                attn_np,
                lyric_mask.numpy(),
                token_to_unit.numpy(),
                unit_boundaries.numpy(),
                scaffold["unit_duration"].numpy(),
                scaffold_lyric_pos,
            )
            mode_attns[mname] = attn_np
            mode_metrics[mname] = metrics

            # Centroid
            lyric_mask_b = lyric_mask.numpy().astype(bool)
            centers = (unit_boundaries.numpy()[:-1] + unit_boundaries.numpy()[1:]) / 2
            if lyric_mask_b.any():
                attn_mean = attn_np.mean(axis=0)  # [T, L]
                c = np.zeros(T_eff, dtype=np.float32)
                for t in range(T_eff):
                    w = attn_mean[t, lyric_mask_b]
                    total = w.sum()
                    if total > 1e-10:
                        valid_pos = centers[
                            np.clip(token_to_unit.numpy(), 0, len(centers) - 1)
                        ][lyric_mask_b]
                        c[t] = (w * valid_pos).sum() / total
                mode_centroids[mname] = c

        except Exception as e:
            print(f"    Mode {mname} error: {e}")
            traceback.print_exc()
            continue

    # ---- Lyric units debug string ----
    unit_lines = []
    unit_lines.append(f"# Lyric Units — {sid}")
    unit_lines.append(f"# T_eff={T_eff}, L_eff={L_eff}, duration={sample['duration']:.1f}s")
    unit_lines.append(f"# Total units: {len(units)}")
    unit_lines.append(f"# p_mae (PMDC zero-init): {p_mae:.6f}")
    unit_lines.append("")
    unit_lines.append(f"{'u_id':<5} {'section':<15} {'occ':<3} {'silence':<8} {'chars':<6} {'text'}")
    unit_lines.append("-" * 80)

    boundaries_np = unit_boundaries.numpy()
    for u in units:
        idx = u.unit_id
        left = boundaries_np[idx] if idx < len(boundaries_np) else 0.0
        right = boundaries_np[min(idx + 1, len(boundaries_np) - 1)]
        token_str = ",".join(str(t) for t in u.token_indices[:5])
        if len(u.token_indices) > 5:
            token_str += f"...({len(u.token_indices)} total)"
        unit_lines.append(
            f"{u.unit_id:<5} {u.section:<15} {u.occurrence_id:<3} "
            f"{str(u.is_silence):<8} {u.char_count:<6} "
            f"interval=[{left:.3f},{right:.3f}] tokens=[{token_str}]"
        )
    unit_lines.append("")
    # Check for any non-lyric tokens that should be lyric
    unit_lines.append(f"# lyric_mask true count: {lyric_mask.sum().item()}")
    unit_lines.append(f"# token_to_unit >= 0 count: {(token_to_unit >= 0).sum().item()}")
    unit_lines.append(f"# p_mae (PMDC zero-init vs p_base): {p_mae:.6f}")

    return {
        "sample_id": sid,
        "T_eff": T_eff,
        "L_eff": L_eff,
        "duration": sample["duration"],
        "units": units,
        "scaffold": {k: v.numpy() if isinstance(v, torch.Tensor) else v
                     for k, v in scaffold.items()},
        "mode_attns": mode_attns,
        "mode_metrics": mode_metrics,
        "mode_centroids": mode_centroids,
        "mode_stats_extra": mode_stats_extra,
        "p_base": p_base_np,
        "p_final": p_final_np,
        "p_mae": p_mae,
        "lyric_units_str": "\n".join(unit_lines),
    }


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="PMDC Static Probe — Stage 2")

    parser.add_argument("--audio-dir", type=str, default="/root/autodl-tmp/musicdata/audios")
    parser.add_argument("--tensor-dir", type=str, default=str(TENSOR_DIR_DEFAULT))
    parser.add_argument("--dataset-dir", type=str, default="/root/autodl-tmp/musicdata/dataset")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-duration", type=float, default=180)
    parser.add_argument("--max-duration", type=float, default=300)

    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["all"] + [m for m in ALL_MODES if m != "baseline"],
                        help="Attention mode (default: all)")

    # PMDC params
    parser.add_argument("--probe-dim", type=int, default=256)
    parser.add_argument("--mem-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.1)

    # Bias params
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--lambda", type=float, dest="lambda_", default=0.25)
    parser.add_argument("--max-bias", type=float, default=0.5)
    parser.add_argument("--gate", type=float, default=0.1,
                        help="Bias gate multiplier for duration interval modes")

    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/pmdc_probe")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir)
    tensor_dir = Path(args.tensor_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    print("=" * 70)
    print("PMDC STATIC PROBE — Stage 2")
    print(f"  Mode: {args.mode}, Layer: {args.layer}")
    print(f"  sigma={args.sigma}, lambda={args.lambda_}, gate={args.gate}")
    print(f"  beta={args.beta}, mem_dim={args.mem_dim}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Scan audio files
    # ------------------------------------------------------------------
    print("\n[1/7] Scanning audio files...", flush=True)
    all_audio = scan_audio_files(audio_dir)
    print(f"  Found {len(all_audio)} audio files")

    audio_durations = {}
    for f in all_audio:
        dur = get_audio_duration_ffprobe(f)
        if dur is not None:
            audio_durations[f.name] = dur

    valid_audio = []
    for f in all_audio:
        dur = audio_durations.get(f.name)
        if dur is None:
            continue
        if args.min_duration <= dur <= args.max_duration:
            valid_audio.append((f, dur))

    print(f"  {len(valid_audio)} files within [{args.min_duration}, {args.max_duration}]s")

    if len(valid_audio) == 0:
        print("  No valid audio files found. Aborting.")
        sys.exit(1)

    selected = random.sample(valid_audio, min(args.num_samples, len(valid_audio)))
    print(f"  Selected {len(selected)} samples")

    # ------------------------------------------------------------------
    # Step 2: Find lyrics
    # ------------------------------------------------------------------
    print("\n[2/7] Finding lyrics and metadata...", flush=True)

    samples = []
    skipped_log = []
    for f, dur in selected:
        sid = generate_sample_id(f)
        meta_files = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta_files)
        if lyrics is None:
            skipped_log.append((sid, "no_lyrics"))
            continue
        pt_path = find_matching_pt(f.stem, tensor_dir)
        samples.append({
            "sample_id": sid,
            "audio_path": str(f),
            "duration": dur,
            "lyrics": lyrics,
            "pt_path": str(pt_path) if pt_path else None,
        })

    print(f"  Samples with lyrics: {len(samples)}")

    # ------------------------------------------------------------------
    # Step 3: Load model
    # ------------------------------------------------------------------
    print("\n[3/7] Loading baseline model (all adapters off)...", flush=True)
    dit_handler = setup_model(device=args.device)
    model = dit_handler.model
    D = model.config.hidden_size

    # ------------------------------------------------------------------
    # Step 4: Teacher-forcing forward + collect for each sample
    # ------------------------------------------------------------------
    print(f"\n[4/7] Running teacher-forcing forward on {len(samples)} samples...", flush=True)

    all_results = []
    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] {sample['sample_id']}  (dur={sample['duration']:.1f}s)", flush=True)

        result = run_single_sample(model, args, sample, device=args.device)
        if result is None:
            print(f"    Skip")
            skipped_log.append((sample["sample_id"], "processing_failed"))
            continue

        T_eff = result["T_eff"]
        L_eff = result["L_eff"]
        n_modes = len(result["mode_metrics"])
        print(f"    T={T_eff}, L={L_eff}, modes={n_modes}", flush=True)

        all_results.append(result)

    print(f"\n  Successfully processed: {len(all_results)} samples")
    print(f"  Skipped: {len(skipped_log)} samples")
    for sid, reason in skipped_log:
        print(f"    {sid}: {reason}")

    if len(all_results) < 1:
        print("  No samples processed. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Visualize
    # ------------------------------------------------------------------
    print(f"\n[5/7] Generating visualizations...", flush=True)

    fig_base = OUTPUT_DIR / "figures"
    for result in all_results:
        sid = result["sample_id"]
        fig_dir = fig_base / sid

        # Build lyric_units_str from result
        lyric_units_str = result.get("lyric_units_str", "")

        # Get p_final from result
        p_final = result.get("p_final")

        # Build scaffold dict with numpy arrays
        scaffold_np = result.get("scaffold", {})

        # Get lyric_pos from scaffold_lyric_pos (derived from unit centers)
        sid_actual = result["sample_id"]

        # Generate visualizations
        visualize_sample(
            sample_id=sid,
            fig_dir=fig_dir,
            mode_attns=result.get("mode_attns", {}),
            mode_metrics=result.get("mode_metrics", {}),
            mode_centroids=result.get("mode_centroids", {}),
            lyric_units_str=lyric_units_str,
            scaffold=scaffold_np,
            p_base=result["p_base"],
            p_final=p_final,
            lyric_pos=np.array([]),  # computed inside metrics already
            section_ids=torch.tensor([]),
            T_eff=result["T_eff"],
            L_eff=result["L_eff"],
            duration=result["duration"],
        )

    # ------------------------------------------------------------------
    # Step 6: Aggregate & save results
    # ------------------------------------------------------------------
    print(f"\n[6/7] Aggregating results...", flush=True)

    # Aggregate metrics per mode
    agg_metrics: Dict[str, Dict[str, List[float]]] = {}
    for result in all_results:
        for mname, metrics in result["mode_metrics"].items():
            if mname not in agg_metrics:
                agg_metrics[mname] = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    agg_metrics[mname].setdefault(k, []).append(v)

    # Mean per mode
    summary: Dict[str, Dict[str, float]] = {}
    for mname, metrics_dict in agg_metrics.items():
        summary[mname] = {}
        for k, vals in metrics_dict.items():
            summary[mname][k] = float(np.mean(vals)) if vals else float("nan")

    # Also aggregate lyric_mass_delta from stats
    for result in all_results:
        for mname, stats in result.get("mode_stats_extra", {}).items():
            if stats is not None:
                for k in ("lyric_mass_base", "lyric_mass_new", "lyric_mass_delta"):
                    if k in stats:
                        agg_metrics.setdefault(mname, {}).setdefault(k, []).append(stats[k])

    # Update summary with mass stats
    for mname, metrics_dict in agg_metrics.items():
        if mname not in summary:
            summary[mname] = {}
        for k, vals in metrics_dict.items():
            if k.startswith("lyric_mass"):
                summary[mname][k] = float(np.mean(vals)) if vals else float("nan")

    # Print summary table
    key_metrics = ["coverage_hole_ratio", "coverage_error_to_duration",
                   "reversal_rate", "centroid_range",
                   "centroid_spearman_time", "coverage_gini"]
    mass_metrics = ["lyric_mass_base", "lyric_mass_new", "lyric_mass_delta"]

    print("\n" + "=" * 110)
    print("AGGREGATED RESULTS")
    print("=" * 110)

    header = f"{'Mode':<35s}"
    for km in key_metrics:
        header += f" {km:>20s}"
    for mm in mass_metrics:
        if mm in ["lyric_mass_delta"]:
            header += f" {mm:>18s}"
    print(header)
    print("-" * 110)

    for mname in ALL_MODES:
        if mname not in summary:
            continue
        row = f"{mname:<35s}"
        for km in key_metrics:
            v = summary[mname].get(km, float("nan"))
            row += f" {v:>20.4f}" if not np.isnan(v) else f" {'':>20s}"
        for mm in ["lyric_mass_delta"]:
            v = summary[mname].get(mm, float("nan"))
            row += f" {v:>18.6f}" if not np.isnan(v) else f" {'':>18s}"
        print(row)
    print("-" * 110)

    # Per sample CSV
    csv_rows = []
    for result in all_results:
        for mname, metrics in result["mode_metrics"].items():
            row = {
                "sample_id": result["sample_id"],
                "mode": mname,
                "T_eff": result["T_eff"],
                "L_eff": result["L_eff"],
                "duration": result["duration"],
            }
            row.update(metrics)
            stats_extra = result.get("mode_stats_extra", {}).get(mname)
            if stats_extra:
                row.update(stats_extra)
            csv_rows.append(row)

    csv_path = OUTPUT_DIR / "pmdc_probe_results.csv"
    if csv_rows:
        all_fields = set(csv_rows[0].keys())
        for row in csv_rows[1:]:
            all_fields.update(row.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV -> {csv_path}")

    # JSON summary
    json_path = OUTPUT_DIR / "pmdc_probe_results.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "n_samples": len(all_results),
                    "config": vars(args)}, f, indent=2)
    print(f"  JSON -> {json_path}")

    # Skipped samples
    skipped_path = OUTPUT_DIR / "skipped_samples.json"
    with open(skipped_path, "w") as f:
        json.dump([{"sample_id": s, "reason": r} for s, r in skipped_log], f, indent=2)
    print(f"  Skipped -> {skipped_path}")

    # ------------------------------------------------------------------
    # Step 7: Conclusions
    # ------------------------------------------------------------------
    print(f"\n[7/7] Auto-conclusions...")
    print("\n" + "=" * 110)
    print("AUTO-CONCLUSIONS")
    print("=" * 110)

    # Extract key values for comparison
    bs = summary.get("baseline", {})
    fl = summary.get("fixed_linear", {})
    sd = summary.get("static_duration", {})
    sd_mp = summary.get("static_duration_mass_preserve", {})
    pmdc = summary.get("pmdc_zero_init", {})

    hole_b = bs.get("coverage_hole_ratio", float("nan"))
    hole_sd_mp = sd_mp.get("coverage_hole_ratio", float("nan"))
    cov_err_b = bs.get("coverage_error_to_duration", float("nan"))
    cov_err_sd_mp = sd_mp.get("coverage_error_to_duration", float("nan"))
    mass_delta_sd_mp = sd_mp.get("lyric_mass_delta", float("nan"))
    mass_delta_pmdc = pmdc.get("lyric_mass_delta", float("nan"))
    rev_b = bs.get("reversal_rate", float("nan"))
    rev_sd_mp = sd_mp.get("reversal_rate", float("nan"))

    # Compare PMDC vs static_duration_mass_preserve
    pmae_vals = [r.get("p_mae", 1.0) for r in all_results if "p_mae" in r]
    avg_pmae = float(np.mean(pmae_vals)) if pmae_vals else float("nan")

    print(f"\n  PMDC zero-init average p_mae vs p_base: {avg_pmae:.6f}")

    # Condition checks
    c1 = (hole_sd_mp < hole_b) if not (np.isnan(hole_sd_mp) or np.isnan(hole_b)) else None
    c2 = (cov_err_sd_mp < cov_err_b) if not (np.isnan(cov_err_sd_mp) or np.isnan(cov_err_b)) else None
    c3 = (abs(mass_delta_sd_mp) < 0.01) if not np.isnan(mass_delta_sd_mp) else None
    c4 = (abs(mass_delta_pmdc) < 0.01) if not np.isnan(mass_delta_pmdc) else None
    c5 = (avg_pmae < 0.01) if not np.isnan(avg_pmae) else None

    print(f"\n  Criteria checks:")
    print(f"    sd_mp hole_ratio < baseline hole_ratio:         {c1} "
          f"({hole_sd_mp:.4f} vs {hole_b:.4f})")
    print(f"    sd_mp coverage_error < baseline coverage_error: {c2} "
          f"({cov_err_sd_mp:.4f} vs {cov_err_b:.4f})")
    print(f"    sd_mp lyric_mass_delta ≈ 0:                     {c3} "
          f"(delta={mass_delta_sd_mp:.6f})")
    print(f"    pmdc lyric_mass_delta ≈ 0:                      {c4} "
          f"(delta={mass_delta_pmdc:.6f})")
    print(f"    pmdc p_final ≈ p_base (avg p_mae < 0.01):       {c5} "
          f"(p_mae={avg_pmae:.6f})")

    conclusions = []

    if c1 and c3 and c2:
        conclusions.append(
            "Conclusion A: static_duration_mass_preserve improves hole_ratio and \n"
            "  coverage_error while keeping lyric_mass_delta near 0.\n"
            "  => Duration scaffold + mass-preserving attention is useful."
        )
    elif c1 and c3 and not c2:
        conclusions.append(
            "Conclusion A-: static_duration_mass_preserve improves hole_ratio but not \n"
            "  coverage_error. lyric_mass_delta near 0.\n"
            "  => Duration scaffold helps coverage but token→unit mapping may need work."
        )

    if c5 and c4:
        conclusions.append(
            "Conclusion B: pmdc_zero_init matches static_duration_mass_preserve within tolerance.\n"
            "  => PMDC zero-init is stable and does not change scaffold initially."
        )
    elif c5 and not c4:
        conclusions.append(
            "Conclusion B-: pmdc_zero_init p_final ≈ p_base but lyric_mass_delta differs.\n"
            "  => PMDC zero-init stable; bias construction may differ."
        )

    fl_hole = fl.get("coverage_hole_ratio", float("nan"))
    fl_rev = fl.get("reversal_rate", float("nan"))
    if not np.isnan(fl_rev) and not np.isnan(rev_b):
        fl_rev_improved = fl_rev < rev_b
    else:
        fl_rev_improved = None

    fl_mass_delta = fl.get("lyric_mass_delta", float("nan"))
    if fl_rev_improved and not np.isnan(fl_hole) and fl_hole > hole_b:
        conclusions.append(
            "Conclusion C: fixed_linear improves centroid monotonicity but worsens \n"
            "  hole_ratio. Fixed linear bias is too rigid / unsafe."
        )
    elif fl_rev_improved:
        conclusions.append(
            "Conclusion C-: fixed_linear improves centroid monotonicity.\n"
            "  Still too rigid for final solution."
        )

    if not c1 and not c2:
        conclusions.append(
            "Conclusion D: static_duration_mass_preserve does not improve coverage.\n"
            "  => Duration scaffold or token mapping may be wrong; inspect lyric_units.txt."
        )

    for c in conclusions:
        print(f"\n  {c}")

    if not conclusions:
        print("\n  No clear conclusion — insufficient data or all metrics within noise.")

    print(f"\nDone. All outputs in {OUTPUT_DIR}")


# ===================================================================
#  Entry point
# ===================================================================

if __name__ == "__main__":
    main()
