#!/usr/bin/env python3
"""
pmdc_bias_sweep — Stage 2.5: PMDC Bias Sweep + Lyric Mask Debug.

Sweeps duration-interval bias parameters (sigma, lambda, gate, max_bias)
to find the weakest effective intervention that improves line-level
coverage.  Also provides detailed lyric-mask diagnostics.

Usage
-----
    python scripts/pmdc_bias_sweep.py \\
        --num-samples 20 \\
        --sigmas 0.03 0.05 0.08 0.12 \\
        --lambdas 0.5 1.0 2.0 \\
        --gates 0.1 0.3 0.5 \\
        --max-biases 0.5 1.0 2.0
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
        import tarfile, tempfile
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


# ===================================================================
#  PART 2 — Model loading & logit patching
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
    """Patches eager_attention_forward to capture pre-softmax logits + V."""

    def __init__(self):
        self.logits = None
        self.value = None
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
            collector.logits = attn_weights.detach().cpu()
            collector.value = value_states.detach().cpu()
            attn_weights = F.softmax(attn_weights, dim=-1,
                                     dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
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
    return dit_handler


def run_teacher_forward(model, target_latents, attention_mask,
                        encoder_hidden_states, encoder_attention_mask,
                        context_latents, t_noise=0.0, device="cuda"):
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
            hidden_states=hs, timestep=t_tensor, timestep_r=t_tensor,
            attention_mask=am,
            encoder_hidden_states=enc_hs, encoder_attention_mask=enc_am,
            context_latents=ctx, use_cache=False, output_attentions=True,
        )


# ===================================================================
#  PART 3 — Lyric structure parsing + mask debug
# ===================================================================

def parse_lyrics_to_units(
    lyrics_text: str,
    section_ids: torch.Tensor,
) -> Tuple[List[LyricUnit], np.ndarray, dict]:
    """Parse lyrics into LyricUnits. Returns units, token_pos, debug_info."""
    lines = [l for l in lyrics_text.strip().split("\n") if l.strip()]
    L = len(section_ids)
    ids_np = section_ids.cpu().numpy() if isinstance(section_ids, torch.Tensor) else section_ids

    # ---- Map section per line (uniform chunking) ----
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

    # ---- Determine lyric tokens (sections 1..7) ----
    lyric_mask_raw = (ids_np >= 1) & (ids_np <= 7)
    lyric_indices = np.where(lyric_mask_raw)[0]

    # ---- Unit construction ----
    units: List[LyricUnit] = []
    next_unit_id = 0
    token_pos = np.full(L, -1.0, dtype=np.float32)
    token_unit_id = np.full(L, -1, dtype=np.int32)

    if len(lyric_indices) == 0:
        return units, token_pos, {"lyric_ratio": 0.0, "n_unknown": int((ids_np == 0).sum())}

    # Non-silence lines (section != UNKNOWN/INSTR)
    non_silence_lines = [(i, lines[i], sec) for i, sec in enumerate(line_sections)
                         if sec not in ("UNKNOWN", "INSTRUMENTAL", "INSTR")]
    silence_lines = [(i, lines[i], sec) for i, sec in enumerate(line_sections)
                     if sec in ("UNKNOWN", "INSTRUMENTAL", "INSTR")]

    if len(non_silence_lines) > 0:
        token_splits = np.array_split(lyric_indices, len(non_silence_lines))
        for (orig_idx, line, sec), token_idxs in zip(non_silence_lines, token_splits):
            if len(token_idxs) == 0:
                continue
            char_count = len(line.replace(" ", ""))
            units.append(LyricUnit(
                unit_id=next_unit_id, section=sec, text=line,
                char_count=char_count, token_indices=[int(t) for t in token_idxs],
                occurrence_id=0, is_silence=False,
            ))
            token_unit_id[token_idxs] = next_unit_id
            next_unit_id += 1
    else:
        # All silence
        units.append(LyricUnit(
            unit_id=next_unit_id, section="UNKNOWN", text=lyrics_text,
            char_count=len(lyrics_text), token_indices=list(lyric_indices),
            occurrence_id=0, is_silence=True,
        ))
        next_unit_id += 1
        token_unit_id[lyric_indices] = 0

    # Add silence units for lines with no lyric tokens
    for (orig_idx, line, sec) in silence_lines:
        start, end = chunk_edges[orig_idx], chunk_edges[orig_idx + 1]
        chunk_mask = lyric_mask_raw[start:end]
        if not chunk_mask.any():
            units.append(LyricUnit(
                unit_id=next_unit_id, section=sec, text=line,
                char_count=0, token_indices=[], occurrence_id=0, is_silence=True,
            ))
            next_unit_id += 1

    # Token-level lyric_pos (simple linear)
    valid_indices = np.where(lyric_mask_raw)[0]
    if len(valid_indices) > 1:
        for k, idx in enumerate(valid_indices):
            token_pos[idx] = k / (len(valid_indices) - 1)
    elif len(valid_indices) == 1:
        token_pos[valid_indices[0]] = 0.5

    # ---- Debug info ----
    num_unknown = int((ids_np == 0).sum())
    lyric_ratio = float(lyric_mask_raw.sum()) / max(L, 1)

    return units, token_pos, {
        "lyric_ratio": lyric_ratio,
        "n_unknown": num_unknown,
        "raw_lyric_count": int(lyric_mask_raw.sum()),
        "total_tokens": L,
        "n_non_silence_lines": len(non_silence_lines),
        "n_silence_lines": len(silence_lines),
        "n_units": len(units),
    }


def generate_token_mapping_text(
    units: List[LyricUnit],
    L: int,
    lyric_mask: np.ndarray,
    token_to_unit: np.ndarray,
) -> str:
    """Generate a text file snippet mapping each token to unit info.

    Since we don't have a tokeniser decode here, we show index ranges.
    """
    lines = []
    lines.append(f"# Token Mapping (L={L}, units={len(units)})")
    lines.append("# idx  is_lyric  unit        section         unit_text")
    lines.append("-" * 80)

    # Build token-to-text mapping from units
    token_texts = {}
    for u in units:
        for i, idx in enumerate(u.token_indices):
            # Show position in line text
            if u.text:
                # Rough estimate: char position relative to line
                chars = u.text.replace(" ", "")
                c_total = max(len(chars), 1)
                c_pos = int(i * c_total / max(len(u.token_indices), 1))
                c_end = int((i + 1) * c_total / max(len(u.token_indices), 1))
                snippet = chars[c_pos:c_end] if c_pos < c_total else ""
                token_texts[idx] = snippet
            else:
                token_texts[idx] = ""

    for j in range(L):
        is_lyric = lyric_mask[j] if j < len(lyric_mask) else False
        uid = int(token_to_unit[j]) if j < len(token_to_unit) else -1
        section = ""
        text = ""
        if uid >= 0 and uid < len(units):
            u = units[uid]
            section = f"{u.section:<16}"
            text = token_texts.get(j, u.text[:30])
        else:
            section = "NON-LYRIC"
            text = ""

        lines.append(f"{j:<5} {str(is_lyric):<9} {uid:<4} {section} {text}")

    # Stats
    mask_sum = int(lyric_mask.sum()) if isinstance(lyric_mask, np.ndarray) else 0
    lines.append("")
    lines.append(f"# Total: L={L}, lyric_mask_true={mask_sum}")
    return "\n".join(lines)


# ===================================================================
#  PART 4 — Apply bias and compute attention
# ===================================================================

def apply_bias_mass_preserving(
    logits: torch.Tensor,
    value: torch.Tensor,
    lyric_mask: torch.Tensor,
    bias: torch.Tensor,
    gate: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Apply duration interval bias with mass-preserving split-softmax."""
    output, attn, stats = mass_preserving_attention(
        logits=logits.to(torch.float32),
        value=value.to(torch.float32),
        lyric_mask=lyric_mask,
        bias=bias.to(torch.float32),
        gate=gate,
    )
    return output, attn, stats


# ===================================================================
#  PART 5 — Metrics
# ===================================================================

def compute_line_coverage_metrics(
    A_np: np.ndarray,          # [H, T, L]
    lyric_mask: np.ndarray,    # [L]
    token_to_unit: np.ndarray, # [L]
    unit_boundaries: np.ndarray,
    unit_duration: np.ndarray,
    unit_lyric_mask: np.ndarray,  # [U] bool: whether unit has lyric tokens
) -> dict:
    """Compute line-level coverage metrics.

    Averages attention over heads, aggregates per-unit mass, compares
    to target duration scaffold.
    """
    H, T, L = A_np.shape
    A_mean = A_np.mean(axis=0)  # [T, L]
    U = len(unit_duration)
    eps = 1e-10

    # Total coverage mass per token over all time
    token_coverage = A_mean.sum(axis=0)  # [L]

    # Aggregate per-unit
    coverage_u = np.zeros(U, dtype=np.float32)
    for j in range(L):
        if lyric_mask[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            if 0 <= u < U:
                coverage_u[u] += token_coverage[j]

    # Only non-silence lyric units
    has_lyric = unit_lyric_mask.astype(bool)
    if not has_lyric.any():
        return {
            "line_coverage_error": float("nan"),
            "line_coverage_l1": float("nan"),
            "line_coverage_corr": float("nan"),
            "low_coverage_line_ratio": float("nan"),
            "over_coverage_line_ratio": float("nan"),
            "coverage_entropy": float("nan"),
            "coverage_gini": float("nan"),
        }

    c_lyric = coverage_u[has_lyric] + eps
    c_lyric_norm = c_lyric / c_lyric.sum()

    t_lyric = unit_duration[has_lyric] + eps
    t_lyric_norm = t_lyric / t_lyric.sum()

    # Error
    line_coverage_error = float(np.mean(np.abs(c_lyric_norm - t_lyric_norm)))
    line_coverage_l1 = float(np.sum(np.abs(c_lyric_norm - t_lyric_norm)))

    # Correlation
    from scipy.stats import pearsonr
    try:
        corr, _ = pearsonr(c_lyric_norm, t_lyric_norm)
        line_coverage_corr = float(corr)
    except Exception:
        line_coverage_corr = float("nan")

    # Low/over coverage line ratio
    low_ratio = float(np.mean(c_lyric_norm < 0.25 * t_lyric_norm))
    over_ratio = float(np.mean(c_lyric_norm > 2.0 * t_lyric_norm))

    # Entropy
    entropy = float(-np.sum(c_lyric_norm * np.log(c_lyric_norm + eps)) / np.log(len(c_lyric_norm)))
    gini = float(1 - 2 * np.sum(np.sort(c_lyric_norm) * np.arange(1, len(c_lyric_norm) + 1) / len(c_lyric_norm)) / c_lyric_norm.sum() + eps)

    return {
        "line_coverage_error": line_coverage_error,
        "line_coverage_l1": line_coverage_l1,
        "line_coverage_corr": line_coverage_corr,
        "low_coverage_line_ratio": low_ratio,
        "over_coverage_line_ratio": over_ratio,
        "coverage_entropy": entropy,
        "coverage_gini": gini,
    }


def compute_centroid_metrics(
    A_np: np.ndarray,          # [H, T, L]
    lyric_mask: np.ndarray,    # [L]
    scaffold_pos: np.ndarray,  # [L] unit-centre position per token
) -> dict:
    """Compute centroid metrics using scaffold-based positions."""
    H, T, L = A_np.shape
    lyric_b = lyric_mask.astype(bool)
    eps = 1e-10

    c = np.zeros(T, dtype=np.float32)
    for t in range(T):
        w = A_np[:, t, lyric_b].sum(axis=0)  # sum over heads
        total = w.sum()
        if total > eps:
            c[t] = (w * scaffold_pos[lyric_b]).sum() / total

    delta_c = np.diff(c) if T > 1 else np.array([0.0])
    reversal_rate = float(np.mean(delta_c < -0.005))
    jump_rate = float(np.mean(delta_c > 0.05))
    stagnant_rate = float(np.mean(np.abs(delta_c) < 0.001))

    from scipy.stats import spearmanr
    time_lin = np.linspace(0, 1, T)
    corr, _ = spearmanr(c, time_lin)

    return {
        "reversal_rate": reversal_rate,
        "jump_rate": jump_rate,
        "stagnant_rate": stagnant_rate,
        "centroid_spearman_time": float(corr) if not np.isnan(corr) else 0.0,
        "centroid_range": float(c.max() - c.min()),
        "centroid_std": float(c.std()),
    }


def compute_attention_perturbation(
    A_new_np: np.ndarray,
    A_base_np: np.ndarray,
    bias_np: np.ndarray,
    gate: float,
    logits_np: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    """Compute perturbation metrics between baseline and intervention attention."""
    # L1 difference
    attention_l1 = float(np.mean(np.abs(A_new_np - A_base_np)))

    # KL divergence
    A_new_p = A_new_np + eps
    A_base_p = A_base_np + eps
    A_new_p = A_new_p / A_new_p.sum(axis=-1, keepdims=True)
    A_base_p = A_base_p / A_base_p.sum(axis=-1, keepdims=True)
    kl = (A_new_p * (np.log(A_new_p) - np.log(A_base_p))).sum(axis=-1)
    attention_kl = float(kl.mean())

    # Bias magnitude
    bias_abs_mean = float(np.abs(bias_np).mean())
    bias_abs_max = float(np.abs(bias_np).max())
    effective_bias_abs_mean = gate * bias_abs_mean
    effective_bias_abs_max = gate * bias_abs_max

    # Logit stats
    logit_abs_mean = float(np.abs(logits_np).mean())
    logit_std = float(logits_np.std())
    effective_bias_to_logit_std = effective_bias_abs_mean / max(logit_std, eps)

    return {
        "attention_l1": attention_l1,
        "attention_kl": attention_kl,
        "bias_abs_mean": bias_abs_mean,
        "bias_abs_max": bias_abs_max,
        "effective_bias_abs_mean": effective_bias_abs_mean,
        "effective_bias_abs_max": effective_bias_abs_max,
        "logit_abs_mean": logit_abs_mean,
        "logit_std": logit_std,
        "effective_bias_to_logit_std": effective_bias_to_logit_std,
    }


def compute_lyric_mass(
    A_np: np.ndarray,
    lyric_mask: np.ndarray,
) -> dict:
    """Compute lyric attention mass."""
    lyric_b = lyric_mask.astype(bool)
    mass = A_np[:, :, lyric_b].sum(axis=-1)  # [H, T]
    return {
        "lyric_mass_mean": float(mass.mean()),
        "lyric_mass_std": float(mass.std()),
    }


# ===================================================================
#  PART 6 — Build scaffold positions
# ===================================================================

def compute_scaffold_positions(
    token_to_unit: np.ndarray,
    unit_boundaries: np.ndarray,
    L: int,
    lyric_mask: np.ndarray,
) -> np.ndarray:
    """Unit-centre position for each token (-1 for non-lyric)."""
    pos = np.full(L, -1.0, dtype=np.float32)
    centres = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
    for j in range(L):
        if lyric_mask[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            pos[j] = centres[min(u, len(centres) - 1)]
    return pos


# ===================================================================
#  PART 7 — Process one sample through all configs
# ===================================================================

def process_sample(
    model,
    sample: dict,
    args,
    configs: List[dict],
    device: str,
) -> Optional[dict]:
    """Forward + all sweep configs on one sample."""
    sid = sample["sample_id"]
    lyrics = sample["lyrics"]
    pt_path_str = sample.get("pt_path")
    if pt_path_str is None:
        return None

    pt_data = load_preprocessed_data(Path(pt_path_str))
    if pt_data is None:
        return None

    # ---- Forward + capture logits ----
    hidden_collector = HiddenCollector()
    logit_collector = LogitValueCollector()

    h_handle = model.decoder.layers[args.layer].register_forward_hook(hidden_collector)
    logit_collector.install(model, layer=args.layer)

    try:
        run_teacher_forward(
            model, pt_data["target_latents"], pt_data["attention_mask"],
            pt_data["encoder_hidden_states"], pt_data["encoder_attention_mask"],
            pt_data["context_latents"], device=device,
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

    H_np = H_tensor.squeeze(0).float().numpy()
    logits_gpu = logits_tensor.float().to(device)      # [B, H, T, L] on GPU
    values_gpu = value_tensor.float().to(device)        # [B, H, L, Dh] on GPU
    T_eff, D = H_np.shape
    L_eff = logits_gpu.shape[-1]
    B = logits_gpu.shape[0]

    # ---- Parse section_ids ----
    parser = LyricsStructureParser()
    parsed = parser.parse(lyrics, num_chunks=L_eff)
    section_ids = parsed.section_type_ids

    # ---- Build lyric units ----
    units, raw_lyric_pos_np, debug_info = parse_lyrics_to_units(lyrics, section_ids)
    if len(units) == 0:
        return None

    # ---- Build scaffold ----
    scaffold = build_duration_scaffold(
        units=units, text_len=L_eff, device="cpu",
    )
    unit_boundaries_cpu = scaffold["unit_boundaries"]
    token_to_unit_cpu = scaffold["token_to_unit"]
    lyric_mask_cpu = scaffold["lyric_mask"]
    unit_duration_cpu = scaffold["unit_duration"]

    if not lyric_mask_cpu.any():
        return None

    # Move mask tensors to GPU for attention computation
    lyric_mask_gpu = lyric_mask_cpu.to(device)
    token_to_unit_gpu = token_to_unit_cpu.to(device)

    scaffold_pos_np = compute_scaffold_positions(
        token_to_unit_cpu.numpy(), unit_boundaries_cpu.numpy(), L_eff, lyric_mask_cpu.numpy(),
    )

    # ---- Determine which units are lyric-bearing ----
    unit_lyric_mask_np = np.zeros(len(unit_duration_cpu), dtype=bool)
    for u in units:
        if not u.is_silence and u.unit_id < len(unit_duration_cpu):
            unit_lyric_mask_np[u.unit_id] = True

    # ---- p_base on GPU ----
    p_base_gpu = torch.linspace(0, 1, T_eff, device=device).float().unsqueeze(0).expand(B, T_eff)

    # ---- Token mapping debug file ----
    tok_map = generate_token_mapping_text(units, L_eff, lyric_mask_cpu.numpy(), token_to_unit_cpu.numpy())

    # ---- Baseline attention (GPU) ----
    A0_gpu = F.softmax(logits_gpu, dim=-1, dtype=torch.float32)  # [B, H, T, L]
    A0_np = A0_gpu[0].cpu().numpy()  # [H, T, L] for metrics on CPU

    baseline_metrics = {
        **compute_line_coverage_metrics(
            A0_np, lyric_mask_cpu.numpy(), token_to_unit_cpu.numpy(),
            unit_boundaries_cpu.numpy(), unit_duration_cpu.numpy(), unit_lyric_mask_np,
        ),
        **compute_centroid_metrics(A0_np, lyric_mask_cpu.numpy(), scaffold_pos_np),
        **compute_lyric_mass(A0_np, lyric_mask_cpu.numpy()),
    }

    # ---- Run each config (all on GPU) ----
    config_results = []

    for cfg in configs:
        sigma = cfg["sigma"]
        lambda_ = cfg["lambda_"]
        gate = cfg["gate"]
        max_bias = cfg["max_bias"]

        bias_gpu = build_duration_interval_bias(
            p_final=p_base_gpu,
            unit_boundaries=unit_boundaries_cpu.to(device),
            token_to_unit=token_to_unit_gpu,
            lyric_mask=lyric_mask_gpu,
            sigma=sigma, lambda_=lambda_, max_bias=max_bias,
        )

        try:
            out, A_new_gpu, stats = apply_bias_mass_preserving(
                logits_gpu, values_gpu,
                lyric_mask_gpu, bias_gpu, gate,
            )
        except Exception as e:
            print(f"    Config sigma={sigma},lambda={lambda_},gate={gate} error: {e}")
            continue

        A_new_np = A_new_gpu[0].cpu().numpy()
        bias_np = (gate * bias_gpu[0]).cpu().numpy()  # [T, L]

        line_metrics = compute_line_coverage_metrics(
            A_new_np, lyric_mask_cpu.numpy(), token_to_unit_cpu.numpy(),
            unit_boundaries_cpu.numpy(), unit_duration_cpu.numpy(), unit_lyric_mask_np,
        )
        centroid_metrics = compute_centroid_metrics(
            A_new_np, lyric_mask_cpu.numpy(), scaffold_pos_np,
        )
        mass_metrics = compute_lyric_mass(A_new_np, lyric_mask_cpu.numpy())
        pert_metrics = compute_attention_perturbation(
            A_new_np, A0_np, bias_np, gate,
            logits_gpu[0].cpu().numpy(),
        )
        lyric_mass_delta = mass_metrics["lyric_mass_mean"] - baseline_metrics["lyric_mass_mean"]

        entry = {
            "sample_id": sid,
            "sigma": sigma,
            "lambda_": lambda_,
            "gate": gate,
            "max_bias": max_bias,
            **pert_metrics,
            **line_metrics,
            **centroid_metrics,
            "lyric_mass_delta": lyric_mass_delta,
            "lyric_mass_mean": mass_metrics["lyric_mass_mean"],
        }
        config_results.append(entry)

    return {
        "sample_id": sid,
        "T_eff": T_eff, "L_eff": L_eff,
        "duration": sample["duration"],
        "lyric_ratio": debug_info.get("lyric_ratio", 0),
        "n_units": debug_info.get("n_units", 0),
        "n_unknown": debug_info.get("n_unknown", 0),
        "raw_lyric_count": debug_info.get("raw_lyric_count", 0),
        "total_tokens": L_eff,
        "baseline": baseline_metrics,
        "config_results": config_results,
        "A0_np": A0_np,
        "token_mapping_text": tok_map,
    }


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PMDC Bias Sweep + Lyric Mask Debug — Stage 2.5",
    )
    parser.add_argument("--audio-dir", default="/root/autodl-tmp/musicdata/audios")
    parser.add_argument("--tensor-dir", default=str(TENSOR_DIR_DEFAULT))
    parser.add_argument("--dataset-dir", default="/root/autodl-tmp/musicdata/dataset")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=180)
    parser.add_argument("--max-duration", type=float, default=300)

    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda")

    # Sweep ranges
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.03, 0.05, 0.08, 0.12])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--gates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--max-biases", type=float, nargs="+", dest="max_biases",
                        default=[0.5, 1.0, 2.0])

    parser.add_argument("--output-dir", default="/root/autodl-tmp/pmdc_bias_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Downsample to at most this many audio tokens")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Build all config combos ----
    configs: List[dict] = []
    for sigma in args.sigmas:
        for lambda_ in args.lambdas:
            for gate in args.gates:
                for max_bias in args.max_biases:
                    configs.append({
                        "sigma": sigma, "lambda_": lambda_,
                        "gate": gate, "max_bias": max_bias,
                    })
    print(f"Total configs to sweep: {len(configs)}")

    # ---- Scan audio ----
    print("\n[1/6] Scanning audio...", flush=True)
    audio_dir = Path(args.audio_dir)
    tensor_dir = Path(args.tensor_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    all_audio = scan_audio_files(audio_dir)
    audio_durations = {}
    for f in all_audio:
        dur = get_audio_duration_ffprobe(f)
        if dur is not None:
            audio_durations[f.name] = dur
    valid_audio = [(f, dur) for f in all_audio
                   if (dur := audio_durations.get(f.name)) is not None
                   and args.min_duration <= dur <= args.max_duration]
    print(f"  {len(valid_audio)} files in [{args.min_duration}, {args.max_duration}]s")

    selected = random.sample(valid_audio, min(args.num_samples, len(valid_audio)))

    # ---- Find lyrics ----
    print("\n[2/6] Finding lyrics...", flush=True)
    samples = []
    skipped_log = []
    for f, dur in selected:
        sid = f.stem
        meta = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta)
        if lyrics is None:
            skipped_log.append((sid, "no_lyrics"))
            continue
        pt_path = find_matching_pt(f.stem, tensor_dir)
        samples.append({
            "sample_id": sid, "audio_path": str(f), "duration": dur,
            "lyrics": lyrics, "pt_path": str(pt_path) if pt_path else None,
        })
    print(f"  {len(samples)} samples with lyrics")

    if len(samples) < 1:
        print("  No samples. Aborting.")
        sys.exit(1)

    # ---- Load model ----
    print("\n[3/6] Loading model...", flush=True)
    dit_handler = setup_model(device=args.device)
    model = dit_handler.model

    # ---- Process samples ----
    print(f"\n[4/6] Processing {len(samples)} samples x {len(configs)} configs...", flush=True)
    all_results = []
    mask_debug_rows = []

    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] {sample['sample_id']}", flush=True)

        result = process_sample(model, sample, args, configs, device=args.device)
        if result is None:
            skipped_log.append((sample["sample_id"], "processing_failed"))
            print(f"    Skip")
            continue

        T = result["T_eff"]
        L = result["L_eff"]
        dur = result["duration"]
        lyric_ratio = result["lyric_ratio"]
        n_configs = len(result["config_results"])
        print(f"    T={T}, L={L}, ratio={lyric_ratio:.2f}, configs={n_configs}", flush=True)

        if lyric_ratio > 0.95:
            print(f"    ⚠ WARNING: lyric_ratio > 0.95! Mask may be too broad.", flush=True)

        mask_debug_rows.append({
            "sample_id": result["sample_id"],
            "total_tokens": result["total_tokens"],
            "num_lyric_tokens": result["raw_lyric_count"],
            "num_non_lyric_tokens": result["total_tokens"] - result["raw_lyric_count"],
            "lyric_ratio": lyric_ratio,
            "n_units": result["n_units"],
            "n_unknown": result["n_unknown"],
        })

        all_results.append(result)

    print(f"\n  Processed: {len(all_results)} / {len(samples)}")

    # ---- Save per-sample visualizations ----
    print(f"\n[5/6] Saving visualizations...", flush=True)
    fig_base = OUTPUT_DIR / "figures"
    for result in all_results:
        sid = result["sample_id"]
        fig_dir = fig_base / sid
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Token mapping
        (fig_dir / "token_mapping.txt").write_text(result.get("token_mapping_text", ""))

        # Baseline coverage by unit
        _save_sample_figures(result, fig_dir)

    # ---- Aggregate results ----
    print(f"\n[6/6] Aggregating...", flush=True)

    # Flatten per-config results
    all_rows = []
    for result in all_results:
        for cr in result["config_results"]:
            all_rows.append(cr)

    # CSV
    csv_path = OUTPUT_DIR / "bias_sweep_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  CSV -> {csv_path}")

    # Per-config averages
    config_avg: Dict[str, List[float]] = {}
    for cr in all_rows:
        key = f"sigma={cr['sigma']},lambda={cr['lambda_']},gate={cr['gate']},max_bias={cr['max_bias']}"
        if key not in config_avg:
            config_avg[key] = {"config": {"sigma": cr["sigma"], "lambda_": cr["lambda_"],
                                          "gate": cr["gate"], "max_bias": cr["max_bias"]}, "vals": {}}
        for k, v in cr.items():
            if isinstance(v, (int, float)) and k not in ("sigma", "lambda_", "gate", "max_bias"):
                config_avg[key]["vals"].setdefault(k, []).append(v)

    avg_list = []
    for key, data in config_avg.items():
        avg = data["config"].copy()
        for k, vals in data["vals"].items():
            avg[f"avg_{k}"] = float(np.nanmean(vals)) if vals else float("nan")
        avg_list.append(avg)

    # JSON
    json_path = OUTPUT_DIR / "bias_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_samples": len(all_results),
            "n_configs": len(configs),
            "config_averages": avg_list,
            "config_spec": {
                "sigmas": list(args.sigmas),
                "lambdas": list(args.lambdas),
                "gates": list(args.gates),
                "max_biases": list(args.max_biases),
            },
        }, f, indent=2)
    print(f"  JSON -> {json_path}")

    # Per-sample metrics
    per_sample_path = OUTPUT_DIR / "per_sample_metrics.csv"
    with open(per_sample_path, "w", newline="") as f:
        fieldnames = ["sample_id", "T_eff", "L_eff", "duration", "lyric_ratio", "n_units"]
        for result in all_results:
            for k in fieldnames:
                if k not in result:
                    result[k] = 0
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow({k: result.get(k, "") for k in fieldnames})
    print(f"  Per-sample -> {per_sample_path}")

    # Mask debug CSV
    mask_path = OUTPUT_DIR / "mask_debug.csv"
    if mask_debug_rows:
        with open(mask_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mask_debug_rows[0].keys())
            writer.writeheader()
            writer.writerows(mask_debug_rows)
        print(f"  Mask debug -> {mask_path}")

    # Skipped
    skipped_path = OUTPUT_DIR / "skipped_samples.json"
    with open(skipped_path, "w") as f:
        json.dump([{"sample_id": s, "reason": r} for s, r in skipped_log], f, indent=2)
    print(f"  Skipped -> {skipped_path}")

    # ---- Best config selection ----
    baseline_avg = {}
    if all_results:
        baseline_line_err = np.nanmean([r["baseline"]["line_coverage_error"] for r in all_results])
        baseline_low_ratio = np.nanmean([r["baseline"]["low_coverage_line_ratio"] for r in all_results])
        baseline_rev = np.nanmean([r["baseline"]["reversal_rate"] for r in all_results])
        baseline_centroid_range = np.nanmean([r["baseline"]["centroid_range"] for r in all_results])
        baseline_lyric_mass = np.nanmean([r["baseline"]["lyric_mass_mean"] for r in all_results])
        baseline_entropy = np.nanmean([r["baseline"]["coverage_entropy"] for r in all_results])
    else:
        baseline_line_err = baseline_low_ratio = baseline_rev = baseline_centroid_range = 0
        baseline_lyric_mass = baseline_entropy = 0

    print(f"\n  Baseline: line_coverage_error={baseline_line_err:.4f}, "
          f"low_coverage_ratio={baseline_low_ratio:.4f}, "
          f"reversal_rate={baseline_rev:.4f}")

    # Score each config
    scored = []
    for avg_entry in avg_list:
        akl = avg_entry.get("avg_attention_kl", 0)
        al1 = avg_entry.get("avg_attention_l1", 0)
        lce = avg_entry.get("avg_line_coverage_error", float("nan"))
        lcr = avg_entry.get("avg_low_coverage_line_ratio", float("nan"))
        rev = avg_entry.get("avg_reversal_rate", float("nan"))
        cr = avg_entry.get("avg_centroid_range", float("nan"))
        mass_delta = abs(avg_entry.get("avg_lyric_mass_delta", 0))
        ents = avg_entry.get("avg_coverage_entropy", float("nan"))

        if any(np.isnan(x) for x in [akl, lce, rev, mass_delta]):
            continue

        # Screening
        is_effective = 0.005 <= akl <= 0.3
        is_mass_preserved = mass_delta < 1e-4
        coverage_improved = lce < baseline_line_err
        low_ratio_improved = lcr < baseline_low_ratio if not np.isnan(lcr) else False
        reversal_not_worse = rev <= baseline_rev + 0.02
        centroid_ok = cr > baseline_centroid_range * 0.8  # don't collapse

        score = int(is_effective) + int(is_mass_preserved) + int(coverage_improved) \
                + int(low_ratio_improved) + int(reversal_not_worse) + int(centroid_ok)

        scored.append((score, akl, avg_entry, {
            "is_effective": is_effective,
            "mass_preserved": is_mass_preserved,
            "coverage_improved": coverage_improved,
            "low_ratio_improved": low_ratio_improved,
            "reversal_not_worse": reversal_not_worse,
            "centroid_ok": centroid_ok,
        }))

    # Sort: highest score, then lowest KL
    scored.sort(key=lambda x: (-x[0], x[1]))

    best_configs = []
    for score, akl, entry, flags in scored[:5]:
        best_configs.append({
            "score": score,
            **{k: entry[k] for k in ("sigma", "lambda_", "gate", "max_bias")},
            "avg_attention_kl": akl,
            "avg_line_coverage_error": entry.get("avg_line_coverage_error", float("nan")),
            "avg_low_coverage_line_ratio": entry.get("avg_low_coverage_line_ratio", float("nan")),
            "avg_reversal_rate": entry.get("avg_reversal_rate", float("nan")),
            "avg_centroid_range": entry.get("avg_centroid_range", float("nan")),
            "avg_lyric_mass_delta": entry.get("avg_lyric_mass_delta", float("nan")),
            "flags": flags,
        })

    best_path = OUTPUT_DIR / "best_configs.json"
    with open(best_path, "w") as f:

        def _json_safe(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump({
            "baseline": {
                "line_coverage_error": float(baseline_line_err),
                "low_coverage_line_ratio": float(baseline_low_ratio),
                "reversal_rate": float(baseline_rev),
                "centroid_range": float(baseline_centroid_range),
                "lyric_mass_mean": float(baseline_lyric_mass),
            },
            "best_configs": [{
                k: _json_safe(v) for k, v in bc.items()
            } for bc in best_configs],
        }, f, indent=2, default=_json_safe)
    print(f"  Best configs -> {best_path}")

    # ---- Print summary ----
    print("\n" + "=" * 100)
    print("RESULTS TABLE (top 5 configs)")
    print("=" * 100)
    hdr = f"{'Config':<40s} {'KL':>8s} {'L1':>8s} {'cov_err':>8s} {'low_cov':>8s} {'rev':>8s} {'c_range':>8s} {'mass_d':>8s} {'score':>6s}"
    print(hdr)
    print("-" * 100)
    for entry in best_configs[:5]:
        cfg_str = f"s={entry['sigma']},l={entry['lambda_']},g={entry['gate']},m={entry['max_bias']}"
        print(f"{cfg_str:<40s} {entry['avg_attention_kl']:>8.4f} "
              f"{entry.get('avg_attention_l1', 0):>8.6f} "
              f"{entry.get('avg_line_coverage_error', 0):>8.4f} "
              f"{entry.get('avg_low_coverage_line_ratio', 0):>8.4f} "
              f"{entry.get('avg_reversal_rate', 0):>8.4f} "
              f"{entry.get('avg_centroid_range', 0):>8.4f} "
              f"{entry.get('avg_lyric_mass_delta', 0):>8.6f} "
              f"{entry['score']:>6d}")

    # ---- Print baseline ----
    print(f"\n  Baseline: cov_err={baseline_line_err:.4f}, low_cov={baseline_low_ratio:.4f}, "
          f"rev={baseline_rev:.4f}, c_range={baseline_centroid_range:.4f}, "
          f"lyric_mass={baseline_lyric_mass:.4f}")

    # ---- Conclusions ----
    print("\n" + "=" * 100)
    print("CONCLUSIONS")
    print("=" * 100)

    # Mask check
    high_ratio_samples = sum(1 for r in mask_debug_rows if r["lyric_ratio"] > 0.95)
    if high_ratio_samples == len(mask_debug_rows):
        print("\n  Conclusion A: lyric_ratio=1.0 on all samples. All tokens are")
        print("  section-tagged as lyric. If caption/style prompts exist,")
        print("  lyric_mask is too broad. Inspect token_mapping.txt.")
    elif high_ratio_samples > len(mask_debug_rows) / 2:
        print(f"\n  Conclusion A-: {high_ratio_samples}/{len(mask_debug_rows)} samples have")
        print(f"  lyric_ratio > 0.95. Mask may be too broad on many samples.")
    else:
        print(f"\n  ✓ lyric_mask reasonable: {high_ratio_samples}/{len(mask_debug_rows)} broad samples")

    # Effectiveness
    weak_count = sum(1 for e in avg_list if e.get("avg_attention_kl", 1) < 1e-4)
    if weak_count == len(avg_list):
        print("\n  Conclusion B: All configs produce attention_kl ≈ 0.")
        print("  Bias too weak; metrics unchanged are not meaningful.")
    elif weak_count > len(avg_list) / 2:
        print(f"\n  Conclusion B-: {weak_count}/{len(avg_list)} configs produce near-zero KL.")
        print("  Most configs too weak; higher lambda/gate needed.")

    # Best config
    if best_configs and best_configs[0]["score"] >= 4:
        bc = best_configs[0]
        print(f"\n  Conclusion C: Found weak-effective config:")
        print(f"    sigma={bc['sigma']}, lambda={bc['lambda_']}, gate={bc['gate']}, max_bias={bc['max_bias']}")
        print(f"    attention_kl={bc['avg_attention_kl']:.4f}")
        print(f"    line_coverage_error_delta={bc.get('avg_line_coverage_error', 0) - baseline_line_err:.4f}")
        print(f"    lyric_mass_delta={bc.get('avg_lyric_mass_delta', 0):.6f}")
        print("  => Use this config for next PMDC generation/probe.")
    elif best_configs:
        print(f"\n  Conclusion C-: Best config score={best_configs[0]['score']}/6.")
        print("  Consider wider parameter ranges or bias redesign.")

    if not best_configs or best_configs[0]["score"] < 2:
        print("\n  Conclusion D: No config meaningfully improves line-level coverage.")
        print("  Check duration scaffold, token mapping, or redesign bias construction.")

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


# ===================================================================
#  Visualization helpers
# ===================================================================

def _save_sample_figures(result: dict, fig_dir: Path):
    """Save per-sample diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sid = result["sample_id"]

    # Baseline coverage by unit
    baseline = result.get("baseline", {})
    if not baseline:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_to_show = {
        "line_coverage_error": "Line Cov Error",
        "low_coverage_line_ratio": "Low Cov Ratio",
        "over_coverage_line_ratio": "Over Cov Ratio",
        "reversal_rate": "Reversal Rate",
        "centroid_range": "Centroid Range",
    }
    labels = []
    values = []
    for k, lbl in metrics_to_show.items():
        v = baseline.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            labels.append(lbl)
            values.append(v)
    if values:
        ax.bar(range(len(values)), values, color="steelblue", alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Baseline Metrics — {sid}")
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_metrics.png", dpi=150)
    plt.close(fig)

    # Config overlay
    crs = result.get("config_results", [])
    if crs:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # KL vs lambda*gate
        for ax_i, (metric, ylabel) in enumerate([
            ("attention_kl", "KL divergence"),
            ("attention_l1", "L1 diff"),
            ("line_coverage_error", "Line coverage error"),
            ("reversal_rate", "Reversal rate"),
        ]):
            ax = axes[ax_i // 2, ax_i % 2]
            for cr in crs:
                label = f"g={cr['gate']},l={cr['lambda_']}"
                ax.scatter(cr["sigma"], cr.get(metric, 0),
                           s=80, alpha=0.5, label=label)
            ax.set_xlabel("sigma")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs sigma — {sid}")
            ax.grid(alpha=0.3)
            if ax_i == 0:
                ax.legend(fontsize=6, loc="upper right")

        fig.tight_layout()
        fig.savefig(fig_dir / "config_sweep.png", dpi=150)
        plt.close(fig)

        # Best config coverage comparison
        if len(crs) > 0:
            # Find config with lowest coverage error
            best = min(crs, key=lambda x: x.get("line_coverage_error", float("inf")))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar([0, 1],
                   [baseline.get("line_coverage_error", 0),
                    best.get("line_coverage_error", 0)],
                   color=["gray", "darkgreen"], alpha=0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Baseline", f"Best (g={best['gate']},l={best['lambda_']})"])
            ax.set_ylabel("Line Coverage Error")
            ax.set_title(f"Coverage Error Reduction — {sid}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / "coverage_comparison.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
