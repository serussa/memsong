#!/usr/bin/env python3
"""
pmdc_temporal_localization_probe — Stage 2.6.

Measures whether each lyric unit is attended at the **correct time**
according to its duration-interval scaffold, and whether duration
interval bias improves this temporal localisation.

Modes
-----
baseline                   — raw cross-attention
fixed_linear               — token-level linear progress bias
duration_weak              — s=0.03, l=0.5, g=0.5, m=1.0 (KL ≈ 0.005)
duration_mid               — s=0.03, l=2.0, g=0.5, m=2.0 (KL ≈ 0.019)
duration_strong            — s=0.12, l=0.5, g=0.5, m=2.0 (KL ≈ 0.041)
"""

from __future__ import annotations

import argparse
import csv
import json
import functools
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

# Default configs
WEAK_CONFIG  = {"sigma": 0.03, "lambda_": 0.5, "gate": 0.5, "max_bias": 1.0}
MID_CONFIG   = {"sigma": 0.03, "lambda_": 2.0, "gate": 0.5, "max_bias": 2.0}
STRONG_CONFIG = {"sigma": 0.12, "lambda_": 0.5, "gate": 0.5, "max_bias": 2.0}


# ===================================================================
#  PART 1 — Audio scanning & metadata matching (from earlier scripts)
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
#  PART 3 — Lyric structure parsing
# ===================================================================

def parse_lyrics_to_units(lyrics_text: str, section_ids: torch.Tensor) -> Tuple[List[LyricUnit], np.ndarray, dict]:
    lines = [l for l in lyrics_text.strip().split("\n") if l.strip()]
    L = len(section_ids)
    ids_np = section_ids.cpu().numpy() if isinstance(section_ids, torch.Tensor) else section_ids
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

    lyric_mask_raw = (ids_np >= 1) & (ids_np <= 7)
    lyric_indices = np.where(lyric_mask_raw)[0]
    units: List[LyricUnit] = []
    next_unit_id = 0
    token_pos = np.full(L, -1.0, dtype=np.float32)

    if len(lyric_indices) == 0:
        return units, token_pos, {"lyric_ratio": 0.0}

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
            units.append(LyricUnit(unit_id=next_unit_id, section=sec, text=line,
                char_count=char_count, token_indices=[int(t) for t in token_idxs],
                occurrence_id=0, is_silence=False))
            next_unit_id += 1
    else:
        units.append(LyricUnit(unit_id=next_unit_id, section="UNKNOWN", text=lyrics_text,
            char_count=len(lyrics_text), token_indices=list(lyric_indices),
            occurrence_id=0, is_silence=True))
        next_unit_id += 1

    for (orig_idx, line, sec) in silence_lines:
        start, end = chunk_edges[orig_idx], chunk_edges[orig_idx + 1]
        chunk_mask = lyric_mask_raw[start:end]
        if not chunk_mask.any():
            units.append(LyricUnit(unit_id=next_unit_id, section=sec, text=line,
                char_count=0, token_indices=[], occurrence_id=0, is_silence=True))
            next_unit_id += 1

    valid_indices = np.where(lyric_mask_raw)[0]
    if len(valid_indices) > 1:
        for k, idx in enumerate(valid_indices):
            token_pos[idx] = k / (len(valid_indices) - 1)
    elif len(valid_indices) == 1:
        token_pos[valid_indices[0]] = 0.5

    return units, token_pos, {"lyric_ratio": float(lyric_mask_raw.sum()) / max(L, 1)}


# ===================================================================
#  PART 4 — Bias application (mass-preserving)
# ===================================================================

def apply_bias_mp(logits: torch.Tensor, value: torch.Tensor,
                  lyric_mask: torch.Tensor, bias: torch.Tensor, gate: float):
    return mass_preserving_attention(
        logits=logits.to(torch.float32), value=value.to(torch.float32),
        lyric_mask=lyric_mask, bias=bias.to(torch.float32), gate=gate,
    )


def apply_linear_bias(logits: torch.Tensor, value: torch.Tensor,
                      bias: torch.Tensor, gate: float):
    biased = logits + gate * bias.to(device=logits.device, dtype=logits.dtype)
    attn = F.softmax(biased, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn.to(value.dtype), value)
    return out, attn, {}


# ===================================================================
#  PART 5 — Temporal localisation metrics
# ===================================================================

def compute_temporal_metrics(
    A_np: np.ndarray,           # [H, T, L] attention weights
    lyric_mask: np.ndarray,     # [L] bool
    token_to_unit: np.ndarray,  # [L] unit id (-1 = non-lyric)
    unit_boundaries: np.ndarray, # [U+1]
    unit_lyric_mask: np.ndarray, # [U] bool
    eps: float = 1e-10,
) -> dict:
    """Compute unit-level temporal localisation metrics."""
    H, T, L = A_np.shape
    A_mean = A_np.mean(axis=0)  # [T, L]
    U = len(unit_boundaries) - 1
    time_lin = np.linspace(0, 1, T)

    # Aggregate per-unit attention over time: A_u[t] = sum over tokens in unit u
    unit_attn = np.zeros((T, U), dtype=np.float32)  # [T, U]
    for j in range(L):
        if lyric_mask[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            if 0 <= u < U:
                unit_attn[:, u] += A_mean[:, j]

    # Per-unit metrics
    center_errors = []
    outside_ratios = []
    early_ratios = []
    late_ratios = []
    hit_half = []
    hit_full = []
    pred_centers = []
    target_centers = []

    for u in range(U):
        if not unit_lyric_mask[u]:
            continue

        Au = unit_attn[:, u]  # [T]
        total_mass = Au.sum()
        if total_mass < eps:
            continue

        start_u = float(unit_boundaries[u])
        end_u = float(unit_boundaries[u + 1])
        target_center = (start_u + end_u) / 2
        width = end_u - start_u

        # Time center of mass
        time_center = (Au * time_lin).sum() / total_mass
        center_err = abs(time_center - target_center)

        # In/outside interval
        inside = (time_lin >= start_u) & (time_lin <= end_u)
        inside_mass = Au[inside].sum()
        outside_ratio = 1.0 - inside_mass / total_mass

        # Early/late
        early = time_lin < start_u
        late = time_lin > end_u
        early_ratio = Au[early].sum() / total_mass
        late_ratio = Au[late].sum() / total_mass

        # Hit at half/full width
        hit_half.append(float(center_err <= 0.5 * width))
        hit_full.append(float(center_err <= width))

        center_errors.append(center_err)
        outside_ratios.append(outside_ratio)
        early_ratios.append(early_ratio)
        late_ratios.append(late_ratio)
        pred_centers.append(time_center)
        target_centers.append(target_center)

    if len(center_errors) < 2:
        return {
            "unit_time_center_error_mean": float("nan"),
            "unit_time_center_error_median": float("nan"),
            "unit_time_center_error_p90": float("nan"),
            "out_of_interval_ratio_mean": float("nan"),
            "out_of_interval_ratio_median": float("nan"),
            "out_of_interval_ratio_p90": float("nan"),
            "early_attention_ratio_mean": float("nan"),
            "late_attention_ratio_mean": float("nan"),
            "unit_center_pearson": float("nan"),
            "unit_center_spearman": float("nan"),
            "interval_hit_rate_half_width": float("nan"),
            "interval_hit_rate_full_width": float("nan"),
        }

    err_arr = np.array(center_errors)
    out_arr = np.array(outside_ratios)

    from scipy.stats import pearsonr, spearmanr
    pc = float(pearsonr(pred_centers, target_centers)[0]) if len(pred_centers) > 1 else float("nan")
    sc = float(spearmanr(pred_centers, target_centers)[0]) if len(pred_centers) > 1 else float("nan")

    return {
        "unit_time_center_error_mean": float(np.mean(err_arr)),
        "unit_time_center_error_median": float(np.median(err_arr)),
        "unit_time_center_error_p90": float(np.percentile(err_arr, 90)),
        "out_of_interval_ratio_mean": float(np.mean(out_arr)),
        "out_of_interval_ratio_median": float(np.median(out_arr)),
        "out_of_interval_ratio_p90": float(np.percentile(out_arr, 90)),
        "early_attention_ratio_mean": float(np.mean(early_ratios)),
        "late_attention_ratio_mean": float(np.mean(late_ratios)),
        "unit_center_pearson": pc,
        "unit_center_spearman": sc,
        "interval_hit_rate_half_width": float(np.mean(hit_half)),
        "interval_hit_rate_full_width": float(np.mean(hit_full)),
    }


def compute_standard_metrics(
    A_np: np.ndarray,
    lyric_mask: np.ndarray,
    token_to_unit: np.ndarray,
    unit_boundaries: np.ndarray,
    unit_duration: np.ndarray,
    unit_lyric_mask: np.ndarray,
    A0_np: np.ndarray,
    B_np: np.ndarray,     # bias [T, L] (already gated)
    logits_np: np.ndarray, # [H, T, L]
    gate: float,
    eps: float = 1e-10,
) -> dict:
    """Coverage, centroid, perturbation metrics (from Stage 2.5)."""
    # L1 & KL
    attention_l1 = float(np.mean(np.abs(A_np - A0_np)))
    Ak = A_np + eps; A0k = A0_np + eps
    Ak = Ak / Ak.sum(axis=-1, keepdims=True)
    A0k = A0k / A0k.sum(axis=-1, keepdims=True)
    kl = (Ak * (np.log(Ak) - np.log(A0k))).sum(axis=-1)
    attention_kl = float(kl.mean())
    logit_std = float(logits_np.std())

    # Line-level coverage
    H, T, L = A_np.shape
    A_mean = A_np.mean(axis=0)
    token_cov = A_mean.sum(axis=0)
    U = len(unit_duration)
    coverage_u = np.zeros(U, dtype=np.float32)
    for j in range(L):
        if lyric_mask[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            if 0 <= u < U:
                coverage_u[u] += token_cov[j]
    has_lyric = unit_lyric_mask.astype(bool)
    if has_lyric.any():
        c_lyric = coverage_u[has_lyric] + eps
        t_lyric = unit_duration[has_lyric] + eps
        c_lyric /= c_lyric.sum()
        t_lyric /= t_lyric.sum()
        line_coverage_error = float(np.mean(np.abs(c_lyric - t_lyric)))
        low_cov = float(np.mean(c_lyric < 0.25 * t_lyric))
    else:
        line_coverage_error = low_cov = float("nan")

    # Centroid (scaffold positions)
    centres = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
    scaffold_pos = np.full(L, -1.0, dtype=np.float32)
    for j in range(L):
        if lyric_mask[j] and token_to_unit[j] >= 0:
            u = int(token_to_unit[j])
            scaffold_pos[j] = centres[min(u, len(centres) - 1)]
    lyric_b = lyric_mask.astype(bool)
    c = np.zeros(T, dtype=np.float32)
    for t in range(T):
        w = A_mean[t, lyric_b]
        total = w.sum()
        if total > eps:
            c[t] = (w * scaffold_pos[lyric_b]).sum() / total
    delta_c = np.diff(c) if T > 1 else np.array([0.0])
    reversal_rate = float(np.mean(delta_c < -0.005))

    # Lyric mass
    lyric_mass = float(A_np[:, :, lyric_b].sum(axis=-1).mean())

    # Perturbation
    bias_abs_mean = float(np.abs(B_np).mean())
    eff_bias_to_logit = gate * bias_abs_mean / max(logit_std, eps)

    return {
        "attention_kl": attention_kl,
        "attention_l1": attention_l1,
        "effective_bias_to_logit_std": eff_bias_to_logit,
        "reversal_rate": reversal_rate,
        "line_coverage_error": line_coverage_error,
        "low_coverage_line_ratio": low_cov,
        "lyric_mass_mean": lyric_mass,
    }


# ===================================================================
#  PART 6 — Per-sample processing
# ===================================================================

def process_sample(model, sample: dict, configs: dict, device: str) -> Optional[dict]:
    sid = sample["sample_id"]
    lyrics = sample["lyrics"]
    pt_path_str = sample.get("pt_path")
    if pt_path_str is None:
        return None
    pt_data = load_preprocessed_data(Path(pt_path_str))
    if pt_data is None:
        return None

    # Forward + collect
    hidden_collector = HiddenCollector()
    logit_collector = LogitValueCollector()
    h_handle = model.decoder.layers[12].register_forward_hook(hidden_collector)
    logit_collector.install(model, layer=12)
    try:
        run_teacher_forward(
            model, pt_data["target_latents"], pt_data["attention_mask"],
            pt_data["encoder_hidden_states"], pt_data["encoder_attention_mask"],
            pt_data["context_latents"], device=device,
        )
    except Exception as e:
        h_handle.remove(); logit_collector.uninstall()
        return None
    h_handle.remove(); logit_collector.uninstall()

    H_t = hidden_collector.get()
    logits_t = logit_collector.logits
    value_t = logit_collector.value
    if H_t is None or logits_t is None or value_t is None:
        return None

    H_np = H_t.squeeze(0).float().numpy()
    logits_gpu = logits_t.float().to(device)
    values_gpu = value_t.float().to(device)
    T_eff, D = H_np.shape
    L_eff = logits_gpu.shape[-1]

    # Parse lyrics
    parser = LyricsStructureParser()
    parsed = parser.parse(lyrics, num_chunks=L_eff)
    section_ids = parsed.section_type_ids
    units, raw_lyric_pos_np, debug_info = parse_lyrics_to_units(lyrics, section_ids)
    if len(units) == 0:
        return None

    # Scaffold
    scaffold = build_duration_scaffold(units=units, text_len=L_eff, device=device)
    unit_boundaries = scaffold["unit_boundaries"]
    token_to_unit = scaffold["token_to_unit"]
    lyric_mask = scaffold["lyric_mask"]
    unit_duration = scaffold["unit_duration"]
    if not lyric_mask.any():
        return None

    # Unit lyric mask
    unit_lyric_mask_np = torch.zeros(len(unit_duration), dtype=torch.bool)
    for u in units:
        if not u.is_silence and u.unit_id < len(unit_duration):
            unit_lyric_mask_np[u.unit_id] = True

    # Build fixed linear bias
    lyric_pos = torch.from_numpy(raw_lyric_pos_np).float().to(device)
    fixed_bias = _build_fixed_linear_bias(T_eff, L_eff, lyric_pos, lyric_mask, device=device)

    # p_base
    p_base_gpu = torch.linspace(0, 1, T_eff, device=device).float().unsqueeze(0)

    # Baseline attention
    A0_gpu = F.softmax(logits_gpu, dim=-1, dtype=torch.float32)
    A0_np = A0_gpu[0].cpu().numpy()  # [H, T, L]

    # Run modes
    modes = {
        "baseline": (None, None, None),
    }

    # Fixed linear
    modes["fixed_linear"] = ("linear", fixed_bias, None)

    # Duration modes
    for label, cfg in [("duration_weak", configs["weak"]),
                        ("duration_mid", configs["mid"]),
                        ("duration_strong", configs["strong"])]:
        bias_t = build_duration_interval_bias(
            p_final=p_base_gpu, unit_boundaries=unit_boundaries,
            token_to_unit=token_to_unit, lyric_mask=lyric_mask,
            sigma=cfg["sigma"], lambda_=cfg["lambda_"], max_bias=cfg["max_bias"],
        )
        modes[label] = ("duration", bias_t, cfg["gate"])

    mode_metrics = {}
    mode_attn = {}
    mode_unit_attn = {}

    for mname, (mtype, bias, gate) in modes.items():
        try:
            if mtype is None:
                A_new_gpu = A0_gpu
                out = None
            elif mtype == "linear":
                bias_exp = bias.expand(-1, logits_gpu.shape[1], -1, -1)
                out, A_new_gpu, _ = apply_linear_bias(logits_gpu, values_gpu, bias_exp, 0.3)
            elif mtype == "duration":
                out, A_new_gpu, _ = apply_bias_mp(logits_gpu, values_gpu, lyric_mask, bias, float(gate))
            else:
                continue
        except Exception:
            continue

        A_np = A_new_gpu[0].cpu().numpy()
        bias_np = _get_bias_np(bias, gate, mtype, T_eff, L_eff)

        temp = compute_temporal_metrics(
            A_np, lyric_mask.cpu().numpy(), token_to_unit.cpu().numpy(),
            unit_boundaries.cpu().numpy(), unit_lyric_mask_np.cpu().numpy(),
        )
        std = compute_standard_metrics(
            A_np, lyric_mask.cpu().numpy(), token_to_unit.cpu().numpy(),
            unit_boundaries.cpu().numpy(), unit_duration.cpu().numpy(),
            unit_lyric_mask_np.cpu().numpy(), A0_np, bias_np,
            logits_gpu[0].cpu().numpy(), float(gate or 0),
        )
        metric = {**temp, **std}
        metric["lyric_mass_delta"] = metric.get("lyric_mass_mean", 1.0) - 1.0

        mode_metrics[mname] = metric
        mode_attn[mname] = A_np

        # Per-unit attention matrix for viz [T, U]
        if mname in ("baseline", "duration_strong"):
            A_mean = A_np.mean(axis=0)
            U = len(unit_duration)
            unit_attn = np.zeros((T_eff, U), dtype=np.float32)
            lm = lyric_mask.cpu().numpy().astype(bool)
            t2u = token_to_unit.cpu().numpy()
            for j in range(L_eff):
                if lm[j] and t2u[j] >= 0:
                    u = int(t2u[j])
                    if 0 <= u < U:
                        unit_attn[:, u] += A_mean[:, j]
            mode_unit_attn[mname] = unit_attn

    return {
        "sample_id": sid,
        "T_eff": T_eff, "L_eff": L_eff,
        "duration": sample["duration"],
        "mode_metrics": mode_metrics,
        "mode_unit_attn": mode_unit_attn,
        "unit_boundaries": unit_boundaries.cpu().numpy(),
        "unit_lyric_mask": unit_lyric_mask_np.cpu().numpy(),
        "units": units,
    }


def _build_fixed_linear_bias(T_audio, L_text, lyric_pos, lyric_mask, device="cuda"):
    audio_idx = torch.arange(T_audio, device=device)
    progress = audio_idx.float() / max(T_audio - 1, 1)
    dist = lyric_pos.unsqueeze(0) - progress.unsqueeze(1)
    bias = -1.0 * (dist / 0.15) ** 2
    bias = bias.clamp(min=-3.0, max=0.0)
    bias = bias * lyric_mask.unsqueeze(0).float()
    bias = bias.unsqueeze(0).unsqueeze(0)
    return bias.contiguous()


def _get_bias_np(bias, gate, mtype, T, L):
    if bias is None:
        return np.zeros((T, L))
    b = bias[0].cpu().numpy() if isinstance(bias, torch.Tensor) else bias
    if mtype == "duration":
        return float(gate) * b
    return float(gate or 1.0) * b


# ===================================================================
#  PART 7 — Visualisation
# ===================================================================

def save_figures(result: dict, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    sid = result["sample_id"]
    U_b = result["unit_boundaries"]
    U_l = result["unit_lyric_mask"]
    unit_centers = (U_b[:-1] + U_b[1:]) / 2
    T_eff = result["T_eff"]
    time_lin = np.linspace(0, 1, T_eff)

    for mname in ["baseline", "fixed_linear", "duration_weak", "duration_mid", "duration_strong"]:
        if mname not in result["mode_metrics"]:
            continue
        if "unit_center_pearson" not in result["mode_metrics"][mname]:
            continue
        pc = result["mode_metrics"][mname].get("unit_center_pearson", float("nan"))

        # Build pred/target centers from mode metrics
        # Reconstruct from unit_attn if available
        if mname in result.get("mode_unit_attn", {}):
            unit_attn = result["mode_unit_attn"][mname]
            pred_c = []
            tgt_c = []
            for u in range(unit_attn.shape[1]):
                if not U_l[u]:
                    continue
                Au = unit_attn[:, u]
                total = Au.sum()
                if total > 1e-10:
                    pred_c.append((Au * time_lin).sum() / total)
                    tgt_c.append(unit_centers[u])
            if len(pred_c) > 1:
                fig, ax = plt.subplots(figsize=(10, 5))
                x = list(range(len(pred_c)))
                ax.scatter(x, tgt_c, color="forestgreen", s=40, alpha=0.7, label="Target center")
                ax.scatter(x, pred_c, color="darkorange", s=30, alpha=0.7, label="Predicted center")
                for i in range(len(pred_c)):
                    ax.plot([x[i], x[i]], [tgt_c[i], pred_c[i]], color="gray", linewidth=0.5, alpha=0.5)
                ax.set_xlabel("Unit index")
                ax.set_ylabel("Normalised time")
                ax.set_title(f"Unit Temporal Centres ({mname}) — {sid}  (r={pc:.3f})")
                ax.legend()
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_dir / f"unit_temporal_centers_{mname}.png", dpi=150)
                plt.close(fig)

    # Out-of-interval bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(U_b) - 1)[U_l]
    if len(x) > 0:
        width = 0.15
        for i, mname in enumerate(["baseline", "duration_weak", "duration_mid", "duration_strong"]):
            if mname not in result["mode_metrics"]:
                continue
            ooir = result["mode_metrics"][mname].get("out_of_interval_ratio_mean", float("nan"))
            ax.bar(i, ooir if not np.isnan(ooir) else 0, width, label=mname, alpha=0.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["baseline", "weak", "mid", "strong"], rotation=30)
        ax.set_ylabel("Out-of-interval ratio")
        ax.set_title(f"Out-of-Interval Ratio — {sid}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "out_of_interval_by_unit.png", dpi=150)
    plt.close(fig)

    # Heatmap for baseline and strong
    for mname in ["baseline", "duration_strong"]:
        if mname not in result.get("mode_unit_attn", {}):
            continue
        unit_attn = result["mode_unit_attn"][mname]  # [T, U]
        # Normalise per unit
        unit_attn_norm = unit_attn / (unit_attn.sum(axis=0, keepdims=True) + 1e-10)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(unit_attn_norm.T, aspect="auto", origin="lower",
                       cmap="viridis", interpolation="nearest",
                       extent=[0, 1, 0, unit_attn.shape[1]])
        ax.set_xlabel("Audio time")
        ax.set_ylabel("Unit index")
        ax.set_title(f"Unit-Time Attention ({mname}) — {sid}")
        plt.colorbar(im, ax=ax)
        # Diagonal target line
        ax.plot([0, 1], [0, unit_attn.shape[1] - 1], color="white", linewidth=0.8,
                linestyle="--", alpha=0.5, label="Target diagonal")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"attention_unit_time_heatmap_{mname}.png", dpi=150)
        plt.close(fig)


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="PMDC Temporal Localisation Probe")
    parser.add_argument("--audio-dir", default="/root/autodl-tmp/musicdata/audios")
    parser.add_argument("--tensor-dir", default=str(TENSOR_DIR_DEFAULT))
    parser.add_argument("--dataset-dir", default="/root/autodl-tmp/musicdata/dataset")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=180)
    parser.add_argument("--max-duration", type=float, default=300)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/pmdc_temporal_probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configs
    configs = {"weak": WEAK_CONFIG, "mid": MID_CONFIG, "strong": STRONG_CONFIG}

    # Scan
    print("=" * 70)
    print("PMDC TEMPORAL LOCALISATION PROBE — Stage 2.6")
    print(f"  weak:   {WEAK_CONFIG}")
    print(f"  mid:    {MID_CONFIG}")
    print(f"  strong: {STRONG_CONFIG}")
    print("=" * 70)

    print("\n[1/5] Scanning audio...", flush=True)
    audio_dir = Path(args.audio_dir)
    tensor_dir = Path(args.tensor_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    all_audio = scan_audio_files(audio_dir)
    audio_durations = {}
    for f in all_audio:
        d = get_audio_duration_ffprobe(f)
        if d is not None:
            audio_durations[f.name] = d
    valid_audio = [(f, d) for f in all_audio
                   if (d := audio_durations.get(f.name)) is not None
                   and args.min_duration <= d <= args.max_duration]
    print(f"  {len(valid_audio)} valid")
    selected = random.sample(valid_audio, min(args.num_samples, len(valid_audio)))

    print("\n[2/5] Finding lyrics...", flush=True)
    samples = []; skipped_log = []
    for f, dur in selected:
        sid = f.stem
        meta = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta)
        if lyrics is None:
            skipped_log.append((sid, "no_lyrics")); continue
        pt_path = find_matching_pt(f.stem, tensor_dir)
        samples.append({"sample_id": sid, "duration": dur, "lyrics": lyrics,
                        "pt_path": str(pt_path) if pt_path else None})
    print(f"  {len(samples)} samples")

    print("\n[3/5] Loading model...", flush=True)
    dt = setup_model(device=args.device)
    model = dt.model

    print(f"\n[4/5] Processing {len(samples)} samples...", flush=True)
    all_results = []
    for idx, sample in enumerate(samples):
        print(f"  [{idx+1}/{len(samples)}] {sample['sample_id']}", flush=True)
        result = process_sample(model, sample, configs, device=args.device)
        if result is None:
            skipped_log.append((sample["sample_id"], "processing_failed"))
            continue
        all_results.append(result)
        print(f"    T={result['T_eff']}, modes={list(result['mode_metrics'].keys())}", flush=True)

    print(f"\n  Processed: {len(all_results)} / {len(samples)}")

    # Visualise
    print("\n[5/5] Saving...", flush=True)
    fig_base = OUTPUT_DIR / "figures"
    for result in all_results:
        save_figures(result, fig_base / result["sample_id"])

    # Aggregate metrics
    mode_names = ["baseline", "fixed_linear", "duration_weak", "duration_mid", "duration_strong"]
    key_metrics = [
        "unit_time_center_error_mean", "unit_time_center_error_median",
        "out_of_interval_ratio_mean", "out_of_interval_ratio_median",
        "early_attention_ratio_mean", "late_attention_ratio_mean",
        "unit_center_pearson", "unit_center_spearman",
        "interval_hit_rate_half_width", "interval_hit_rate_full_width",
        "attention_kl", "attention_l1",
        "reversal_rate", "line_coverage_error",
        "lyric_mass_delta",
    ]

    agg = {m: {k: [] for k in key_metrics} for m in mode_names}
    for result in all_results:
        for mname, metrics in result["mode_metrics"].items():
            if mname not in agg:
                continue
            for k in key_metrics:
                v = metrics.get(k, float("nan"))
                if not np.isnan(v):
                    agg[mname][k].append(v)

    summary = {}
    for mname in mode_names:
        summary[mname] = {}
        for k in key_metrics:
            vals = agg[mname].get(k, [])
            summary[mname][k] = float(np.mean(vals)) if vals else float("nan")

    # Print table
    print("\n" + "=" * 130)
    print("AGGREGATED RESULTS")
    print("=" * 130)
    hdr = f"{'Mode':<30s}"
    for k in ["unit_time_center_error_mean", "out_of_interval_ratio_mean",
              "unit_center_spearman", "attention_kl",
              "interval_hit_rate_full_width", "lyric_mass_delta"]:
        hdr += f" {k:>24s}"
    print(hdr)
    print("-" * 130)
    for mname in mode_names:
        if mname not in summary:
            continue
        row = f"{mname:<30s}"
        for k in ["unit_time_center_error_mean", "out_of_interval_ratio_mean",
                   "unit_center_spearman", "attention_kl",
                   "interval_hit_rate_full_width", "lyric_mass_delta"]:
            v = summary[mname].get(k, float("nan"))
            row += f" {v:>24.6f}" if not np.isnan(v) else f" {'':>24s}"
        print(row)

    # CSV
    csv_rows = []
    for result in all_results:
        base = {"sample_id": result["sample_id"], "T_eff": result["T_eff"],
                "L_eff": result["L_eff"], "duration": result["duration"]}
        for mname, metrics in result["mode_metrics"].items():
            row = {**base, "mode": mname, **metrics}
            csv_rows.append(row)
    csv_path = OUTPUT_DIR / "temporal_localization_results.csv"
    if csv_rows:
        all_keys = set(csv_rows[0].keys())
        for r in csv_rows[1:]:
            all_keys.update(r.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader(); writer.writerows(csv_rows)
        print(f"  CSV -> {csv_path}")

    # JSON
    json_path = OUTPUT_DIR / "temporal_localization_results.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "n_samples": len(all_results)}, f, indent=2)
    print(f"  JSON -> {json_path}")

    # Skipped
    skipped_path = OUTPUT_DIR / "skipped_samples.json"
    with open(skipped_path, "w") as f:
        json.dump([{"sample_id": s, "reason": r} for s, r in skipped_log], f, indent=2)

    # ---- Best temporal config ----
    bs = summary.get("baseline", {})
    best_score = -1
    best_config = None
    best_metrics = {}

    for cname, cfg in [("weak", WEAK_CONFIG), ("mid", MID_CONFIG), ("strong", STRONG_CONFIG)]:
        mname = f"duration_{cname}"
        if mname not in summary:
            continue
        s = summary[mname]
        akl = s.get("attention_kl", 0)
        ooir = s.get("out_of_interval_ratio_mean", float("nan"))
        booir = bs.get("out_of_interval_ratio_mean", float("nan"))
        cerr = s.get("unit_time_center_error_mean", float("nan"))
        bcerr = bs.get("unit_time_center_error_mean", float("nan"))
        sp = s.get("unit_center_spearman", float("nan"))
        bsp = bs.get("unit_center_spearman", float("nan"))
        md = s.get("lyric_mass_delta", 0)
        rev = s.get("reversal_rate", float("nan"))
        brev = bs.get("reversal_rate", float("nan"))

        score = 0
        if not np.isnan(ooir) and not np.isnan(booir) and ooir < booir: score += 1
        if not np.isnan(cerr) and not np.isnan(bcerr) and cerr < bcerr: score += 1
        if not np.isnan(sp) and not np.isnan(bsp) and sp > bsp: score += 1
        if 0.005 <= akl <= 0.08: score += 1
        if abs(md) < 1e-5: score += 1
        if not np.isnan(rev) and not np.isnan(brev) and rev <= brev + 0.02: score += 1

        if score > best_score:
            best_score = score
            best_config = {**cfg, "name": cname}
            best_metrics = {
                "attention_kl": akl, "out_of_interval_ratio_mean": ooir,
                "unit_time_center_error_mean": cerr, "unit_center_spearman": sp,
                "lyric_mass_delta": md,
            }

    best_path = OUTPUT_DIR / "best_temporal_config.json"
    with open(best_path, "w") as f:
        json.dump({
            "best_config": best_config, "score": best_score,
            "baseline": {
                "out_of_interval_ratio_mean": bs.get("out_of_interval_ratio_mean", float("nan")),
                "unit_time_center_error_mean": bs.get("unit_time_center_error_mean", float("nan")),
                "unit_center_spearman": bs.get("unit_center_spearman", float("nan")),
                "reversal_rate": bs.get("reversal_rate", float("nan")),
            },
            "best_metrics": best_metrics,
        }, f, indent=2)
    print(f"  Best temporal config -> {best_path}")

    # ---- Conclusions ----
    print("\n" + "=" * 130)
    print("CONCLUSIONS")
    print("=" * 130)

    for mname in ["fixed_linear", "duration_weak", "duration_mid", "duration_strong"]:
        if mname not in summary:
            continue
        s = summary[mname]
        cerr = s.get("unit_time_center_error_mean", float("nan"))
        bcerr = bs.get("unit_time_center_error_mean", float("nan"))
        ooir = s.get("out_of_interval_ratio_mean", float("nan"))
        booir = bs.get("out_of_interval_ratio_mean", float("nan"))
        sp = s.get("unit_center_spearman", float("nan"))
        bsp = bs.get("unit_center_spearman", float("nan"))
        akl = s.get("attention_kl", float("nan"))
        cerr_delta = (cerr - bcerr) if not (np.isnan(cerr) or np.isnan(bcerr)) else float("nan")
        ooir_delta = (ooir - booir) if not (np.isnan(ooir) or np.isnan(booir)) else float("nan")
        sp_delta = (sp - bsp) if not (np.isnan(sp) or np.isnan(bsp)) else float("nan")
        print(f"  {mname}: KL={akl:.4f}, center_err_delta={cerr_delta:.4f}, "
              f"ooir_delta={ooir_delta:.4f}, sp_delta={sp_delta:.4f}")

    # Check if duration improves temporal localisation
    dur_improves_ooir = False
    dur_improves_center = False
    dur_improves_sp = False
    for cname in ["duration_weak", "duration_mid", "duration_strong"]:
        if cname not in summary:
            continue
        s = summary[cname]
        booir = bs.get("out_of_interval_ratio_mean", float("nan"))
        bcerr = bs.get("unit_time_center_error_mean", float("nan"))
        bsp = bs.get("unit_center_spearman", float("nan"))
        if not np.isnan(s.get("out_of_interval_ratio_mean", float("nan"))) and \
           not np.isnan(booir) and s["out_of_interval_ratio_mean"] < booir:
            dur_improves_ooir = True
        if not np.isnan(s.get("unit_time_center_error_mean", float("nan"))) and \
           not np.isnan(bcerr) and s["unit_time_center_error_mean"] < bcerr:
            dur_improves_center = True
        if not np.isnan(s.get("unit_center_spearman", float("nan"))) and \
           not np.isnan(bsp) and s["unit_center_spearman"] > bsp:
            dur_improves_sp = True

    fl = summary.get("fixed_linear", {})
    fl_ooir = fl.get("out_of_interval_ratio_mean", float("nan"))
    fl_center = fl.get("unit_time_center_error_mean", float("nan"))
    fl_sp = fl.get("unit_center_spearman", float("nan"))

    print()
    if dur_improves_ooir and dur_improves_center:
        print("  Conclusion A: Duration interval bias reduces unit_time_center_error")
        print("  and out_of_interval_ratio. It improves temporal lyric localisation.")
    elif dur_improves_sp and not dur_improves_ooir:
        print("  Conclusion B: Coverage error similar but temporal correlation improves.")
        print("  PMDC target is temporal retrieval geometry (beyond global coverage).")
    elif not np.isnan(fl_sp) and fl_sp > bs.get("unit_center_spearman", 0):
        print("  Conclusion C: Fixed_linear improves centroid_spearman but does not")
        print("  improve out_of_interval_ratio as much. Weaker unit-level localisation.")
    else:
        print("  Conclusion D: No config improves temporal localisation. Inspect heatmaps.")

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
