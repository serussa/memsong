#!/usr/bin/env python3
"""
pm_speed_probe — Stage 1: PM speed probe.

Verifies that PhaseMemory can read lyric retrieval progress speed from
ACE-Step/ACE-Step-1.5 teacher-forcing hidden dynamics.

Core claims tested:
  1. hidden / Δhidden contains lyric progress speed signal
  2. PhaseMemory can beat simple baselines at predicting this speed
  3. Shuffling Δh degrades performance (temporal tangent matters)

Usage:
    python scripts/pm_speed_probe.py \\
        --num-samples 20 \\
        --layer 12 \\
        --output-dir /root/autodl-tmp/pm_speed_probe \\
        --epochs 20 \\
        --batch-size 1 \\
        --device cuda

Smoke test:
    python scripts/pm_speed_probe.py \\
        --num-samples 5 \\
        --layer 12 \\
        --output-dir /root/autodl-tmp/pm_speed_probe_smoke \\
        --epochs 2 \\
        --batch-size 1 \\
        --device cuda

Output layout:
    {output_dir}/
        pm_speed_probe_results.json
        pm_speed_probe_results.csv
        per_sample_metrics.csv
        pm_speed_probe_ckpt.pt
        figures/
            sample_xxx/
                centroid_curve.png
                target_speed.png
                pred_speed_pm.png
                pred_speed_mlp.png
                pred_speed_gru.png
                speed_overlay.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}

PROBE_NAMES = [
    "linear_time",
    "MLP_h",
    "MLP_h_delta",
    "GRU_h_delta",
    "PM_h",
    "PM_h_delta",
    "PM_h_shuffled_delta",
]


# ===================================================================
#  PART 1 — Audio scanning & metadata matching
#  (adapted from manifold_progress_probe.py)
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


def find_matching_pt(audio_stem: str, tensor_dir: Path) -> Optional[Path]:
    """Find matching preprocessed .pt by hash prefix."""
    pt_path = tensor_dir / f"{audio_stem}.pt"
    if pt_path.is_file():
        return pt_path
    base = audio_stem.rsplit("_", 1)[0]
    for f in tensor_dir.glob(f"{base}_*.pt"):
        return f
    # Check tar archive
    if TENSOR_TAR_PATH.is_file():
        pt_name = f"{audio_stem}.pt"
        base_pt_name = f"{base}.pt"
        try:
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                for name in tar.getnames():
                    leaf = name.split("/")[-1] if "/" in name else name
                    if leaf == pt_name or leaf == base_pt_name:
                        return Path(tensor_dir) / leaf  # virtual path, we'll load from tar
        except Exception:
            pass
    return None


def load_preprocessed_data(pt_path: Path) -> Optional[dict]:
    """Load all tensors from a preprocessed .pt file.

    Supports on-demand extraction from the train_tensors.tar archive
    when the .pt file does not exist on disk.

    Returns None if the file cannot be found or loaded.
    """
    pt_name = pt_path.name

    # First try direct file load
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

    # Try extracting from tar
    if TENSOR_TAR_PATH.is_file():
        try:
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                # Try to find the file in the tar
                tar_path = None
                for name in tar.getnames():
                    leaf = name.split("/")[-1] if "/" in name else name
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


def find_metadata_files(audio_stem: str, audio_dir: Path, dataset_dir: Path) -> dict:
    """Find sidecar metadata for an audio file."""
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
    """Try metadata files in priority order, return lyrics text or None."""
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




def generate_sample_id(audio_path: Path) -> str:
    """Short unique sample ID derived from audio filename."""
    return audio_path.stem


# ===================================================================
#  PART 2 — Model & hooks
#  (adapted from manifold_progress_probe.py)
# ===================================================================

class HiddenCollector:
    """Forward hook: collect layer hidden state output."""
    def __init__(self):
        self.hidden = []

    def __call__(self, module, input, output):
        hs = output[0]  # [B, T, D]
        self.hidden.append(hs.detach().cpu())

    def get(self):
        if not self.hidden:
            return None
        return torch.cat(self.hidden, dim=0)


class CrossAttnCollector:
    """Forward hook on cross-attention module: capture attention weights."""
    def __init__(self):
        self.weights = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.weights.append(output[1].detach().cpu())

    def get(self):
        if not self.weights:
            return None
        stacked = torch.cat(self.weights, dim=0)
        return stacked.squeeze()  # [H, T, L]


def setup_model(device: str = "cuda") -> AceStepHandler:
    """Load baseline SFT model, disable all adapters."""
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

    # Disable Section-RoPE and PM everywhere
    model.config.use_section_rope_offset = False
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_section_rope", False):
            layer_mod.use_section_rope = False
        if getattr(layer_mod, "use_phase_memory", False):
            layer_mod.use_phase_memory = False

    print(f"  Model: {type(model).__name__}, hidden_size={model.config.hidden_size}")
    print(f"  All adapters disabled (clean baseline)")
    return dit_handler


def register_hooks(model, layer: int = 12) -> Tuple[HiddenCollector, CrossAttnCollector, List]:
    """Register forward hooks on specified layer's decoder layer and cross-attn."""
    hidden_collector = HiddenCollector()
    attn_collector = CrossAttnCollector()

    handles = []
    h = model.decoder.layers[layer].register_forward_hook(hidden_collector)
    handles.append(h)

    h = model.decoder.layers[layer].cross_attn.register_forward_hook(attn_collector)
    handles.append(h)

    ca_module = model.decoder.layers[layer].cross_attn
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
    """Run a single decoder forward pass with teacher-forced latents."""
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    hs = target_latents.unsqueeze(0).to(device=model_device, dtype=model_dtype)          # [1, T, 64]
    am = attention_mask.unsqueeze(0).to(device=model_device, dtype=model_dtype)           # [1, T]
    enc_hs = encoder_hidden_states.unsqueeze(0).to(device=model_device, dtype=model_dtype) # [1, L, D]
    enc_am = encoder_attention_mask.unsqueeze(0).to(device=model_device, dtype=model_dtype) # [1, L]
    ctx = context_latents.unsqueeze(0).to(device=model_device, dtype=model_dtype)          # [1, T', 128]

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
#  PART 3 — Lyric position and attention centroid
# ===================================================================

def compute_lyric_pos(section_ids: torch.Tensor) -> np.ndarray:
    """Compute normalized lyric position for each non-UNKNOWN token.

    lyric_pos[j] = position of token j among lyric tokens / (count - 1)
    lyric tokens: section_ids in [1, 7].
    Returns float array of shape [L], non-lyric tokens set to -1.
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
#  PART 4 — Target speed construction from attention centroid
# ===================================================================

def build_target_speed(
    centroid: np.ndarray,
    smooth_kernel: int = 31,
    eps: float = 1e-4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build target speed from attention centroid.

    Args:
        centroid: [T] values in [0, 1]
        smooth_kernel: moving average kernel size (odd)
        eps: small constant to avoid zero speed

    Returns:
        speed: [T] positive, mean-normalized
        log_speed: [T] log(speed)
        Both None if centroid has NaN.
    """
    if np.any(np.isnan(centroid)):
        return None, None

    T = len(centroid)

    # Smooth centroid with moving average
    if smooth_kernel > 1 and T > smooth_kernel:
        kernel = np.ones(smooth_kernel) / smooth_kernel
        centroid_smooth = np.convolve(centroid, kernel, mode="same")
    else:
        centroid_smooth = centroid.copy()

    # Delta
    delta = np.diff(centroid_smooth)  # [T-1]
    delta = np.concatenate([delta[:1], delta])  # [T]

    # Clamp negative (remove backtracking)
    delta = np.clip(delta, 0.0, None)

    # Add epsilon
    speed = delta + eps

    # Mean normalize
    speed = speed / (speed.mean() + 1e-6)

    # Clip extreme values
    speed = np.clip(speed, 0.05, 5.0)
    speed = speed / (speed.mean() + 1e-6)

    log_speed = np.log(speed)

    return speed, log_speed


# ===================================================================
#  PART 5 — Probe model definitions
# ===================================================================

class LinearTimeProbe(nn.Module):
    """Baseline: predict speed from normalized time position alone."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: [B, T, 1] or [B, T] or [T, 1] — predict from normalized time."""
        return (self.a * x + self.b).squeeze(-1)


class MLPSpeedProbe(nn.Module):
    """MLP speed probe."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRUSpeedProbe(nn.Module):
    """GRU speed probe (temporal)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y).squeeze(-1)


class PhaseMemorySpeedProbe(nn.Module):
    """Progress-only PhaseMemory — no hidden residual, no K/V modification.

    Uses complex-valued phase memory to track lyric retrieval progress
    and predict log-speed residual from the recurrent state.
    """

    def __init__(self, in_dim: int, mem_dim: int = 128, hidden_dim: int = 256, alpha: float = 1.0):
        super().__init__()
        self.mem_dim = mem_dim
        self.alpha = alpha

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        self.proj_r = nn.Linear(hidden_dim, mem_dim)
        self.proj_i = nn.Linear(hidden_dim, mem_dim)

        self.omega = nn.Linear(hidden_dim + 2 * mem_dim, mem_dim)

        self.speed_head = nn.Sequential(
            nn.LayerNorm(2 * mem_dim),
            nn.Linear(2 * mem_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Init final layer with small weights (not zero, so initial output varies)
        nn.init.normal_(self.speed_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.speed_head[-1].bias)

    def forward(self, x):
        """
        x: [B, T, in_dim]
        return: log_speed_pred: [B, T]
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        x = self.input_proj(x)

        zr = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)
        zi = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)

        outs = []
        for t in range(T):
            xt = x[:, t]

            inp_r = self.proj_r(xt)
            inp_i = self.proj_i(xt)

            omega_in = torch.cat([xt, zr, zi], dim=-1)
            omega = torch.tanh(self.omega(omega_in))

            cos = torch.cos(omega)
            sin = torch.sin(omega)

            new_zr = zr * cos - zi * sin + inp_r
            new_zi = zr * sin + zi * cos + inp_i

            zr, zi = new_zr, new_zi

            state = torch.cat([zr, zi], dim=-1)
            s = self.speed_head(state).squeeze(-1)
            s = self.alpha * torch.tanh(s)
            outs.append(s)

            # Detach state: no gradient through recurrence dynamics.
            # Gradient only flows through speed_head at each step,
            # treating the accumulated state as a fixed representation.
            zr = zr.detach()
            zi = zi.detach()

        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Reset internal recurrent state (for eval with batch_size=1)."""
        # This model doesn't persist state, so this is a no-op
        pass


# ===================================================================
#  PART 6 — Probe training utilities
# ===================================================================

def build_probe_input(
    H: torch.Tensor,
    delta_H: Optional[torch.Tensor] = None,
    probe_dim: int = 256,
) -> Tuple[torch.Tensor, int]:
    """Build probe input from hidden states.

    Args:
        H: [T, D] hidden states
        delta_H: [T, D] or None
        probe_dim: projection dimension

    Returns:
        x: [1, T, in_dim]
        in_dim: input dimension for probe
    """
    T, D = H.shape

    if delta_H is not None:
        x = torch.cat([H, delta_H], dim=-1)  # [T, 2*D]
    else:
        x = H

    return x.unsqueeze(0)  # [1, T, in_dim]


class ProbeDataset(torch.utils.data.Dataset):
    """Dataset of song-level features for speed probe training.

    Each item is one song with all its time steps.
    """

    def __init__(
        self,
        songs: List[dict],
        probe_dim: int = 256,
        smooth_kernel: int = 31,
        seed: int = 42,
    ):
        self.songs = songs
        self.probe_dim = probe_dim
        self.smooth_kernel = smooth_kernel
        self.seed = seed
        self.valid_indices = []

        # Pre-validate songs
        for idx, song in enumerate(songs):
            H = song.get("H")  # [T, D]
            c = song.get("centroid")  # [T]
            if H is None or c is None:
                continue
            if np.any(np.isnan(c)):
                continue
            speed, log_speed = build_target_speed(c, smooth_kernel=smooth_kernel)
            if speed is None:
                continue
            self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        song_idx = self.valid_indices[idx]
        song = self.songs[song_idx]

        H = song["H"]  # [T, D]
        c = song["centroid"]
        T = H.shape[0]

        speed, log_speed = build_target_speed(c, smooth_kernel=self.smooth_kernel)
        # speed, log_speed: [T]

        delta_H = np.zeros_like(H)
        delta_H[1:] = H[1:] - H[:-1]

        time_pos = np.linspace(0, 1, T, dtype=np.float32)

        return {
            "H": torch.from_numpy(H).float(),                                # [T, D]
            "delta_H": torch.from_numpy(delta_H).float(),                    # [T, D]
            "speed": torch.from_numpy(speed).float(),                         # [T]
            "log_speed": torch.from_numpy(log_speed).float(),                 # [T]
            "time_pos": torch.from_numpy(time_pos).float(),                   # [T]
            "T": T,
            "sample_id": song.get("sample_id", f"song_{song_idx}"),
        }


def collate_probe(batch: List[dict]) -> dict:
    """Collate list of song dicts into batched dicts with padding.

    Since songs have different T, we use packing (batch_first with padding).
    """
    batch_size = len(batch)
    max_T = max(item["T"] for item in batch)
    D = batch[0]["H"].shape[-1]

    device = batch[0]["H"].device if isinstance(batch[0]["H"], torch.Tensor) else "cpu"

    H_batch = torch.zeros(batch_size, max_T, D, device=device)
    delta_H_batch = torch.zeros(batch_size, max_T, D, device=device)
    speed_batch = torch.zeros(batch_size, max_T, device=device)
    log_speed_batch = torch.zeros(batch_size, max_T, device=device)
    time_batch = torch.zeros(batch_size, max_T, device=device)
    mask_batch = torch.zeros(batch_size, max_T, dtype=torch.bool, device=device)
    sample_ids = []

    for i, item in enumerate(batch):
        T = item["T"]
        H_batch[i, :T] = item["H"]
        delta_H_batch[i, :T] = item["delta_H"]
        speed_batch[i, :T] = item["speed"]
        log_speed_batch[i, :T] = item["log_speed"]
        time_batch[i, :T] = item["time_pos"]
        mask_batch[i, :T] = True
        sample_ids.append(item["sample_id"])

    return {
        "H": H_batch,
        "delta_H": delta_H_batch,
        "speed": speed_batch,
        "log_speed": log_speed_batch,
        "time_pos": time_batch,
        "mask": mask_batch,
        "sample_ids": sample_ids,
    }


def compute_metrics(
    pred_log_speed: torch.Tensor,
    target_log_speed: torch.Tensor,
    mask: torch.Tensor,
    pred_speed: Optional[torch.Tensor] = None,
    target_speed: Optional[torch.Tensor] = None,
) -> dict:
    """Compute speed prediction metrics on masked tokens.

    Args:
        pred_log_speed: [B, T]
        target_log_speed: [B, T]
        mask: [B, T] bool
        pred_speed: [B, T] optional (exp of log if not provided)
        target_speed: [B, T] optional

    Returns:
        dict of scalar metrics
    """
    p = pred_log_speed[mask]
    t = target_log_speed[mask]

    if p.numel() == 0 or t.numel() == 0:
        return {k: float("nan") for k in
                ["speed_log_mae", "speed_log_huber", "speed_pearson",
                 "speed_spearman", "speed_r2", "progress_mae", "progress_spearman"]}

    # Log-scale MAE
    log_mae = F.l1_loss(p, t).item()

    # Log-scale Huber
    log_huber = F.smooth_l1_loss(p, t).item()

    # R²
    t_mean = t.mean()
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t_mean) ** 2).sum()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10)).item()

    # Pearson correlation on log scale (squeeze to 1D for safety)
    p_np = p.detach().cpu().numpy().ravel()
    t_np = t.detach().cpu().numpy().ravel()
    pearson = float(np.corrcoef(p_np, t_np)[0, 1]) if len(p_np) > 1 else 0.0

    # Spearman
    from scipy.stats import spearmanr
    sp = spearmanr(p_np.ravel(), t_np.ravel())[0] if len(p_np) > 1 else 0.0

    # Progress MAE (if speed tensors provided)
    progress_mae = float("nan")
    progress_spearman = float("nan")
    if pred_speed is not None and target_speed is not None:
        ps = pred_speed[mask]
        ts = target_speed[mask]

        # Cumulative progress — handle 2D [B, T] or 1D [N]
        if pred_speed.dim() == 1:
            pred_speed = pred_speed.unsqueeze(0)
        if target_speed.dim() == 1:
            target_speed = target_speed.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        B, T_full = pred_speed.shape
        pred_progress = torch.zeros_like(pred_speed)
        target_progress = torch.zeros_like(target_speed)
        for b in range(B):
            row_mask = mask[b]
            if row_mask.sum() < 2:
                continue
            pred_progress[b, row_mask] = torch.cumsum(pred_speed[b, row_mask], dim=0)
            target_progress[b, row_mask] = torch.cumsum(target_speed[b, row_mask], dim=0)
            # Normalize to [0, 1]
            pred_progress[b, row_mask] = (
                (pred_progress[b, row_mask] - pred_progress[b, row_mask][0])
                / (pred_progress[b, row_mask][-1] - pred_progress[b, row_mask][0] + 1e-6)
            )
            target_progress[b, row_mask] = (
                (target_progress[b, row_mask] - target_progress[b, row_mask][0])
                / (target_progress[b, row_mask][-1] - target_progress[b, row_mask][0] + 1e-6)
            )

        pp = pred_progress[mask].detach().cpu().numpy().ravel()
        tp = target_progress[mask].detach().cpu().numpy().ravel()
        progress_mae = float(np.mean(np.abs(pp - tp)))
        progress_spearman = spearmanr(pp.ravel(), tp.ravel())[0] if len(pp) > 1 else 0.0

    return {
        "speed_log_mae": log_mae,
        "speed_log_huber": log_huber,
        "speed_pearson": pearson,
        "speed_spearman": sp,
        "speed_r2": r2,
        "progress_mae": progress_mae,
        "progress_spearman": progress_spearman,
    }


def normalize_features(
    H: torch.Tensor,
    delta_H: torch.Tensor,
    probe_dim: int = 256,
    H_dim: int = 2048,
    proj: Optional[nn.Module] = None,
    delta_proj: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, nn.Module, nn.Module]:
    """Layer-norm + project features to probe_dim.

    Args:
        H: [B, T, D] or [T, D]
        delta_H: [B, T, D] or [T, D]
        proj, delta_proj: optional existing projection layers

    Returns:
        H_proj: [B, T, probe_dim]
        delta_H_proj: [B, T, probe_dim]
        proj, delta_proj
    """
    if H.dim() == 2:
        H = H.unsqueeze(0)
    if delta_H.dim() == 2:
        delta_H = delta_H.unsqueeze(0)

    B, T, D = H.shape

    if proj is None:
        proj = nn.Linear(D, probe_dim)
        # Use orthogonal init for stable projection
        nn.init.orthogonal_(proj.weight)
        nn.init.zeros_(proj.bias)
    if delta_proj is None:
        delta_proj = nn.Linear(D, probe_dim)
        nn.init.orthogonal_(delta_proj.weight)
        nn.init.zeros_(delta_proj.bias)

    # LayerNorm + project
    H_proj = proj(H)  # [B, T, probe_dim]
    delta_H_proj = delta_proj(delta_H)

    return H_proj, delta_H_proj, proj, delta_proj


# ===================================================================
#  PART 7 — Training loop
# ===================================================================

def train_probe(
    model: nn.Module,
    name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    input_keys: Tuple[str, ...],
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    use_shuffled_delta: bool = False,
    seed: int = 42,
) -> Tuple[nn.Module, dict]:
    """Train a single probe model.

    Args:
        model: probe model instance
        name: probe name for logging
        train_loader: DataLoader yielding dicts with keys
        val_loader: same structure for validation
        input_keys: which keys to use as input (e.g., ("H",) or ("H", "delta_H"))
        epochs: training epochs
        lr: learning rate
        device: torch device
        use_shuffled_delta: if True, shuffle delta_H along time axis (per sample)
        seed: random seed for shuffling

    Returns:
        model: trained probe
        val_metrics: best validation metrics
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_metrics = {}
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            mask = batch["mask"].to(device)
            target = batch["log_speed"].to(device)

            # Build input from specified keys
            if use_shuffled_delta and "delta_H" in input_keys:
                # Shuffle delta_H along time dim for each sample in batch
                delta_H = batch["delta_H"].clone()
                B, T, D = delta_H.shape
                for b in range(B):
                    t_valid = mask[b].sum().int().item()
                    if t_valid > 3:
                        idx_shuf = list(range(1, t_valid))
                        random.Random(seed + epoch + b).shuffle(idx_shuf)
                        # Keep delta_H[0] = 0, shuffle rest
                        perm = [0] + idx_shuf
                        delta_H[b, :t_valid] = delta_H[b, perm]
                input_tensors = []
                for k in input_keys:
                    if k == "delta_H":
                        input_tensors.append(delta_H.to(device))
                    else:
                        input_tensors.append(batch[k].to(device))
            else:
                input_tensors = [batch[k].to(device) for k in input_keys]

            # Concatenate along feature dim
            if len(input_tensors) == 1:
                x = input_tensors[0]
            else:
                x = torch.cat(input_tensors, dim=-1)

            pred = model(x)  # [B, T]

            loss = F.smooth_l1_loss(pred[mask], target[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        val_masks = []

        with torch.no_grad():
            for batch in val_loader:
                mask = batch["mask"].to(device)
                target = batch["log_speed"].to(device)

                input_tensors = [batch[k].to(device) for k in input_keys]
                if len(input_tensors) == 1:
                    x = input_tensors[0]
                else:
                    x = torch.cat(input_tensors, dim=-1)

                # For shuffled delta, we shuffle at val time too (same logic)
                if use_shuffled_delta and "delta_H" in input_keys:
                    delta_H = batch["delta_H"].clone()
                    B_v, T_v, D_v = delta_H.shape
                    for b in range(B_v):
                        t_valid = mask[b].sum().int().item()
                        if t_valid > 3:
                            idx_shuf = list(range(1, t_valid))
                            random.Random(seed + epoch + 999 + b).shuffle(idx_shuf)
                            perm = [0] + idx_shuf
                            delta_H[b, :t_valid] = delta_H[b, perm]
                    # Rebuild x with shuffled delta
                    input_tensors_shuf = []
                    for k in input_keys:
                        if k == "delta_H":
                            input_tensors_shuf.append(delta_H.to(device))
                        else:
                            input_tensors_shuf.append(batch[k].to(device))
                    if len(input_tensors_shuf) == 1:
                        x = input_tensors_shuf[0]
                    else:
                        x = torch.cat(input_tensors_shuf, dim=-1)

                pred = model(x)
                loss = F.smooth_l1_loss(pred[mask], target[mask])
                val_losses.append(loss.item())
                val_preds.append(pred.cpu())
                val_targets.append(target.cpu())
                val_masks.append(mask.cpu())

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  train_loss={np.mean(train_losses):.6f}  val_loss={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Compute full val metrics at best epoch
    model.eval()
    all_pred = []
    all_target = []
    all_mask = []
    with torch.no_grad():
        for batch in val_loader:
            mask = batch["mask"].to(device)
            target = batch["log_speed"].to(device)
            input_tensors = [batch[k].to(device) for k in input_keys]

            if use_shuffled_delta and "delta_H" in input_keys:
                delta_H = batch["delta_H"].clone()
                B_v, T_v, D_v = delta_H.shape
                for b in range(B_v):
                    t_valid = mask[b].sum().int().item()
                    if t_valid > 3:
                        idx_shuf = list(range(1, t_valid))
                        random.Random(seed + 9999).shuffle(idx_shuf)
                        perm = [0] + idx_shuf
                        delta_H[b, :t_valid] = delta_H[b, perm]
                input_tensors_shuf = []
                for k in input_keys:
                    if k == "delta_H":
                        input_tensors_shuf.append(delta_H.to(device))
                    else:
                        input_tensors_shuf.append(batch[k].to(device))
                if len(input_tensors_shuf) == 1:
                    x = input_tensors_shuf[0]
                else:
                    x = torch.cat(input_tensors_shuf, dim=-1)
            else:
                if len(input_tensors) == 1:
                    x = input_tensors[0]
                else:
                    x = torch.cat(input_tensors, dim=-1)

            pred = model(x)
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_mask.append(mask.cpu())

    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    mask_cat = torch.cat(all_mask, dim=0)

    pred_speed = torch.exp(pred_cat)
    target_speed = torch.exp(target_cat)

    metrics = compute_metrics(pred_cat, target_cat, mask_cat, pred_speed, target_speed)
    metrics["val_loss"] = best_val_loss

    return model, metrics


# ===================================================================
#  PART 8 — Visualization
# ===================================================================

def visualize_sample(
    sample_id: str,
    centroid: np.ndarray,
    target_speed: np.ndarray,
    target_log_speed: np.ndarray,
    predictions: Dict[str, np.ndarray],
    T_eff: int,
    duration: float,
    L_eff: int,
    output_dir: Path,
):
    """Generate diagnostic plots per sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures" / sample_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(T_eff) / 50.0  # seconds (50Hz internal rate)
    norm_time = np.arange(T_eff) / T_eff

    # 1. Centroid curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(norm_time, centroid, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Attention centroid (lyric pos)")
    ax.set_title(f"Attention Centroid — {sample_id}")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "centroid_curve.png", dpi=150)
    plt.close(fig)

    # 2. Target speed
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(norm_time, target_speed, color="forestgreen", linewidth=1.2)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Target speed (normalized)")
    ax.set_title(f"Target Speed — {sample_id}  (T={T_eff}, L={L_eff}, dur={duration:.1f}s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "target_speed.png", dpi=150)
    plt.close(fig)

    # 3-5. Individual prediction plots
    model_colors = {
        "linear_time": "gray",
        "MLP_h": "coral",
        "MLP_h_delta": "tomato",
        "GRU_h_delta": "darkorange",
        "PM_h": "mediumblue",
        "PM_h_delta": "darkviolet",
        "PM_h_shuffled_delta": "lightcoral",
    }

    for name, color in model_colors.items():
        if name not in predictions:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(norm_time, target_speed, color="forestgreen", linewidth=0.8,
                alpha=0.6, label="Target")
        ax.plot(norm_time, predictions[name], color=color, linewidth=1.2,
                label=name)
        ax.set_xlabel("Normalized audio time")
        ax.set_ylabel("Speed")
        ax.set_title(f"Predicted Speed ({name}) — {sample_id}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"pred_speed_{name}.png", dpi=150)
        plt.close(fig)

    # 6. Speed overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(norm_time, target_speed, color="forestgreen", linewidth=1.5,
            alpha=0.8, label="Target")
    overlay_models = ["MLP_h_delta", "GRU_h_delta", "PM_h_delta", "PM_h_shuffled_delta"]
    for name in overlay_models:
        if name in predictions:
            ax.plot(norm_time, predictions[name],
                    color=model_colors.get(name, "gray"), linewidth=1.0,
                    linestyle="--", alpha=0.7, label=name)
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Speed")
    ax.set_title(f"Speed Overlay — {sample_id}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "speed_overlay.png", dpi=150)
    plt.close(fig)

    # 7. Progress overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    # Target progress
    target_progress = np.cumsum(target_speed)
    target_progress = (target_progress - target_progress[0]) / (target_progress[-1] - target_progress[0] + 1e-6)
    ax.plot(norm_time, target_progress, color="forestgreen", linewidth=1.5,
            alpha=0.8, label="Target progress")
    # Linear progress baseline
    ax.plot(norm_time, norm_time, color="gray", linewidth=1.0,
            linestyle=":", alpha=0.6, label="Linear progress")
    # Model predictions
    for name in overlay_models + ["linear_time"]:
        if name not in predictions:
            continue
        pred_progress = np.cumsum(predictions[name])
        pred_progress = (pred_progress - pred_progress[0]) / (pred_progress[-1] - pred_progress[0] + 1e-6)
        ax.plot(norm_time, pred_progress,
                color=model_colors.get(name, "gray"), linewidth=1.0,
                linestyle="--", alpha=0.7, label=f"{name} progress")
    ax.set_xlabel("Normalized audio time")
    ax.set_ylabel("Progress")
    ax.set_title(f"Progress Overlay — {sample_id}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "progress_overlay.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved -> {fig_dir}")


# ===================================================================
#  PART 9 — Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="PM Speed Probe — Stage 1")

    # Data
    parser.add_argument("--audio-dir", type=str, default="/root/autodl-tmp/musicdata/audios")
    parser.add_argument("--tensor-dir", type=str, default=str(TENSOR_DIR_DEFAULT))
    parser.add_argument("--dataset-dir", type=str, default="/root/autodl-tmp/musicdata/dataset")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of songs to sample")
    parser.add_argument("--min-duration", type=float, default=180, help="Min duration in seconds")
    parser.add_argument("--max-duration", type=float, default=300, help="Max duration in seconds")

    # Model
    parser.add_argument("--layer", type=int, default=12, help="Cross-attention layer to hook")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Probe training
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per probe")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (songs)")
    parser.add_argument("--probe-dim", type=int, default=256, help="Projection dimension")
    parser.add_argument("--mem-dim", type=int, default=128, help="PhaseMemory mem_dim")
    parser.add_argument("--smooth-kernel", type=int, default=31, help="Centroid smoothing kernel")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test fraction")

    # Output
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/pm_speed_probe")
    parser.add_argument("--seed", type=int, default=42)

    # Downsampling (for long sequences)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Downsample to at most this many tokens per sample")
    parser.add_argument("--pm-truncate", type=int, default=64,
                        help="Truncate to this many timesteps for PM probe training (default: 64)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir)
    tensor_dir = Path(args.tensor_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    print("=" * 70)
    print("PM SPEED PROBE — Stage 1")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Scan audio files
    # ------------------------------------------------------------------
    print("\n[1/8] Scanning audio files...")
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
    print(f"  Selected {len(selected)} samples randomly")

    # ------------------------------------------------------------------
    # Step 2: Find lyrics
    # ------------------------------------------------------------------
    print("\n[2/8] Finding lyrics and metadata...")

    samples = []
    skipped_no_lyrics = 0
    skipped_reasons = []

    for f, dur in selected:
        sid = generate_sample_id(f)
        meta_files = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta_files)

        if lyrics is None:
            print(f"  Skip {f.name}: no lyrics")
            skipped_no_lyrics += 1
            skipped_reasons.append((sid, "no_lyrics"))
            continue

        pt_path = find_matching_pt(f.stem, tensor_dir)

        samples.append({
            "sample_id": sid,
            "audio_path": str(f),
            "duration": dur,
            "lyrics": lyrics,
            "pt_path": str(pt_path) if pt_path else None,
        })

    print(f"  Samples with lyrics: {len(samples)} (skipped {skipped_no_lyrics})")

    # ------------------------------------------------------------------
    # Step 3: Load model
    # ------------------------------------------------------------------
    print("\n[3/8] Loading baseline model (all adapters off)...")
    dit_handler = setup_model(device=args.device)
    model = dit_handler.model
    hidden_size = model.config.hidden_size  # 2048
    D = hidden_size

    # ------------------------------------------------------------------
    # Step 4: Teacher-forcing forward on each sample
    # ------------------------------------------------------------------
    print(f"\n[4/8] Running teacher-forcing forward on {len(samples)} samples...")

    song_data = []  # list of dicts with H, delta_H, centroid, speed, etc.

    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] {sample['sample_id']}")
        print(f"    Duration: {sample['duration']:.1f}s")

        sid = sample["sample_id"]
        lyrics = sample["lyrics"]
        pt_path_str = sample.get("pt_path")

        if pt_path_str is None:
            print(f"    Skip: no preprocessed .pt path")
            skipped_reasons.append((sid, "no_pt"))
            continue

        try:
            pt_data = load_preprocessed_data(Path(pt_path_str))
            if pt_data is None:
                print(f"    Skip: cannot load preprocessed .pt (not on disk or in tar)")
                skipped_reasons.append((sid, "no_pt"))
                continue
            target_latents = pt_data["target_latents"]
            attention_mask = pt_data["attention_mask"]
            encoder_hidden_states = pt_data["encoder_hidden_states"]
            encoder_attention_mask = pt_data["encoder_attention_mask"]
            context_latents = pt_data["context_latents"]

            T_raw = target_latents.shape[0]
            L_raw = encoder_hidden_states.shape[0]
            print(f"    Raw T={T_raw}, L={L_raw}")

            # Teacher-forcing forward
            hidden_collector, attn_collector, handles = register_hooks(model, layer=args.layer)
            try:
                run_teacher_forward(
                    model, target_latents, attention_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    context_latents, device=args.device,
                )
            except Exception as e:
                remove_hooks(handles)
                print(f"    Forward error: {e}")
                skipped_reasons.append((sid, "forward_error"))
                continue

            H_tensor = hidden_collector.get()
            A_tensor = attn_collector.get()
            remove_hooks(handles)

            if H_tensor is None or A_tensor is None:
                print(f"    No hook data")
                skipped_reasons.append((sid, "no_hooks"))
                continue

            H_np = H_tensor.squeeze(0).float().numpy()  # [T_eff, D]
            A_np = A_tensor.float().numpy()              # [H, T_eff, L_eff] or [T_eff, L_eff]
            if A_np.ndim == 2:
                A_np = A_np[np.newaxis, :, :]  # [1, T, L]
            T_eff, L_eff = H_np.shape[0], A_np.shape[2]

            print(f"    T_eff={T_eff}, L_eff={L_eff}, H={H_np.shape}, A={A_np.shape}")

            # Parse section_ids
            parser = LyricsStructureParser()
            parsed = parser.parse(lyrics, num_chunks=L_eff)
            section_ids = parsed.section_type_ids

            # Lyric position
            lyric_pos = compute_lyric_pos(section_ids)

            # Attention centroid
            c_h = []
            for h in range(A_np.shape[0]):
                centroid_h = compute_attention_centroid(
                    A_np[np.newaxis, h], lyric_pos
                ).squeeze(0)
                c_h.append(centroid_h)
            c = np.mean(c_h, axis=0)  # [T_eff]
            print(f"    Centroid range: [{c.min():.4f}, {c.max():.4f}]")

            # Check for NaN
            if np.any(np.isnan(c)):
                print(f"    Skip: NaN centroid")
                skipped_reasons.append((sid, "nan_centroid"))
                continue

            # Build target speed
            speed, log_speed = build_target_speed(c, smooth_kernel=args.smooth_kernel)
            if speed is None:
                print(f"    Skip: speed construction failed")
                skipped_reasons.append((sid, "speed_failed"))
                continue

            # Check speed variance
            if np.var(log_speed) < 0.001:
                print(f"    Skip: log_speed variance too low ({np.var(log_speed):.6f})")
                skipped_reasons.append((sid, "low_speed_variance"))
                continue

            print(f"    Speed stats: mean={speed.mean():.3f}, var(log)={np.var(log_speed):.4f}")

            # Delta H
            delta_H = np.zeros_like(H_np)
            delta_H[1:] = H_np[1:] - H_np[:-1]

            song_data.append({
                "sample_id": sid,
                "duration": sample["duration"],
                "T": T_eff,
                "L": L_eff,
                "H": H_np,
                "delta_H": delta_H,
                "centroid": c,
                "speed": speed,
                "log_speed": log_speed,
                "lyric_pos": lyric_pos,
            })

        except Exception as e:
            print(f"    Error: {e}")
            traceback.print_exc()
            skipped_reasons.append((sid, "exception"))
            continue

    print(f"\n  Successfully processed: {len(song_data)} songs")
    print(f"  Skipped: {len(skipped_reasons)} songs")
    for sid, reason in skipped_reasons:
        print(f"    {sid}: {reason}")

    if len(song_data) < 5:
        print(f"\n  Too few samples ({len(song_data)}) for probe training. Need at least 5.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Downsample if needed
    # ------------------------------------------------------------------
    if args.max_tokens is not None and args.max_tokens > 0:
        max_T = args.max_tokens
        for song in song_data:
            T = song["T"]
            if T > max_T:
                # Uniform downsample to max_T
                indices = np.linspace(0, T - 1, max_T, dtype=int)
                song["H"] = song["H"][indices]
                song["delta_H"] = song["delta_H"][indices]
                song["centroid"] = song["centroid"][indices]
                # Recompute speed after downsampling
                speed, log_speed = build_target_speed(
                    song["centroid"], smooth_kernel=min(args.smooth_kernel, max_T // 4)
                )
                if speed is not None:
                    song["speed"] = speed
                    song["log_speed"] = log_speed
                song["T"] = max_T
                print(f"    Downsampled {song['sample_id']}: T={T} -> {max_T}")

    # ------------------------------------------------------------------
    # Step 5b: Song-level split
    # ------------------------------------------------------------------
    random.Random(args.seed).shuffle(song_data)
    n_total = len(song_data)
    n_test = max(1, int(n_total * args.test_split))
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_test - n_val

    test_songs = song_data[:n_test]
    val_songs = song_data[n_test:n_test + n_val]
    train_songs = song_data[n_test + n_val:]

    print(f"\n  Song-level split: {n_train} train / {n_val} val / {n_test} test")

    if n_train < 1:
        print("  No valid training samples. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Build input feature projections (D -> probe_dim)
    # ------------------------------------------------------------------
    print(f"\n[5/8] Building feature projections (D={D} -> probe_dim={args.probe_dim})...", flush=True)

    proj_H = nn.Linear(D, args.probe_dim)
    nn.init.orthogonal_(proj_H.weight)
    nn.init.zeros_(proj_H.bias)
    proj_dH = nn.Linear(D, args.probe_dim)
    nn.init.orthogonal_(proj_dH.weight)
    nn.init.zeros_(proj_dH.bias)

    H_proj_all: Dict[str, Dict[str, torch.Tensor]] = {}
    for song_list in (train_songs, val_songs, test_songs):
        for song in song_list:
            sid = song["sample_id"]
            H_t = torch.from_numpy(song["H"].copy()).float()
            dH_t = torch.from_numpy(song["delta_H"].copy()).float()
            H_proj_all[sid] = {
                "H_proj": proj_H(H_t).detach(),
                "dH_proj": proj_dH(dH_t).detach(),
            }

    def get_probe_input(sid, use_keys):
        """Get probe input tensors for a given sample."""
        p = H_proj_all[sid]
        tensors = []
        for k in use_keys:
            if k == "H":
                tensors.append(p["H_proj"])
            elif k == "delta_H":
                tensors.append(p["dH_proj"])
        return torch.cat(tensors, dim=-1)

    # Build pre-featurized datasets for faster training
    def make_featurized_dataset(song_list, use_keys, shuffle_delta=False,
                                 truncate: int = 0):
        """Create a simple list-based dataset with pre-computed features.

        Args:
            truncate: If > 0, randomly truncate each song to at most this
                      many timesteps (for PM probe gradient stability).
        """
        data = []
        for song in song_list:
            sid = song["sample_id"]
            if sid not in H_proj_all:
                continue
            x = get_probe_input(sid, use_keys)  # [T, in_dim]
            T = song["T"]
            y = torch.from_numpy(song["log_speed"]).float()  # [T]
            s = torch.from_numpy(song["speed"]).float()      # [T]
            t_pos = torch.linspace(0, 1, T)

            # Optional truncation to shorter subsequence
            if truncate > 0 and T > truncate:
                start = random.randint(0, T - truncate)
                x = x[start:start + truncate]
                y = y[start:start + truncate]
                s = s[start:start + truncate]
                t_pos = t_pos[start:start + truncate]
                T = truncate
            mask = torch.ones(T, dtype=torch.bool)
            data.append({
                "x": x, "log_speed": y, "speed": s,
                "mask": mask, "time_pos": t_pos,
                "sample_id": sid, "T": T,
            })
        return data

    probe_designs = [
        ("linear_time", {}, LinearTimeProbe(), ("time_pos",)),
        ("MLP_h", {"in_dim": args.probe_dim},
         MLPSpeedProbe(in_dim=args.probe_dim, hidden_dim=args.probe_dim), ("H",)),
        ("MLP_h_delta", {"in_dim": 2 * args.probe_dim},
         MLPSpeedProbe(in_dim=2 * args.probe_dim, hidden_dim=args.probe_dim), ("H", "delta_H")),
        ("GRU_h_delta", {"in_dim": 2 * args.probe_dim},
         GRUSpeedProbe(in_dim=2 * args.probe_dim, hidden_dim=args.probe_dim), ("H", "delta_H")),
        ("PM_h", {"in_dim": args.probe_dim, "mem_dim": args.mem_dim},
         PhaseMemorySpeedProbe(in_dim=args.probe_dim, mem_dim=args.mem_dim, hidden_dim=args.probe_dim),
         ("H",)),
        ("PM_h_delta", {"in_dim": 2 * args.probe_dim, "mem_dim": args.mem_dim},
         PhaseMemorySpeedProbe(in_dim=2 * args.probe_dim, mem_dim=args.mem_dim, hidden_dim=args.probe_dim),
         ("H", "delta_H")),
        ("PM_h_shuffled_delta", {"in_dim": 2 * args.probe_dim, "mem_dim": args.mem_dim},
         PhaseMemorySpeedProbe(in_dim=2 * args.probe_dim, mem_dim=args.mem_dim, hidden_dim=args.probe_dim),
         ("H", "delta_H")),
    ]

    # ------------------------------------------------------------------
    # Step 6: Train probes
    # ------------------------------------------------------------------
    print(f"\n[6/8] Training {len(probe_designs)} probes ({args.epochs} epochs each)...")

    all_metrics = {}
    trained_models = {}

    for name, kwargs, probe_model, input_keys in probe_designs:
        print(f"\n  --- Training {name} ---")

        if name == "linear_time":
            # Special: use time_pos as input
            train_data = []
            for song in train_songs:
                T = song["T"]
                train_data.append({
                    "x": torch.linspace(0, 1, T).unsqueeze(-1),
                    "log_speed": torch.from_numpy(song["log_speed"]).float(),
                    "speed": torch.from_numpy(song["speed"]).float(),
                    "mask": torch.ones(T, dtype=torch.bool),
                    "time_pos": torch.linspace(0, 1, T),
                    "sample_id": song["sample_id"],
                    "T": T,
                })
            val_data = []
            for song in val_songs:
                T = song["T"]
                val_data.append({
                    "x": torch.linspace(0, 1, T).unsqueeze(-1),
                    "log_speed": torch.from_numpy(song["log_speed"]).float(),
                    "speed": torch.from_numpy(song["speed"]).float(),
                    "mask": torch.ones(T, dtype=torch.bool),
                    "time_pos": torch.linspace(0, 1, T),
                    "sample_id": song["sample_id"],
                    "T": T,
                })
            test_data = []
            for song in test_songs:
                T = song["T"]
                test_data.append({
                    "x": torch.linspace(0, 1, T).unsqueeze(-1),
                    "log_speed": torch.from_numpy(song["log_speed"]).float(),
                    "speed": torch.from_numpy(song["speed"]).float(),
                    "mask": torch.ones(T, dtype=torch.bool),
                    "time_pos": torch.linspace(0, 1, T),
                    "sample_id": song["sample_id"],
                    "T": T,
                })
        elif name == "PM_h_shuffled_delta":
            train_data = make_featurized_dataset(train_songs, ("H", "delta_H"),
                                                  truncate=args.pm_truncate)
            val_data = make_featurized_dataset(val_songs, ("H", "delta_H"),
                                                truncate=args.pm_truncate)
            test_data = make_featurized_dataset(test_songs, ("H", "delta_H"),
                                                 truncate=args.pm_truncate)
        elif name.startswith("PM_"):
            train_data = make_featurized_dataset(train_songs, input_keys,
                                                  truncate=args.pm_truncate)
            val_data = make_featurized_dataset(val_songs, input_keys,
                                                truncate=args.pm_truncate)
            test_data = make_featurized_dataset(test_songs, input_keys,
                                                 truncate=args.pm_truncate)
        else:
            train_data = make_featurized_dataset(train_songs, input_keys)
            val_data = make_featurized_dataset(val_songs, input_keys)
            test_data = make_featurized_dataset(test_songs, input_keys)

        if len(train_data) < 1:
            print(f"    No training data, skipping")
            continue

        # Simple batch collation for training (all samples concatenated along time)
        def _collate(data_list):
            xs, ys, ss, masks, tpos = [], [], [], [], []
            for d in data_list:
                xs.append(d["x"])
                ys.append(d["log_speed"])
                ss.append(d["speed"])
                masks.append(d["mask"])
                tpos.append(d["time_pos"])
            return {
                "x": torch.stack(xs) if xs else torch.empty(0),
                "log_speed": torch.stack(ys) if ys else torch.empty(0),
                "speed": torch.stack(ss) if ss else torch.empty(0),
                "mask": torch.stack(masks) if masks else torch.empty(0),
                "time_pos": torch.stack(tpos) if tpos else torch.empty(0),
            }

        tr_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, collate_fn=_collate,
        )
        vl_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
        )
        te_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, collate_fn=_collate,
        )

        use_shuf = (name == "PM_h_shuffled_delta")

        model_out, val_metrics = train_probe(
            probe_model, name, tr_loader, vl_loader,
            ("x",), epochs=args.epochs, lr=args.lr,
            device=args.device, use_shuffled_delta=use_shuf,
            seed=args.seed,
        )

        trained_models[name] = model_out
        all_metrics[name] = val_metrics

        print(f"    Val metrics: R²={val_metrics['speed_r2']:.4f}, "
              f"Spearman={val_metrics['speed_spearman']:.4f}, "
              f"log_MAE={val_metrics['speed_log_mae']:.4f}")

    # ------------------------------------------------------------------
    # Step 7: Evaluate on test set, compute predictions, visualize
    # ------------------------------------------------------------------
    print(f"\n[7/8] Evaluating on test set and visualizing...")

    test_metrics = {}
    test_predictions = {}  # sample_id -> {model_name -> pred_speed}

    for name, kwargs, probe_model, input_keys in probe_designs:
        if name not in trained_models:
            test_metrics[name] = {k: float("nan") for k in
                                  ["speed_log_mae", "speed_log_huber", "speed_pearson",
                                   "speed_spearman", "speed_r2", "progress_mae",
                                   "progress_spearman"]}
            continue

        model_out = trained_models[name]
        model_out.eval()

        if name == "linear_time":
            test_data = []
            for song in test_songs:
                T = song["T"]
                test_data.append({
                    "x": torch.linspace(0, 1, T).unsqueeze(-1),
                    "log_speed": torch.from_numpy(song["log_speed"]).float(),
                    "speed": torch.from_numpy(song["speed"]).float(),
                    "mask": torch.ones(T, dtype=torch.bool),
                    "sample_id": song["sample_id"],
                    "T": T,
                })
        elif name == "PM_h_shuffled_delta":
            test_data = make_featurized_dataset(test_songs, ("H", "delta_H"),
                                                 truncate=args.pm_truncate)
        elif name.startswith("PM_"):
            test_data = make_featurized_dataset(test_songs, input_keys,
                                                 truncate=args.pm_truncate)
        else:
            test_data = make_featurized_dataset(test_songs, input_keys)

        def _collate_test(data_list):
            xs, ys, ss, masks = [], [], [], []
            sids = []
            for d in data_list:
                xs.append(d["x"])
                ys.append(d["log_speed"])
                ss.append(d["speed"])
                masks.append(d["mask"])
                sids.append(d["sample_id"])
            return {
                "x": torch.stack(xs) if xs else torch.empty(0),
                "log_speed": torch.stack(ys) if ys else torch.empty(0),
                "speed": torch.stack(ss) if ss else torch.empty(0),
                "mask": torch.stack(masks) if masks else torch.empty(0),
                "sample_ids": sids,
            }

        te_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, collate_fn=_collate_test,
        )

        with torch.no_grad():
            all_pred = []
            all_target = []
            all_mask = []
            all_speed = []
            for batch in te_loader:
                x = batch["x"].to(args.device)
                mask = batch["mask"].to(args.device)
                target = batch["log_speed"].to(args.device)

                if use_shuf:
                    # Shuffle delta along time
                    B_t, T_t, D_t = x.shape
                    for b in range(B_t):
                        t_valid = mask[b].sum().int().item()
                        if t_valid > 3:
                            idx_shuf = list(range(1, t_valid))
                            random.Random(args.seed + 9999).shuffle(idx_shuf)
                            perm = [0] + idx_shuf
                            x[b, :t_valid] = x[b, perm]

                pred = model_out(x)
                all_pred.append(pred.cpu())
                all_target.append(target.cpu())
                all_mask.append(mask.cpu())
                all_speed.append(batch["speed"].cpu())

        pred_cat = torch.cat(all_pred, dim=0)
        target_cat = torch.cat(all_target, dim=0)
        mask_cat = torch.cat(all_mask, dim=0)
        speed_cat = torch.cat(all_speed, dim=0)

        pred_speed = torch.exp(pred_cat)
        target_speed = speed_cat

        metrics = compute_metrics(pred_cat, target_cat, mask_cat, pred_speed, target_speed)
        test_metrics[name] = metrics
        print(f"  {name:30s}  R²={metrics['speed_r2']:.4f}  "
              f"Spearman={metrics['speed_spearman']:.4f}  "
              f"log_MAE={metrics['speed_log_mae']:.4f}  "
              f"progress_MAE={metrics.get('progress_mae', float('nan')):.4f}")

        # Collect per-sample predictions for visualization
        model_out.eval()
        with torch.no_grad():
            for batch in te_loader:
                sids = batch["sample_ids"]
                x = batch["x"].to(args.device)
                mask = batch["mask"].to(args.device)
                log_pred = model_out(x)
                speed_pred = torch.exp(log_pred).cpu().numpy()
            for i, sid in enumerate(sids):
                if sid not in test_predictions:
                    test_predictions[sid] = {}
                test_predictions[sid][name] = speed_pred[i, mask[i].cpu().numpy()]

    # ------------------------------------------------------------------
    # Generate test set visualizations
    # ------------------------------------------------------------------
    for song in test_songs:
        sid = song["sample_id"]
        if sid not in test_predictions:
            continue
        # Ensure T matches
        T = song["T"]
        speed_preds = {}
        for model_name in PROBE_NAMES:
            if model_name in test_predictions.get(sid, {}):
                pred_arr = test_predictions[sid][model_name]
                if len(pred_arr) == T:
                    speed_preds[model_name] = pred_arr
                else:
                    print(f"    Warning: {sid} {model_name} pred len {len(pred_arr)} != T {T}")

        # Also add target speed
        target_speed = song["speed"]

        # Find matching test song for centroid
        centroid = song["centroid"]

        visualize_sample(
            sid,
            centroid=centroid,
            target_speed=target_speed,
            target_log_speed=song["log_speed"],
            predictions=speed_preds,
            T_eff=T,
            duration=song["duration"],
            L_eff=song.get("L", 0),
            output_dir=OUTPUT_DIR,
        )

    # ------------------------------------------------------------------
    # Step 8: Save results and print conclusions
    # ------------------------------------------------------------------
    print(f"\n[8/8] Saving results and printing conclusions...")

    # Save probe checkpoint
    ckpt = {}
    for name in PROBE_NAMES:
        if name in trained_models:
            ckpt[name] = trained_models[name].state_dict()
    torch.save(ckpt, OUTPUT_DIR / "pm_speed_probe_ckpt.pt")
    print(f"  Checkpoint -> {OUTPUT_DIR / 'pm_speed_probe_ckpt.pt'}")

    # Save results JSON
    results = {"test_metrics": {}, "config": vars(args)}
    for name in PROBE_NAMES:
        m = test_metrics.get(name, {})
        results["test_metrics"][name] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in m.items()
        }

    with open(OUTPUT_DIR / "pm_speed_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = OUTPUT_DIR / "pm_speed_probe_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["model"] + list(next(iter(test_metrics.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in PROBE_NAMES:
            row = {"model": name, **test_metrics.get(name, {})}
            writer.writerow(row)
    print(f"  CSV -> {csv_path}")

    # Per-sample metrics
    per_sample_path = OUTPUT_DIR / "per_sample_metrics.csv"
    with open(per_sample_path, "w", newline="") as f:
        fieldnames = ["sample_id", "T", "L", "duration", "centroid_mean", "centroid_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for song in song_data:
            writer.writerow({
                "sample_id": song["sample_id"],
                "T": song["T"],
                "L": song.get("L", 0),
                "duration": song.get("duration", 0),
                "centroid_mean": float(np.mean(song["centroid"])),
                "centroid_std": float(np.std(song["centroid"])),
            })
    print(f"  Per-sample metrics -> {per_sample_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST SET RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'Model':30s} {'R²':>8s} {'Spearman':>10s} {'log_MAE':>10s} {'progress_MAE':>12s}"
    print(header)
    print("-" * 90)
    for name in PROBE_NAMES:
        m = test_metrics.get(name, {})
        r2 = m.get("speed_r2", float("nan"))
        sp = m.get("speed_spearman", float("nan"))
        lmae = m.get("speed_log_mae", float("nan"))
        pmae = m.get("progress_mae", float("nan"))
        print(f"{name:30s} {r2:>8.4f} {sp:>10.4f} {lmae:>10.4f} {pmae:>12.4f}")

    # ------------------------------------------------------------------
    # Auto-conclusions
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("AUTO-CONCLUSIONS")
    print("=" * 90)

    pm_h = test_metrics.get("PM_h", {}).get("speed_r2", float("nan"))
    pm_hd = test_metrics.get("PM_h_delta", {}).get("speed_r2", float("nan"))
    pm_shuf = test_metrics.get("PM_h_shuffled_delta", {}).get("speed_r2", float("nan"))
    mlp_h = test_metrics.get("MLP_h", {}).get("speed_r2", float("nan"))
    mlp_hd = test_metrics.get("MLP_h_delta", {}).get("speed_r2", float("nan"))
    gru_hd = test_metrics.get("GRU_h_delta", {}).get("speed_r2", float("nan"))
    linear = test_metrics.get("linear_time", {}).get("speed_r2", float("nan"))

    cond1 = pm_hd > pm_shuf if not (np.isnan(pm_hd) or np.isnan(pm_shuf)) else False
    cond2 = pm_hd > pm_h if not (np.isnan(pm_hd) or np.isnan(pm_h)) else False
    cond3 = mlp_hd > mlp_h if not (np.isnan(mlp_hd) or np.isnan(mlp_h)) else False
    cond4 = pm_hd >= gru_hd if not (np.isnan(pm_hd) or np.isnan(gru_hd)) else False
    cond5 = pm_hd >= mlp_hd if not (np.isnan(pm_hd) or np.isnan(mlp_hd)) else False

    print(f"\n  Minimal success criteria:")
    print(f"    PM_h_delta > PM_h_shuffled_delta: {cond1}  ({pm_hd:.4f} vs {pm_shuf:.4f})")
    print(f"    PM_h_delta > PM_h:               {cond2}  ({pm_hd:.4f} vs {pm_h:.4f})")
    print(f"    MLP_h_delta > MLP_h:             {cond3}  ({mlp_hd:.4f} vs {mlp_h:.4f})")

    print(f"\n  Strong success criteria:")
    print(f"    PM_h_delta >= GRU_h_delta: {cond4}  ({pm_hd:.4f} vs {gru_hd:.4f})")
    print(f"    PM_h_delta >= MLP_h_delta: {cond5}  ({pm_hd:.4f} vs {mlp_hd:.4f})")

    conclusions = []
    if cond1 and cond2:
        conclusions.append(
            "Conclusion A: PM_h_delta beats shuffled delta and PM_h alone.\n"
            "  => Hidden tangent dynamics are useful, and PhaseMemory reads\n"
            "     progress-relevant temporal signals."
        )
    elif cond1 and not cond2:
        conclusions.append(
            "Conclusion A-: PM_h_delta beats shuffled delta but doesn't beat PM_h.\n"
            "  => Tangent signal exists but PhaseMemory may not be fully utilizing it."
        )

    if cond4 and cond5:
        conclusions.append(
            "Conclusion B: PM_h_delta improves over MLP_h_delta and GRU_h_delta.\n"
            "  => Phase recurrence provides additional benefit over standard speed probes."
        )
    elif cond5 and not cond4:
        conclusions.append(
            "Conclusion B-: PM_h_delta improves over MLP_h_delta but not GRU_h_delta.\n"
            "  => PM matches GRU temporally but recurrence advantage over feedforward not proven."
        )
    elif cond1 and not cond4 and not cond5:
        conclusions.append(
            "Conclusion C: PM_h_delta only matches MLP/GRU but beats shuffled delta.\n"
            "  => Tangent signal is valid, but phase recurrence advantage is not yet proven."
        )
    elif not cond1:
        conclusions.append(
            "Conclusion D: PM_h_delta does not beat shuffled delta.\n"
            "  => PM is not reliably reading progress; do not use PM as main module yet."
        )

    for c in conclusions:
        print(f"\n  {c}")

    print(f"\n  Baseline linear_time R²: {linear:.4f}")
    print(f"\nDone. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
