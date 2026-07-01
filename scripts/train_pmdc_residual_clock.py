#!/usr/bin/env python3
"""
train_pmdc_residual_clock — Trainable PMDC Residual Lyric Clock.

Trains a lightweight MLP ``PMDCResidualClock`` that reads per-timestep
hidden states from the frozen DiT decoder and outputs a bounded
log-speed residual.  The residual warps the linear progress schedule
``p_base`` into ``p_final``, which drives duration-interval attention
bias via ``text-mass-preserving`` split-softmax.

Only the PMDCResidualClock parameters are trained — the entire DiT
backbone, text encoder, and VAE remain frozen.

Usage
-----
    python scripts/train_pmdc_residual_clock.py \\
        --num-train-samples 500 --epochs 3 --batch-size 8 --lr 5e-5 \\
        --output-dir /root/autodl-tmp/pmdc_residual_clock_smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

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
    PMDCResidualClock,
    build_duration_scaffold,
    build_duration_interval_bias,
    mass_preserving_attention,
    parse_lyrics_to_units,
    freeze_except_pmdc_clock,
    get_scheduled_gate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


# ===================================================================
#  Data loading
# ===================================================================

def find_matching_pt(audio_stem: str, tensor_dir: Path) -> Optional[Path]:
    pt_path = tensor_dir / f"{audio_stem}.pt"
    if pt_path.is_file():
        return pt_path
    base = audio_stem.rsplit("_", 1)[0]
    for f in tensor_dir.glob(f"{base}_*.pt"):
        return f
    if TENSOR_TAR_PATH.is_file():
        pt_name = f"{audio_stem}.pt"
        try:
            import tarfile
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                for name in tar.getnames():
                    if name.split("/")[-1] == pt_name or name.split("/")[-1] == f"{base}.pt":
                        return Path(tensor_dir) / pt_path.name
        except Exception:
            pass
    return None


def load_preprocessed_data(pt_path: Path) -> Optional[dict]:
    pt_name = pt_path.name
    if pt_path.is_file():
        try:
            data = torch.load(str(pt_path), weights_only=True, map_location="cpu")
            return data
        except Exception:
            return None
    if TENSOR_TAR_PATH.is_file():
        import tarfile, tempfile
        try:
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                for name in tar.getnames():
                    if name.split("/")[-1] == pt_name:
                        m = tar.getmember(name)
                        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                            with tar.extractfile(m) as src:
                                tmp.write(src.read())
                        data = torch.load(tmp.name, weights_only=True, map_location="cpu")
                        os.unlink(tmp.name)
                        return data
        except Exception:
            pass
    return None


def find_metadata_files(audio_stem: str, audio_dir: Path, dataset_dir: Path) -> dict:
    result = {"caption_txt": None, "lyrics_txt": None}
    for f in audio_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name == f"{audio_stem}.lyrics.txt":
            result["lyrics_txt"] = f
        elif name == f"{audio_stem}.caption.txt":
            result["caption_txt"] = f
    if dataset_dir is not None:
        for key, sub in [("caption_txt", "caption.txt"), ("lyrics_txt", "lyrics.txt")]:
            p = dataset_dir / f"{audio_stem}.{sub}"
            if p.is_file():
                result[key] = p
    return result


def read_lyrics(meta_files: dict) -> Optional[str]:
    try:
        if meta_files.get("lyrics_txt"):
            return meta_files["lyrics_txt"].read_text().strip()
    except Exception:
        pass
    return None


def get_audio_duration_ffprobe(path: Path) -> Optional[float]:
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return None


# ===================================================================
#  Flow-matching timestep sampling
# ===================================================================

def sample_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    data_proportion: float = 0.0,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample continuous timesteps from a logit-normal distribution."""
    if data_proportion > 0 and random.random() < data_proportion:
        t = torch.full((batch_size,), 0.0, device=device, dtype=dtype)
    else:
        u = torch.empty(batch_size, device=device, dtype=dtype).uniform_(0, 1)
        t = 1.0 / (1.0 + torch.exp(-(torch.log(u / (1.0 - u + 1e-8)) * timestep_sigma + timestep_mu)))
        # Clamp for numerical stability
        t = torch.clamp(t, 1e-4, 1.0 - 1e-4)
    return t, t


# ===================================================================
#  Model setup with hooks
# ===================================================================

class PMDCTrainWrapper(nn.Module):
    """Wrapper that manages model, PMDC clock, hooks, and attention patching.

    During forward:
      1. Hook layer 12 hidden state.
      2. Patch ``eager_attention_forward`` to inject duration bias
         computed from ``p_final`` (output of ``PMDCResidualClock``).
      3. Run decoder forward.
      4. Unpatch attention.
      5. Return flow-matching loss + PMDC losses.
    """

    def __init__(
        self,
        model: nn.Module,
        clock: PMDCResidualClock,
        sigma: float = 0.03,
        lambda_: float = 0.5,
        max_bias: float = 1.0,
        gate_init: float = 0.35,
        w_pbase: float = 0.02,
        w_res: float = 0.001,
        w_smooth: float = 0.001,
    ):
        super().__init__()
        self.model = model
        self.clock = clock
        self.sigma = sigma
        self.lambda_ = lambda_
        self.max_bias = max_bias
        self.gate_init = gate_init
        self.w_pbase = w_pbase
        self.w_res = w_res
        self.w_smooth = w_smooth

        # Learnable gate
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        # Reset to gate_init: sigmoid(logit) = gate_init
        init_logit = math.log(max(gate_init / (1.0 - gate_init + 1e-6), 1e-6))
        with torch.no_grad():
            self.gate_logit.fill_(init_logit)
        # Ensure gate_logit requires grad by default
        self.gate_logit.requires_grad_(True)

        # Scaffold cache (built per-sample, reused across steps)
        self._scaffold = None
        self._p_base = None
        self._T_cache = 0

        self._hidden_collector = None
        self._handles = []
        self._orig_eaf = None
        self._patched_mod = None

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def set_scaffold(self, scaffold: dict, T: int, device: torch.device):
        """Precompute scaffold for an epoch of batches."""
        self._scaffold = {}
        for k, v in scaffold.items():
            if isinstance(v, torch.Tensor):
                # Keep int/bool types as-is, convert float to f32
                if v.dtype in (torch.float16, torch.bfloat16, torch.float64):
                    self._scaffold[k] = v.to(device, dtype=torch.float32)
                else:
                    self._scaffold[k] = v.to(device)
            else:
                self._scaffold[k] = v
        self._T_cache = T
        self._p_base = torch.linspace(0, 1, T, device=device).float().unsqueeze(0)

    def _install_hooks_and_patch(self):
        """Install hidden collector + patch attention forward."""
        device = next(self.model.parameters()).device

        # Hidden collector
        hs_list = []

        def _hook(module, input, output):
            hs_list.append(output[0])

        handle = self.model.decoder.layers[12].register_forward_hook(_hook)
        self._handles.append(handle)
        self._hs_list = hs_list

        # Patch eager_attention_forward
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        ca_module = self.model.decoder.layers[12].cross_attn
        import sys
        cls = type(ca_module)
        attn_mod = sys.modules[cls.__module__]
        self._orig_eaf = getattr(attn_mod, "eager_attention_forward", None)

        wrapper_ref = self

        def _pmdc_train_forward(*args, **kwargs):
            # Normal attention first
            mod = args[0]
            q = args[1]; k = args[2]; v = args[3]
            am = args[4] if len(args) > 4 else kwargs.get("attention_mask")
            sc = kwargs.get("scaling", args[5] if len(args) > 5 else None)
            dr = kwargs.get("dropout", 0.0)

            ks = repeat_kv(k, mod.num_key_value_groups)
            vs = repeat_kv(v, mod.num_key_value_groups)
            aw = torch.matmul(q, ks.transpose(2, 3)) * sc
            if am is not None and isinstance(am, torch.Tensor):
                causal_mask = am[:, :, :, :ks.shape[-2]]
                aw = aw + causal_mask

            # Check if we have a scaffold and bias to apply
            if wrapper_ref._scaffold is not None and hasattr(wrapper_ref, '_bias_cache'):
                gate_val = wrapper_ref.gate
                bias_t = wrapper_ref._bias_cache
                if bias_t.shape[0] != aw.shape[0]:
                    b_factor = aw.shape[0] // max(bias_t.shape[0], 1)
                    if b_factor > 1:
                        bias_t = bias_t.repeat(b_factor, 1, 1, 1)
                bias_exp = bias_t.unsqueeze(1)  # [B, 1, T, L]

                # Text-mass-preserving attention
                amask = wrapper_ref._scaffold.get("attendable_mask",
                            wrapper_ref._scaffold.get("lyric_mask"))
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

        setattr(attn_mod, "eager_attention_forward", _pmdc_train_forward)
        self._patched_mod = attn_mod

    def _install_patch_only(self):
        """Patch eager_attention_forward without installing hooks."""
        if self._orig_eaf is not None:
            return  # already patched
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        ca_module = self.model.decoder.layers[12].cross_attn
        import sys
        cls = type(ca_module)
        attn_mod = sys.modules[cls.__module__]
        self._orig_eaf = getattr(attn_mod, "eager_attention_forward", None)

        wrapper_ref = self
        def _pmdc_train_forward(*args, **kwargs):
            mod = args[0]; q = args[1]; k = args[2]; v = args[3]
            am = args[4] if len(args) > 4 else kwargs.get("attention_mask")
            sc = kwargs.get("scaling", args[5] if len(args) > 5 else None)
            dr = kwargs.get("dropout", 0.0)
            ks = repeat_kv(k, mod.num_key_value_groups)
            vs = repeat_kv(v, mod.num_key_value_groups)
            aw = torch.matmul(q, ks.transpose(2, 3)) * sc
            if am is not None and isinstance(am, torch.Tensor):
                aw = aw + am[:, :, :, :ks.shape[-2]]

            if wrapper_ref._bias_cache is not None:
                gate_val = wrapper_ref.gate
                bias_t = wrapper_ref._bias_cache
                if bias_t.shape[0] != aw.shape[0]:
                    bf = aw.shape[0] // max(bias_t.shape[0], 1)
                    if bf > 1:
                        bias_t = bias_t.repeat(bf, 1, 1, 1)
                bias_exp = bias_t.unsqueeze(1)

                amask = wrapper_ref._scaffold.get("attendable_mask",
                            wrapper_ref._scaffold.get("lyric_mask"))
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

        setattr(attn_mod, "eager_attention_forward", _pmdc_train_forward)
        self._patched_mod = attn_mod

    def _uninstall_patch(self):
        """Restore original eager_attention_forward."""
        if self._patched_mod is not None and self._orig_eaf is not None:
            setattr(self._patched_mod, "eager_attention_forward", self._orig_eaf)
        self._patched_mod = None
        self._orig_eaf = None

    def _remove_hooks_and_patch(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._hs_list = []
        if self._patched_mod is not None and self._orig_eaf is not None:
            setattr(self._patched_mod, "eager_attention_forward", self._orig_eaf)
        self._patched_mod = None
        self._orig_eaf = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with target_latents, attention_mask, encoder_hidden_states,
                   encoder_attention_mask, context_latents.

        Returns:
            dict with loss, flow_loss, pbase_loss, res_loss, smooth_loss, etc.
        """
        device = next(self.model.parameters()).device
        nb = True  # non_blocking
        dtype = next(self.model.parameters()).dtype

        target_latents = batch["target_latents"].to(device, dtype=dtype, non_blocking=nb)
        attention_mask = batch["attention_mask"].to(device, dtype=dtype, non_blocking=nb)
        encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=dtype, non_blocking=nb)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=dtype, non_blocking=nb)
        context_latents = batch["context_latents"].to(device, dtype=dtype, non_blocking=nb)

        bsz = target_latents.shape[0]
        T = target_latents.shape[1]

        # ---- Recompute p_final if T changes --------------------------------
        if T != self._T_cache and self._scaffold is not None:
            self._p_base = torch.linspace(0, 1, T, device=device).float().unsqueeze(0)

        # ---- Flow matching setup -------------------------------------------
        x1 = torch.randn_like(target_latents)  # noise
        x0 = target_latents  # data

        t, r = sample_timesteps(
            batch_size=bsz, device=device, dtype=dtype,
            data_proportion=0.0,
            timestep_mu=-0.4, timestep_sigma=1.0,
        )
        t_ = t.unsqueeze(-1).unsqueeze(-1)
        xt = t_ * x1 + (1.0 - t_) * x0

        # ---- Step 1: Warmup forward (no PMDC bias) to get H ---------------
        # Run the decoder WITHOUT the PMDC patch to collect layer 12 H.
        # Important: set up hook BEFORE calling decoder.
        hs_list = []
        def _warmup_hook(m, i, o):
            hs_list.append(o[0])
        handle = self.model.decoder.layers[12].register_forward_hook(_warmup_hook)
        try:
            _ = self.model.decoder(
                hidden_states=xt,
                timestep=t, timestep_r=r,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            handle.remove()
        if hs_list:
            H_warm = hs_list[0].detach().float()
        else:
            H_warm = torch.zeros(bsz, T, 2048, device=device, dtype=torch.float32)

        # Hidden states may have different T than target_latents due to DiT
        # patch_size=2. Use the hidden state's T for clock computation.
        T_pmdc = H_warm.shape[1]

        # ---- PMDC forward (uses H from warmup) -----------------------------
        # p_base at hidden-state T (T_pmdc, not target_latents T)
        p_base = torch.linspace(0, 1, T_pmdc, device=device, dtype=torch.float32)
        p_base = p_base.unsqueeze(0).expand(bsz, -1)
        speed_residual, p_final, log_v = self.clock(H_warm, p_base)

        # ---- Build bias from p_final (synced with this step) ---------------
        if self._scaffold is not None:
            bias = build_duration_interval_bias(
                p_final=p_final,
                unit_boundaries=self._scaffold["unit_boundaries"],
                token_to_unit=self._scaffold["token_to_unit"],
                attendable_mask=self._scaffold.get("attendable_mask",
                                                     self._scaffold["lyric_mask"]),
                sigma=self.sigma, lambda_=self.lambda_, max_bias=self.max_bias,
            )
            if next(self.model.parameters()).dtype == torch.bfloat16:
                bias = bias.to(torch.bfloat16)
            self._bias_cache = bias
        else:
            self._bias_cache = None

        # ---- Step 2: Main decoder forward with PMDC bias -------------------
        # Install ONLY the attention patch (no hook needed — we have H from warmup)
        self._install_patch_only()
        try:
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t, timestep_r=r,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=False, output_attentions=False,
            )
        finally:
            self._uninstall_patch()

        # ---- Losses --------------------------------------------------------
        flow = x1 - x0
        flow_loss = F.mse_loss(decoder_outputs[0], flow)
        pbase_loss = F.smooth_l1_loss(p_final, p_base)
        res_loss = speed_residual.pow(2).mean()
        smooth_loss = F.mse_loss(log_v[:, 1:], log_v[:, :-1])

        loss = flow_loss + \
               self.w_pbase * pbase_loss + \
               self.w_res * res_loss + \
               self.w_smooth * smooth_loss

        metrics = {
            "loss": loss,
            "flow_loss": flow_loss,
            "pbase_loss": pbase_loss,
            "res_loss": res_loss,
            "smooth_loss": smooth_loss,
            "gate": self.gate,
            "beta": self.clock.beta,
            "mean_abs_p_delta": (p_final - p_base).abs().mean(),
            "speed_residual_std": speed_residual.std(),
            "log_v_std": log_v.std(),
            "p_final_min": p_final.min(),
            "p_final_max": p_final.max(),
        }

        return metrics


# ===================================================================
#  Dataset
# ===================================================================

class PMDCTrainDataset(Dataset):
    """Dataset that loads preprocessed .pt files and builds scaffolds."""

    def __init__(
        self,
        pt_paths: List[Path],
        lyrics_dict: Dict[str, str],
        max_samples: int = -1,
        seed: int = 42,
    ):
        self.items = []
        rng = random.Random(seed)

        for pt_path in pt_paths:
            sid = pt_path.stem
            lyrics = lyrics_dict.get(sid)
            if lyrics is None:
                continue

            # Parse to get section_ids for scaffold
            parser = LyricsStructureParser()
            parsed = parser.parse(lyrics, num_chunks=128)
            section_ids = parsed.section_type_ids

            units, _, debug = parse_lyrics_to_units(lyrics, section_ids)
            if len(units) == 0:
                continue

            tcm = debug.get("tag_control_mask", None)
            scaffold = build_duration_scaffold(units, text_len=769, tag_control_mask=tcm)

            self.items.append({
                "pt_path": pt_path,
                "sample_id": sid,
                "scaffold": scaffold,
            })

        if max_samples > 0 and len(self.items) > max_samples:
            rng.shuffle(self.items)
            self.items = self.items[:max_samples]

        print(f"  Dataset: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        pt_data = load_preprocessed_data(item["pt_path"])
        if pt_data is None:
            # Return a dummy — skip in collation
            return None
        return {
            "target_latents": pt_data["target_latents"],          # [T, 64]
            "attention_mask": pt_data["attention_mask"],          # [T]
            "encoder_hidden_states": pt_data["encoder_hidden_states"],  # [L, D]
            "encoder_attention_mask": pt_data["encoder_attention_mask"],  # [L]
            "context_latents": pt_data["context_latents"],        # [T', 128]
            "scaffold": item["scaffold"],
            "sample_id": item["sample_id"],
        }


def collate_pmdc(batch):
    """Collate into batched dicts, handling None skips."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Pad to max T
    max_T = max(b["target_latents"].shape[0] for b in batch)
    max_L = max(b["encoder_hidden_states"].shape[0] for b in batch)

    target_latents = []
    attention_mask = []
    encoder_hidden_states = []
    encoder_attention_mask = []
    context_latents = []
    scaffolds = []
    sample_ids = []

    for b in batch:
        T = b["target_latents"].shape[0]
        L = b["encoder_hidden_states"].shape[0]

        pad_T = max_T - T
        pad_L = max_L - L

        tl = b["target_latents"]
        if pad_T > 0:
            tl = F.pad(tl, (0, 0, 0, pad_T))
        target_latents.append(tl.unsqueeze(0))

        am = b["attention_mask"]
        if pad_T > 0:
            am = F.pad(am, (0, pad_T))
        attention_mask.append(am.unsqueeze(0))

        eh = b["encoder_hidden_states"]
        if pad_L > 0:
            eh = F.pad(eh, (0, 0, 0, pad_L))
        encoder_hidden_states.append(eh.unsqueeze(0))

        ea = b["encoder_attention_mask"]
        if pad_L > 0:
            ea = F.pad(ea, (0, pad_L))
        encoder_attention_mask.append(ea.unsqueeze(0))

        ctx = b["context_latents"]
        # context_latents: [T', 128], pad independently
        C = ctx.shape[0]
        max_C = max(b2["context_latents"].shape[0] for b2 in batch)
        if C < max_C:
            ctx = F.pad(ctx, (0, 0, 0, max_C - C))
        context_latents.append(ctx.unsqueeze(0))

        scaffolds.append(b["scaffold"])
        sample_ids.append(b["sample_id"])

    return {
        "target_latents": torch.cat(target_latents, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
        "encoder_hidden_states": torch.cat(encoder_hidden_states, dim=0),
        "encoder_attention_mask": torch.cat(encoder_attention_mask, dim=0),
        "context_latents": torch.cat(context_latents, dim=0),
        "scaffolds": scaffolds,
        "sample_ids": sample_ids,
    }


# ===================================================================
#  Training step helper
# ===================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    wrapper: PMDCTrainWrapper,
    epoch: int,
    args: Any,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    wrapper.train()

    total_metrics: Dict[str, float] = {}
    count = 0

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        # Set scaffold for this batch (use first sample's scaffold)
        # In a real training setup, each sample in the batch should have its own
        # scaffold, but for PMDC where all samples share the same text length,
        # we use the first one's scaffold
        scaffold = batch["scaffolds"][0]
        T = batch["target_latents"].shape[1]
        wrapper.set_scaffold(scaffold, T, device)

        optimizer.zero_grad(set_to_none=True)

        try:
            metrics = wrapper(batch)
        except Exception as e:
            print(f"  Step {step} error: {e}")
            traceback.print_exc()
            # Try to recover
            wrapper._uninstall_patch()
            continue

        loss = metrics["loss"]
        loss.backward()

        # Clip gradients (only PMDC + gate params)
        trainable_params = [p for p in wrapper.clock.parameters()] + [wrapper.gate_logit]
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in total_metrics:
                total_metrics[k] = 0.0
            total_metrics[k] += v
        count += 1

        if step % max(args.log_every, 1) == 0:
            log_v = {k: f"{v/count:.6f}" if count > 0 else "N/A"
                     for k, v in total_metrics.items()}
            lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            print(f"  Epoch {epoch+1} Step {step}: loss={log_v.get('loss', 'N/A')} "
                  f"flow={log_v.get('flow_loss', 'N/A')} "
                  f"pbase={log_v.get('pbase_loss', 'N/A')} "
                  f"gate={log_v.get('gate', 'N/A')} "
                  f"beta={log_v.get('beta', 'N/A')} "
                  f"pdelta={log_v.get('mean_abs_p_delta', 'N/A')} "
                  f"lr={lr:.2e}", flush=True)

    # Average
    avg = {k: v / max(count, 1) for k, v in total_metrics.items()}
    return avg


# ===================================================================
#  Batch sanity check
# ===================================================================

def run_forward_check(wrapper: PMDCTrainWrapper, batch: dict, device: torch.device):
    """Run one forward, print p_final stats, check gradients."""
    print("\n--- Forward check ---")
    wrapper.eval()
    with torch.no_grad():
        scaffold = batch["scaffolds"][0]
        T = batch["target_latents"].shape[1]
        wrapper.set_scaffold(scaffold, T, device)

        metrics = wrapper(batch)

    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"  {k}: {v:.6f}")

    # Check p_final from the real forward (already computed above in metrics)
    pdelta_actual = metrics.get("mean_abs_p_delta", 0)
    if isinstance(pdelta_actual, torch.Tensor):
        pdelta_actual = pdelta_actual.item()
    print(f"  mean_abs_p_delta (actual forward): {pdelta_actual:.6f}")
    # The actual p_delta should be ~0 since H is detached and speed_head is zero-init
    if pdelta_actual < 0.01:
        print("  ✓ p_final ≈ p_base at init (zero-init working)")
    else:
        print(f"  ⚠ p_delta={pdelta_actual:.6f} > 0.01 (may be okay if pretrained)")
    print("  ✓ Forward check passed")


def run_backward_check(wrapper: PMDCTrainWrapper, batch: dict, device: torch.device, args):
    """Run forward+backward, check which params got gradients."""
    print("\n--- Backward check ---")
    wrapper.train()

    scaffold = batch["scaffolds"][0]
    T = batch["target_latents"].shape[1]
    wrapper.set_scaffold(scaffold, T, device)

    metrics = wrapper(batch)
    loss = metrics["loss"]
    loss.backward()

    pmdc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in wrapper.clock.parameters())
    gate_grad = wrapper.gate_logit.grad is not None and wrapper.gate_logit.grad.abs().sum() > 0

    # Check backbone frozen
    backbone_grad = False
    for name, param in wrapper.model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            backbone_grad = True
            break

    print(f"  PMDC params have gradient: {pmdc_grad}")
    print(f"  Gate param has gradient: {gate_grad}")
    print(f"  Backbone params have gradient: {backbone_grad}")
    assert pmdc_grad, "PMDC has no gradient!"
    assert not backbone_grad, "Backbone should be frozen!"
    print("  ✓ Backward check passed")

    wrapper.zero_grad(set_to_none=True)


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Train PMDC Residual Clock")
    parser.add_argument("--data-root", default=str(TENSOR_DIR_DEFAULT))
    parser.add_argument("--audio-root", default="/root/autodl-tmp/musicdata/audios")
    parser.add_argument("--dataset-dir", default="/root/autodl-tmp/musicdata/dataset")
    parser.add_argument("--checkpoint", default=str(MODEL_ROOT))
    parser.add_argument("--output-dir", default="/root/autodl-tmp/pmdc_residual_clock")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--num-train-samples", type=int, default=500)
    parser.add_argument("--min-duration", type=float, default=180)
    parser.add_argument("--max-duration", type=float, default=300)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--precision", default="bf16")

    # PMDC
    parser.add_argument("--pmdc-hidden-dim", type=int, default=128)
    parser.add_argument("--beta-init", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=0.15)
    parser.add_argument("--use-delta-h", action="store_true", default=True)

    # Duration bias
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--lambda", type=float, dest="lambda_", default=0.5)
    parser.add_argument("--gate-init", type=float, default=0.35)
    parser.add_argument("--max-bias", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--w-pbase", type=float, default=0.02)
    parser.add_argument("--w-res", type=float, default=0.001)
    parser.add_argument("--w-smooth", type=float, default=0.001)

    # Overfit mode
    parser.add_argument("--overfit", action="store_true",
                        help="Overfit on 10 steps for quick smoke test")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("TRAIN PMDC RESIDUAL CLOCK")
    print("=" * 70)
    print(f"  output: {OUTPUT_DIR}")
    print(f"  epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")
    print(f"  sigma={args.sigma}, lambda={args.lambda_}, gate_init={args.gate_init}")
    print(f"  beta_init={args.beta_init}, beta_max={args.beta_max}")
    print(f"  w_pbase={args.w_pbase}, w_res={args.w_res}, w_smooth={args.w_smooth}")

    # ================================================================
    # Step 1: Find samples
    # ================================================================
    print("\n[1/6] Finding training samples...", flush=True)
    audio_dir = Path(args.audio_root)
    tensor_dir = Path(args.data_root)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    all_audio = []
    for ext in AUDIO_EXTENSIONS:
        all_audio.extend(audio_dir.rglob(f"*{ext}"))
    all_audio = sorted(all_audio)

    audio_durations = {}
    for f in all_audio:
        dur = get_audio_duration_ffprobe(f)
        if dur is not None:
            audio_durations[f.name] = dur

    # Find samples with lyrics + .pt
    valid_samples = []
    for f in all_audio:
        dur = audio_durations.get(f.name)
        if dur is None or not (args.min_duration <= dur <= args.max_duration):
            continue
        meta = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta)
        if lyrics is None:
            continue
        pt_path = find_matching_pt(f.stem, tensor_dir)
        if pt_path is None:
            continue
        valid_samples.append((f.stem, pt_path, lyrics))

    print(f"  {len(valid_samples)} valid samples")

    if len(valid_samples) < 1:
        print("  No samples. Aborting.")
        sys.exit(1)

    if args.overfit:
        valid_samples = valid_samples[:1]
        print(f"  Overfit mode: 1 sample")

    # ================================================================
    # Step 2: Build dataset
    # ================================================================
    print("\n[2/6] Building dataset...", flush=True)

    lyrics_dict = {sid: lyrics for sid, _, lyrics in valid_samples}
    pt_paths = [pt for _, pt, _ in valid_samples]

    n_train = min(args.num_train_samples, len(pt_paths))
    train_pt = pt_paths[:n_train]

    dataset = PMDCTrainDataset(
        train_pt, lyrics_dict,
        max_samples=args.num_train_samples,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pmdc, drop_last=True, num_workers=0,
    )

    # ================================================================
    # Step 3: Load model
    # ================================================================
    print("\n[3/6] Loading baseline model...", flush=True)

    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(args.checkpoint),
        config_path="acestep-v15-sft",
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit_handler.model.eval()
    D = model.config.hidden_size

    # Disable all adapters
    model.config.use_section_rope_offset = False
    for lm in model.decoder.layers:
        if getattr(lm, "use_section_rope", False):
            lm.use_section_rope = False
        if getattr(lm, "use_phase_memory", False):
            lm.use_phase_memory = False

    model = model.to(device)

    # ================================================================
    # Step 4: Create PMDC clock + wrapper
    # ================================================================
    print("\n[4/6] Creating PMDC residual clock...", flush=True)

    clock = PMDCResidualClock(
        dim=D,
        hidden_dim=args.pmdc_hidden_dim,
        beta_init=args.beta_init,
        beta_max=args.beta_max,
        use_delta_h=args.use_delta_h,
    ).to(device)

    wrapper = PMDCTrainWrapper(
        model=model,
        clock=clock,
        sigma=args.sigma,
        lambda_=args.lambda_,
        max_bias=args.max_bias,
        gate_init=args.gate_init,
        w_pbase=args.w_pbase,
        w_res=args.w_res,
        w_smooth=args.w_smooth,
    ).to(device)

    # ================================================================
    # Step 5: Freeze backbone
    # ================================================================
    print("\n[5/6] Freezing backbone...", flush=True)

    trainable, total, trainable_names = freeze_except_pmdc_clock(model, pmdc_clock=clock)
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,} ({100*trainable/max(total,1):.3f}%)")
    print(f"  Trainable modules:")
    for name in sorted(set(n.split(".")[-1] for n in trainable_names)):
        print(f"    - {name}")

    # ================================================================
    # Step 6: Forward + backward checks, then train
    # ================================================================
    print("\n[6/6] Checks + training...", flush=True)

    # Get a batch for checks
    check_batch = None
    for batch in loader:
        if batch is not None:
            check_batch = batch
            break

    if check_batch is not None:
        run_forward_check(wrapper, check_batch, device)
        run_backward_check(wrapper, check_batch, device, args)
    else:
        print("  No data for checks!")
        sys.exit(1)

    # ---- Training ----
    if args.overfit:
        # Overfit: single batch, repeated
        print("\n--- Overfit mode: 10 steps on single batch ---")
        optimizer = torch.optim.AdamW(
            list(clock.parameters()) + [wrapper.gate_logit],
            lr=args.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        for step in range(30):
            wrapper.train()
            optimizer.zero_grad(set_to_none=True)

            scaffold = check_batch["scaffolds"][0]
            T = check_batch["target_latents"].shape[1]
            wrapper.set_scaffold(scaffold, T, device)

            metrics = wrapper(check_batch)
            loss = metrics["loss"]
            loss.backward()

            trainable_params = list(clock.parameters()) + [wrapper.gate_logit]
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % 2 == 0:
                log = {k: f"{v.item() if isinstance(v, torch.Tensor) else v:.6f}"
                       for k, v in metrics.items()
                       if isinstance(v, (torch.Tensor, float, int))}
                print(f"  Step {step}: loss={log.get('loss', 'N/A')} "
                      f"pdelta={log.get('mean_abs_p_delta', 'N/A')} "
                      f"beta={log.get('beta', 'N/A')} "
                      f"gate={log.get('gate', 'N/A')}", flush=True)

        print("  ✓ Overfit complete")

    else:
        # Full training
        print(f"\n--- Full training: {args.epochs} epochs ---")
        optimizer = torch.optim.AdamW(
            list(clock.parameters()) + [wrapper.gate_logit],
            lr=args.lr, weight_decay=1e-4,
        )
        steps_per_epoch = len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch,
        )

        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(
                model=model, loader=loader, optimizer=optimizer,
                scheduler=scheduler, wrapper=wrapper,
                epoch=epoch, args=args, device=device,
            )
            print(f"  Epoch {epoch+1}/{args.epochs} avg: "
                  f"loss={train_metrics.get('loss', 0):.6f} "
                  f"flow={train_metrics.get('flow_loss', 0):.6f} "
                  f"pdelta={train_metrics.get('mean_abs_p_delta', 0):.6f} "
                  f"beta={train_metrics.get('beta', 0):.6f} "
                  f"gate={train_metrics.get('gate', 0):.6f}", flush=True)

            # Save checkpoint
            ckpt_dir = OUTPUT_DIR / "checkpoints" / f"epoch_{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "clock_state_dict": clock.state_dict(),
                "gate_logit": wrapper.gate_logit.data.cpu(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": train_metrics,
            }, ckpt_dir / "pmdc_clock.pt")
            print(f"  Checkpoint -> {ckpt_dir / 'pmdc_clock.pt'}")

        # Final save
        torch.save({
            "clock_state_dict": clock.state_dict(),
            "gate_logit": wrapper.gate_logit.data.cpu(),
            "args": vars(args),
        }, OUTPUT_DIR / "pmdc_clock_final.pt")
        print(f"\n  Final model -> {OUTPUT_DIR / 'pmdc_clock_final.pt'}")

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
