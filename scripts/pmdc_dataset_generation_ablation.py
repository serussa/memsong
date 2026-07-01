#!/usr/bin/env python3
"""
pmdc_dataset_generation_ablation — Stage 3.

Generates audio with PMDC duration-scaffold attention intervention
and compares modes: baseline, fixed_linear, duration_weak/mid/strong.

Usage
-----
    python scripts/pmdc_dataset_generation_ablation.py \\
        --num-samples 5 \\
        --seeds 1234 5678 \\
        --duration 240 \\
        --steps 50 \\
        --output-dir /root/autodl-tmp/pmdc_stage3_generation \\
        --device cuda

Smoke:
    python scripts/pmdc_dataset_generation_ablation.py \\
        --num-samples 3 \\
        --seeds 1234 \\
        --duration 120 --steps 30 \\
        --output-dir /root/autodl-tmp/pmdc_stage3_generation_smoke \\
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
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

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.tgca.lyrics_parser import LyricsStructureParser

from acestep.phase_memory import (
    LyricUnit,
    build_duration_scaffold,
    build_duration_interval_bias,
    mass_preserving_attention,
    parse_lyrics_to_units,
    is_natural_control_line,
    make_control_tag,
    insert_control_lines,
    get_scheduled_gate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

WEAK_CONFIG  = {"sigma": 0.03, "lambda_": 0.5, "gate": 0.5, "max_bias": 1.0}
MID_CONFIG   = {"sigma": 0.03, "lambda_": 2.0, "gate": 0.5, "max_bias": 2.0}
STRONG_CONFIG = {"sigma": 0.12, "lambda_": 0.5, "gate": 0.5, "max_bias": 2.0}

MODES = ["baseline", "fixed_linear", "duration_weak", "duration_mid", "duration_strong",
         "old_parser_duration_weak", "clean_parser_no_control", "clean_parser_intro_outro_control",
         "clean_parser_intro_outro_control_fadeout"]


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
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
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
        for key, sub in [("caption_txt", "caption.txt"), ("lyrics_txt", "lyrics.txt")]:
            p = dataset_dir / f"{audio_stem}.{sub}"
            if p.is_file():
                result[key] = p
    return result


def read_lyrics(meta_files: dict) -> Optional[str]:
    for kwargs in [
        (lambda: meta_files["json"] and json.load(open(meta_files["json"])).get("lyrics"),),
        (lambda: meta_files["json"] and json.load(open(meta_files["json"])).get("text"),),
        (lambda: meta_files["lrc"] and meta_files["lrc"].read_text(),),
        (lambda: meta_files["txt"] and meta_files["txt"].read_text(),),
        (lambda: meta_files["lyrics_txt"] and meta_files["lyrics_txt"].read_text(),),
    ]:
        try:
            val = kwargs[0]()
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass
    return None


def read_caption(meta_files: dict) -> Optional[str]:
    try:
        if meta_files.get("caption_txt"):
            val = meta_files["caption_txt"].read_text().strip()
            if val:
                return val
    except Exception:
        pass
    try:
        if meta_files.get("json"):
            data = json.load(open(meta_files["json"]))
            for key in ("caption", "prompt", "tags"):
                if key in data and data[key]:
                    return str(data[key])
    except Exception:
        pass
    return None


# ===================================================================
#  PART 2 — Lyric unit parsing (imported from PMDC)
# ===================================================================
# Note: parse_lyrics_to_units is now imported from acestep.phase_memory.
# The old local copy has been removed in favour of the tag-aware version.
# Make sure to update the import above when syncing scripts.


# ===================================================================
#  PART 3 — PMDC attention patcher
# ===================================================================

# ===================================================================
#  Old parser simulation (uniform line split, all tokens = lyric)
# ===================================================================

def old_parse_units(lyrics_text: str, section_ids: torch.Tensor, L: int = 769) -> Tuple[List[LyricUnit], np.ndarray]:
    """Simulate the old parser: uniform line split, all tokens as lyric."""
    lines = [l for l in lyrics_text.strip().split("\n") if l.strip()]
    ids_np = section_ids.cpu().numpy() if isinstance(section_ids, torch.Tensor) else section_ids
    chunk_edges = np.linspace(0, L, len(lines) + 1, dtype=int)
    lyric_mask_raw = (ids_np >= 1) & (ids_np <= 7)
    lyric_indices = np.where(lyric_mask_raw)[0]
    units = []
    section_map = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                   4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}
    if len(lyric_indices) > 0:
        token_splits = np.array_split(lyric_indices, len(lines))
        for (i, line), tidxs in zip(enumerate(lines), token_splits):
            if len(tidxs) == 0:
                continue
            smode = np.bincount(ids_np[tidxs[0]:tidxs[-1]+1][ids_np[tidxs[0]:tidxs[-1]+1] >= 0]).argmax() if len(tidxs) > 0 else 0
            units.append(LyricUnit(unit_id=len(units), section=section_map.get(smode, "UNKNOWN"), text=line,
                char_count=len(line.replace(" ", "")), token_indices=[int(t) for t in tidxs],
                occurrence_id=0, is_silence=False))
    return units, np.full(L, 0.0, dtype=np.float32)


# ===================================================================
#  PMDC injection context
# ===================================================================

class PMDCInjectionContext:
    """Context manager that patches cross-attention to inject PMDC bias
    during inference generation.

    On entry:
      - Builds duration scaffold from lyrics
      - Builds bias for each mode
      - Patches eager_attention_forward in the module namespace

    On exit:
      - Restores original eager_attention_forward
    """

    def __init__(self, model, units, text_len, T_audio, device="cuda",
                 tag_control_mask: Optional[np.ndarray] = None):
        self.model = model
        self.ca_module = model.decoder.layers[12].cross_attn
        self._orig_fn = None
        self._patched_mod = None

        # CRITICAL: force eager attention so our patch on eager_attention_forward runs
        model.config._attn_implementation = "eager"
        model.config._attn_implementation_compiled = None
        for layer in model.decoder.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.config._attn_implementation = "eager"
            if hasattr(layer, "cross_attn"):
                layer.cross_attn.config._attn_implementation = "eager"

        # Build scaffold (with tag_control_mask if available)
        scaffold = build_duration_scaffold(units=units, text_len=text_len, device=device,
                                            tag_control_mask=tag_control_mask)
        self.unit_boundaries = scaffold["unit_boundaries"]
        self.token_to_unit = scaffold["token_to_unit"]
        self.lyric_mask = scaffold["lyric_mask"]
        self.control_mask = scaffold.get("control_mask", None)
        self.attendable_mask = scaffold.get("attendable_mask", None)
        self.unit_duration = scaffold["unit_duration"]

        # p_base (linear progress)
        self.p_base = torch.linspace(0, 1, T_audio, device=device).float().unsqueeze(0)

        # Pre-build biases for all modes using attendable_mask
        attendable_for_bias = self.attendable_mask if self.attendable_mask is not None else self.lyric_mask
        self.bias_cache = {}
        for name, cfg in [("duration_weak", WEAK_CONFIG),
                           ("duration_mid", MID_CONFIG),
                           ("duration_strong", STRONG_CONFIG)]:
            self.bias_cache[name] = build_duration_interval_bias(
                p_final=self.p_base, unit_boundaries=self.unit_boundaries,
                token_to_unit=self.token_to_unit, attendable_mask=attendable_for_bias,
                sigma=cfg["sigma"], lambda_=cfg["lambda_"], max_bias=cfg["max_bias"],
            )
        # Fixed linear bias
        self.bias_cache["fixed_linear"] = build_fixed_linear_bias(
            T_audio, text_len,
            self._make_lyric_pos(text_len, device),
            self.lyric_mask.to(device),
            device=device,
        )

    def _make_lyric_pos(self, L_text, device):
        lyric_pos = torch.zeros(L_text, device=device)
        lm = self.lyric_mask if self.lyric_mask.device == device else self.lyric_mask.to(device)
        lm_np = lm.cpu().numpy()
        indices = np.where(lm_np)[0]
        if len(indices) > 1:
            for k, idx in enumerate(indices):
                lyric_pos[idx] = k / (len(indices) - 1)
        elif len(indices) == 1:
            lyric_pos[indices[0]] = 0.5
        return lyric_pos

    def install(self, mode: str, gate: float):
        """Install patched forward for a given mode."""
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        if mode == "baseline":
            return

        # Store bias on module
        if mode in ("fixed_linear",):
            bias = self.bias_cache.get(mode)
            self.ca_module._pmdc_use_mp = False
            gate = 0.3
        else:
            bias = self.bias_cache.get(mode)
            self.ca_module._pmdc_use_mp = True

        self.ca_module._pmdc_bias = bias.to(device=next(self.ca_module.parameters()).device,
                                             dtype=next(self.ca_module.parameters()).dtype)
        self.ca_module._pmdc_gate = gate
        self.ca_module._pmdc_lyric_mask = self.lyric_mask.to(device=next(self.ca_module.parameters()).device)
        if self.attendable_mask is not None:
            self.ca_module._pmdc_attendable_mask = self.attendable_mask.to(
                device=next(self.ca_module.parameters()).device)
        else:
            self.ca_module._pmdc_attendable_mask = self.lyric_mask.to(
                device=next(self.ca_module.parameters()).device)

        # Patch eager_attention_forward (only save original once)
        import sys
        cls = type(self.ca_module)
        attn_mod = sys.modules[cls.__module__]
        if self._orig_fn is None:
            self._orig_fn = getattr(attn_mod, "eager_attention_forward", None)
        self._patched_mod = attn_mod

        def _pmdc_patched_forward(module, query, key, value, attention_mask,
                                   scaling, dropout=0.0, **kwargs):
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            bias_t = getattr(module, "_pmdc_bias", None)
            if bias_t is not None and hasattr(module, "_pmdc_lyric_mask"):
                use_mp = getattr(module, "_pmdc_use_mp", True)
                gate = getattr(module, "_pmdc_gate", 0.1)

                # Expand bias if needed (CFG doubles batch)
                if bias_t.shape[0] != attn_weights.shape[0]:
                    b_factor = attn_weights.shape[0] // bias_t.shape[0]
                    bias_exp = bias_t.repeat(b_factor, 1, 1, 1) if bias_t.dim() == 4 else bias_t
                    bias_exp = bias_exp.unsqueeze(1) if bias_exp.dim() == 3 else bias_exp
                else:
                    bias_exp = bias_t.unsqueeze(1) if bias_t.dim() == 3 else bias_t

                if use_mp:
                    # Text-mass-preserving split-softmax
                    # Use attendable_mask (lyric + control tokens)
                    am = module._pmdc_attendable_mask  # [L]

                    def _expand(m):
                        m2 = m.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        if m2.shape[0] != attn_weights.shape[0]:
                            m2 = m2.expand(attn_weights.shape[0], -1, -1, -1)
                        return m2

                    am_exp = _expand(am).bool()

                    # Base: total attendable mass
                    attn_base = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    text_mass_base = (attn_base * am_exp.float()).sum(dim=-1)

                    # Text split: ALL attendable tokens get bias based on their unit
                    text_logits = (attn_weights + gate * bias_exp).masked_fill(~am_exp, float("-inf"))
                    attn_text = F.softmax(text_logits, dim=-1, dtype=torch.float32)
                    attn_text = attn_text.masked_fill(~am_exp, 0.0)

                    # Non-text split
                    non_text_logits = attn_weights.masked_fill(am_exp, float("-inf"))
                    attn_non_text = F.softmax(non_text_logits, dim=-1, dtype=torch.float32)
                    attn_non_text = attn_non_text.masked_fill(am_exp, 0.0)

                    # Recombine preserving total text mass
                    attn_weights = (attn_text * text_mass_base.unsqueeze(-1)
                                    + attn_non_text * (1.0 - text_mass_base).unsqueeze(-1))
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
                    attn_weights = attn_weights.to(query.dtype)
                else:
                    # Plain softmax with bias
                    attn_weights = attn_weights + gate * bias_exp
                    attn_weights = F.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query.dtype)
            else:
                attn_weights = F.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query.dtype)

            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        setattr(attn_mod, "eager_attention_forward", _pmdc_patched_forward)

    def uninstall(self):
        """Restore original eager_attention_forward."""
        if self._patched_mod is not None and self._orig_fn is not None:
            setattr(self._patched_mod, "eager_attention_forward", self._orig_fn)
        if hasattr(self.ca_module, "_pmdc_bias"):
            del self.ca_module._pmdc_bias
        if hasattr(self.ca_module, "_pmdc_lyric_mask"):
            del self.ca_module._pmdc_lyric_mask
        if hasattr(self.ca_module, "_pmdc_attendable_mask"):
            del self.ca_module._pmdc_attendable_mask
        if hasattr(self.ca_module, "_pmdc_gate"):
            del self.ca_module._pmdc_gate
        if hasattr(self.ca_module, "_pmdc_use_mp"):
            del self.ca_module._pmdc_use_mp
        self._patched_mod = None
        self._orig_fn = None



# ===================================================================
#  PART 4 — Build linear bias helper
# ===================================================================

def build_fixed_linear_bias(
    T_audio: int, L_text: int,
    lyric_pos: torch.Tensor, lyric_mask: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    audio_idx = torch.arange(T_audio, device=device)
    progress = audio_idx.float() / max(T_audio - 1, 1)
    dist = lyric_pos.unsqueeze(0) - progress.unsqueeze(1)
    bias = -1.0 * (dist / 0.15) ** 2
    bias = bias.clamp(min=-3.0, max=0.0)
    bias = bias * lyric_mask.unsqueeze(0).float()
    bias = bias.unsqueeze(0).unsqueeze(0)
    return bias.contiguous()


# ===================================================================
#  PART 5 — Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="PMDC Dataset Generation Ablation — Stage 3")
    parser.add_argument("--audio-dir", default="/root/autodl-tmp/musicdata/audios")
    parser.add_argument("--dataset-dir", default="/root/autodl-tmp/musicdata/dataset")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=180)
    parser.add_argument("--max-duration", type=float, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234])
    parser.add_argument("--duration", type=int, default=240, help="Target duration in seconds")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["baseline", "fixed_linear", "duration_weak",
                                 "duration_mid", "duration_strong"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/pmdc_stage3_generation")
    parser.add_argument("--seed", type=int, default=42, help="Sample selection seed")
    parser.add_argument("--max-samples-to-skip", type=int, default=5,
                        help="Skip samples without captions up to this many")
    # Stage 3.5 — Control lines & gate fadeout
    parser.add_argument("--control-line-mode", type=str, default="intro_outro",
                        choices=["none", "intro_outro", "intro_outro_break"],
                        help="Natural control line insertion mode")
    parser.add_argument("--intro-control-ratio", type=float, default=0.025,
                        help="Target intro control duration fraction")
    parser.add_argument("--outro-control-ratio", type=float, default=0.035,
                        help="Target outro control duration fraction")
    parser.add_argument("--max-control-ratio", type=float, default=0.08,
                        help="Max total control duration fraction")
    parser.add_argument("--auto-insert-breaks", action="store_true",
                        help="Auto-insert [Instrumental Break] at transitions")
    parser.add_argument("--gate-fadeout", action="store_true",
                        help="Enable denoising gate fadeout")
    parser.add_argument("--gate-fadeout-start", type=float, default=0.60)
    parser.add_argument("--gate-fadeout-end", type=float, default=0.85)
    parser.add_argument("--final-gate", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.03,
                        help="Duration bias sigma (default for Stage 3.5)")
    parser.add_argument("--lambda", type=float, dest="lambda_", default=0.5,
                        help="Duration bias lambda")
    parser.add_argument("--gate", type=float, default=0.35,
                        help="Duration bias gate")
    parser.add_argument("--max-bias", type=float, default=1.0,
                        help="Duration bias max clamp")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save mode configs
    (OUTPUT_DIR / "configs").mkdir(exist_ok=True)
    for name, cfg in [("duration_weak", WEAK_CONFIG), ("duration_mid", MID_CONFIG),
                       ("duration_strong", STRONG_CONFIG)]:
        with open(OUTPUT_DIR / "configs" / f"{name}.json", "w") as f:
            json.dump(cfg, f, indent=2)

    audio_dir = Path(args.audio_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    # Limit modes
    active_modes = [m for m in MODES if m in args.modes]
    print(f"Active modes: {active_modes}")

    # ---- Step 1: Scan audio files and find captions ----
    print("\n[1/5] Scanning audio & finding captions...", flush=True)
    all_audio = scan_audio_files(audio_dir)
    audio_durations = {}
    for f in all_audio:
        d = get_audio_duration_ffprobe(f)
        if d is not None:
            audio_durations[f.name] = d

    # Find samples with lyrics + caption
    candidates = []
    for f in all_audio:
        dur = audio_durations.get(f.name)
        if dur is None or not (args.min_duration <= dur <= args.max_duration):
            continue
        meta = find_metadata_files(f.stem, audio_dir, dataset_dir)
        lyrics = read_lyrics(meta)
        caption = read_caption(meta)
        if lyrics and caption:
            candidates.append({"sample_id": f.stem, "audio_path": str(f),
                               "duration": dur, "lyrics": lyrics, "caption": caption})
    print(f"  {len(candidates)} samples with lyrics + caption")

    if len(candidates) < 1:
        print("  No candidates. Aborting.")
        sys.exit(1)

    selected = random.sample(candidates, min(args.num_samples, len(candidates)))
    print(f"  Selected {len(selected)} samples for generation")

    # ---- Step 2: Init models ----
    print("\n[2/5] Initializing models...", flush=True)
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit_handler.model.eval()
    model.config.use_section_rope_offset = False
    for lm in model.decoder.layers:
        if getattr(lm, "use_section_rope", False):
            lm.use_section_rope = False
        if getattr(lm, "use_phase_memory", False):
            lm.use_phase_memory = False
    print(f"  Model loaded: {type(model).__name__}")

    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device=args.device,
    )
    print(f"  LLM loaded")

    # ---- Step 3: For each sample, parse lyrics, build scaffold, generate ----
    print(f"\n[3/5] Generating up to {len(selected) * len(active_modes) * len(args.seeds)} audio files...", flush=True)

    skipped_log = []
    summary_rows = []

    for idx, sample in enumerate(selected):
        sid = sample["sample_id"]
        lyrics = sample["lyrics"]
        caption = sample["caption"]
        duration_s = min(sample["duration"], float(args.duration))
        print(f"\n  Sample [{idx}] {sid}  (dur={duration_s:.0f}s, caption={caption[:60]}...)", flush=True)

        sample_dir = OUTPUT_DIR / f"sample_{idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # ---- Parse lyrics (multiple variants for different modes) ----
        parser = LyricsStructureParser()
        parsed = parser.parse(lyrics, num_chunks=128)
        section_ids = parsed.section_type_ids

        # 1. Clean parser (tag-aware)
        units_clean, token_pos, debug_info = parse_lyrics_to_units(lyrics, section_ids)

        # 2. Clean parser + natural control lines
        control_lyrics, control_info = insert_control_lines(
            lyrics, intro_ratio=args.intro_control_ratio,
            outro_ratio=args.outro_control_ratio,
            auto_insert_breaks=args.auto_insert_breaks,
        )
        if control_info.get("intro_inserted") or control_info.get("outro_inserted"):
            parsed_ctrl = parser.parse(control_lyrics, num_chunks=128 if len(control_lyrics) < 5000 else 192)
            units_control, _, debug_ctrl = parse_lyrics_to_units(control_lyrics, parsed_ctrl.section_type_ids)
        else:
            units_control = list(units_clean)
            debug_ctrl = dict(debug_info)

        # 3. Old parser (uniform split, all tokens lyric)
        units_old, _ = old_parse_units(lyrics, parsed.section_type_ids, L=769)

        use_units = units_clean
        use_debug = debug_info
        # Determine which unit set to use (will be overridden per-mode if needed)

        if len(units_clean) == 0:
            print(f"    Skip: no lyric units")
            skipped_log.append({"sample_id": sid, "reason": "no_lyric_units"})
            continue

        T_target = int(duration_s * 50)

        # Build PMDC contexts
        tag_cm_clean = debug_info.get("tag_control_mask", None)
        tag_cm_ctrl = debug_ctrl.get("tag_control_mask", None)

        pmdc_ctx_clean = PMDCInjectionContext(model, units_clean, 777, T_target,
                                               device=args.device, tag_control_mask=tag_cm_clean)
        pmdc_ctx_ctrl = PMDCInjectionContext(model, units_control, 777, T_target,
                                              device=args.device, tag_control_mask=tag_cm_ctrl)
        pmdc_ctx_old = PMDCInjectionContext(model, units_old, 777, T_target, device=args.device)

        # Save lyric_units.txt (from best available units)
        unit_lines = [f"# Lyric Units — {sid}", f"# Duration: {duration_s}s",
                       f"# Clean units: {len(units_clean)} (lyric={debug_info.get('n_lyric_units', 0)}, "
                       f"silence={debug_info.get('n_silence_units', 0)})",
                       f"# Old units: {len(units_old)} (all lyric)",
                       f"# Control lines inserted: {control_info.get('intro_inserted', False)} intro, "
                       f"{control_info.get('outro_inserted', False)} outro",
                       ""]
        # Add detailed unit info
        for u in units_clean:
            unit_lines.append(f"  [{u.unit_id:<3}] {u.section:<12} L={str(u.is_lyric):5} C={str(u.is_control):5} "
                              f"chars={u.char_count:<4} tokens={len(u.token_indices)} {u.text[:50]}")
        (sample_dir / "lyric_units.txt").write_text("\n".join(unit_lines))

        # ---- Generate per seed ----
        for seed in args.seeds:
            for mode in active_modes:
                out_dir = sample_dir / f"seed_{seed}" / mode.replace(" ", "_").lower()
                out_dir.mkdir(parents=True, exist_ok=True)

                config_info = {"sample_id": sid, "seed": seed, "mode": mode,
                               "duration": duration_s, "caption": caption}

                # Determine mode configuration
                mode_gate = 0.0
                install_mode = None
                if mode == "fixed_linear":
                    install_mode = "fixed_linear"
                    mode_gate = 0.3
                    config_info.update({"gate": 0.3, "lambda": 1.0, "sigma": 0.15, "max_bias": 3.0})
                elif mode == "duration_weak":
                    install_mode = "duration_weak"
                    mode_gate = 0.5
                    config_info.update({**WEAK_CONFIG})
                elif mode == "duration_mid":
                    install_mode = "duration_mid"
                    mode_gate = 0.5
                    config_info.update({**MID_CONFIG})
                elif mode == "duration_strong":
                    install_mode = "duration_strong"
                    mode_gate = 0.5
                    config_info.update({**STRONG_CONFIG})
                elif mode == "old_parser_duration_weak":
                    install_mode = "duration_weak"
                    mode_gate = args.gate
                    config_info.update({"gate": args.gate, "sigma": args.sigma, "lambda_": args.lambda_, "max_bias": args.max_bias})
                elif mode == "clean_parser_no_control":
                    install_mode = "duration_weak"
                    mode_gate = args.gate
                    config_info.update({"gate": args.gate, "sigma": args.sigma, "lambda_": args.lambda_, "max_bias": args.max_bias})
                elif mode.startswith("clean_parser_intro_outro"):
                    install_mode = "duration_weak"
                    fadeout = "fadeout" in mode
                    mode_gate = args.gate
                    config_info.update({"gate": args.gate, "sigma": args.sigma, "lambda_": args.lambda_,
                                        "max_bias": args.max_bias, "fadeout": fadeout})

                # Select PMDC context and lyrics per mode
                if mode == "old_parser_duration_weak":
                    ctx = pmdc_ctx_old
                    mode_lyrics = lyrics
                elif mode.startswith("clean_parser_intro_outro"):
                    ctx = pmdc_ctx_ctrl
                    mode_lyrics = control_lyrics
                elif mode == "clean_parser_no_control":
                    ctx = pmdc_ctx_clean
                    mode_lyrics = lyrics
                else:
                    ctx = pmdc_ctx_clean
                    mode_lyrics = lyrics

                # Save config
                with open(out_dir / "config.json", "w") as f:
                    json.dump(config_info, f, indent=2)

                # Generate
                output_flac = out_dir / "audio.flac"
                if output_flac.is_file():
                    print(f"    {mode} seed={seed}: already exists", flush=True)
                    summary_rows.append({"sample_id": sid, "seed": seed, "mode": mode,
                                         "result": "exists", "audio_path": str(output_flac)})
                    continue

                try:
                    params = GenerationParams(
                        task_type="text2music",
                        caption=caption,
                        lyrics=mode_lyrics,
                        instrumental=False,
                        duration=duration_s,
                        inference_steps=args.steps,
                        guidance_scale=7.0,
                        seed=seed,
                        thinking=True,
                        use_cot_metas=True,
                        use_cot_caption=True,
                        lm_temperature=0.75,
                    )
                    gen_config = GenerationConfig(
                        batch_size=1,
                        audio_format="flac",
                        use_random_seed=False,
                    )

                    # Install PMDC patch if not baseline
                    if install_mode is not None:
                        ctx.install(install_mode, mode_gate)

                    result = generate_music(
                        dit_handler=dit_handler,
                        llm_handler=llm_handler,
                        params=params,
                        config=gen_config,
                        save_dir=str(out_dir),
                    )

                    # Restore original attention
                    if install_mode is not None:
                        ctx.uninstall()

                    # Rename audio
                    audio_saved = False
                    for af in list(out_dir.glob("*.flac")) + list(out_dir.glob("*.wav")):
                        if "audio" not in af.name:
                            af.rename(out_dir / "audio.flac")
                            audio_saved = True
                        elif af.name == "audio.flac":
                            audio_saved = True
                    if not audio_saved and result.audios:
                        import shutil
                        for a in result.audios:
                            src = Path(a["path"])
                            if src.exists():
                                shutil.copy2(src, out_dir / "audio.flac")
                                break

                    if result.success:
                        print(f"    {mode} seed={seed}: generated", flush=True)
                        summary_rows.append({"sample_id": sid, "seed": seed, "mode": mode,
                                             "result": "success", "audio_path": str(out_dir / "audio.flac")})
                    else:
                        print(f"    {mode} seed={seed}: {str(result.error)[:80]}", flush=True)
                        summary_rows.append({"sample_id": sid, "seed": seed, "mode": mode,
                                             "result": "error", "audio_path": ""})

                except Exception as e:
                    try:
                        ctx.uninstall()
                    except Exception:
                        pass
                    print(f"    {mode} seed={seed}: error: {e}", flush=True)
                    traceback.print_exc()
                    summary_rows.append({"sample_id": sid, "seed": seed, "mode": mode,
                                         "result": "exception", "audio_path": ""})

    # ---- Step 4: Save summary ----
    print(f"\n[4/5] Saving summary...", flush=True)

    csv_path = OUTPUT_DIR / "generation_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary CSV -> {csv_path}")

    json_path = OUTPUT_DIR / "generation_summary.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary_rows, "config": {
            "num_samples": args.num_samples, "seeds": args.seeds,
            "modes": active_modes, "duration": args.duration,
            "steps": args.steps}}, f, indent=2)
    print(f"  Summary JSON -> {json_path}")

    # ---- Step 5: Manual eval sheet ----
    print(f"\n[5/5] Creating manual eval sheet...", flush=True)

    eval_rows = []
    for row in summary_rows:
        eval_rows.append({
            "sample_id": row.get("sample_id", ""),
            "seed": row.get("seed", ""),
            "mode": row.get("mode", ""),
            "audio_path": row.get("audio_path", ""),
            "prompt": "",
            "repeat_score": "",
            "skip_score": "",
            "chorus_timing_score": "",
            "bridge_recovery_score": "",
            "outro_score": "",
            "vocal_loudness_score": "",
            "pronunciation_score": "",
            "overall_score": "",
            "notes": "",
        })

    eval_path = OUTPUT_DIR / "manual_eval_sheet.csv"
    with open(eval_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"  Manual eval sheet -> {eval_path}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    success_count = sum(1 for r in summary_rows if r.get("result") == "success")
    error_count = sum(1 for r in summary_rows if r.get("result", "").startswith(("error", "exception")))
    existing_count = sum(1 for r in summary_rows if r.get("result") == "exists")
    print(f"  Success: {success_count}, Already existed: {existing_count}, Errors: {error_count}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Per-mode stats
    for mode in active_modes:
        mode_rows = [r for r in summary_rows if r.get("mode") == mode]
        s = sum(1 for r in mode_rows if r.get("result") == "success")
        e = sum(1 for r in mode_rows if r.get("result") == "exists")
        f_count = sum(1 for r in mode_rows if r.get("result", "").startswith(("error", "exception")))
        print(f"  {mode:20s}: success={s}, existing={e}, errors={f_count}")

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
