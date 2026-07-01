#!/usr/bin/env python3
"""
PMDC attention debug — teacher-forcing attention trace analysis.

Loads the baseline model, runs teacher-forcing forward on a sample,
captures cross-attention logits and values, then applies each PMDC
mode's attention patch and analyses:

  - lyric_mass vs control_mass over time
  - attention centroid trajectory
  - control interval mass allocation
  - per-step KL divergence vs baseline

Usage:
    python scripts/pmdc_attention_debug.py \\
        --sample-id 0 \\
        --output-dir /root/autodl-tmp/pmdc_attention_debug
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
    LyricUnit, parse_lyrics_to_units, build_duration_scaffold,
    build_duration_interval_bias, mass_preserving_attention,
    insert_control_lines, is_natural_control_line,
    get_scheduled_gate,
)

MODE_CONFIGS = {
    "baseline": {},
    "clean_parser_no_control": {"sigma": 0.03, "lambda_": 0.5, "gate": 0.35, "max_bias": 1.0},
    "clean_parser_intro_outro_control": {"sigma": 0.03, "lambda_": 0.5, "gate": 0.35, "max_bias": 1.0},
    "clean_parser_intro_outro_control_fadeout": {"sigma": 0.03, "lambda_": 0.5, "gate": 0.35, "max_bias": 1.0},
}

WEAK_CONFIG = {"sigma": 0.03, "lambda_": 0.5, "gate": 0.5, "max_bias": 1.0}


def setup_model(device="cuda"):
    dt = AceStepHandler()
    dt.initialize_service(
        project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
        device=device, use_flash_attention=False, compile_model=False, offload_to_cpu=False,
    )
    model = dt.model.eval()
    model.config.use_section_rope_offset = False
    for lm in model.decoder.layers:
        if getattr(lm, "use_section_rope", False): lm.use_section_rope = False
        if getattr(lm, "use_phase_memory", False): lm.use_phase_memory = False
    return model


class HiddenCollector:
    def __init__(self): self.hidden = []
    def __call__(self, m, i, o): self.hidden.append(o[0].detach().cpu())
    def get(self): return torch.cat(self.hidden, dim=0) if self.hidden else None


class LogitValueCollector:
    def __init__(self):
        self.logits = None; self.value = None; self._orig_fn = None; self._patched_mod = None
    def _make_patched_fn(self):
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        c = self
        def fn(*args, **kwargs):
            # Extract args flexibly
            mod = args[0]
            q = args[1]
            k = args[2]
            v = args[3]
            am = args[4] if len(args) > 4 else kwargs.get("attention_mask")
            sc = kwargs.get("scaling", kwargs.get("sc", args[5] if len(args) > 5 else None))
            dr = kwargs.get("dropout", 0.0)
            ks = repeat_kv(k, mod.num_key_value_groups)
            vs = repeat_kv(v, mod.num_key_value_groups)
            aw = torch.matmul(q, ks.transpose(2, 3)) * sc
            if am is not None and isinstance(am, torch.Tensor):
                causal_mask = am[:, :, :, :ks.shape[-2]]
                aw = aw + causal_mask
            c.logits = aw.detach().cpu(); c.value = vs.detach().cpu()
            aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
            aw = F.dropout(aw, p=dr, training=mod.training)
            return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw
        return fn
    def install(self, model, layer=12):
        import sys
        ca = model.decoder.layers[layer].cross_attn
        mod = sys.modules[type(ca).__module__]
        self._orig_fn = getattr(mod, "eager_attention_forward", None)
        self._patched_mod = mod
        setattr(mod, "eager_attention_forward", self._make_patched_fn())
    def uninstall(self):
        if self._patched_mod and self._orig_fn:
            setattr(self._patched_mod, "eager_attention_forward", self._orig_fn)
        self._patched_mod = None; self._orig_fn = None


def run_teacher_forward(model, pt_data, device="cuda"):
    md = next(model.parameters()).dtype; mv = next(model.parameters()).device
    hs = pt_data["target_latents"].unsqueeze(0).to(device=mv, dtype=md)
    am = pt_data["attention_mask"].unsqueeze(0).to(device=mv, dtype=md)
    eh = pt_data["encoder_hidden_states"].unsqueeze(0).to(device=mv, dtype=md)
    ea = pt_data["encoder_attention_mask"].unsqueeze(0).to(device=mv, dtype=md)
    ctx = pt_data["context_latents"].unsqueeze(0).to(device=mv, dtype=md)
    tt = torch.full((1,), 0.0, device=mv, dtype=md)
    with torch.no_grad():
        model.decoder(hidden_states=hs, timestep=tt, timestep_r=tt,
                      attention_mask=am, encoder_hidden_states=eh,
                      encoder_attention_mask=ea, context_latents=ctx,
                      use_cache=False, output_attentions=True)


def load_preprocessed_data(pt_path):
    pt_name = pt_path.name
    if pt_path.is_file():
        try:
            d = torch.load(str(pt_path), weights_only=True, map_location="cpu")
            return d
        except: return None
    if TENSOR_TAR_PATH.is_file():
        import tarfile, tempfile
        try:
            with tarfile.open(str(TENSOR_TAR_PATH), "r") as tar:
                for name in tar.getnames():
                    if name.split("/")[-1] == pt_name:
                        m = tar.getmember(name)
                        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                            with tar.extractfile(m) as src: tmp.write(src.read())
                        d = torch.load(tmp.name, weights_only=True, map_location="cpu")
                        os.unlink(tmp.name)
                        return d
        except: pass
    return None


def get_audio_duration_ffprobe(path):
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip(): return float(r.stdout.strip())
    except: pass
    return None


def compute_attention_trace(
    logits: torch.Tensor, value: torch.Tensor,
    lyric_mask: torch.Tensor,
    control_mask: Optional[torch.Tensor],
    attendable_mask: Optional[torch.Tensor],
    bias: Optional[torch.Tensor], gate: float, T_eff: int,
) -> Dict:
    # Move masks to same device as logits
    device = logits.device
    lyric_mask = lyric_mask.to(device)
    if control_mask is not None:
        control_mask = control_mask.to(device)
    if attendable_mask is not None:
        attendable_mask = attendable_mask.to(device)
    """Run text-mass-preserving attention and return traces."""
    B, H, T, L = logits.shape

    def _expand(m):
        m2 = m.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return m2.expand(B, H, T, -1).bool()

    lm = _expand(lyric_mask)
    if control_mask is not None: cm = _expand(control_mask)
    else: cm = None

    if attendable_mask is not None: am = _expand(attendable_mask)
    else: am = lm

    am_f = am.float()
    lm_f = lm.float()

    # Baseline
    attn_base = F.softmax(logits, dim=-1, dtype=torch.float32)
    lyric_mass_base = (attn_base * lm_f).sum(dim=-1)  # [B, H, T]
    text_mass_base = (attn_base * am_f).sum(dim=-1)

    # Apply bias
    if bias is not None:
        bias_4d = bias.unsqueeze(1)
        biased = logits + gate * bias_4d
    else:
        biased = logits

    # Text split
    text_logits = biased.masked_fill(~am, float("-inf"))
    attn_text = F.softmax(text_logits, dim=-1, dtype=torch.float32)
    attn_text = attn_text.masked_fill(~am, 0.0)

    non_text_logits = logits.masked_fill(am, float("-inf"))
    attn_non = F.softmax(non_text_logits, dim=-1, dtype=torch.float32)
    attn_non = attn_non.masked_fill(am, 0.0)

    attn_new = attn_text * text_mass_base.unsqueeze(-1) + \
               attn_non * (1.0 - text_mass_base).unsqueeze(-1)
    attn_new = attn_new / (attn_new.sum(dim=-1, keepdim=True) + 1e-10)

    # Traces (mean over heads)
    text_mass = (attn_new * am_f).sum(dim=-1).mean(dim=(0, 1))  # [T]
    lyric_mass = (attn_new * lm_f).sum(dim=-1).mean(dim=(0, 1))
    if cm is not None: control_mass = (attn_new * cm.float()).sum(dim=-1).mean(dim=(0, 1))
    else: control_mass = torch.zeros(T)

    # KL vs baseline
    eps = 1e-10
    kl = (attn_new * ((attn_new + eps).log() - (attn_base + eps).log())).sum(dim=-1).mean(dim=(0, 1))

    return {
        "attn_new": attn_new[0].cpu().numpy(),
        "text_mass": text_mass.cpu().numpy(),
        "lyric_mass": lyric_mass.cpu().numpy(),
        "control_mass": control_mass.cpu().numpy(),
        "kl_vs_baseline": kl.cpu().numpy(),
        "lyric_mass_base_mean": (attn_base * lm_f).sum(dim=-1).mean().item(),
        "lyric_mass_new_mean": lyric_mass.mean().item(),
        "lyric_mass_delta": (lyric_mass.mean() - (attn_base * lm_f).sum(dim=-1).mean()).item(),
        "text_mass_base_mean": text_mass_base.mean().item(),
        "text_mass_new_mean": text_mass.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="PMDC attention debug")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/pmdc_attention_debug")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find sample
    audio_dir = Path("/root/autodl-tmp/musicdata/audios")
    all_audio = sorted(audio_dir.glob("*.mp3")) + sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac"))
    random.Random(42).shuffle(all_audio)
    candidates = []
    for f in all_audio:
        dur = get_audio_duration_ffprobe(f)
        if dur is None or not (180 <= dur <= 300): continue
        lrc = audio_dir / f"{f.stem}.lyrics.txt"
        cap = audio_dir / f"{f.stem}.caption.txt"
        if lrc.exists() and cap.exists():
            candidates.append((f, lrc, cap, dur))
    if not candidates:
        print("No candidates"); sys.exit(1)
    selected = min(args.sample_idx, len(candidates) - 1)
    audio_path, lrc_path, cap_path, duration = candidates[selected]
    sid = audio_path.stem
    lyrics = lrc_path.read_text()
    print(f"Sample: {sid} ({duration:.0f}s)")

    # Load model
    print("Loading model…", flush=True)
    model = setup_model(device=args.device)

    # Find preprocessed .pt
    tensor_dir = TENSOR_DIR_DEFAULT
    pt_path = tensor_dir / f"{sid}.pt"
    if not pt_path.is_file():
        base = sid.rsplit("_", 1)[0]
        for f in tensor_dir.glob(f"{base}_*.pt"):
            pt_path = f; break
    print(f"PT: {pt_path}")
    pt_data = load_preprocessed_data(pt_path)
    if pt_data is None:
        print("No pt data"); sys.exit(1)

    T_raw = pt_data["target_latents"].shape[0]

    # Forward + capture
    hc = HiddenCollector()
    lc = LogitValueCollector()
    hh = model.decoder.layers[12].register_forward_hook(hc)
    lc.install(model, layer=12)
    run_teacher_forward(model, pt_data, device=args.device)
    hh.remove(); lc.uninstall()

    H_t = hc.get(); logits_t = lc.logits; value_t = lc.value
    if H_t is None or logits_t is None or value_t is None:
        print("No hooks"); sys.exit(1)

    H_np = H_t.squeeze(0).float().numpy()
    logits_gpu = logits_t.float().to(args.device)
    values_gpu = value_t.float().to(args.device)
    T_eff, D = H_np.shape
    L_eff = logits_gpu.shape[-1]
    print(f"T={T_eff}, L={L_eff}")

    # Parse lyrics — clean + control variants
    parser = LyricsStructureParser()
    section_ids = parser.parse(lyrics, num_chunks=L_eff).section_type_ids

    units_clean, _, debug_clean = parse_lyrics_to_units(lyrics, section_ids)
    tcm_clean = debug_clean.get("tag_control_mask", None)
    scaffold_clean = build_duration_scaffold(units_clean, text_len=L_eff, tag_control_mask=tcm_clean)

    control_lyrics, _ = insert_control_lines(lyrics, intro_ratio=0.025, outro_ratio=0.035)
    section_ids_ctrl = parser.parse(control_lyrics, num_chunks=L_eff if len(control_lyrics) < 5000 else 256).section_type_ids
    units_ctrl, _, debug_ctrl = parse_lyrics_to_units(control_lyrics, section_ids_ctrl)
    tcm_ctrl = debug_ctrl.get("tag_control_mask", None)
    scaffold_ctrl = build_duration_scaffold(units_ctrl, text_len=scaffold_clean["token_to_unit"].shape[-1],
                                             tag_control_mask=tcm_ctrl)

    p_base = torch.linspace(0, 1, T_eff, device=args.device).float().unsqueeze(0)

    # Helper to move scaffold to device
    def to_device(d):
        return {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    sc_clean = to_device(scaffold_clean)
    sc_ctrl = to_device(scaffold_ctrl)

    # Run modes
    results = {}
    for mode_name in ["baseline", "clean_parser_no_control",
                       "clean_parser_intro_outro_control",
                       "clean_parser_intro_outro_control_fadeout"]:
        cfg = MODE_CONFIGS.get(mode_name, {})

        if mode_name == "baseline":
            bias = None; gate = 0.0
            lm = sc_clean["lyric_mask"]; cm = sc_clean.get("control_mask")
            am = sc_clean.get("attendable_mask")
        elif mode_name == "clean_parser_no_control":
            bias = build_duration_interval_bias(p_base, sc_clean["unit_boundaries"],
                       sc_clean["token_to_unit"], sc_clean["attendable_mask"],
                       sigma=cfg.get("sigma", 0.03), lambda_=cfg.get("lambda_", 0.5), max_bias=cfg.get("max_bias", 1.0))
            gate = cfg.get("gate", 0.35)
            lm = sc_clean["lyric_mask"]; cm = sc_clean.get("control_mask")
            am = sc_clean.get("attendable_mask")
        elif "intro_outro" in mode_name:
            bias = build_duration_interval_bias(p_base, sc_ctrl["unit_boundaries"],
                       sc_ctrl["token_to_unit"], sc_ctrl["attendable_mask"],
                       sigma=cfg.get("sigma", 0.03), lambda_=cfg.get("lambda_", 0.5), max_bias=cfg.get("max_bias", 1.0))
            gate = cfg.get("gate", 0.35)
            lm = sc_ctrl["lyric_mask"]; cm = sc_ctrl.get("control_mask")
            am = sc_ctrl.get("attendable_mask")

        trace = compute_attention_trace(logits_gpu, values_gpu, lm, cm, am, bias, gate, T_eff)
        trace["config"] = cfg
        trace["mode"] = mode_name
        results[mode_name] = trace

    # Print summary
    print("\n" + "=" * 70)
    print("ATTENTION TRACE SUMMARY")
    print("=" * 70)
    for mode, r in results.items():
        print(f"\n{mode}:")
        print(f"  lyric_mass:        {r['lyric_mass'].mean():.4f}  (min={r['lyric_mass'].min():.4f}, max={r['lyric_mass'].max():.4f})")
        if r['control_mass'] is not None and len(r['control_mass']) > 0 and r['control_mass'].sum() > 0:
            print(f"  control_mass:      {r['control_mass'].mean():.4f}  (min={r['control_mass'].min():.4f}, max={r['control_mass'].max():.4f})")
        print(f"  text_mass:         {r['text_mass'].mean():.4f}")
        print(f"  lyric_mass_delta:  {r['lyric_mass_delta']:.6f}")
        print(f"  text_mass_delta:   {r['lyric_mass_new_mean'] - r['text_mass_base_mean']:.6f}")
        if len(r['kl_vs_baseline']) > 0:
            print(f"  KL vs baseline:    {r['kl_vs_baseline'].mean():.6f}")

    # Find control intervals from scaffold_ctrl
    if 'control_units' in dir():
        pass
    control_unit_ids = [u for u in units_ctrl if u.is_silence and len(u.token_indices) > 0]
    print(f"\nControl units with tokens: {len(control_unit_ids)}")
    for u in control_unit_ids:
        idx = u.unit_id
        if idx < len(scaffold_ctrl["unit_boundaries"]) - 1:
            start = scaffold_ctrl["unit_boundaries"][idx].item()
            end = scaffold_ctrl["unit_boundaries"][idx + 1].item()
            mid = (start + end) / 2
            ct = int(mid * T_eff)
            # Check mass in control interval
            for mode, r in results.items():
                if mode == "baseline": continue
                if ct < len(r['lyric_mass']):
                    lm_val = r['lyric_mass'][max(0, ct - 5):min(T_eff, ct + 5)].mean()
                    cm_val = r.get('control_mass', None)
                    if cm_val is not None and ct < len(cm_val):
                        cm_val = cm_val[max(0, ct - 5):min(T_eff, ct + 5)].mean()
                        print(f"  [{u.section}] t≈{mid:.2f}  lyric_mass={lm_val:.4f}  control_mass={cm_val:.4f}")

    # Save JSON
    json_out = {}
    for mode, r in results.items():
        json_out[mode] = {
            "lyric_mass_mean": float(r['lyric_mass'].mean()),
            "lyric_mass_min": float(r['lyric_mass'].min()),
            "lyric_mass_max": float(r['lyric_mass'].max()),
            "text_mass_mean": float(r['text_mass'].mean()),
            "lyric_mass_delta": float(r['lyric_mass_delta']),
            "kl_mean": float(r['kl_vs_baseline'].mean()),
        }
        if r.get('control_mass') is not None and r['control_mass'].sum() > 0:
            json_out[mode]["control_mass_mean"] = float(r['control_mass'].mean())
    with open(OUTPUT_DIR / "attention_trace.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nJSON -> {OUTPUT_DIR / 'attention_trace.json'}")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(T_eff) / T_eff
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. Lyric mass over time
        ax = axes[0]
        for mode, r in results.items():
            lbl = mode.replace("_", " ")
            ax.plot(t, r['lyric_mass'], label=lbl, linewidth=1.2, alpha=0.8)
        # Mark control intervals
        for u in control_unit_ids:
            idx = u.unit_id
            if idx < len(scaffold_ctrl["unit_boundaries"]) - 1:
                st = scaffold_ctrl["unit_boundaries"][idx].item()
                en = scaffold_ctrl["unit_boundaries"][idx + 1].item()
                ax.axvspan(st, en, color='gray', alpha=0.15)
        ax.set_ylabel("Lyric mass")
        ax.set_title(f"Lyric attention mass over time — {sid}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # 2. Control mass over time
        ax = axes[1]
        for mode, r in results.items():
            if r.get('control_mass') is not None and r['control_mass'].sum() > 0:
                lbl = mode.replace("_", " ")
                ax.plot(t, r['control_mass'], label=lbl, linewidth=1.2)
        for u in control_unit_ids:
            idx = u.unit_id
            if idx < len(scaffold_ctrl["unit_boundaries"]) - 1:
                st = scaffold_ctrl["unit_boundaries"][idx].item()
                en = scaffold_ctrl["unit_boundaries"][idx + 1].item()
                ax.axvspan(st, en, color='gray', alpha=0.15)
        ax.set_ylabel("Control mass")
        ax.set_title("Control attention mass (tag/silence tokens)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # 3. KL divergence vs baseline
        ax = axes[2]
        for mode, r in results.items():
            if mode == "baseline": continue
            lbl = mode.replace("_", " ")
            ax.plot(t, r['kl_vs_baseline'], label=lbl, linewidth=1.2)
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("KL divergence")
        ax.set_title("Per-step KL vs baseline")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "attention_trace.png", dpi=150)
        print(f"Plot -> {OUTPUT_DIR / 'attention_trace.png'}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available, skipping plot")

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
