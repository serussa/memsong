#!/usr/bin/env python3
"""
Experiment 3: RoPE Frequency Band Ablation.

Selectively masks specific frequency bands of the RoPE embedding to identify
which frequencies are critical for maintaining attention quality.

Conditions:
  - High:     keep first 16 pairs (dim 0-31)
  - Mid-High: keep pairs 16-32 (dim 32-63)
  - Mid-Low:  keep pairs 32-48 (dim 64-95)
  - Low:      keep last 16 pairs (dim 96-127)
  - Full:     keep all 64 pairs (baseline)
  - NoRoPE:   mask all pairs

Usage:
    python scripts/exp_rope_band_ablation.py [--duration 60]
"""

import sys, os, json, math, copy
from pathlib import Path

import torch
import numpy as np

SRC = Path("/root/ACE-Step-1.5")
OUT = SRC / "output" / "rope_exp3"
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(SRC))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

sys.path.insert(0, str(SRC / "scripts"))
from exp_rope_analyze import compute_attention_entropy, compute_receptive_field

# Band definitions: (name, start_pair, end_pair)
# head_dim=128 -> 64 frequency pairs, each pair controls 2 dims
BANDS = [
    ("High",     0,  16),   # dims 0-31   (local positional precision)
    ("MidHigh",  16, 32),   # dims 32-63
    ("MidLow",   32, 48),   # dims 64-95
    ("Low",      48, 64),   # dims 96-127 (long-range signal)
    ("Full",     0,  64),   # all pairs (baseline)
    ("NoRoPE",   0,   0),   # no position info
]


class BandAblationProfiler:
    """Apply band mask to rotary_emb and hook attention."""

    def __init__(self, model, band_name: str, start_pair: int, end_pair: int,
                 capture_every: int = 5):
        self.band_name = band_name
        self.capture_every = capture_every
        self.stats = {}
        self.diff_step = [0]
        self._patch_rotary_emb(model, start_pair, end_pair)
        self._register_hooks(model)

    def _patch_rotary_emb(self, model, start_pair: int, end_pair: int):
        """Monkey-patch rotary_emb.forward to mask out-of-band frequencies."""
        # Use the ORIGINAL forward saved before any profiler modifications
        orig_forward = getattr(model.decoder.rotary_emb, '_original_forward', model.decoder.rotary_emb.forward)
        if not hasattr(model.decoder.rotary_emb, '_original_forward'):
            model.decoder.rotary_emb._original_forward = orig_forward
        orig_forward = model.decoder.rotary_emb._original_forward
        head_dim = 128  # 64 pairs, each pair = 2 dims

        # Create band mask: [64] boolean
        band_mask = torch.zeros(head_dim // 2, dtype=torch.bool)
        if end_pair > start_pair:
            band_mask[start_pair:end_pair] = True

        def masked_forward(x, position_ids):
            cos, sin = orig_forward(x, position_ids)
            # cos/sin shape: [B, T, D] where D = head_dim
            if self.band_name == "NoRoPE":
                # No position info at all: identity rotation (keep content, no rotation)
                cos = torch.ones_like(cos)
                sin = torch.zeros_like(sin)
            elif self.band_name != "Full":
                # Expand mask to full dims: each pair controls 2 consecutive dims
                dim_mask = band_mask.repeat_interleave(2).to(cos.device)  # [128] bool
                dim_mask = dim_mask.view(1, 1, head_dim)
                # Kept bands: original cos/sin (full position info)
                # Masked bands: cos=1, sin=0 (identity - keep content, no rotation)
                cos = torch.where(dim_mask, cos, torch.ones_like(cos))
                sin = torch.where(dim_mask, sin, torch.zeros_like(sin))
            return cos, sin

        model.decoder.rotary_emb.forward = masked_forward
        # Store for reference
        self.band_mask = band_mask

    def _register_hooks(self, model):
        decoder = model.decoder
        orig_decoder_fwd = decoder.forward
        def patched_decoder_fwd(*args, **kwargs):
            self.diff_step[0] += 1
            return orig_decoder_fwd(*args, **kwargs)
        decoder.forward = patched_decoder_fwd

        for layer_idx, layer in enumerate(decoder.layers):
            self._hook_self_attn(layer_idx, layer)

    def _hook_self_attn(self, layer_idx, layer):
        orig_fwd = layer.self_attn.forward
        def patched_fwd(*args, **kwargs):
            kwargs["output_attentions"] = True
            output = orig_fwd(*args, **kwargs)
            attn_weights = output[1]
            step = self.diff_step[0]
            if attn_weights is not None and (step % self.capture_every == 0):
                attn = attn_weights.detach().cpu().float()
                entropy = compute_attention_entropy(attn)
                rf = compute_receptive_field(attn)
                self.stats[(layer_idx, step)] = {
                    "entropy": entropy,
                    "receptive_field": rf,
                    "seq_len": attn.shape[-1],
                }
            return output
        layer.self_attn.forward = patched_fwd


def run_band_condition(
    dit, llm, band_name: str, start_pair: int, end_pair: int,
    duration: float, steps: int, seed: int, output_dir: Path,
) -> dict:
    """Run generation with band mask and capture attention stats."""
    band_out = output_dir / f"band_{band_name}"
    band_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Band: {band_name} (pairs {start_pair}-{end_pair})")
    print(f"{'='*60}")

    # Force eager
    dit.model.config._attn_implementation = "eager"
    for mod in dit.model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "eager"

    profiler = BandAblationProfiler(dit.model, band_name, start_pair, end_pair, capture_every=5)

    # Report
    n_patches = int(duration * 25 // 2)
    print(f"  Patches: {n_patches}")

    params = GenerationParams(
        task_type="text2music",
        caption="pop, female vocal, piano, guitar, drums, bass, 120 bpm, C major, emotional",
        lyrics="""[Verse]
漫天的星光 照亮了夜晚
微风轻轻吹 带来你的温暖
走过的路上 花开又花落
每一刻都是 最美的时光

[Chorus]
你就是我心中 最亮的光
照亮所有黑暗 不再迷茫
不管未来多远 路有多长
握紧你的手 一起飞翔

[Verse]
城市的灯火 闪烁着希望
你的微笑 是我最暖的阳光
风雨中前行 有你陪在身旁
每一天都是 最美的篇章

[Chorus]
你就是我心中 最亮的光
照亮所有黑暗 不再迷茫
不管未来多远 路有多长
握紧你的手 一起飞翔

[Bridge]
时光流转 不会改变
这份爱永远 在心间
就算世界 沧海桑田
你依然是我 最亮的星

[Chorus]
你就是我心中 最亮的光
照亮所有黑暗 不再迷茫
不管未来多远 路有多长
握紧你的手 一起飞翔""",
        instrumental=False,
        bpm=120,
        keyscale="C major",
        timesignature="4",
        vocal_language="zh",
        duration=duration,
        inference_steps=steps,
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
        seeds=[seed],
    )

    result = generate_music(
        dit_handler=dit,
        llm_handler=llm,
        params=params,
        config=gen_config,
        save_dir=str(band_out),
    )

    # Add analysis
    n_captured = len(profiler.stats)
    print(f"  Captured {n_captured} (layer,step) entries")

    # Save audio path for later comparison
    audio_paths = []
    if result.success and result.audios:
        audio_paths = [a['path'] for a in result.audios]
        print(f"  Audio: {audio_paths[0]}")

    if n_captured == 0:
        return {"band": band_name, "n_captured": 0, "error": "no attention data"}

    full_deltas = []
    sliding_deltas = []
    all_entropies = {"first10pct": [], "last10pct": [], "global": []}

    for (layer_idx, step), data in profiler.stats.items():
        ent = data["entropy"].mean(dim=(0, 1)).numpy()
        T = len(ent)
        head = ent[:max(1, T//10)].mean()
        tail = ent[-max(1, T//10):].mean()
        delta = tail - head
        if layer_idx % 2 == 0:
            sliding_deltas.append(delta)
        else:
            full_deltas.append(delta)
        all_entropies["first10pct"].append(head)
        all_entropies["last10pct"].append(tail)
        all_entropies["global"].append(data["entropy"].mean().item())

    summary = {
        "band": band_name,
        "n_captured": n_captured,
        "avg_full_delta": float(np.mean(full_deltas)) if full_deltas else None,
        "avg_sliding_delta": float(np.mean(sliding_deltas)) if sliding_deltas else None,
        "avg_global_entropy": float(np.mean(all_entropies["global"])),
        "avg_head_entropy": float(np.mean(all_entropies["first10pct"])),
        "avg_tail_entropy": float(np.mean(all_entropies["last10pct"])),
        "audio_path": audio_paths[0] if audio_paths else None,
    }

    json_path = band_out / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    stats_pt = {}
    for key, data in profiler.stats.items():
        stats_pt[f"L{key[0]}_S{key[1]}"] = {
            "entropy": data["entropy"],
            "receptive_field": data["receptive_field"],
        }
    torch.save(stats_pt, band_out / "stats.pt")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--long-duration", type=float, default=0.0,
                        help="Second pass at longer duration (e.g. 270). If 0, only runs --duration.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    durations = [d for d in [args.duration, args.long_duration] if d > 0]

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 3: Frequency Band Ablation")
    print(f"  Durations: {durations}s")
    print(f"  Bands: {[b[0] for b in BANDS]}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit.model.eval()
    model.config.use_section_rope_offset = False
    model.config._attn_implementation = "eager"
    for l in model.decoder.layers:
        if getattr(l, "use_section_rope", False):
            l.use_section_rope = False
        if getattr(l, "use_phase_memory", False):
            l.use_phase_memory = False

    print("[2/3] Initializing 5Hz LM...")
    llm = LLMHandler()
    ok = llm.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not ok:
        print("  FAILED")
        sys.exit(1)
    print("  OK")

    # Run each band at each duration
    print("[3/3] Running band ablation...")
    all_results = {}  # {f"{band}_{dur}": summary}

    for dur in durations:
        # Use same number of steps regardless of duration (fair comparison)
        steps = args.steps
        per_dur_out = OUT / f"dur_{int(dur)}s"
        per_dur_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"# Duration = {int(dur)}s ({int(dur * 25 // 2)} patches), steps = {steps}")
        print(f"{'#'*60}")

        # Restore original rotary_emb at start of each duration block
        if hasattr(dit.model.decoder.rotary_emb, '_original_forward'):
            dit.model.decoder.rotary_emb.forward = dit.model.decoder.rotary_emb._original_forward

        for band_name, start, end in BANDS:
            summary = run_band_condition(
                dit, llm, band_name, start, end,
                dur, steps, args.seed, per_dur_out,
            )
            all_results[f"{band_name}_{int(dur)}s"] = summary
            print(f"  Result: full_delta={summary.get('avg_full_delta', 'N/A')}, "
                  f"head={summary.get('avg_head_entropy', 'N/A'):.4f}, "
                  f"tail={summary.get('avg_tail_entropy', 'N/A'):.4f}")

        # Summary table per duration
        print(f"\n{'='*60}")
        print(f"BAND ABLATION SUMMARY ({int(dur)}s)")
        print(f"{'='*60}")
        print(f"{'Band':<10} {'FullDelta':>10} {'SlidDelta':>10} {'HeadEnt':>8} {'TailEnt':>8}")
        print("-" * 46)
        for band_name, _, _ in BANDS:
            s = all_results.get(f"{band_name}_{int(dur)}s", {})
            if s.get("n_captured", 0) == 0:
                print(f"{band_name:<10} {'N/A':>10}")
                continue
            print(f"{band_name:<10} {s.get('avg_full_delta', 0):>+10.4f} "
                  f"{s.get('avg_sliding_delta', 0):>+10.4f} "
                  f"{s.get('avg_head_entropy', 0):>8.4f} "
                  f"{s.get('avg_tail_entropy', 0):>8.4f}")
        print(f"{band_name:<10} {s.get('avg_full_delta', 0):>+10.4f} "
              f"{s.get('avg_sliding_delta', 0):>+10.4f} "
              f"{s.get('avg_head_entropy', 0):>8.4f} "
              f"{s.get('avg_tail_entropy', 0):>8.4f}")

    combined = {}
    for k, v in all_results.items():
        combined[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, type(None)))}
    json_path = OUT / "band_ablation_results.json"
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"\nResults saved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
