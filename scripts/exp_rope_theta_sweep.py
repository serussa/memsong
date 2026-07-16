#!/usr/bin/env python3
"""
Experiment 2: RoPE Base Frequency (Theta) Sweep.

Tests multiple theta values and measures their effect on attention
decay over long sequences. For each theta, generates audio (60s by default)
and captures attention statistics.

Usage:
    python scripts/exp_rope_theta_sweep.py [--duration 60] [--thetas 10000,100000,1000000,10000000]
"""

import sys, os, json, math, copy
from pathlib import Path

import torch
import numpy as np

SRC = Path("/root/ACE-Step-1.5")
OUT = SRC / "output" / "rope_exp2"
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(SRC))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

sys.path.insert(0, str(SRC / "scripts"))
from exp_rope_analyze import compute_attention_entropy, compute_receptive_field


class ThetaAttentionProfiler:
    """Hooks self-attention to capture entropy stats for a given theta config."""

    def __init__(self, model, theta: float, capture_every: int = 5):
        self.theta = theta
        self.capture_every = capture_every
        self.stats = {}          # {(layer_idx, step): {...}}
        self.diff_step = [0]
        self._patch_rotary_emb(model)
        self._register_hooks(model)

    def _patch_rotary_emb(self, model):
        """Replace decoder.rotary_emb with one using the given theta."""
        new_config = copy.deepcopy(model.decoder.config)
        new_config.rope_theta = self.theta
        # Create new rotary embedding with desired theta
        new_rotary = Qwen3RotaryEmbedding(new_config, device=model.device)
        model.decoder.rotary_emb = new_rotary
        self.rotary_emb = new_rotary

    def _register_hooks(self, model):
        decoder = model.decoder

        # Track diffusion step
        orig_decoder_fwd = decoder.forward
        def patched_decoder_fwd(*args, **kwargs):
            self.diff_step[0] += 1
            return orig_decoder_fwd(*args, **kwargs)
        decoder.forward = patched_decoder_fwd

        # Hook self-attention on all layers
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


def run_theta_condition(
    dit, llm, theta: float, duration: float, steps: int, seed: int,
    output_dir: Path,
) -> dict:
    """Run generation with given theta and capture attention stats."""
    theta_out = output_dir / f"theta_{theta:.0e}"
    theta_out.mkdir(parents=True, exist_ok=True)

    # Reset model for this theta
    print(f"\n{'='*60}")
    print(f"  Theta = {theta:>10,.0f}")
    print(f"{'='*60}")

    # Force eager mode
    dit.model.config._attn_implementation = "eager"
    for module in dit.model.modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "eager"

    # Profiler patches the rotary_emb
    profiler = ThetaAttentionProfiler(dit.model, theta, capture_every=5)

    # Report frequency status
    head_dim = 128
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    n_patches = int(duration * 25 // 2)
    cycles = (inv_freq * n_patches) / (2 * math.pi)
    first_cycle = (cycles < 1).sum().item()
    beyond_5 = (cycles >= 5).sum().item()
    print(f"  Patches: {n_patches}, first_cycle={first_cycle}/64, >5cyc={beyond_5}/64")

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
        save_dir=str(theta_out),
    )

    # Analyze
    n_captured = len(profiler.stats)
    print(f"  Captured {n_captured} (layer,step) entries")

    if n_captured == 0:
        return {"theta": theta, "n_captured": 0, "error": "no attention data"}

    # Aggregate across layers and steps
    full_deltas = []
    sliding_deltas = []
    all_entropies = {"first10pct": [], "last10pct": [], "global": []}
    steps_captured = sorted(set(s for _, s in profiler.stats.keys()))

    for step in steps_captured:
        for layer_idx in range(24):
            key = (layer_idx, step)
            if key not in profiler.stats:
                continue
            data = profiler.stats[key]
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
        "theta": theta,
        "n_captured": n_captured,
        "avg_full_delta": float(np.mean(full_deltas)) if full_deltas else None,
        "avg_sliding_delta": float(np.mean(sliding_deltas)) if sliding_deltas else None,
        "avg_global_entropy": float(np.mean(all_entropies["global"])),
        "avg_head_entropy": float(np.mean(all_entropies["first10pct"])),
        "avg_tail_entropy": float(np.mean(all_entropies["last10pct"])),
        "first_cycle_pairs": first_cycle,
        "beyond_5_cycles": beyond_5,
    }

    # Save
    json_path = theta_out / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Save raw stats
    stats_pt = {}
    for key, data in profiler.stats.items():
        stats_pt[f"L{key[0]}_S{key[1]}"] = {
            "entropy": data["entropy"],
            "receptive_field": data["receptive_field"],
        }
    torch.save(stats_pt, theta_out / "stats.pt")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--thetas", type=str, default="10000,100000,500000,1000000,5000000,10000000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    thetas = [float(x) for x in args.thetas.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment 2: Theta Sweep")
    print(f"  Duration: {args.duration}s per run")
    print(f"  Thetas: {[f'{t:.0e}' for t in thetas]}")
    print("=" * 60)

    # Load model once
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

    # Run each theta
    print("[3/3] Running theta sweep...")
    all_results = {}
    for theta in thetas:
        summary = run_theta_condition(dit, llm, theta, args.duration, args.steps, args.seed, OUT)
        all_results[f"{theta:.0e}"] = summary
        print(f"  Result: full_delta={summary.get('avg_full_delta', 'N/A')}, "
              f"head={summary.get('avg_head_entropy', 'N/A'):.4f}, "
              f"tail={summary.get('avg_tail_entropy', 'N/A'):.4f}")

    # Overall comparison
    print("\n" + "=" * 60)
    print("THETA SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'Theta':>12} {'FullDelta':>10} {'SlidDelta':>10} {'HeadEnt':>8} {'TailEnt':>8} {'1stCyc':>8} {'>5Cyc':>8}")
    print("-" * 64)
    for theta in thetas:
        key = f"{theta:.0e}"
        s = all_results.get(key, {})
        if s.get("n_captured", 0) == 0:
            print(f"{key:>12} {'N/A':>10}")
            continue
        print(f"{key:>12} {s.get('avg_full_delta', 0):>+10.4f} "
              f"{s.get('avg_sliding_delta', 0):>+10.4f} "
              f"{s.get('avg_head_entropy', 0):>8.4f} "
              f"{s.get('avg_tail_entropy', 0):>8.4f} "
              f"{s.get('first_cycle_pairs', 0):>8d} "
              f"{s.get('beyond_5_cycles', 0):>8d}")

    # Save combined results
    combined = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, type(None)))}
                for k, v in all_results.items()}
    json_path = OUT / "theta_sweep_results.json"
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"\nResults saved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
