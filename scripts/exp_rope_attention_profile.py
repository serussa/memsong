#!/usr/bin/env python3
"""
Experiment 1: Attention Decay Profiling.

Hooks into DiT self-attention layers during generation to measure
how attention patterns degrade at long positions.

Measures per position:
  - Attention entropy (higher = more blurred)
  - Effective receptive field (expected distance to attended keys)
  - Layer-wise comparison (shallow vs deep layers)
  - Sliding vs full attention comparison

Usage:
    python scripts/exp_rope_attention_profile.py [--duration 270]
"""

import sys, os, json, math, pickle
from pathlib import Path

import torch
import numpy as np

SRC = Path("/root/ACE-Step-1.5")
OUT = SRC / "output" / "rope_exp1"
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(SRC))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

sys.path.insert(0, str(SRC / "scripts"))
from exp_rope_analyze import compute_attention_entropy, compute_receptive_field


# ---------------------------------------------------------------------------
# Attention profiler
# ---------------------------------------------------------------------------

class AttentionProfiler:
    """Capture attention weights during diffusion and compute per-position stats.

    Only stores summary statistics (entropy, receptive field), not raw weights,
    to avoid OOM on long sequences.
    """

    def __init__(self, model, capture_every: int = 1):
        """
        Args:
            model: AceStepConditionGenerationModel
            capture_every: capture every N-th diffusion step (1=all, 5=every 5th)
        """
        self.model = model
        self.capture_every = capture_every
        self.stats = {}          # {(layer_idx, step): {'entropy': [B,H,T], 'receptive_field': [B,H,T], ...}}
        self.raw_weights = {}    # {(layer_idx, step): [B,H,T,S]} — only stored for early/late steps
        self.raw_capture_steps = set()  # which steps also store raw weights
        self.diff_step = [0]     # counter, mutable for closure
        self._register_hooks()

    def _register_hooks(self):
        """Monkey-patch decoder forward to track step, then hook each self-attn."""
        decoder = self.model.decoder

        # --- Track diffusion step via decoder entry count ---
        orig_decoder_fwd = decoder.forward

        def patched_decoder_fwd(*args, **kwargs):
            self.diff_step[0] += 1
            return orig_decoder_fwd(*args, **kwargs)

        decoder.forward = patched_decoder_fwd

        # --- Hook each layer's self-attention ---
        for layer_idx, layer in enumerate(decoder.layers):
            self._hook_self_attn(layer_idx, layer)

    def _hook_self_attn(self, layer_idx, layer):
        orig_fwd = layer.self_attn.forward

        def patched_fwd(*args, **kwargs):
            kwargs["output_attentions"] = True
            output = orig_fwd(*args, **kwargs)
            # output = (attn_output, attn_weights)
            attn_weights = output[1]
            step = self.diff_step[0]

            if attn_weights is not None and (step % self.capture_every == 0):
                attn = attn_weights.detach().cpu().float()  # [B, H, T, S]
                entropy = compute_attention_entropy(attn)   # [B, H, T]
                rf = compute_receptive_field(attn)          # [B, H, T]
                self.stats[(layer_idx, step)] = {
                    "entropy": entropy,
                    "receptive_field": rf,
                    "seq_len": attn.shape[-1],
                }
                # Also store raw weights for first/last capture
                if step in self.raw_capture_steps:
                    self.raw_weights[(layer_idx, step)] = attn

            return output

        layer.self_attn.forward = patched_fwd

    def set_raw_capture_steps(self, steps):
        self.raw_capture_steps = set(steps)

    @property
    def total_diff_steps(self):
        return self.diff_step[0]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_attention_trends(stats, total_layers: int, output_dir: Path):
    """Generate plots showing attention decay across positions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    # Separate stats by full vs sliding attention layers
    full_stats = {}
    sliding_stats = {}
    for (layer_idx, step), data in stats.items():
        # Even-numbered layers (0-indexed) that use "sliding_attention" or "full_attention"
        # From config: sliding_attention if (i+1)%2 else full_attention
        # So layer 0 = sliding, layer 1 = full, etc.
        if layer_idx % 2 == 1:
            full_stats[(layer_idx, step)] = data
        else:
            sliding_stats[(layer_idx, step)] = data

    # Collect by diffusion step
    steps = sorted(set(s for _, s in stats.keys()))
    if not steps:
        print("[WARN] no attention stats captured")
        return

    # For each step, analyze how entropy changes with position
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Entropy vs position (first diffusion step) ---
    ax = axes[0, 0]
    first_step = steps[0] if steps else None
    if first_step is not None:
        for (layer_idx, step), data in stats.items():
            if step != first_step:
                continue
            ent = data["entropy"].mean(dim=(0, 1)).numpy()  # [T]
            T = len(ent)
            label = f"L{layer_idx} ({'full' if layer_idx%2==1 else 'sliding'})"
            ax.plot(np.arange(T), ent, label=label, alpha=0.7)
        ax.set_xlabel("Position (patch index)")
        ax.set_ylabel("Attention Entropy")
        ax.set_title(f"Attention Entropy vs Position (step {first_step})")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # --- Plot 2: Receptive field vs position (first diffusion step) ---
    ax = axes[0, 1]
    if first_step is not None:
        for (layer_idx, step), data in stats.items():
            if step != first_step:
                continue
            rf = data["receptive_field"].mean(dim=(0, 1)).numpy()  # [T]
            T = len(rf)
            label = f"L{layer_idx}"
            ax.plot(np.arange(T), rf, label=label, alpha=0.7)
        ax.set_xlabel("Position (patch index)")
        ax.set_ylabel("Expected Key Distance")
        ax.set_title(f"Receptive Field vs Position (step {first_step})")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # --- Plot 3: Entropy trend: tail vs head across diffusion steps ---
    ax = axes[1, 0]
    head_entropy = {}  # step -> avg entropy in first 10%
    tail_entropy = {}  # step -> avg entropy in last 10%
    for (layer_idx, step), data in stats.items():
        ent = data["entropy"].mean(dim=(0, 1)).numpy()  # [T]
        T = len(ent)
        head = ent[:max(1, T // 10)].mean()
        tail = ent[-max(1, T // 10):].mean()
        head_entropy.setdefault(step, []).append(head)
        tail_entropy.setdefault(step, []).append(tail)

    steps_plot = sorted(set(head_entropy.keys()))
    if steps_plot:
        head_means = [np.mean(head_entropy[s]) for s in steps_plot]
        tail_means = [np.mean(tail_entropy[s]) for s in steps_plot]
        ax.plot(steps_plot, head_means, "b-o", label="Head (first 10%)", markersize=3)
        ax.plot(steps_plot, tail_means, "r-o", label="Tail (last 10%)", markersize=3)
        # Ratio
        ratio = [t / h for t, h in zip(tail_means, head_means)]
        ax2 = ax.twinx()
        ax2.plot(steps_plot, ratio, "g--", alpha=0.5, label="Tail/Head ratio")
        ax2.set_ylabel("Tail/Head ratio", color="g")
        ax.set_xlabel("Diffusion Step")
        ax.set_ylabel("Avg Attention Entropy")
        ax.set_title("Head vs Tail Attention Entropy over Diffusion")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    # --- Plot 4: Full attention layers only - entropy heatmap ---
    ax = axes[1, 1]
    if first_step is not None:
        full_entropies = []
        full_labels = []
        for (layer_idx, step), data in stats.items():
            if step != first_step or layer_idx % 2 == 0:
                continue
            ent = data["entropy"].mean(dim=(0, 1)).numpy()  # [T]
            # Downsample for visualization
            T = len(ent)
            if T > 2000:
                ds = T // 1000
                ent = ent[::ds][:1000]
            full_entropies.append(ent)
            full_labels.append(f"L{layer_idx}")

        if full_entropies:
            # Pad to same length
            max_len = max(len(e) for e in full_entropies)
            padded = np.zeros((len(full_entropies), max_len))
            for i, e in enumerate(full_entropies):
                padded[i, :len(e)] = e
            im = ax.imshow(padded, aspect="auto", cmap="hot", interpolation="nearest")
            ax.set_yticks(range(len(full_labels)))
            ax.set_yticklabels(full_labels)
            ax.set_xlabel("Position (downsampled)")
            ax.set_ylabel("Full Attention Layer")
            ax.set_title(f"Entropy Heatmap (step {first_step})")
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path = output_dir / "attention_decay.png"
    plt.savefig(plot_path, dpi=120)
    print(f"  → saved {plot_path}")
    plt.close()


def save_analysis_data(stats, output_dir: Path):
    """Save aggregated stats as JSON and raw data as .pt for later analysis."""
    # Save summary JSON
    summary = {}
    for (layer_idx, step), data in stats.items():
        key = f"L{layer_idx}_S{step}"
        entropy = data["entropy"].mean(dim=(0, 1)).numpy().tolist()  # [T]
        rf = data["receptive_field"].mean(dim=(0, 1)).numpy().tolist()
        summary[key] = {
            "seq_len": data["seq_len"],
            "entropy_mean": float(np.mean(entropy)),
            "entropy_std": float(np.std(entropy)),
            "entropy_first10pct": float(np.mean(entropy[:max(1, len(entropy)//10)])),
            "entropy_last10pct": float(np.mean(entropy[-max(1, len(entropy)//10):])),
            "rf_mean": float(np.mean(rf)),
            "rf_std": float(np.std(rf)),
        }

    json_path = output_dir / "attention_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  → saved {json_path}")

    # Save raw per-step stats as .pt
    stats_tensor = {}
    for (layer_idx, step), data in stats.items():
        stats_tensor[(layer_idx, step)] = {
            "entropy": data["entropy"],  # [B, H, T]
            "receptive_field": data["receptive_field"],
        }
    pt_path = output_dir / "attention_stats.pt"
    torch.save(stats_tensor, pt_path)
    print(f"  → saved {pt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Attention Decay Profiling")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Generation duration in seconds (default: 60)")
    parser.add_argument("--capture-every", type=int, default=5,
                        help="Capture every N-th diffusion step (default: 5)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion inference steps (default: 50)")
    args = parser.parse_args()

    dur = args.duration
    OUT.mkdir(parents=True, exist_ok=True)

    # Compute expected sequence length for report
    audio_frames = int(dur * 25)
    n_patches = audio_frames // 2
    print("=" * 60)
    print(f"Rope Exp 1: Attention Decay Profiling")
    print(f"  Duration:        {dur}s ({audio_frames} frames, ~{n_patches} patches)")
    print(f"  Capture every:   {args.capture_every}th step")
    print(f"  Diff steps:      {args.steps}")

    # Pre-generation analysis
    print()
    print("--- RoPE Frequency Status ---")
    head_dim = 128
    rope_theta = 1_000_000.0
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    desc = [
        ("First cycle", lambda c: c < 1),
        ("1-2 cycles",  lambda c: (c >= 1) & (c < 2)),
        ("2-5 cycles",  lambda c: (c >= 2) & (c < 5)),
        (">5 cycles",   lambda c: c >= 5),
    ]
    cycles_at_patches = (inv_freq * n_patches) / (2 * math.pi)
    for label, cond in desc:
        n = cond(cycles_at_patches).sum().item()
        print(f"  {label}: {n}/64 frequency pairs")
    print(f"  Max cycles: {cycles_at_patches[0].item():.0f}")

    # ---- Load model ----
    print()
    print("[1/3] Loading model...")
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

    # Disable Section-RoPE and PhaseMemory
    model.config.use_section_rope_offset = False
    for l in model.decoder.layers:
        if getattr(l, "use_section_rope", False):
            l.use_section_rope = False
        if getattr(l, "use_phase_memory", False):
            l.use_phase_memory = False

    # ---- Init LLM ----
    print("[2/3] Initializing 5Hz LM...")
    llm = LLMHandler()
    ok = llm.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not ok:
        print("  ❌ LM init failed")
        sys.exit(1)
    print("  ✓ LM ready")

    # ---- Force eager attention to capture weights ----
    print("[3/3] Registering attention hooks...")
    # Force all attention modules to use eager mode so weights are returned
    model.config._attn_implementation = "eager"
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "eager"

    profiler = AttentionProfiler(model, capture_every=args.capture_every)
    # Store raw weights for first and last captured steps
    first_captured = 1
    last_captured = args.steps  # approximate
    profiler.set_raw_capture_steps({first_captured, last_captured})
    print(f"  ✓ {len(model.decoder.layers)} layers hooked, capturing every {args.capture_every}th step")

    # ---- Generate ----
    print()
    print("=" * 60)
    print("Generating music...")
    print("=" * 60)

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
        duration=dur,
        inference_steps=args.steps,
        guidance_scale=7.0,
        seed=42,
        thinking=True,
        use_cot_metas=True,
        use_cot_caption=True,
        lm_temperature=0.75,
    )
    gen_config = GenerationConfig(
        batch_size=1,
        audio_format="flac",
        use_random_seed=False,
        seeds=[42],
    )

    result = generate_music(
        dit_handler=dit,
        llm_handler=llm,
        params=params,
        config=gen_config,
        save_dir=str(OUT),
    )

    if result.success:
        audio_path = result.audios[0]["path"] if result.audios else None
        print(f"  ✓ Generation complete: {audio_path}")
    else:
        print(f"  ❌ Generation failed: {result.error}")
        # Still try to analyze whatever we captured

    # ---- Analyze ----
    print()
    print("=" * 60)
    print("Analyzing attention data...")
    print("=" * 60)

    total_captured = len(profiler.stats)
    print(f"  Captured {total_captured} (layer, step) entries")

    if total_captured > 0:
        analyze_attention_trends(profiler.stats, len(model.decoder.layers), OUT)
        save_analysis_data(profiler.stats, OUT)

        # Print quick summary
        print()
        print("--- Quick Summary ---")
        # Compare first and last captured diffusion step
        steps = sorted(set(s for _, s in profiler.stats.keys()))
        print(f"  Diffusion steps captured: min={min(steps)}, max={max(steps)}")
        for step in [min(steps), max(steps)]:
            ents = []
            rfs = []
            for (layer_idx, s), data in profiler.stats.items():
                if s != step:
                    continue
                ents.append(data["entropy"].mean().item())
                rfs.append(data["receptive_field"].mean().item())
            if ents:
                print(f"  Step {step}: avg_entropy={np.mean(ents):.4f}, avg_rf={np.mean(rfs):.1f}, "
                      f"head_entropy={np.mean([e for e in ents]):.4f}")
    else:
        print("  ⚠ No attention data captured")

    print()
    print(f"Results saved to {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
