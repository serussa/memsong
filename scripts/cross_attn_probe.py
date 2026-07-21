#!/usr/bin/env python3
"""Experiment 2: Does cross-attention lose timing info? Probe attention maps."""

import os, sys, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="output/cross_attn_probe")
    args = parser.parse_args()

    OUT = Path(args.output)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
        device="cuda", use_flash_attention=False, compile_model=False,
    )
    model = dit.model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Hook cross-attention weights for all layers that have cross-attention
    cross_attn_weights = {}  # layer_idx → list of attn maps

    def make_cross_attn_hook(layer_idx):
        def hook(module, input, output):
            # output: (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                # attn_weights shape: [B, heads, T_audio, T_enc]
                w = output[1].detach().cpu()
                if layer_idx not in cross_attn_weights:
                    cross_attn_weights[layer_idx] = []
                cross_attn_weights[layer_idx].append(w)
        return hook

    hooks = []
    for i, layer in enumerate(model.decoder.layers):
        if hasattr(layer, 'cross_attn'):
            h = layer.cross_attn.register_forward_hook(make_cross_attn_hook(i))
            hooks.append(h)

    # Quick conditioning
    from acestep.phase_memory import make_synthetic_beat_phase, reset_phase_memory

    B, T = 1, 250
    text_hs = torch.randn(B, 77, 1024, device=device, dtype=dtype)
    text_am = torch.ones(B, 77, device=device, dtype=dtype)
    lyric_hs = torch.randn(B, 50, 1024, device=device, dtype=dtype)
    lyric_am = torch.ones(B, 50, device=device, dtype=dtype)
    timbre_packed = torch.randn(1, 750, 64, device=device, dtype=dtype)
    timbre_order = torch.zeros(1, device=timbre_packed.device, dtype=torch.long)
    src_latents = torch.randn(B, T, 64, device=device, dtype=dtype)
    chunk_masks = torch.ones(B, T, 64, device=device, dtype=dtype)
    attention_mask = torch.ones(B, T, device=device, dtype=dtype)
    silence_latent = torch.randn(B, T, 64, device=device, dtype=dtype)
    is_covers = torch.zeros(B, device=device, dtype=torch.long)

    reset_phase_memory(model)
    enc_hs, enc_am, ctx = model.prepare_condition(
        text_hidden_states=text_hs, text_attention_mask=text_am,
        lyric_hidden_states=lyric_hs, lyric_attention_mask=lyric_am,
        refer_audio_acoustic_hidden_states_packed=timbre_packed,
        refer_audio_order_mask=timbre_order,
        hidden_states=src_latents, attention_mask=attention_mask,
        silence_latent=silence_latent, src_latents=src_latents,
        chunk_masks=chunk_masks, is_covers=is_covers,
    )

    # Run diffusion
    noise = torch.randn(B, T, 64, device=device, dtype=dtype)
    xt = noise.clone()
    timesteps = torch.linspace(1.0, 0.0, args.steps + 1, device=device, dtype=dtype)

    for i in range(args.steps):
        t_curr = timesteps[i]
        with torch.no_grad():
            out = model.decoder(
                hidden_states=xt, timestep=t_curr.unsqueeze(0), timestep_r=t_curr.unsqueeze(0),
                attention_mask=attention_mask, encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_am, context_latents=ctx,
                use_cache=False, beat_phase=None,
                output_attentions=True,
            )
        dt = (t_curr - timesteps[i + 1]).unsqueeze(-1).unsqueeze(-1)
        xt = xt - out[0] * dt

    for h in hooks:
        h.remove()

    # Analyze cross-attention patterns
    print(f"\n{'='*60}")
    print(f"  CROSS-ATTENTION PROBE")
    print(f"{'='*60}")

    for layer_idx in sorted(cross_attn_weights.keys()):
        weights = cross_attn_weights[layer_idx]  # list of [B, heads, T_audio, T_enc]
        if not weights:
            continue

        # Stack: [steps, B, heads, T_audio, T_enc]
        stacked = torch.stack(weights, dim=0)

        # Mean over heads
        attn_mean = stacked.mean(dim=2)  # [steps, B, T_audio, T_enc]

        # 1. Temporal focus: does the model attend to different encoder positions
        #    at different diffusion steps?
        attn_over_steps = attn_mean[:, 0]  # [steps, T_audio, T_enc]

        # Entropy over encoder positions per audio frame, per step
        p_enc = attn_over_steps / (attn_over_steps.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p_enc * torch.log(p_enc.clamp(min=1e-10))).sum(dim=-1)  # [steps, T_audio]
        norm_entropy = entropy / math.log(attn_over_steps.shape[-1])

        # 2. Does the same encoder frame mean the same thing across steps?
        #    Compute frame-to-frame similarity of the attention pattern across steps
        attn_flat = attn_over_steps.reshape(args.steps, -1)  # [steps, T_audio * T_enc]
        attn_norm = attn_flat / (attn_flat.norm(dim=-1, keepdim=True) + 1e-8)
        step_sim = (attn_norm[:-1] * attn_norm[1:]).sum(dim=-1)  # [steps-1]

        # 3. Temporal dynamics of attention (per audio position, which encoder position)
        attn_mode = attn_over_steps.argmax(dim=-1)  # [steps, T_audio]
        mode_shift = (attn_mode[1:] - attn_mode[:-1]).abs().float().mean().item()

        # 4. Average entropy per step
        entropy_by_step = norm_entropy.mean(dim=-1)  # [steps]
        early_entropy = entropy_by_step[:args.steps//3].mean().item()
        late_entropy = entropy_by_step[2*args.steps//3:].mean().item()

        print(f"\n  Layer {layer_idx} (cross-attention):")
        print(f"    Step-to-step attention similarity: {step_sim.mean().item():.4f}±{step_sim.std().item():.4f}")
        print(f"    Attended encoder position shift: {mode_shift:.2f} tokens/step")
        print(f"    Normalized attention entropy: early={early_entropy:.4f}, late={late_entropy:.4f}")
        print(f"    Attn entropy over all: mean={norm_entropy.mean().item():.4f}")

        # Save per-layer data
        np.save(OUT / f"attn_entropy_l{layer_idx}.npy", norm_entropy.float().numpy())
        np.save(OUT / f"attn_over_steps_l{layer_idx}.npy", attn_over_steps.float().numpy())

    print(f"\n{'='*60}")
    print(f"  KEY: entropy ~1.0 = uniform (no selective attention)")
    print(f"       entropy ~0.0 = focused on 1 encoder token")
    print(f"  mode_shift ~0 = attention pattern is FROZEN across steps")
    print(f"  mode_shift >1 = attention shifts over time (dynamic)")
    print(f"{'='*60}")
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
