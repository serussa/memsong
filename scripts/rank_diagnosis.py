#!/usr/bin/env python3
"""Diagnose rank collapse: which sub-layer causes it and does AdaLN matter."""

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


def compute_effective_rank(X, cumvar_threshold=0.95):
    """Compute effective rank from SVD."""
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var_exp = (s ** 2) / (s ** 2).sum()
        cumvar = np.cumsum(var_exp)
        eff_rank = int((cumvar < cumvar_threshold).sum()) + 1
        return eff_rank, float(var_exp[0]), float(var_exp[:3].sum())
    except np.linalg.LinAlgError:
        return None, None, None


class LayerHookSet:
    """Hook specific points inside a DiT layer without device conflicts."""

    def __init__(self, layer):
        self.hooks = []
        self.records = {
            "input": [], "pre_attn": [], "post_attn": [],
            "post_cross_attn": [], "post_pm": [], "pre_mlp": [], "post_mlp": [],
        }

        # Input to the whole layer
        self.hooks.append(layer.register_forward_hook(
            lambda m, i, o: self.records["input"].append(
                i[0][:1].detach().cpu() if isinstance(o, tuple) else i[0][:1].detach().cpu()
                if not isinstance(o, tuple) else i[0][:1].detach().cpu()
            )
        ))

        # After self-attention norm (pre-attention normalized input)
        self.hooks.append(layer.self_attn_norm.register_forward_hook(
            lambda m, i, o: self.records["pre_attn"].append(
                o[:1].detach().cpu() if isinstance(o, torch.Tensor) else i[0][:1].detach().cpu()
            )
        ))

        # After self-attention output projection
        self.hooks.append(layer.self_attn.o_proj.register_forward_hook(
            lambda m, i, o: self.records["post_attn"].append(o[:1].detach().cpu())
        ))

        # After cross-attention output projection
        if hasattr(layer, 'cross_attn') and hasattr(layer.cross_attn, 'o_proj'):
            self.hooks.append(layer.cross_attn.o_proj.register_forward_hook(
                lambda m, i, o: self.records["post_cross_attn"].append(o[:1].detach().cpu())
            ))

        # After PhaseMemory hidden state output (output[0] is the main hidden state)
        if hasattr(layer, 'phase_memory'):
            self.hooks.append(layer.phase_memory.register_forward_hook(
                lambda m, i, o: self.records["post_pm"].append(
                    (o[0] if isinstance(o, tuple) else o)[:1].detach().cpu()
                )
            ))

        # After MLP norm (pre-MLP normalized input)
        self.hooks.append(layer.mlp_norm.register_forward_hook(
            lambda m, i, o: self.records["pre_mlp"].append(
                o[:1].detach().cpu() if isinstance(o, torch.Tensor) else i[0][:1].detach().cpu()
            )
        ))

        # After MLP output
        self.hooks.append(layer.mlp.register_forward_hook(
            lambda m, i, o: self.records["post_mlp"].append(o[:1].detach().cpu())
        ))

    def remove(self):
        for h in self.hooks:
            h.remove()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=str, default="output/rank_diagnosis")
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

    # Hook Layer
    hook = LayerHookSet(model.decoder.layers[args.layer])

    # Quick conditioning (no LLM needed for diagnosis)
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

    from acestep.phase_memory import reset_phase_memory
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

    # Run diffusion steps
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
            )
        dt = (t_curr - timesteps[i + 1]).unsqueeze(-1).unsqueeze(-1)
        xt = xt - out[0] * dt

    hook.remove()

    # ---- Analyze ----
    print(f"\n{'='*60}")
    print(f"  RANK DIAGNOSIS — Layer {args.layer}")
    print(f"{'='*60}")

    results = {}
    for name, states in hook.records.items():
        if not states:
            continue
        ranks, top1s, top3s = [], [], []
        for s in states:
            h = s[0].float().numpy()  # [T, D]
            r, t1, t3 = compute_effective_rank(h)
            if r is not None:
                ranks.append(r)
                top1s.append(t1)
                top3s.append(t3)

        results[name] = {
            "rank_mean": float(np.mean(ranks)),
            "rank_std": float(np.std(ranks)),
            "rank_min": int(np.min(ranks)),
            "rank_max": int(np.max(ranks)),
            "top1_mean": float(np.mean(top1s)),
            "top3_mean": float(np.mean(top3s)),
        }

        print(f"  {name:>20}: rank={np.mean(ranks):.1f}±{np.std(ranks):.1f} "
              f"(min={np.min(ranks)}, max={np.max(ranks)}) "
              f"top1={np.mean(top1s):.3f} top3={np.mean(top3s):.3f}")

    # Compare cross-attention enabled layers vs not
    # Run the same on a non-cross-attention layer (e.g., layer 13 or 0)
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Cross-Attention layers vs Sliding-Attention layers")
    print(f"{'='*60}")

    for layer_idx in [0, 5, 11, 13, 17, 23]:
        reset_phase_memory(model)
        hook2 = LayerHookSet(model.decoder.layers[layer_idx])
        xt = noise.clone()
        for i in range(min(args.steps, 5)):  # 5 steps enough for rank analysis
            t_curr = timesteps[i]
            with torch.no_grad():
                out = model.decoder(
                    hidden_states=xt, timestep=t_curr.unsqueeze(0), timestep_r=t_curr.unsqueeze(0),
                    attention_mask=attention_mask, encoder_hidden_states=enc_hs,
                    encoder_attention_mask=enc_am, context_latents=ctx,
                    use_cache=False, beat_phase=None,
                )
            dt = (t_curr - timesteps[i + 1]).unsqueeze(-1).unsqueeze(-1)
            xt = xt - out[0] * dt

        hook2.remove()
        # Report input rank for this layer
        states = hook2.records.get("input", [])
        if states:
            h = states[-1][0].float().numpy()
            r, t1, t3 = compute_effective_rank(h)
            attn_type = "full" if model.decoder.layers[layer_idx].attention_type == "full_attention" else "sliding"
            cross = "Y" if model.decoder.layers[layer_idx].use_cross_attention else "N"
            print(f"  Layer {layer_idx:>2d} ({attn_type:>8s}, cross={cross}): "
                  f"input_rank={r}, top1={t1:.3f}")

    # Also check encoder output rank
    print(f"\n  Encoder hidden states: ", end="")
    h = enc_hs[0].float().cpu().numpy()
    r, t1, t3 = compute_effective_rank(h)
    print(f"rank={r}, top1={t1:.3f}")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}/")


if __name__ == "__main__":
    main()
