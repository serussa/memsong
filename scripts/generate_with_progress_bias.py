#!/usr/bin/env python3
"""
Generate music with Fixed Linear Progress Bias intervention.

Patches layer-12 cross-attention to inject a monotonic progress bias
during the full diffusion generation process (not just teacher-forcing).

Usage:
    python scripts/generate_with_progress_bias.py \
        --prompt "pop, female vocal, piano" \
        --lyrics "[Verse] ..." \
        --duration 30 \
        --sigma 0.15 --gate 1.0 \
        --seeds 42,123

    3 prompts x 2 seeds each (default):
    python scripts/generate_with_progress_bias.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# ---------------------------------------------------------------------------
# Dynamic progress bias — patches eager_attention_forward
# ---------------------------------------------------------------------------

def _install_dynamic_progress_bias(model, sigma=0.15, lambda_=1.0,
                                    max_bias=3.0, gate=1.0, layer=12):
    """Patch ``eager_attention_forward`` in the model's runtime module to
    inject a Gaussian progress bias built dynamically from the actual T, L
    shapes at each forward call.

    The bias uses uniform lyric positions ``j/(L-1)`` for all text tokens,
    since we do not have parsed ``section_ids`` during generation.
    """
    import sys as _sys
    ca_module = model.decoder.layers[layer].cross_attn
    runtime_mod = _sys.modules[type(ca_module).__module__]
    orig_eaf = runtime_mod.eager_attention_forward

    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    # Store params on the module so biased_forward can read them
    ca_module._pb_params = {"sigma": sigma, "lambda_": lambda_,
                            "max_bias": max_bias, "gate": gate}

    def _biased_eaf(module, query, key, value, attention_mask,
                    scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # ---- Dynamic progress bias injection ----
        pb_params = getattr(module, "_pb_params", None)
        if pb_params is not None:
            B, H, T, L = attn_weights.shape
            dev = attn_weights.device
            # Audio progress
            p = torch.arange(T, device=dev, dtype=torch.float32) / max(T - 1, 1)
            # Uniform lyric positions (all tokens)
            r = torch.arange(L, device=dev, dtype=torch.float32) / max(L - 1, 1)
            # Distance
            dist = r[None, :] - p[:, None]  # [T, L]
            # Gaussian bias
            pb = -pb_params["lambda_"] * (dist / pb_params["sigma"]) ** 2
            pb = pb.clamp(min=-pb_params["max_bias"], max=0.0)
            pb = pb * pb_params["gate"]
            pb = pb[None, None, :, :]  # [1, 1, T, L]
            attn_weights = attn_weights + pb.to(dtype=attn_weights.dtype,
                                                 device=attn_weights.device)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                    dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout,
                                                    training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    runtime_mod.eager_attention_forward = _biased_eaf
    return runtime_mod, orig_eaf


def _restore_eaf(runtime_mod, orig_eaf, model, layer=12):
    """Restore original eager_attention_forward and clean up."""
    runtime_mod.eager_attention_forward = orig_eaf
    ca = model.decoder.layers[layer].cross_attn
    if hasattr(ca, "_pb_params"):
        del ca._pb_params


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    {
        "caption": "pop, female vocal, piano, strings, drums, romantic, 120 bpm, C major",
        "lyrics": "",
        "duration": 30,
        "bpm": 120,
        "keyscale": "C",
    },
    {
        "caption": "jazz, saxophone, double bass, drums, piano, slow swing, smoky atmosphere, 90 bpm, F major",
        "lyrics": "",
        "duration": 30,
        "bpm": 90,
        "keyscale": "F",
    },
    {
        "caption": "EDM, electronic, synth, heavy bass, drums, energetic, 128 bpm, A minor",
        "lyrics": "",
        "duration": 30,
        "bpm": 128,
        "keyscale": "Am",
    },
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate music with progress bias")
    # Bias params
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--max-bias", type=float, default=3.0)
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--layer", type=int, default=12)
    # Generation
    parser.add_argument("--seeds", type=str, default="42,123",
                        help="Comma-separated seeds for each prompt (default: 42,123)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Audio duration in seconds (default: 30)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Diffusion inference steps (default: 8)")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/progress_bias_generations")
    # Prompts
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single caption prompt (overrides defaults)")
    parser.add_argument("--lyrics", type=str, default="",
                        help="Lyrics for single prompt mode")
    parser.add_argument("--bpm", type=int, default=None,
                        help="BPM for single prompt mode")
    parser.add_argument("--keyscale", type=str, default="",
                        help="Key scale for single prompt mode")
    # Baseline comparison
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run without bias for comparison")
    parser.add_argument("--no-bias", action="store_true",
                        help="Run WITHOUT bias (baseline control)")

    args = parser.parse_args()
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt list
    if args.prompt is not None:
        prompts = [{"caption": args.prompt, "lyrics": args.lyrics or "",
                    "duration": args.duration, "bpm": args.bpm, "keyscale": args.keyscale}]
    else:
        prompts = DEFAULT_PROMPTS

    total_jobs = len(prompts) * len(seed_list)
    label = "no-bias" if args.no_bias else f"bias_s{args.sigma}_g{args.gate}"
    print("=" * 70)
    print("ACE-Step 1.5 音乐生成 with Fixed Progress Bias")
    print(f"  Bias: sigma={args.sigma}, lambda={args.lambda_}, "
          f"max_bias={args.max_bias}, gate={args.gate}, layer={args.layer}")
    print(f"  Label: {label}")
    print(f"  Prompts: {len(prompts)}, Seeds per prompt: {len(seed_list)} = {total_jobs} total")
    print("=" * 70)

    # ---- Init handlers ----
    print("\n[1/3] 初始化 DiT...")
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit_handler.model.eval()
    # Disable Section-RoPE / PM
    model.config.use_section_rope_offset = False
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_section_rope", False):
            layer_mod.use_section_rope = False
        if getattr(layer_mod, "use_phase_memory", False):
            layer_mod.use_phase_memory = False
    print("  DiT ready, adapters disabled.")

    print("\n[2/3] 初始化 5Hz LM...")
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    print("  LM ready.")

    # ---- Patch bias if enabled ----
    runtime_mod, orig_eaf = None, None
    if not args.no_bias:
        print(f"\n  Installing progress bias (sigma={args.sigma}, gate={args.gate})...")
        runtime_mod, orig_eaf = _install_dynamic_progress_bias(
            model, sigma=args.sigma, lambda_=args.lambda_,
            max_bias=args.max_bias, gate=args.gate, layer=args.layer,
        )

    # ---- Generate ----
    print(f"\n[3/3] 生成 {total_jobs} 条音频...\n")
    all_results = []
    job_idx = 0

    for pi, prompt in enumerate(prompts):
        for seed in seed_list:
            job_idx += 1
            print(f"\n{'─' * 60}")
            print(f"[{job_idx}/{total_jobs}] Prompt {pi+1} | Seed {seed}")
            print(f"  Caption: {prompt['caption'][:80]}...")
            if prompt.get("lyrics"):
                print(f"  Lyrics:  {prompt['lyrics'][:60]}...")
            print(f"  Duration: {prompt['duration']}s | BPM: {prompt.get('bpm','auto')}")

            params = GenerationParams(
                task_type="text2music",
                caption=prompt["caption"],
                lyrics=prompt.get("lyrics", ""),
                instrumental=not bool(prompt.get("lyrics", "")),
                bpm=prompt.get("bpm"),
                keyscale=prompt.get("keyscale", ""),
                vocal_language="en",
                duration=prompt["duration"],
                inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=seed,
                thinking=False,  # skip LM reasoning for speed
                use_cot_metas=False,
                use_cot_caption=False,
                use_cot_language=False,
            )
            config = GenerationConfig(
                batch_size=1,
                audio_format="flac",
                use_random_seed=False,
                seeds=[seed],
            )

            t0 = time.time()
            result = generate_music(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                params=params,
                config=config,
                save_dir=str(out_dir),
            )
            elapsed = time.time() - t0

            # Save metadata
            entry = {
                "job": job_idx,
                "prompt_idx": pi + 1,
                "seed": seed,
                "caption": prompt["caption"],
                "sigma": args.sigma,
                "gate": args.gate,
                "duration_target": prompt["duration"],
                "success": result.success,
                "time_sec": round(elapsed, 1),
            }
            if result.success and result.audios:
                entry["audio_path"] = result.audios[0].get("path", "")
            else:
                entry["error"] = result.error or "unknown"

            all_results.append(entry)
            status = "✓" if result.success else "✗"
            print(f"  {status} {elapsed:.1f}s  → {entry.get('audio_path', 'FAILED')}")

    # Restore original forward
    if runtime_mod is not None and orig_eaf is not None:
        _restore_eaf(runtime_mod, orig_eaf, model, layer=args.layer)

    # ---- Save manifest ----
    manifest = {
        "label": label,
        "bias_params": {"sigma": args.sigma, "lambda": args.lambda_,
                        "max_bias": args.max_bias, "gate": args.gate},
        "results": all_results,
    }
    mpath = out_dir / f"manifest_{label}.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {mpath}")

    # Summary
    n_ok = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"Done: {n_ok}/{total_jobs} succeeded")
    print(f"Output: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
