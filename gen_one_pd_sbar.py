#!/usr/bin/env python3
"""
Generate one song with PD-SBAR rewiring.
Patches prepare_condition to get actual L_enc, then installs
the SinkhornBregmanReparameterizer before decoder forward.
"""
import os, sys, argparse, uuid, time
from pathlib import Path

ACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ACE_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
os.environ["SIDESTEP_SAFE_ROOT"] = "/root/autodl-tmp"

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="baseline", choices=["baseline", "pd_sbar", "pd_sbar_trained"])
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--lyrics", type=str, required=True)
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pd-sbar-ckpt", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="PD-SBAR alpha (Bregman weight, 0-1). Default 0.05 for inference.")
    parser.add_argument("--gamma-leak", type=float, default=0.5,
                        help="PD-SBAR gamma_leak (leak penalty). Default 0.5 for inference.")
    parser.add_argument("--beta-dual", type=float, default=0.1,
                        help="PD-SBAR beta_dual (dual potential). Default 0.1 for inference.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or uuid.uuid4().hex[:8]

    # ---- Load models ---------------------------------------------------
    dt = AceStepHandler()
    dt.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5",
        config_path="acestep-v15-sft", device="cuda",
        use_flash_attention=False, compile_model=False, offload_to_cpu=False,
    )
    model = dt.model.eval()
    device = next(model.parameters()).device
    model.config.use_section_rope_offset = False
    for l in model.decoder.layers:
        if getattr(l, "use_section_rope", False): l.use_section_rope = False
        if getattr(l, "use_phase_memory", False): l.use_phase_memory = False

    # Force eager attention for PD-SBAR
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            if m.config._attn_implementation != "eager":
                m.config._attn_implementation = "eager"

    # ---- Baseline: just generate and return ----------------------------
    if args.method == "baseline":
        llm = LLMHandler()
        llm.initialize(checkpoint_dir="/root/autodl-tmp/Ace-Step1.5",
                       lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
        params = GenerationParams(
            task_type="text2music", caption=args.caption, lyrics=args.lyrics,
            instrumental=False, bpm=120, keyscale="C major", timesignature="4",
            vocal_language="en", duration=args.duration, inference_steps=50,
            guidance_scale=7.0, seed=args.seed, thinking=True,
            use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
        )
        config = GenerationConfig(batch_size=1, audio_format="flac",
                                   use_random_seed=False, seeds=[args.seed])
        result = generate_music(dt, llm, params, config, save_dir=str(output_dir))
        if result.success and result.audios:
            gen_path = Path(result.audios[0]['path'])
            rename_to = gen_path.parent / f"{name}.flac"
            if gen_path.exists() and not rename_to.exists():
                gen_path.rename(rename_to)
            print(f"SUCCESS: {rename_to if rename_to.exists() else gen_path}")
        else:
            print(f"FAIL: {result.error if hasattr(result, 'error') else 'unknown'}")
            sys.exit(1)
        return

    # ---- PD-SBAR setup via prepare_condition patch --------------------
    from acestep.phase_memory import (
        PDSBARPlanner, SinkhornBregmanReparameterizer,
        parse_lyrics_to_units, build_duration_scaffold,
    )
    T_eff = int(args.duration * 25)
    T_audio = T_eff // 2

    # Build planner lazily after prepare_condition reveals L_enc
    planner = None
    reparam = None
    reparam_installed = [False]
    orig_prepare = model.prepare_condition

    def _patched_prepare(*a, **kw):
        nonlocal planner, reparam
        result = orig_prepare(*a, **kw)
        if result[0] is not None and not reparam_installed[0]:
            enc_hs = result[0]
            L_enc = enc_hs.shape[1]
            print(f"  [PD-SBAR] L_enc={L_enc}, T_audio={T_audio}", flush=True)

            # Build planner with actual L_enc
            section_ids = torch.zeros(L_enc, dtype=torch.long, device=device)
            units, _, debug = parse_lyrics_to_units(
                args.lyrics, section_ids.cpu(),
                auto_transition_ratios=dict(intro=0., outro=0., chorus_to_verse=0.,
                                            chorus_to_bridge=0., bridge_to_chorus=0.),
            )
            tcm = debug.get("tag_control_mask", None)
            scaffold = build_duration_scaffold(units, text_len=L_enc, tag_control_mask=tcm)
            scaffold = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}

            planner = PDSBARPlanner(
                sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
                slack_ratio=0.08, budget_smoothing=0.1, eta_dual=1.0, lambda_max=3.0,
            ).to(device).float()
            planner.build_from_scaffold(scaffold, T_audio, batch_size=1, device=device)

            reparam = SinkhornBregmanReparameterizer(
                planner=planner, rewired_layers=[8, 12, 16, 20],
                alpha=args.alpha, gamma_leak=args.gamma_leak,
                beta_dual=args.beta_dual,
            )

            # Load LoRA
            if args.method == "pd_sbar_trained" and args.pd_sbar_ckpt:
                from acestep.training_v2.fixed_lora_module import _inject_lora_on_rewired_layers
                from safetensors.torch import load_file as sf_load
                _inject_lora_on_rewired_layers(
                    model, rewired_layers=[8, 12, 16, 20],
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                    rank=16, alpha=32,
                )
                sd = model.state_dict()
                sd_load = sf_load(args.pd_sbar_ckpt)
                for k, v in sd_load.items():
                    if k in sd and sd[k].shape == v.shape:
                        sd[k].copy_(v)
                model.load_state_dict(sd, strict=False)
                print(f"  [PD-SBAR] LoRA loaded from {args.pd_sbar_ckpt}")

            reparam.install(model)
            reparam_installed[0] = True
            print(f"  [PD-SBAR] Installed (inference: alpha={reparam.alpha}, gamma_leak={reparam.gamma_leak}, beta_dual={reparam.beta_dual})", flush=True)

        return result

    model.prepare_condition = _patched_prepare

    # ---- Generate -----------------------------------------------------
    llm = LLMHandler()
    llm.initialize(checkpoint_dir="/root/autodl-tmp/Ace-Step1.5",
                   lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
    params = GenerationParams(
        task_type="text2music", caption=args.caption, lyrics=args.lyrics,
        instrumental=False, bpm=120, keyscale="C major", timesignature="4",
        vocal_language="en", duration=args.duration, inference_steps=50,
        guidance_scale=7.0, seed=args.seed, thinking=True,
        use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
    )
    config = GenerationConfig(batch_size=1, audio_format="flac",
                               use_random_seed=False, seeds=[args.seed])
    result = generate_music(dt, llm, params, config, save_dir=str(output_dir))

    if reparam_installed[0] and reparam is not None:
        reparam.remove()
        model.prepare_condition = orig_prepare

    if result.success and result.audios:
        gen_path = Path(result.audios[0]['path'])
        # Rename to prompt name for easy identification
        rename_to = gen_path.parent / f"{name}.flac"
        if gen_path.exists() and not rename_to.exists():
            gen_path.rename(rename_to)
            print(f"SUCCESS: {rename_to}")
        else:
            print(f"SUCCESS: {gen_path}")
    else:
        print(f"FAIL: {result.error if hasattr(result, 'error') else 'unknown'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
