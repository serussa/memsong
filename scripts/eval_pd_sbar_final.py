#!/usr/bin/env python3
"""
PD-SBAR Final Evaluation — compare baseline vs inference-only vs epoch1/2/3.
"""
import argparse, json, os, sys, time, math
from pathlib import Path

import torch
import numpy as np

ACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.training_v2.model_loader import load_decoder_for_training
from acestep.phase_memory import (
    PDSBARPlanner, SinkhornBregmanReparameterizer,
    parse_lyrics_to_units, build_duration_scaffold,
    pd_sbar_verify_forward,
)
from acestep.training_v2.fixed_lora_module import _inject_lora_on_rewired_layers


def load_lora(model, checkpoint_path: str, rewired_layers: list, rank: int):
    """Load PD-SBAR LoRA weights onto the model."""
    from safetensors.torch import load_file as sf_load
    state_dict = sf_load(checkpoint_path)
    _inject_lora_on_rewired_layers(
        model, rewired_layers=rewired_layers,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        rank=rank, alpha=rank * 2,
    )
    # Map safetensors keys to model parameters
    model_sd = model.state_dict()
    for sd_key, sd_tensor in state_dict.items():
        if sd_key in model_sd and model_sd[sd_key].shape == sd_tensor.shape:
            model_sd[sd_key].copy_(sd_tensor)
    model.load_state_dict(model_sd, strict=False)
    print(f"  LoRA loaded from {checkpoint_path}")


def evaluate_config(model, planner, reparam, batch, device, label: str) -> dict:
    """Run PD-SBAR evaluation for one configuration."""
    from acestep.phase_memory import pd_sbar_verify_forward as _verify
    t0 = time.time()
    diag = _verify(model, planner, reparam, batch, device)
    elapsed = time.time() - t0
    diag["label"] = label
    diag["time_seconds"] = round(elapsed, 1)
    return diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/root/autodl-tmp/Ace-Step1.5/checkpoints")
    parser.add_argument("--model-variant", type=str, default="sft")
    parser.add_argument("--dataset-dir", type=str,
                        default="/root/autodl-tmp/musicdata/train_tensors")
    parser.add_argument("--exp-dir", type=str,
                        default="/root/autodl-tmp/pd_sbar_exp_final1")
    parser.add_argument("--output-dir", type=str,
                        default=str(ACE_ROOT / "eval_prompts" / "pd_sbar_final"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rewired_layers = [8, 12, 16, 20]
    lora_rank = 16

    # ---- 1. Load model -----------------------------------------------------
    print("[1/4] Loading model...")
    model = load_decoder_for_training(
        checkpoint_dir=args.checkpoint_dir, variant=args.model_variant,
        device=str(device), precision="bf16",
    )
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    if hasattr(model.config, "_attn_implementation_compiled"):
        model.config._attn_implementation_compiled = None
    model.eval()
    print(f"  Model loaded on {device}")

    # ---- 2. Load data ------------------------------------------------------
    print("[2/4] Loading dataset...")
    from acestep.training.data_module import PreprocessedTensorDataset, collate_preprocessed_batch
    dataset = PreprocessedTensorDataset(args.dataset_dir)
    batch = collate_preprocessed_batch([dataset[i] for i in range(min(4, len(dataset)))])
    batch = {k: v[:1] if isinstance(v, (torch.Tensor, list)) else v for k, v in batch.items()}
    T_raw = batch["target_latents"].shape[1]
    T_audio = T_raw // max(getattr(model.config, "patch_size", 2), 1)
    L_enc = batch["encoder_hidden_states"].shape[1]
    print(f"  T_audio={T_audio} L_enc={L_enc}")

    # ---- 3. Build scaffold + planner ---------------------------------------
    print("[3/4] Building scaffold and planner...")
    meta = (batch.get("metadata") or batch.get("metadatas") or [{}])[0]
    lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""
    section_ids = batch.get("section_ids", torch.zeros(L_enc, dtype=torch.long))
    if section_ids is not None:
        section_ids = section_ids.to(device)

    units, _, debug = parse_lyrics_to_units(
        lyrics_text, section_ids[0].cpu() if section_ids is not None else torch.zeros(L_enc),
        auto_transition_ratios=dict(intro=0., outro=0., chorus_to_verse=0., chorus_to_bridge=0., bridge_to_chorus=0.),
    )
    tcm = debug.get("tag_control_mask", None)
    scaffold = build_duration_scaffold(units, text_len=L_enc, tag_control_mask=tcm)
    scaffold = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}

    planner = PDSBARPlanner(sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
                             slack_ratio=0.08, budget_smoothing=0.1, eta_dual=1.0, lambda_max=3.0).to(device).float()
    planner.build_from_scaffold(scaffold, T_audio, batch_size=1, device=device)
    print(f"  Units: {len(units)}")

    # ---- 4. Evaluate configurations ----------------------------------------
    print("[4/4] Running evaluations...\n")
    results = {}

    # Baseline (no PD-SBAR)
    reparam_baseline = SinkhornBregmanReparameterizer(planner, rewired_layers, alpha=0.0, gamma_leak=0.0, beta_dual=0.0)
    diag = evaluate_config(model, planner, reparam_baseline, batch, device, "baseline")
    results["baseline"] = diag
    print(f"  baseline: Δattn={diag.get('attention_delta',0):.4f} Δlatent={diag.get('latent_delta',0):.4f}")

    # PD-SBAR inference-only (α=0.6, γ=3.0, β=1.0)
    reparam = SinkhornBregmanReparameterizer(planner, rewired_layers, alpha=0.6, gamma_leak=3.0, beta_dual=1.0)
    diag = evaluate_config(model, planner, reparam, batch, device, "pd_sbar_inference_only")
    results["pd_sbar_inference_only"] = diag
    print(f"  inference_only: Δattn={diag.get('attention_delta',0):.4f} Δlatent={diag.get('latent_delta',0):.4f}")

    # PD-SBAR + LoRA epoch 1
    model1 = load_decoder_for_training(
        checkpoint_dir=args.checkpoint_dir, variant=args.model_variant,
        device=str(device), precision="bf16",
    )
    for m in model1.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    if hasattr(model1.config, "_attn_implementation_compiled"):
        model1.config._attn_implementation_compiled = None
    load_lora(model1, f"{args.exp_dir}/checkpoints/epoch_1_loss_1.0541/pd_sbar_lora.safetensors", rewired_layers, lora_rank)
    model1.eval()
    pl1 = PDSBARPlanner(sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
                         slack_ratio=0.08, budget_smoothing=0.1, eta_dual=1.0, lambda_max=3.0).to(device).float()
    pl1.build_from_scaffold(scaffold, T_audio, batch_size=1, device=device)
    rp1 = SinkhornBregmanReparameterizer(pl1, rewired_layers, alpha=0.6, gamma_leak=3.0, beta_dual=1.0)
    diag = evaluate_config(model1, pl1, rp1, batch, device, "pd_sbar_epoch1")
    results["pd_sbar_epoch1"] = diag
    print(f"  epoch1: Δattn={diag.get('attention_delta',0):.4f} Δlatent={diag.get('latent_delta',0):.4f}")

    # PD-SBAR + LoRA epoch 2
    model2 = load_decoder_for_training(
        checkpoint_dir=args.checkpoint_dir, variant=args.model_variant,
        device=str(device), precision="bf16",
    )
    for m in model2.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    if hasattr(model2.config, "_attn_implementation_compiled"):
        model2.config._attn_implementation_compiled = None
    load_lora(model2, f"{args.exp_dir}/checkpoints/epoch_2_loss_0.9965/pd_sbar_lora.safetensors", rewired_layers, lora_rank)
    model2.eval()
    pl2 = PDSBARPlanner(sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
                         slack_ratio=0.08, budget_smoothing=0.1, eta_dual=1.0, lambda_max=3.0).to(device).float()
    pl2.build_from_scaffold(scaffold, T_audio, batch_size=1, device=device)
    rp2 = SinkhornBregmanReparameterizer(pl2, rewired_layers, alpha=0.6, gamma_leak=3.0, beta_dual=1.0)
    diag = evaluate_config(model2, pl2, rp2, batch, device, "pd_sbar_epoch2")
    results["pd_sbar_epoch2"] = diag
    print(f"  epoch2: Δattn={diag.get('attention_delta',0):.4f} Δlatent={diag.get('latent_delta',0):.4f}")

    # PD-SBAR + LoRA epoch 3 (best loss overall)
    model3 = load_decoder_for_training(
        checkpoint_dir=args.checkpoint_dir, variant=args.model_variant,
        device=str(device), precision="bf16",
    )
    for m in model3.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    if hasattr(model3.config, "_attn_implementation_compiled"):
        model3.config._attn_implementation_compiled = None
    load_lora(model3, f"{args.exp_dir}/checkpoints/epoch_3_loss_0.9945/pd_sbar_lora.safetensors", rewired_layers, lora_rank)
    model3.eval()
    pl3 = PDSBARPlanner(sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
                         slack_ratio=0.08, budget_smoothing=0.1, eta_dual=1.0, lambda_max=3.0).to(device).float()
    pl3.build_from_scaffold(scaffold, T_audio, batch_size=1, device=device)
    rp3 = SinkhornBregmanReparameterizer(pl3, rewired_layers, alpha=0.6, gamma_leak=3.0, beta_dual=1.0)
    diag = evaluate_config(model3, pl3, rp3, batch, device, "pd_sbar_epoch3")
    results["pd_sbar_epoch3"] = diag
    print(f"  epoch3: Δattn={diag.get('attention_delta',0):.4f} Δlatent={diag.get('latent_delta',0):.4f}")

    # ---- Save results ------------------------------------------------------
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n Results saved to {output_dir / 'results.json'}")

    # ---- Print comparison table --------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'Config':<30} {'Δattn':>8} {'Δlogit':>8} {'Δlatent':>8} {'ent_bef':>8} {'ent_aft':>8} "
          f"{'leak_b':>8} {'leak_a':>8} {'Λ_mean':>8}")
    print("-" * 100)
    for k in ["baseline", "pd_sbar_inference_only", "pd_sbar_epoch1", "pd_sbar_epoch2", "pd_sbar_epoch3"]:
        d = results.get(k, {})
        print(f"{k:<30} {d.get('attention_delta',0):>8.4f} {d.get('logit_delta',0):>8.2f} "
              f"{d.get('latent_delta',0):>8.4f} {d.get('entropy_before',0):>8.2f} "
              f"{d.get('entropy_after',0):>8.2f} {d.get('leak_mass_before',0):>8.6f} "
              f"{d.get('leak_mass_after',0):>8.6f} {d.get('Lambda_mean',0):>8.4f}")
    print("=" * 100)

    # ---- Verdict -----------------------------------------------------------
    attn_ok = any(results.get(k, {}).get("attention_delta", 0) > 0.05 for k in results)
    latent_ok = any(results.get(k, {}).get("latent_delta", 0) > 0.01 for k in results)
    leak_reduction = (
        results.get("baseline", {}).get("leak_mass_after", 1) -
        results.get("pd_sbar_epoch3", {}).get("leak_mass_after", 0)
    ) > 0
    pref_improvement = (
        results.get("baseline", {}).get("prefix_drift", 0) -
        results.get("pd_sbar_epoch3", {}).get("prefix_drift", 0)
    ) > 0

    if attn_ok and latent_ok and (leak_reduction or pref_improvement):
        verdict = "PASS_STRUCTURAL"
    elif attn_ok and latent_ok:
        verdict = "PASS_EFFECT_ONLY"
    elif not attn_ok:
        verdict = "FAIL_EFFECT"
    else:
        verdict = "FAIL_EFFECT"

    print(f"\n  ==== FINAL VERDICT: {verdict} ====")
    results["verdict"] = verdict
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"  Full results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
