#!/usr/bin/env python3
"""
PD-SBAR Inference-Only Verification — Run BEFORE training.

Checks:
  - attention_delta > 0.05
  - logit_delta > 0.5
  - latent_delta > 0.01

If ALL pass → safe to train.
If ANY fail → stop, output FAIL_EFFECT / FAIL_HOOK.
"""
import argparse, json, os, sys, math, time
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
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/root/autodl-tmp/Ace-Step1.5/checkpoints")
    parser.add_argument("--model-variant", type=str, default="sft")
    parser.add_argument("--dataset-dir", type=str,
                        default="/root/autodl-tmp/musicdata/train_tensors")
    parser.add_argument("--output-dir", type=str,
                        default=str(ACE_ROOT / "eval_prompts" / "pd_sbar_verify"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PD-SBAR Inference-Only Verification")
    print("  GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    print("=" * 70)

    # ---- 1. Load model -----------------------------------------------------
    print("\n[1/5] Loading ACE-Step model (forcing eager attention)...")
    model = load_decoder_for_training(
        checkpoint_dir=args.checkpoint_dir,
        variant=args.model_variant,
        device=str(device),
        precision="bf16",
    )
    # Force eager attention (required for PD-SBAR monkey-patch)
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            if m.config._attn_implementation != "eager":
                m.config._attn_implementation = "eager"
                print(f"  ⚡ Forced {type(m).__name__} to eager attention")
    if hasattr(model.config, "_attn_implementation_compiled"):
        model.config._attn_implementation_compiled = None
    print(f"  Model loaded: {type(model).__name__}, device={device}")

    # ---- 2. Load dataset batch ----------------------------------------------
    print("\n[2/5] Loading dataset batch...")
    from acestep.training.data_module import PreprocessedTensorDataset, collate_preprocessed_batch
    dataset = PreprocessedTensorDataset(args.dataset_dir)
    batch = collate_preprocessed_batch([dataset[i] for i in range(min(4, len(dataset)))])
    # Select first sample from batch
    batch = {k: v[:1] if isinstance(v, torch.Tensor) else v[:1] if isinstance(v, list) else v
             for k, v in batch.items()}
    B = batch["target_latents"].shape[0]
    T_raw = batch["target_latents"].shape[1]
    patch_size = getattr(model.config, "patch_size", 2)
    T_audio = T_raw // max(patch_size, 1)
    L_enc = batch["encoder_hidden_states"].shape[1]
    print(f"  Batch: T_raw={T_raw} T_audio={T_audio} L_enc={L_enc}")

    # ---- 3. Build scaffold + planner ---------------------------------------
    print("\n[3/5] Building scaffold and planner...")
    meta = (batch.get("metadata") or batch.get("metadatas") or [{}])[0]
    lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""
    section_ids = batch.get("section_ids")
    if section_ids is not None:
        section_ids = section_ids.to(device)
    else:
        section_ids = torch.zeros(B, L_enc, dtype=torch.long, device=device)

    print(f"  Lyrics length: {len(lyrics_text)} chars")
    units, _, debug = parse_lyrics_to_units(
        lyrics_text, section_ids[0].cpu(),
        auto_transition_ratios={
            "intro": 0.0, "outro": 0.0,
            "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
            "bridge_to_chorus": 0.0,
        },
    )
    tcm = debug.get("tag_control_mask", None)
    scaffold = build_duration_scaffold(units, text_len=L_enc, tag_control_mask=tcm)
    scaffold = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}

    planner = PDSBARPlanner(
        sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
        slack_ratio=0.08, budget_smoothing=0.1,
        eta_dual=1.0, lambda_max=3.0,
    ).to(device).float()
    planner.build_from_scaffold(scaffold, T_audio, batch_size=B, device=device)

    reparam = SinkhornBregmanReparameterizer(
        planner=planner, rewired_layers=[8, 12, 16, 20],
        alpha=0.6, gamma_leak=3.0, beta_dual=1.0,
    )

    unit_info = getattr(planner, "_unit_info", {})
    print(f"  Units: {unit_info.get('num_units', '?')} lyric, {unit_info.get('num_silence_units', '?')} silence")
    print(f"  Empty units: {unit_info.get('empty_unit_ids', [])}")
    print(f"  S matrix rows all >0: {all(planner.S[j].sum() > 1e-10 for j in range(planner.S.shape[0]))}")

    # ---- 4/5. Run verification with escalation -----------------------------
    print("\n[4/5] Running forward verification with escalation...")
    from acestep.phase_memory import pd_sbar_verify_or_escalate as _escalate

    t0 = time.time()
    try:
        diag = _escalate(model, planner, reparam, batch, device)
        t_elapsed = time.time() - t0
        print(f"\n  Verification took {t_elapsed:.1f}s")
    except RuntimeError as e:
        print(f"\n  ==== VERDICT: FAIL_HOOK ====")
        print(f"  {e}")
        diag = getattr(reparam, '_diag_captures', {})
        report = {"verdict": "FAIL_HOOK", "error": str(e),
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        (output_dir / "pd_sbar_verify_report.json").write_text(json.dumps(report, indent=2))
        sys.exit(1)

    attn_final = diag.get("attention_delta", 0)
    logit_final = diag.get("logit_delta", 0)
    latent_final = diag.get("latent_delta", 0)
    attn_ok = attn_final > 0.05
    logit_ok = logit_final > 0.5
    latent_ok = latent_final > 0.01
    all_ok = attn_ok and logit_ok and latent_ok

    verdict = "PASS" if all_ok else "FAIL_HOOK"

    print(f"\n  ==== VERDICT: {verdict} ====")
    if not all_ok:
        print(f"  Reason: {reason}")
    else:
        print(f"  All thresholds met — safe to train.")

    # Save diagnostics
    diag["verdict"] = verdict
    diag["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    diag["device"] = str(device)
    diag["model_variant"] = args.model_variant

    report_path = output_dir / "pd_sbar_verify_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pd_sbar_verify_report.json").write_text(json.dumps(diag, indent=2))
    print(f"  Report saved to {report_path}")

    # Also save a clean txt
    txt = []
    txt.append("=" * 60)
    txt.append("PD-SBAR Inference-Only Verification Report")
    txt.append("=" * 60)
    txt.append(f"Device: {device}")
    txt.append(f"Model: acestep-v15-{args.model_variant}")
    txt.append(f"Timestamp: {diag['timestamp']}")
    txt.append("")
    txt.append("Internal Metrics:")
    txt.append(f"  attention_delta: {diag.get('attention_delta', 0):.6f}  (need >0.05)")
    txt.append(f"  logit_delta:     {diag.get('logit_delta', 0):.6f}  (need >0.5)")
    txt.append(f"  latent_delta:    {diag.get('latent_delta', 0):.6f}  (need >0.01)")
    txt.append(f"  entropy_before:  {diag.get('entropy_before', 0):.4f}")
    txt.append(f"  entropy_after:   {diag.get('entropy_after', 0):.4f}")
    txt.append(f"  leak_mass:       {diag.get('leak_mass_before', 0):.6f} → {diag.get('leak_mass_after', 0):.6f}")
    txt.append(f"  R_row_sum:       {diag.get('R_row_sum_min', 0):.4f} – {diag.get('R_row_sum_max', 0):.4f}")
    txt.append(f"  R_entropy:       {diag.get('R_entropy', 0):.4f}")
    txt.append(f"  Sinkhorn row err: {diag.get('sinkhorn_row_error', 0):.2e}")
    txt.append(f"  Sinkhorn col err: {diag.get('sinkhorn_col_error', 0):.2e}")
    txt.append("")
    txt.append("Per-Layer:")
    for lidx in sorted(reparam._diag_captures.keys()):
        txt.append(f"  Layer {lidx}: "
                   f"Δattn={diag.get(f'attn_delta_layer{lidx}', 0):.4f} "
                   f"Δlogit={diag.get(f'logit_delta_layer{lidx}', 0):.2f} "
                   f"ent={diag.get(f'ent_before_layer{lidx}', 0):.2f}→{diag.get(f'ent_after_layer{lidx}', 0):.2f} "
                   f"leak={diag.get(f'leak_before_layer{lidx}', 0):.4f}→{diag.get(f'leak_after_layer{lidx}', 0):.4f}")
    txt.append("")
    txt.append(f"VERDICT: {verdict}")
    if not all_ok:
        txt.append(f"REASON: {reason}")
    txt.append("=" * 60)
    (output_dir / "pd_sbar_verify_report.txt").write_text("\n".join(txt))
    print(f"  TXT saved")
    print(f"\n  Full report: {report_path}")

    # Exit with code
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
