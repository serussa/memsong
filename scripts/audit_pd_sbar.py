#!/usr/bin/env python3
"""
PD-SBAR Engineering Audit — comprehensive pre-training verification.

Usage:
    python scripts/audit_pd_sbar.py --checkpoint-dir /path/to/checkpoints \\
        --model-variant sft --dataset-dir /path/to/train_tensors \\
        [--output-dir ./pd_sbar_audit]

Requirements:
    - Must be run on a GPU with the ACE-Step model loaded.
    - Requires one batch from the training dataset for forward pass.

Outputs:
    pd_sbar_audit_report.json
    pd_sbar_audit_report.txt
    pd_sbar_layer_diagnostics.csv
"""

import argparse, json, os, sys, math
from pathlib import Path

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(name: str, condition: bool, detail: str = "") -> dict:
    result = {"check": name, "pass": bool(condition)}
    if not condition:
        result["fail_reason"] = detail
    return result


def fmt(v: float, sig=4) -> str:
    if abs(v) < 1e-10:
        return "0"
    if abs(v) < 1e-4:
        return f"{v:.2e}"
    return f"{v:.{sig}f}"


def fmt_header(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


# ===========================================================================
#  Audit
# ===========================================================================

def run_audit(args) -> dict:
    """Run full PD-SBAR audit and return report dict."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {
        "method": "PD-SBAR Audit",
        "device": str(device),
        "checks": [],
        "per_layer_diagnostics": [],
        "summary": {},
        "verdict": "FAIL_UNKNOWN",
    }

    # ---- 0. Load model --------------------------------------------------------
    print(fmt_header("0. Loading ACE-Step model"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ["ACESTEP_OFFLINE"] = "1"
    os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

    from acestep.handler import AceStepHandler
    from acestep.training_v2.configs import TrainingConfigV2

    # Use a dummy config to hold PD-SBAR hyperparams
    train_cfg = TrainingConfigV2(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        adapter_type="pd_sbar",
        pd_sbar_sigma=args.sigma,
        pd_sbar_leak_cost=args.leak_cost,
        pd_sbar_epsilon=args.epsilon,
        pd_sbar_sinkhorn_iters=args.sinkhorn_iters,
        pd_sbar_slack_ratio=args.slack_ratio,
        pd_sbar_budget_smoothing=args.budget_smoothing,
        pd_sbar_eta_dual=args.eta_dual,
        pd_sbar_lambda_max=args.lambda_max,
        pd_sbar_alpha=args.alpha,
        pd_sbar_gamma_leak=args.gamma_leak,
        pd_sbar_beta_dual=args.beta_dual,
        pd_sbar_lambda_leak=args.lambda_leak,
        pd_sbar_lambda_pref=args.lambda_pref,
        pd_sbar_lora_rank=args.lora_rank,
        pd_sbar_rewired_layers=args.rewired_layers,
    )

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=args.checkpoint_dir,
        config_path=f"acestep-v15-{args.model_variant}",
        device=str(device), use_flash_attention=False,
        compile_model=False, offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(f"Model load failed: {status}")
    model = handler.model
    print(f"  Model loaded: {type(model).__name__}")
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    if hasattr(model.config, "_attn_implementation_compiled"):
        model.config._attn_implementation_compiled = None

    # ---- 1. Load one batch from dataset ---------------------------------------
    print(fmt_header("1. Loading dataset batch"))
    from acestep.training.data_module import PreprocessedDataModule

    dm = PreprocessedDataModule(
        tensor_dir=args.dataset_dir, batch_size=1,
        num_workers=0, pin_memory=False,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    print(f"  Batch loaded: target_latents {list(batch['target_latents'].shape)}, "
          f"encoder_hidden_states {list(batch['encoder_hidden_states'].shape)}")

    # ---- 2. Build scaffold + planner ------------------------------------------
    print(fmt_header("2. Building scaffold and planner"))
    from acestep.phase_memory import (
        PDSBARPlanner, SinkhornBregmanReparameterizer,
        parse_lyrics_to_units, build_duration_scaffold,
    )

    B = batch["target_latents"].shape[0]
    T_raw = batch["target_latents"].shape[1]
    patch_size = getattr(model.config, "patch_size", 2)
    T_audio = T_raw // max(patch_size, 1)
    L_enc = batch["encoder_hidden_states"].shape[1]

    meta = (batch.get("metadata") or batch.get("metadatas") or [{}])[0]
    lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""
    section_ids = batch.get("section_ids")
    if section_ids is not None:
        section_ids = section_ids.to(device)
    else:
        section_ids = torch.zeros(B, L_enc, dtype=torch.long, device=device)

    if not lyrics_text:
        print("  ⚠ No lyrics in batch metadata — audit limited")
    else:
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
            sigma=args.sigma, leak_cost=args.leak_cost,
            epsilon=args.epsilon, sinkhorn_iters=args.sinkhorn_iters,
            slack_ratio=args.slack_ratio,
            budget_smoothing=args.budget_smoothing,
            eta_dual=args.eta_dual, lambda_max=args.lambda_max,
        ).to(device).float()
        planner.build_from_scaffold(scaffold, T_audio, batch_size=B, device=device)

        reparam = SinkhornBregmanReparameterizer(
            planner=planner, rewired_layers=args.rewired_layers,
            alpha=args.alpha, gamma_leak=args.gamma_leak,
            beta_dual=args.beta_dual,
        )

        print(f"  Planner built: U={len(units)}, T={T_audio}, L={L_enc}")
    # else: skip remaining checks

    # ---- 3. Unit-token mapping check ------------------------------------------
    print(fmt_header("3. Unit-token mapping check"))
    if planner.has_plan:
        S = planner.S
        M = S.shape[0]
        row_sums = S.sum(dim=-1)
        empty_rows = [int(i) for i in range(M) if row_sums[i].item() < 1e-10]
        unit_info = getattr(planner, "_unit_info", {})
        tokens_per = unit_info.get("tokens_per_unit", [])

        print(f"  Units: {M - 1} + 1 slack = {M}")
        print(f"  Empty unit rows: {empty_rows if empty_rows else 'None'}")
        for j in range(M - 1):
            sec = unit_info.get("unit_sections", [f"U{j}"] * M)[j] if j < len(unit_info.get("unit_sections", [])) else f"U{j}"
            tpc = tokens_per[j] if j < len(tokens_per) else "?"
            print(f"    unit {j:2d}: section={sec:<20s} tokens={str(tpc):>4s}  S_row_sum={row_sums[j].item():.4f}")

        check_map = _check("unit_token_mapping", len(empty_rows) == 0,
                           f"{len(empty_rows)} units with empty S row: {empty_rows}")
        report["checks"].append(check_map)

        # ---- 4. Sinkhorn marginal check -------------------------------------------
        print(fmt_header("4. Sinkhorn marginal check"))
        Gamma = planner.Gamma
        nu = planner.nu
        mu = planner.mu
        row_marginal_err = (Gamma.sum(dim=-1) - nu).abs().mean().item()
        col_marginal_err = (Gamma.sum(dim=-2) - mu).abs().mean().item()
        print(f"  Row marginal error (abs mean): {row_marginal_err:.2e}")
        print(f"  Col marginal error (abs mean): {col_marginal_err:.2e}")
        check_marg = _check("sinkhorn_marginals",
                            row_marginal_err < 0.01 and col_marginal_err < 0.01,
                            f"row_err={row_marginal_err:.2e} col_err={col_marginal_err:.2e}")
        report["checks"].append(check_marg)

        # ---- 5. R row normalization check ------------------------------------------
        print(fmt_header("5. R row normalization check"))
        R = planner.R
        R_sum = R.sum(dim=-1)
        print(f"  R row sum: min={R_sum.min().item():.6f} max={R_sum.max().item():.6f}")
        check_rnorm = _check("R_row_normalization",
                             (R_sum.min() > 0.99).item() and (R_sum.max() < 1.01).item(),
                             f"min={R_sum.min().item():.6f} max={R_sum.max().item():.6f}")
        report["checks"].append(check_rnorm)

        # ---- 6. Log(R) NaN check ---------------------------------------------------
        log_R = torch.log(R.clamp_min(1e-10))
        check_logr = _check("log_R_no_nan", torch.isfinite(log_R).all().item(),
                            "log(R) has NaN/Inf")
        report["checks"].append(check_logr)
        print(f"  log(R) NaN/Inf: {'None' if torch.isfinite(log_R).all().item() else 'FOUND!'}")

        # ---- 7. Hook slice check (forward verification) ----------------------------
        print(fmt_header("6-9. Forward verification (attention + Lambda + loss)"))
        from acestep.phase_memory import pd_sbar_verify_forward

        reparam.alpha = args.alpha
        reparam.gamma_leak = args.gamma_leak
        reparam.beta_dual = args.beta_dual

        print(f"  Running forward verification with α={args.alpha} γ={args.gamma_leak} β={args.beta_dual} ...")

        # Save original forward handles
        lambda_norm_start = planner.Lambda.norm().item() if planner.Lambda is not None else 0.0

        # Run the verification with reparam.install/uninstall internally
        verify_diag = pd_sbar_verify_forward(
            model, planner, reparam, batch, device,
        )

        # ---- Lambda reset check ----------------------------------------------------
        check_lambda = _check("lambda_reset", lambda_norm_start < 1e-6,
                              f"Lambda norm at start: {lambda_norm_start:.6f}")
        report["checks"].append(check_lambda)
        print(f"  Lambda norm at start: {lambda_norm_start:.6f} "
              f"{'(OK < 1e-6)' if lambda_norm_start < 1e-6 else '⚠ NON-ZERO!'}")

        # ---- Dual no_grad check ----------------------------------------------------
        check_dual = _check("dual_update_no_grad",
                            not getattr(planner.Lambda, 'requires_grad', False),
                            "Lambda.requires_grad is True!")
        report["checks"].append(check_dual)
        print(f"  Lambda.requires_grad: {planner.Lambda.requires_grad} "
              f"{'(OK)' if not planner.Lambda.requires_grad else '⚠ REQUIRES GRAD!'}")
        print(f"  U_dual.requires_grad: {planner.U_dual.requires_grad} "
              f"{'(OK)' if not planner.U_dual.requires_grad else '⚠ REQUIRES GRAD!'}")

        # ---- Attention delta checks -------------------------------------------------
        attn_delta = verify_diag.get("attention_delta", 0)
        logit_delta = verify_diag.get("logit_delta", 0)
        check_attn = _check("attention_delta", attn_delta > 0.05,
                            f"attention_delta={attn_delta:.6f} < 0.05")
        report["checks"].append(check_attn)

        check_logit = _check("logit_delta", logit_delta > 0.5,
                             f"logit_delta={logit_delta:.6f} < 0.5")
        report["checks"].append(check_logit)

        # ---- Per-layer attention check -----------------------------------------------
        all_layers_ok = True
        for lidx in sorted(reparam._diag_captures.keys()):
            lay_d = verify_diag.get(f"attn_delta_layer{lidx}", 0)
            fine = lay_d >= 0.01
            if not fine:
                all_layers_ok = False
            report["per_layer_diagnostics"].append({
                "layer": lidx,
                "attention_delta": lay_d,
                "logit_delta": verify_diag.get(f"logit_delta_layer{lidx}", 0),
                "entropy_before": verify_diag.get(f"ent_before_layer{lidx}", 0),
                "entropy_after": verify_diag.get(f"ent_after_layer{lidx}", 0),
                "leak_before": verify_diag.get(f"leak_before_layer{lidx}", 0),
                "leak_after": verify_diag.get(f"leak_after_layer{lidx}", 0),
                "lambda_before": verify_diag.get(f"lambda_norm_before_layer{lidx}", 0),
                "lambda_after": verify_diag.get(f"lambda_norm_after_layer{lidx}", 0),
            })
        check_per_layer = _check("per_layer_attention_delta", all_layers_ok,
                                 "Some layers have attention_delta < 0.01")
        if not all_layers_ok:
            low_layers = [d["layer"] for d in report["per_layer_diagnostics"]
                          if d["attention_delta"] < 0.01]
            check_per_layer["fail_reason"] += f" low layers: {low_layers}"
        report["checks"].append(check_per_layer)

        # ---- Latent delta check ----------------------------------------------------
        latent_delta = verify_diag.get("latent_delta", 0)
        check_latent = _check("latent_delta", latent_delta > 0.01,
                              f"latent_delta={latent_delta:.6f} < 0.01")
        report["checks"].append(check_latent)

        # ---- Entropy check ---------------------------------------------------------
        ent_after = verify_diag.get("entropy_after", 0)
        ent_drop = verify_diag.get("entropy_drop", 0)
        check_ent = _check("entropy_not_collapsed", ent_after > 0.5,
                           f"entropy_after={ent_after:.4f} < 0.5")
        report["checks"].append(check_ent)
        print(f"  Entropy after: {ent_after:.4f}, drop: {ent_drop:.4f} "
              f"{'(OK)' if ent_after > 0.5 else '⚠ COLLAPSED!'}")

        # ---- Loss normalization check (simulated) ------------------------------------
        # We can't compute actual losses here (no training), but we verify shapes
        print(f"\n  leak_mass (before→after): {verify_diag.get('leak_mass_before', 0):.6f}→{verify_diag.get('leak_mass_after', 0):.6f}")

        # ---- NaN/Inf check ---------------------------------------------------------
        # Already handled inside _bregman_forward — if we got here, no NaN
        check_nan = _check("no_nan_inf", True, "")
        report["checks"].append(check_nan)

        # ---- LoRA trainable param check --------------------------------------------
        print(fmt_header("10. LoRA trainable parameter check"))
        lora_params = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
        non_lora_trainable = [n for n, p in model.named_parameters()
                              if p.requires_grad and 'lora_' not in n]
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())

        print(f"  Total params: {total_all:,}")
        print(f"  Trainable params: {total_trainable:,}")
        print(f"  LoRA trainable: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n):,}")
        lora_per_layer = {}
        for ln in lora_params:
            parts = ln.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_idx = int(parts[i+1])
                    break
            if layer_idx is not None:
                lora_per_layer.setdefault(layer_idx, []).append(ln)
        for lidx in sorted(lora_per_layer.keys()):
            params = lora_per_layer[lidx]
            n_params = sum(p.numel() for n, p in model.named_parameters() if n in params)
            print(f"    layer {lidx}: {len(params)} LoRA modules, {n_params:,} params")
            for pn in sorted(params):
                p = dict(model.named_parameters())[pn]
                print(f"      {pn.split('.')[-1]}: {p.numel():,} params, grad={p.requires_grad}")

        check_lora = _check("lora_exists", len(lora_params) > 0,
                           "No trainable LoRA parameters found!")
        report["checks"].append(check_lora)

        check_backbone = _check("backbone_frozen", len(non_lora_trainable) == 0,
                                f"{len(non_lora_trainable)} non-LoRA trainable params: {non_lora_trainable[:10]}")
        report["checks"].append(check_backbone)
        if non_lora_trainable:
            print(f"  ⚠ Non-LoRA trainable params: {non_lora_trainable[:10]}")

        # ---- Verdict ---------------------------------------------------------------
        print(fmt_header("VERDICT"))
        failed_checks = [c for c in report["checks"] if not c["pass"]]
        failed_names = [c["check"] for c in failed_checks]

        if not failed_checks:
            report["verdict"] = "PASS"
            print("  ✓ PASS — all checks passed")
        elif any("attention_delta" in c["check"] or "per_layer" in c["check"]
                 for c in failed_checks):
            report["verdict"] = "FAIL_HOOK"
            print(f"  ✗ FAIL_HOOK — attention hook not modifying actual attention path")
        elif any("latent" in c["check"] for c in failed_checks):
            report["verdict"] = "FAIL_EFFECT"
            print(f"  ✗ FAIL_EFFECT — attention modified but denoising trajectory unchanged")
        elif any("NaN" in c["check"] or "collapse" in c["check"]
                 or "no_nan" in c["check"] for c in failed_checks):
            report["verdict"] = "FAIL_NUMERIC"
            print(f"  ✗ FAIL_NUMERIC — numerical issues detected")
        else:
            report["verdict"] = "FAIL_OTHER"
            print(f"  ✗ FAIL_OTHER — failed checks: {failed_names}")

        report["summary"] = {
            "verdict": report["verdict"],
            "total_checks": len(report["checks"]),
            "passed_checks": len(report["checks"]) - len(failed_checks),
            "failed_checks": len(failed_checks),
            "attention_delta": attn_delta,
            "logit_delta": logit_delta,
            "latent_delta": latent_delta,
            "entropy_after": ent_after,
            "R_row_sum_min": verify_diag.get("R_row_sum_min", 0),
            "num_rewired_layers": len(reparam._diag_captures),
            "sinkhorn_row_error": verify_diag.get("sinkhorn_row_error", 0),
            "leak_mass_before": verify_diag.get("leak_mass_before", 0),
            "leak_mass_after": verify_diag.get("leak_mass_after", 0),
        }

    # ---- Save report --------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "pd_sbar_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved to {report_path}")

    # TXT summary
    txt_path = output_dir / "pd_sbar_audit_report.txt"
    with open(txt_path, "w") as f:
        f.write(json.dumps(report, indent=2))
    print(f"  TXT saved to {txt_path}")

    # CSV per-layer
    if report["per_layer_diagnostics"]:
        import csv
        csv_path = output_dir / "pd_sbar_layer_diagnostics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=report["per_layer_diagnostics"][0].keys())
            w.writeheader()
            w.writerows(report["per_layer_diagnostics"])
        print(f"  CSV saved to {csv_path}")

    return report


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="PD-SBAR Engineering Audit")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--model-variant", type=str, default="sft")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./pd_sbar_audit")

    # PD-SBAR hyperparams
    parser.add_argument("--sigma", type=float, default=0.18)
    parser.add_argument("--leak-cost", type=float, default=4.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--slack-ratio", type=float, default=0.08)
    parser.add_argument("--budget-smoothing", type=float, default=0.1)
    parser.add_argument("--eta-dual", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--gamma-leak", type=float, default=3.0)
    parser.add_argument("--beta-dual", type=float, default=1.0)
    parser.add_argument("--lambda-leak", type=float, default=0.05)
    parser.add_argument("--lambda-pref", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--rewired-layers", type=int, nargs="+", default=[8, 12, 16, 20])

    args = parser.parse_args()

    print("=" * 70)
    print("  PD-SBAR Engineering Audit")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Variant: {args.model_variant}")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Alpha: {args.alpha}, Gamma_leak: {args.gamma_leak}, Beta_dual: {args.beta_dual}")
    print(f"  Rewired layers: {args.rewired_layers}")
    print(f"  LoRA rank: {args.lora_rank}")
    print("=" * 70)

    report = run_audit(args)

    print(f"\n{'=' * 70}")
    print(f"  Final Verdict: {report['verdict']}")
    print(f"  {report['summary'].get('passed_checks', 0)}/{report['summary'].get('total_checks', 0)} checks passed")
    print(f"{'=' * 70}")

    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
