#!/usr/bin/env python3
"""
Cold-start diagnosis & minimal fix experiment for TransportRetrievalAdapter.
Tests A/B/C/D configs, logs grads, and does inference with forced alpha.

Usage:
    python scripts/diagnose_transport_coldstart.py --output /root/ACE-Step-1.5/diag_output
"""

from __future__ import annotations
import os, sys, json, math, time, copy
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.phase_memory import (
    TransportRetrievalAdapter, PMRetrievalPhaseMemory,
    parse_lyrics_to_units, build_duration_scaffold, scaffold_progress,
)
from acestep.training_v2.timestep_sampling import sample_timesteps

# ── Sample ─────────────────────────────────────────────────────────────
SAMPLE_ID = "b704dd0b245aa2ebaf6229399ca942a13bc907cb_1754"
DATASET_DIR = Path("/root/autodl-tmp/musicdata/dataset")
TENSOR_DIR = Path("/root/autodl-tmp/musicdata/train_tensors")

# ── Configs ────────────────────────────────────────────────────────────
@dataclass
class AlphaConfig:
    label: str
    write_alpha_max: float
    write_alpha_init: float
    out_proj_init_std: float = 0.01
    # Option D: no weight decay on these groups
    no_wd_on_adapter: bool = False
    # Option C: fixed alpha warmup steps (0 = disabled)
    fixed_warmup_steps: int = 0
    fixed_alpha: float = 0.01

CONFIGS = [
    AlphaConfig("A_current", write_alpha_max=0.01, write_alpha_init=0.001),
    AlphaConfig("B_larger",  write_alpha_max=0.05, write_alpha_init=0.01),
    AlphaConfig("C_fixed_warmup", write_alpha_max=0.01, write_alpha_init=0.001, fixed_warmup_steps=500, fixed_alpha=0.01),
    AlphaConfig("D_no_wd",    write_alpha_max=0.01, write_alpha_init=0.001, no_wd_on_adapter=True),
    AlphaConfig("E_aggressive", write_alpha_max=0.10, write_alpha_init=0.03, out_proj_init_std=0.05),
]

INFERENCE_ALPHAS = [0.001, 0.01, 0.03]

LR = 1e-4
WEIGHT_DECAY = 0.01


def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return math.sqrt(total) if total > 0 else 0.0


def build_adapter(D: int, cfg: AlphaConfig, device: torch.device, dtype: torch.dtype) -> tuple:
    """Build fresh TransportRetrievalAdapter + PMRetrievalPhaseMemory."""
    pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True).to(device).float()
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        transport_mode="sinkhorn", sinkhorn_iters=5,
        transport_sigma=0.18, transport_qk_scale=1.0,
        write_alpha_init=cfg.write_alpha_init,
        write_alpha_max=cfg.write_alpha_max,
        out_proj_init_std=cfg.out_proj_init_std,
    ).to(device).float()
    adapt.train()
    pm.train()
    return pm, adapt


def get_param_groups(adapt, pm, cfg: AlphaConfig, lr: float, wd: float):
    """Build optimizer param groups with optional no-wd on adapter."""
    if not cfg.no_wd_on_adapter:
        return [
            {"params": pm.parameters(), "lr": lr},
            {"params": adapt.parameters(), "lr": lr},
        ]
    # Option D: no weight decay on adapter's core projections
    wd_names = {"write_logit", "q_mlp", "k_mlp", "v_mlp", "out_proj"}
    adapt_wd = []
    adapt_no_wd = []
    for name, p in adapt.named_parameters():
        base = name.split(".")[0]  # e.g. "q_mlp" from "q_mlp.0.weight"
        if base in wd_names:
            adapt_no_wd.append(p)
        else:
            adapt_wd.append(p)
    return [
        {"params": pm.parameters(), "lr": lr},
        {"params": adapt_wd, "lr": lr, "weight_decay": wd},
        {"params": adapt_no_wd, "lr": lr, "weight_decay": 0.0},
    ]


def run_training_steps(
    model, pm, adapt, data, cfg: AlphaConfig,
    device, dtype, output_dir, run_label: str,
    num_steps: int = 30,
) -> Dict[str, Any]:
    """Run num_steps training steps with given adapter config, log all metrics."""
    D = model.config.hidden_size
    B = 1
    T = data["target_latents"].shape[0]
    L = data["encoder_hidden_states"].shape[0]

    param_groups = get_param_groups(adapt, pm, cfg, LR, WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY)

    # Pre-compute scaffold (shared across steps)
    lyrics_text = (DATASET_DIR / f"{SAMPLE_ID}.lyrics.txt").read_text(encoding="utf-8").strip()
    parser = LyricsStructureParser()
    parsed = parser.parse(lyrics_text, num_chunks=L)
    section_ids_cpu = parsed.section_type_ids
    units, _, debug = parse_lyrics_to_units(lyrics_text, section_ids_cpu, auto_transition_ratios={})
    scaffold = build_duration_scaffold(units, text_len=L, tag_control_mask=debug.get("tag_control_mask"))

    # Per-step logged metrics
    history = []

    for step in range(num_steps):
        # ── Fixed-alpha warmup (Option C) ──────────────────────────────────
        if cfg.fixed_warmup_steps > 0 and step < cfg.fixed_warmup_steps:
            # Bypass learned write_alpha: directly set it
            with torch.no_grad():
                adapt.write_logit.fill_(math.log(cfg.fixed_alpha / (cfg.write_alpha_max - cfg.fixed_alpha + 1e-10)))

        # ── Prepare batch ─────────────────────────────────────────────────
        t, r = sample_timesteps(B, device, dtype, data_proportion=0.5, timestep_mu=-0.4, timestep_sigma=1.0, use_meanflow=False)
        x1 = torch.randn_like(data["target_latents"]).unsqueeze(0).to(device=device, dtype=dtype)
        x0 = data["target_latents"].unsqueeze(0).to(device=device, dtype=dtype)
        xt = t.view(-1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1)) * x0

        eh = data["encoder_hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
        eam = data["encoder_attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
        am = data["attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
        ctx = data["context_latents"].unsqueeze(0).to(device=device, dtype=dtype)

        # ── Warmup forward → collect H from layer 12 ─────────────────────
        hs_list = []
        def _whook(m, i, o): hs_list.append(o[0])
        handle = model.decoder.layers[12].register_forward_hook(_whook)
        with torch.no_grad():
            _ = model.decoder(hidden_states=xt, timestep=t, timestep_r=t,
                              attention_mask=am, encoder_hidden_states=eh,
                              encoder_attention_mask=eam, context_latents=ctx,
                              use_cache=False, output_attentions=False)
        handle.remove()

        H = hs_list[0].detach().float()
        pm_state = pm(H, t)

        # ── Build unit tensors ───────────────────────────────────────────
        token_to_unit = scaffold["token_to_unit"].to(device)
        lyric_mask_s = scaffold["lyric_mask"].to(device)
        unit_boundaries = scaffold["unit_boundaries"].to(device)
        unit_duration = scaffold["unit_duration"].to(device)
        u_section_ids = scaffold["unit_section_ids"].to(device)
        lyric_unit_mask = scaffold["lyric_unit_mask"].to(device)

        U = len(unit_boundaries) - 1
        c_all = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
        mu_all = unit_duration / unit_duration.sum()

        # Pool encoder hidden per unit
        eh_f = eh.float()
        unit_text_hidden_list = []
        for uid in range(U):
            is_lyric = lyric_unit_mask[uid].item()
            tmask = (token_to_unit == uid) & lyric_mask_s if is_lyric else (token_to_unit == uid)
            tmask_b = tmask.unsqueeze(0).expand(B, -1)
            if tmask_b.any():
                pooled = eh_f[tmask_b].view(B, -1, D).mean(dim=1)
            else:
                pooled = torch.zeros(B, D, device=device, dtype=torch.float32)
            unit_text_hidden_list.append(pooled)
        unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)

        # ── Adapter forward ──────────────────────────────────────────────
        p_audio = torch.linspace(0, 1, H.shape[1], device=device, dtype=torch.float32).unsqueeze(0)
        delta_h, Pi, diag = adapt(
            hidden_states=H, text_hidden=eh, pm_state=pm_state,
            p_audio=p_audio, unit_text_hidden=unit_text_hidden,
            unit_c_pos=c_all.unsqueeze(0), unit_mass=mu_all.unsqueeze(0),
            unit_section_id=u_section_ids.unsqueeze(0),
            unit_is_lyric=lyric_unit_mask.unsqueeze(0),
        )
        final_h = H + delta_h

        # ── Inject and forward decoder ───────────────────────────────────
        def _inject(m, i, o):
            return (final_h.to(dtype=o[0].dtype, device=o[0].device), *o[1:])
        inj_handle = model.decoder.layers[12].register_forward_hook(_inject)
        decoder_outputs = model.decoder(
            hidden_states=xt, timestep=t, timestep_r=t,
            attention_mask=am, encoder_hidden_states=eh,
            encoder_attention_mask=eam, context_latents=ctx,
            use_cache=False, output_attentions=False,
        )
        inj_handle.remove()

        flow = x1 - x0
        loss = torch.nn.functional.mse_loss(decoder_outputs[0], flow)

        # ── Backward ─────────────────────────────────────────────────────
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # ── Capture gradients ────────────────────────────────────────────
        with torch.no_grad():
            grad_info = {
                "step": step,
                "loss": loss.item(),
                "write_alpha": adapt.write_alpha.item(),
                "write_logit": adapt.write_logit.item(),
                "delta_h_rms": delta_h.pow(2).mean().sqrt().item(),
                "hidden_rms": H.pow(2).mean().sqrt().item(),
                "delta_h_ratio": delta_h.pow(2).mean().sqrt().item() / (H.pow(2).mean().sqrt().item() + 1e-10),
                "raw_res_norm": diag.get("raw_res_norm", 0),
                "raw_res_std": diag.get("raw_res_std", 0),
                "Pi_entropy": diag.get("entropy", 0),
                "Pi_mean": diag.get("transport_mean", 0),
                "row_error": diag.get("row_error", 0),
                "col_error": diag.get("col_error", 0),
                "R_mean": diag.get("R_mean", 0),
                "R_std": diag.get("R_std", 0),
                "base_logit_std": diag.get("base_logit_std", 0),
                "qk_ratio": diag.get("effective_qk_ratio", 0),
                "write_logit_grad": adapt.write_logit.grad.norm().item() if adapt.write_logit.grad is not None else 0,
            }
            for name, mod in [("pm", pm), ("q_mlp", adapt.q_mlp), ("k_mlp", adapt.k_mlp),
                              ("v_mlp", adapt.v_mlp), ("out_proj", adapt.out_proj)]:
                g = _grad_norm(mod)
                grad_info[f"{name}_grad_norm"] = g

            # Coupling diff: Pi @ value minus baseline (mean Pi over all time)
            Pi_flat = Pi.mean(dim=1, keepdim=True).expand(-1, Pi.shape[1], -1)
            coupling_diff = (Pi - Pi_flat).abs().mean().item()
            grad_info["coupling_diff"] = coupling_diff

            # Non-lyric-to-lyric leakage: fraction of mass on silence units
            if lyric_unit_mask is not None:
                silence_mask = (~lyric_unit_mask).float().unsqueeze(0).unsqueeze(0)
                leakage = (Pi * silence_mask).sum(dim=(-1, -2)) / Pi.sum(dim=(-1, -2))
                grad_info["silence_leakage"] = leakage.mean().item()

            # Conditional marginal error: column-wise relative error
            col_mass = Pi.sum(dim=1)
            target_mass = mu_all.unsqueeze(0)
            col_err = (col_mass - target_mass).abs().mean().item()
            grad_info["col_marginal_error"] = col_err

            history.append(grad_info)

        # ── Optimizer step ───────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(list(pm.parameters()) + list(adapt.parameters()), 1.0)
        opt.step()

    # ── Aggregate ──────────────────────────────────────────────────────
    agg = {}
    for k in history[0]:
        vals = [h[k] for h in history]
        agg[f"{k}_first"] = vals[0]
        agg[f"{k}_last"] = vals[-1]
        agg[f"{k}_mean"] = float(np.mean(vals))
        if k not in ("step",):
            agg[f"{k}_trend"] = vals[-1] - vals[0] if len(vals) > 1 else 0

    return {"history": history, "aggregate": agg, "final_state": {
        "write_alpha": adapt.write_alpha.item(),
        "write_logit": adapt.write_logit.item(),
        "out_proj_std": adapt.out_proj.weight.std().item(),
    }}


def run_inference_test(
    model, data, device, dtype, output_dir,
):
    """Run generation-like forward passes with forced alpha values.

    Measures decoder output difference between PM-on and PM-off.
    Builds a fresh adapter with write_alpha_max=0.1 to allow all test alphas.
    """
    D = model.config.hidden_size
    B = 1

    # Use a high alpha_max so forced alphas 0.001, 0.01, 0.03 all work
    pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True).to(device).float()
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        transport_mode="sinkhorn", sinkhorn_iters=5,
        transport_sigma=0.18, transport_qk_scale=1.0,
        write_alpha_init=0.001, write_alpha_max=0.1,  # allow up to alpha=0.03
        out_proj_init_std=0.01,
    ).to(device).float()
    adapt.eval()
    pm.eval()

    lyrics_text = (DATASET_DIR / f"{SAMPLE_ID}.lyrics.txt").read_text(encoding="utf-8").strip()
    L = data["encoder_hidden_states"].shape[0]
    parser = LyricsStructureParser()
    parsed = parser.parse(lyrics_text, num_chunks=L)
    section_ids_cpu = parsed.section_type_ids
    units, _, debug = parse_lyrics_to_units(lyrics_text, section_ids_cpu, auto_transition_ratios={})
    scaffold = build_duration_scaffold(units, text_len=L, tag_control_mask=debug.get("tag_control_mask"))

    results = {}

    xt = data["target_latents"].unsqueeze(0).to(device=device, dtype=dtype)
    eh = data["encoder_hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
    eam = data["encoder_attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
    am = data["attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
    ctx = data["context_latents"].unsqueeze(0).to(device=device, dtype=dtype)
    t_tensor = torch.full((1,), 0.5, device=device, dtype=dtype)  # mid-timestep

    # ── Baseline (PM-off, no adapter) ──────────────────────────────────
    with torch.no_grad():
        out_base = model.decoder(hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                                  attention_mask=am, encoder_hidden_states=eh,
                                  encoder_attention_mask=eam, context_latents=ctx,
                                  use_cache=False, output_attentions=False)
    base_output = out_base[0]

    # ── Warmup for adapter ─────────────────────────────────────────────
    with torch.no_grad():
        hs_list = []
        def _whook(m, i, o): hs_list.append(o[0])
        handle = model.decoder.layers[12].register_forward_hook(_whook)
        model.decoder(hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                      attention_mask=am, encoder_hidden_states=eh,
                      encoder_attention_mask=eam, context_latents=ctx,
                      use_cache=False, output_attentions=False)
        handle.remove()
        H = hs_list[0].float()
        pm_state = pm(H, t_tensor)

    # ── Build unit tensors ─────────────────────────────────────────────
    token_to_unit = scaffold["token_to_unit"].to(device)
    lyric_mask_s = scaffold["lyric_mask"].to(device)
    unit_boundaries = scaffold["unit_boundaries"].to(device)
    unit_duration = scaffold["unit_duration"].to(device)
    u_section_ids = scaffold["unit_section_ids"].to(device)
    lyric_unit_mask = scaffold["lyric_unit_mask"].to(device)
    U = len(unit_boundaries) - 1
    c_all = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
    mu_all = unit_duration / unit_duration.sum()

    eh_f = eh.float()
    unit_text_hidden_list = []
    for uid in range(U):
        is_lyric = lyric_unit_mask[uid].item()
        tmask = (token_to_unit == uid) & lyric_mask_s if is_lyric else (token_to_unit == uid)
        tmask_b = tmask.unsqueeze(0).expand(B, -1)
        if tmask_b.any():
            pooled = eh_f[tmask_b].view(B, -1, D).mean(dim=1)
        else:
            pooled = torch.zeros(B, D, device=device, dtype=torch.float32)
        unit_text_hidden_list.append(pooled)
    unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)
    p_audio = torch.linspace(0, 1, H.shape[1], device=device, dtype=torch.float32).unsqueeze(0)

    for alpha in INFERENCE_ALPHAS:
        # Forced alpha override
        if alpha >= adapt.write_alpha_max:
            continue  # skip if forced alpha >= max
        with torch.no_grad():
            old_logit = adapt.write_logit.clone()
            adapt.write_logit.fill_(math.log(alpha / (adapt.write_alpha_max - alpha + 1e-10)))

        with torch.no_grad():
            delta_h, Pi, _ = adapt(
                hidden_states=H, text_hidden=eh, pm_state=pm_state,
                p_audio=p_audio, unit_text_hidden=unit_text_hidden,
                unit_c_pos=c_all.unsqueeze(0), unit_mass=mu_all.unsqueeze(0),
                unit_section_id=u_section_ids.unsqueeze(0),
                unit_is_lyric=lyric_unit_mask.unsqueeze(0),
            )
            final_h = H + delta_h

            def _inject(m, i, o):
                return (final_h.to(dtype=o[0].dtype, device=o[0].device), *o[1:])
            inj_handle = model.decoder.layers[12].register_forward_hook(_inject)
            out_pm = model.decoder(hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                                    attention_mask=am, encoder_hidden_states=eh,
                                    encoder_attention_mask=eam, context_latents=ctx,
                                    use_cache=False, output_attentions=False)
            inj_handle.remove()

            diff = (out_pm[0] - base_output).abs()
            results[str(alpha)] = {
                "output_diff_mean": diff.mean().item(),
                "output_diff_max": diff.max().item(),
                "output_diff_rms": diff.pow(2).mean().sqrt().item(),
                "delta_h_rms": delta_h.pow(2).mean().sqrt().item(),
                "Pi_mean": Pi.mean().item(),
                "Pi_entropy": (-Pi * (Pi + 1e-10).log()).sum(dim=-1).mean().item(),
            }

        # Restore
        with torch.no_grad():
            adapt.write_logit.copy_(old_logit)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/root/ACE-Step-1.5/diag_output")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--configs", type=str, default="A,B,C,D,E",
                        help="Comma-separated config labels to run")
    args = parser.parse_args()
    num_steps = args.steps

    OUTPUT = Path(args.output)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    selected = [c for c in CONFIGS if c.label[0].upper() in args.configs.split(",")]

    # ── Load DATA ─────────────────────────────────────────────────────
    pt_data = torch.load(str(TENSOR_DIR / f"{SAMPLE_ID}.pt"), map_location="cpu", weights_only=True)
    print(f"Data: T={pt_data['target_latents'].shape[0]}, L={pt_data['encoder_hidden_states'].shape[0]}")

    # ── Load MODEL ────────────────────────────────────────────────────
    print("Loading SFT model...")
    handler = AceStepHandler()
    handler.initialize_service(
        project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
        device="cuda", use_flash_attention=False, compile_model=False,
    )
    model = handler.model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    D = model.config.hidden_size

    model.config.use_section_rope_offset = False
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_section_rope", False): layer_mod.use_section_rope = False
        if getattr(layer_mod, "use_phase_memory", False): layer_mod.use_phase_memory = False

    # ── Run each config ───────────────────────────────────────────────
    all_results = {}
    for cfg in selected:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {cfg.label}")
        print(f"  write_alpha_max={cfg.write_alpha_max}, alpha_init={cfg.write_alpha_init}")
        print(f"  out_proj_std={cfg.out_proj_init_std}")
        if cfg.fixed_warmup_steps > 0:
            print(f"  fixed_alpha warmup: {cfg.fixed_alpha} for {cfg.fixed_warmup_steps} steps")
        if cfg.no_wd_on_adapter:
            print(f"  no weight decay on q/k/v/out_proj/write_logit")

        pm, adapt = build_adapter(D, cfg, device, dtype)
        result = run_training_steps(model, pm, adapt, pt_data, cfg, device, dtype, OUTPUT, cfg.label, num_steps=num_steps)
        all_results[cfg.label] = result

        # Print summary
        a = result["aggregate"]
        print(f"\n  Results after {num_steps} steps:")
        print(f"    write_alpha:   {a['write_alpha_first']:.6f} -> {a['write_alpha_last']:.6f} (trend={a['write_alpha_trend']:+.6f})")
        print(f"    write_logit:   {a['write_logit_first']:.4f} -> {a['write_logit_last']:.4f} (trend={a['write_logit_trend']:+.4f})")
        print(f"    delta_h_ratio: {a['delta_h_ratio_first']:.6f} -> {a['delta_h_ratio_last']:.6f}")
        print(f"    loss:          {a['loss_first']:.4f} -> {a['loss_last']:.4f}")
        print(f"    col_error:     {a['col_error_first']:.6f} -> {a['col_error_last']:.6f}")
        print(f"    qk_ratio:      {a.get('qk_ratio_first',0):.4f} -> {a.get('qk_ratio_last',0):.4f}")
        print(f"    write_logit_grad mean: {a.get('write_logit_grad_mean',0):.6f}")
        print(f"    q_mlp_grad mean:       {a.get('q_mlp_grad_norm_mean',0):.6f}")
        print(f"    v_mlp_grad mean:       {a.get('v_mlp_grad_norm_mean',0):.6f}")
        print(f"    out_proj_grad mean:    {a.get('out_proj_grad_norm_mean',0):.6f}")
        print(f"    coupling_diff mean:    {a.get('coupling_diff_mean',0):.6f}")
        print(f"    Pi_entropy mean:       {a.get('Pi_entropy_mean',0):.6f}")

    # ── Inference test (use first config's trained adapter) ──────────
    print(f"\n{'='*70}")
    print(f"INFERENCE TEST (forced alpha)")
    inf_results = run_inference_test(model, pt_data, device, dtype, OUTPUT)
    all_results["inference"] = inf_results
    for alpha_str, metrics in inf_results.items():
        print(f"  alpha={alpha_str}: output_diff_rms={metrics['output_diff_rms']:.6f}, "
              f"delta_h_rms={metrics['delta_h_rms']:.6f}, "
              f"Pi_entropy={metrics['Pi_entropy']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────
    save = {}
    for label, r in all_results.items():
        if isinstance(r, dict) and "aggregate" in r:
            save[label] = {"aggregate": r["aggregate"], "final_state": r.get("final_state", {}),
                           "history": r["history"]}
        else:
            save[label] = r
    with open(OUTPUT / "diagnosis_results.json", "w") as f:
        json.dump(save, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nAll results -> {OUTPUT / 'diagnosis_results.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
