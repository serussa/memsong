#!/usr/bin/env python3
"""
10-step real-data smoke test for PMRetrievalPhaseMemory + LyricRetrievalAdapter.
"""
import os, sys, math, gc, time, glob
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from acestep.handler import AceStepHandler
from acestep.phase_memory import (
    PMRetrievalPhaseMemory, LyricRetrievalAdapter, build_duration_scaffold,
    parse_lyrics_to_units, scaffold_progress,
)
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps

print("=" * 70)
print("PM-Retrieval 10-step real-data smoke test")
print("=" * 70)

device = torch.device("cuda")
torch.set_float32_matmul_precision("medium")

# ── 1. Load model ──────────────────────────────────────────────────────────
dt = AceStepHandler()
dt.initialize_service(
    project_root='/root/autodl-tmp/Ace-Step1.5', config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False, compile_model=False, offload_to_cpu=False,
)
model = dt.model.eval()
D = model.config.hidden_size
print(f"Model hidden_size={D}, layers={len(model.decoder.layers)}")
null_cond = model.null_condition_emb

# ── 2. Freeze backbone FIRST (before adding new modules) ──────────────────
for p in model.parameters():
    p.requires_grad = False

# ── 3. Create PMRetrievalPhaseMemory + LyricRetrievalAdapter ──────────────
pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256).to(device).float()
adapt = LyricRetrievalAdapter(
    hidden_dim=D, text_dim=D, pm_dim=256, d_r=64,
    residual_scale=0.1, gamma_init=0.01,
    use_adapter_scaffold_prior=True,
    adapter_prior_sigma=0.18, adapter_prior_lambda=0.2,
    adapter_prior_clamp_min=-2.0, adapter_prior_dropout=0.3,
).to(device).float()

model.add_module("pm_retrieval_pm", pm)
model.add_module("pm_retrieval_adapter", adapt)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Backbone frozen: {sum(p.numel() for p in model.parameters()):,} total, {sum(p.numel() for p in trainable):,} trainable")
print(f"PM params: {sum(p.numel() for p in pm.parameters()):,}")
print(f"Adapter params: {sum(p.numel() for p in adapt.parameters()):,}")

# ── 4. Load data manually ─────────────────────────────────────────────────
tensor_dir = '/root/autodl-tmp/musicdata/train_tensors'
pt_files = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))[:10]
print(f"Loading {len(pt_files)} data files from {tensor_dir}")

batches = []
for f in pt_files:
    data = torch.load(f, map_location='cpu', weights_only=False)
    # Ensure expected keys exist
    if "target_latents" not in data:
        data["target_latents"] = data.get("latents", data.get("x0", data.get("audio_latents", None)))
        if data["target_latents"] is None:
            print(f"  Skipping {f}: no target_latents")
            continue
    if "encoder_hidden_states" not in data:
        data["encoder_hidden_states"] = data.get("text_embeddings", data.get("encoder_hidden", None))
        if data["encoder_hidden_states"] is None:
            print(f"  Skipping {f}: no encoder_hidden_states")
            continue

    # Add batch dim (data files are single samples)
    for key in ["target_latents", "attention_mask", "encoder_hidden_states",
                "encoder_attention_mask", "context_latents"]:
        if key in data and data[key] is not None and data[key].dim() == 2:
            data[key] = data[key].unsqueeze(0)
        elif key in data and data[key] is not None and data[key].dim() == 1:
            data[key] = data[key].unsqueeze(0)

    # Fill missing defaults
    if "attention_mask" not in data or data["attention_mask"] is None:
        data["attention_mask"] = torch.ones(1, data["target_latents"].shape[2])
    if "encoder_attention_mask" not in data or data["encoder_attention_mask"] is None:
        data["encoder_attention_mask"] = torch.ones(1, data["encoder_hidden_states"].shape[1])
    if "context_latents" not in data or data["context_latents"] is None:
        data["context_latents"] = torch.zeros(1, 1, 2048)

    # Generate section_ids for scaffold (all UNKNOWN=0, just need correct length)
    text_len = data["encoder_hidden_states"].shape[1]
    if "section_ids" not in data:
        data["section_ids"] = torch.zeros(1, text_len, dtype=torch.long)

    batches.append(data)

if not batches:
    raise RuntimeError("No valid batches loaded!")

print(f"Loaded {len(batches)} batches")

# ── 5. Optimizer ──────────────────────────────────────────────────────────
optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999),
)

# ── 6. Training loop (10 steps) ──────────────────────────────────────────
ADAPTER_LAYER = 12
model.decoder.train()
model.decoder.config.use_cache = False

step = 0
losses = []
last_diag = {}

pm_retrieval_scaffold = None

for batch_idx in range(min(10, len(batches))):
    optim.zero_grad(set_to_none=True)

    batch = batches[batch_idx]

    # ── Move to device ─────────────────────────────────────────────────
    nb = False
    target_latents = batch["target_latents"].to(device, dtype=torch.bfloat16, non_blocking=nb)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.bfloat16, non_blocking=nb)
    encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=torch.bfloat16, non_blocking=nb)
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.bfloat16, non_blocking=nb)
    context_latents = batch["context_latents"].to(device, dtype=torch.bfloat16, non_blocking=nb)
    section_ids = batch.get("section_ids")
    if section_ids is not None:
        section_ids = section_ids.to(device, non_blocking=nb)

    bsz = target_latents.shape[0]

    # ── CFG dropout ─────────────────────────────────────────────────────
    if null_cond is not None:
        encoder_hidden_states = apply_cfg_dropout(
            encoder_hidden_states, null_cond, cfg_ratio=0.15,
        )

    # ── Noise & timesteps ───────────────────────────────────────────────
    x1 = torch.randn_like(target_latents)
    x0 = target_latents
    t, r = sample_timesteps(
        batch_size=bsz, device=device, dtype=torch.bfloat16,
        data_proportion=0.5, timestep_mu=-0.4, timestep_sigma=1.0,
        use_meanflow=False,
    )
    t_ = t.unsqueeze(-1).unsqueeze(-1)
    xt = t_ * x1 + (1.0 - t_) * x0

    # ── Step 1: Warmup forward → collect H ─────────────────────────────
    hs_list = []
    def _whook(m, i, o):
        hs_list.append(o[0].float())
    handle = model.decoder.layers[ADAPTER_LAYER].register_forward_hook(_whook)
    with torch.no_grad():
        _ = model.decoder(
            hidden_states=xt, timestep=t, timestep_r=t,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False, output_attentions=False,
        )
    handle.remove()
    H = hs_list[0]  # [B, T_h, D]

    # ── Step 2: PM forward → pm_state ──────────────────────────────────
    pm_state = pm(H, t)

    # ── Step 3: Build scaffold ─────────────────────────────────────────
    sc = None
    if section_ids is not None:
        meta_raw = batch.get("metadata", batch.get("metadatas", {}))
        if isinstance(meta_raw, list) and len(meta_raw) > 0:
            meta_data = meta_raw[0]
        elif isinstance(meta_raw, dict):
            meta_data = meta_raw
        else:
            meta_data = {}
        lyrics_text = meta_data.get("lyrics", "") if isinstance(meta_data, dict) else ""
        if lyrics_text:
            L_eff = section_ids.shape[-1]
            units, _, debug = parse_lyrics_to_units(
                lyrics_text, section_ids[0].cpu(),
                auto_transition_ratios={
                    "intro": 0.0, "outro": 0.0,
                    "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
                    "bridge_to_chorus": 0.0,
                },
            )
            tcm = debug.get("tag_control_mask", None)
            sc = build_duration_scaffold(units, text_len=L_eff, tag_control_mask=tcm)

    # ── Step 4: Adapter forward ─────────────────────────────────────────
    adapt.train()
    adapt_diag = {}
    if sc is not None:
        sc_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
        T_h = H.shape[1]
        p_audio, c_text, ttid = scaffold_progress(sc_d, T_h, device=device)
        p_audio = p_audio.unsqueeze(0).expand(bsz, -1)
        c_text = c_text.unsqueeze(0).expand(bsz, -1)
        ttid = ttid.unsqueeze(0).expand(bsz, -1)
        sec_id = torch.zeros(sc_d["lyric_mask"].shape[-1], device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        t_emb = torch.zeros(bsz, 128, device=device, dtype=torch.float32)
        amask_text = encoder_attention_mask.bool() if encoder_attention_mask is not None else None

        ret_res, attn_r, adapt_diag = adapt(
            hidden_states=H,
            text_hidden=encoder_hidden_states.float(),
            pm_state=pm_state,
            p_audio=p_audio, c_text=c_text,
            section_id=sec_id, token_type_id=ttid,
            timestep_emb=t_emb, attention_mask=amask_text,
            use_scaffold_prior=True,
        )
    else:
        ret_res = torch.zeros_like(H)
        attn_r = None

    gamma_r_val = adapt.gamma_r
    final_h = H + gamma_r_val * ret_res

    with torch.no_grad():
        retrieval_res_norm = ret_res.norm(dim=-1).mean().item()
        hidden_delta = (final_h - H).norm(dim=-1).mean().item()

    # ── Step 5: Inject hook ─────────────────────────────────────────────
    def _inject_hook(m, i, o):
        return (final_h.to(dtype=o[0].dtype, device=o[0].device), *o[1:])
    ih = model.decoder.layers[ADAPTER_LAYER].register_forward_hook(_inject_hook)

    # ── Step 6: Forward with injected hidden ───────────────────────────
    decoder_outputs = model.decoder(
        hidden_states=xt, timestep=t, timestep_r=t,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        use_cache=False, output_attentions=False,
    )
    ih.remove()

    flow = x1 - x0
    flow_loss = F.mse_loss(decoder_outputs[0].float(), flow.float())
    loss = flow_loss

    # ── Step 7: Backward ────────────────────────────────────────────────
    loss.backward()

    # Gradient clipping
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)

    optim.step()
    step += 1
    loss_val = loss.item()
    losses.append(loss_val)
    last_diag = adapt_diag

    # ── Diagnostics ────────────────────────────────────────────────────
    def _gn(mod):
        total = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                total += p.grad.norm(2).item() ** 2
        return math.sqrt(total)

    with torch.no_grad():
        print(f"\n[Step {step}/{10}]")
        print(f"  flow_loss={loss_val:.6f}")
        print(f"  gamma_r={gamma_r_val.item():.6f}")
        print(f"  pm_state_norm_before={adapt_diag.get('pm_state_norm_before', 0):.4f}")
        print(f"  pm_state_norm_after={adapt_diag.get('pm_state_norm_after', 0):.4f}")
        print(f"  score_qk_std={adapt_diag.get('score_qk_std', 0):.6f}")
        print(f"  score_prior_std={adapt_diag.get('score_prior_std', 0):.6f}")
        print(f"  score_total_std={adapt_diag.get('score_total_std', 0):.6f}")
        print(f"  prior_enabled={adapt_diag.get('prior_enabled', 0):.0f}")
        print(f"  retrieval_residual_norm={retrieval_res_norm:.6f}")
        print(f"  hidden_delta_norm={hidden_delta:.6f}")
        print(f"  attn_r_entropy={adapt_diag.get('attn_entropy', 0):.4f}")
        print(f"  center_p_audio_corr={adapt_diag.get('center_p_audio_corr', 0):.4f}")
        print(f"  delta_center_abs_mean={adapt_diag.get('delta_center_abs_mean', 0):.4f}")
        print(f"  PM_grad_norm={_gn(pm):.6f}")
        print(f"  q_mlp_grad_norm={_gn(adapt.q_mlp):.6f}")
        print(f"  k_mlp_grad_norm={_gn(adapt.k_mlp):.6f}")
        print(f"  v_mlp_grad_norm={_gn(adapt.v_mlp):.6f}")
        print(f"  out_proj_grad_norm={_gn(adapt.out_proj):.6f}")
        print(f"  audio_coord_grad_norm={_gn(adapt.audio_coord_mlp):.6f}")
        print(f"  text_coord_grad_norm={_gn(adapt.text_coord_mlp):.6f}")
        print(f"  has_nan={not math.isfinite(loss_val)}")

    # Clean up
    del loss, decoder_outputs, H, pm_state, ret_res, final_h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Completed {step} steps")
losses_arr = np.array(losses)
print(f"Loss: min={losses_arr.min():.6f} max={losses_arr.max():.6f} mean={losses_arr.mean():.6f}")
print(f"NaN/Inf in loss: {np.any(~np.isfinite(losses_arr))}")

def _gn(mod):
    total = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return math.sqrt(total)

# Check criteria
has_nan = not all(math.isfinite(l) for l in losses)
hidden_delta_ok = hidden_delta > 0
retrieval_res_ok = retrieval_res_norm > 0
pm_grad_ok = _gn(pm) > 0
q_mlp_grad_ok = _gn(adapt.q_mlp) > 0
k_mlp_grad_ok = _gn(adapt.k_mlp) > 0

print(f"\nCriteria:")
print(f"  No NaN/Inf:               {'✓' if not has_nan else '✗'}")
print(f"  hidden_delta_norm > 0:    {'✓' if hidden_delta_ok else '✗'}")
print(f"  retrieval_residual_norm>0:{'✓' if retrieval_res_ok else '✗'}")
print(f"  PM grad norm > 0:         {'✓' if pm_grad_ok else '✗'}")
print(f"  q_mlp / k_mlp grad > 0:   {'✓' if (q_mlp_grad_ok and k_mlp_grad_ok) else '✗'}")
print(f"  score_qk/prior_std ok:    {'✓' if (adapt_diag.get('score_qk_std', 0) > 1e-6 and adapt_diag.get('score_prior_std', 0) > 1e-6) else '✗'}")

# ── Save checkpoint ─────────────────────────────────────────────────────
os.makedirs("/root/autodl-tmp/pm_retrieval_smoke", exist_ok=True)
pm_sd = pm.state_dict()
adapt_sd = adapt.state_dict()
state = {"phase_memory": pm_sd, "retrieval_adapter": adapt_sd}
tensor_count = len(pm_sd) + len(adapt_sd)
param_count = sum(t.numel() for t in pm_sd.values()) + sum(t.numel() for t in adapt_sd.values())
ckpt_path = "/root/autodl-tmp/pm_retrieval_smoke/pm_retrieval.pt"
torch.save(state, ckpt_path)
file_size = os.path.getsize(ckpt_path)
all_keys = list(pm_sd.keys()) + list(adapt_sd.keys())
print(f"\nCheckpoint:")
print(f"  saved tensor count: {tensor_count}")
print(f"  saved parameter count: {param_count}")
print(f"  file size: {file_size / 1024:.1f} KB")
print(f"  first 20 keys: {all_keys[:20]}")
print(f"  checkpoint non-empty: {'✓' if tensor_count > 0 else '✗'}")
all_pass = not has_nan and hidden_delta_ok and retrieval_res_ok and pm_grad_ok and q_mlp_grad_ok and k_mlp_grad_ok and tensor_count > 0
print(f"\n{'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
print("=" * 70)
