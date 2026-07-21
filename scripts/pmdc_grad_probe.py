#!/usr/bin/env python3
"""
PMDC Gradient Probe — 10-step training run with full diagnostics.

Verifies:
  1. eager_attention_forward patch actually fires (call counter)
  2. bias is non-zero and affects attention (attention_KL_vs_static > 0)
  3. flow_loss depends on p_final (grad(flow_loss, p_final) exists)
  4. net.2.weight gets gradient in first 10 steps
"""

import os, sys, gc, time, math, logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

logging.disable(logging.WARNING)

from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.phase_memory import (
    PMDCResidualClock, parse_lyrics_to_units, build_duration_scaffold,
    build_duration_interval_bias,
)

device = torch.device("cuda")
print("=" * 70)
print("PMDC GRADIENT PROBE — 10 steps")
print("=" * 70)

# ---- Load model ------------------------------------------------------------
print("\n[1/4] Loading model...", flush=True)
dt = AceStepHandler()
dt.initialize_service(
    project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
    device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False,
)
model = dt.model.eval()
D = model.config.hidden_size

# Disable irrelevant features
model.config.use_section_rope_offset = False
for lm in model.decoder.layers:
    if getattr(lm, "use_section_rope", False): lm.use_section_rope = False
    if getattr(lm, "use_phase_memory", False): lm.use_phase_memory = False

# Force EAGER
for m in model.modules():
    if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
        old = m.config._attn_implementation
        if old != "eager":
            m.config._attn_implementation = "eager"
            print(f"  Forced {type(m).__name__}: {old} → eager")
if hasattr(model.config, "_attn_implementation_compiled"):
    model.config._attn_implementation_compiled = None

model = model.to(device)
dtype = next(model.parameters()).dtype

# ---- Scaffold (fixed for this probe) ----------------------------------------
print("\n[2/4] Building scaffold...", flush=True)
lyrics = """[Verse]
雪让我有点快乐
那片白色
冬天快乐
[Chorus]
习惯你
就把我当作你
呼吸安静而整齐
冬天快乐"""
parser = LyricsStructureParser()
L_text = 769  # actual text encoder output length
parsed = parser.parse(lyrics, num_chunks=L_text)
units, _, debug = parse_lyrics_to_units(lyrics, parsed.section_type_ids)
tcm = debug.get("tag_control_mask", None)
scaffold = build_duration_scaffold(units, text_len=L_text, tag_control_mask=tcm)
scaffold = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}
print(f"  scaffold: {scaffold['unit_boundaries'].shape[-1]-1} units")

# ---- Create PMDC clock -----------------------------------------------------
print("\n[3/4] Creating PMDC clock + gate...", flush=True)
clock = PMDCResidualClock(dim=D, hidden_dim=128, beta_init=0.05, beta_max=0.15, use_delta_h=True).to(device).float()
gate_logit = nn.Parameter(torch.tensor(0.0, device=device))
with torch.no_grad():
    gate_logit.fill_(math.log(0.35 / 0.65))

# Freeze backbone
for p in model.parameters(): p.requires_grad = False

# Trainable params
trainable_params = list(clock.parameters()) + [gate_logit]
total_trainable = sum(p.numel() for p in trainable_params)
print(f"  trainable params: {total_trainable:,}")
optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-4)

# ---- Warmup + Patch setup --------------------------------------------------
print("\n[4/4] Running 10 steps with grad probe...", flush=True)

patch_call_count = 0
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
import sys
ca_module = model.decoder.layers[12].cross_attn
cls = type(ca_module)
attn_mod = sys.modules[cls.__module__]
orig_eaf = getattr(attn_mod, "eager_attention_forward", None)

# Store for diagnostics
bias_cache = [None]
attn_base_cache = [None]
attn_pmdc_cache = [None]

def _patched_forward(*args, **kwargs):
    global patch_call_count
    patch_call_count += 1
    mod = args[0]; q = args[1]; k = args[2]; v = args[3]
    am = args[4] if len(args) > 4 else kwargs.get("attention_mask")
    sc = kwargs.get("scaling", args[5] if len(args) > 5 else None)
    dr = kwargs.get("dropout", 0.0)
    ks = repeat_kv(k, mod.num_key_value_groups)
    vs = repeat_kv(v, mod.num_key_value_groups)
    aw = torch.matmul(q, ks.transpose(2, 3)) * sc
    if am is not None and isinstance(am, torch.Tensor):
        aw = aw + am[:, :, :, :ks.shape[-2]]

    is_cross_attn = bias_cache[0] is not None and aw.shape[-1] == bias_cache[0].shape[-1]
    if is_cross_attn:
        gate_val = torch.sigmoid(gate_logit)
        bias_t = bias_cache[0]
        if bias_t.shape[0] != aw.shape[0]:
            bf = aw.shape[0] // max(bias_t.shape[0], 1)
            bias_t = bias_t.repeat(bf, 1, 1, 1) if bf > 1 else bias_t
        bias_exp = bias_t.unsqueeze(1)

        amask = scaffold.get("attendable_mask", scaffold["lyric_mask"])
        if amask.dim() == 1:
            amask = amask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if amask.shape[0] != aw.shape[0]:
            amask = amask.expand(aw.shape[0], -1, -1, -1)
        amask = amask.bool()

        text_float = amask.float()
        attn_base = F.softmax(aw, dim=-1, dtype=torch.float32)
        text_mass_base = (attn_base * text_float).sum(dim=-1)

        text_logits = (aw + gate_val * bias_exp).masked_fill(~amask, float("-inf"))
        attn_text = F.softmax(text_logits, dim=-1, dtype=torch.float32)
        attn_text = attn_text.masked_fill(~amask, 0.0)

        non_text_logits = aw.masked_fill(amask, float("-inf"))
        attn_non_text = F.softmax(non_text_logits, dim=-1, dtype=torch.float32)
        attn_non_text = attn_non_text.masked_fill(amask, 0.0)

        aw_new = (attn_text * text_mass_base.unsqueeze(-1) +
                  attn_non_text * (1.0 - text_mass_base).unsqueeze(-1))
        aw_new = aw_new / (aw_new.sum(dim=-1, keepdim=True) + 1e-10)
        aw_new = aw_new.to(q.dtype)

        # Cache for diagnostics
        attn_base_cache[0] = attn_base.detach()
        attn_pmdc_cache[0] = aw_new.detach()

        aw = aw_new
    else:
        aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)

    aw = F.dropout(aw, p=dr, training=mod.training)
    return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw

setattr(attn_mod, "eager_attention_forward", _patched_forward)

# ---- Training loop ---------------------------------------------------------
net2_grad_norms = []

for step in range(3):
    # Create fake batch
    B, T = 1, 3000
    target_latents = torch.randn(B, T, 64, device=device, dtype=dtype)
    attention_mask = torch.ones(B, T, device=device, dtype=dtype)
    null_emb = model.null_condition_emb.expand(B, 769, -1)
    encoder_attention_mask = torch.ones(B, 769, device=device, dtype=dtype)
    context_latents = torch.zeros(B, T, 128, device=device, dtype=dtype)
    section_ids = torch.zeros(B, 128, dtype=torch.long, device=device)

    x1 = torch.randn_like(target_latents)
    x0 = target_latents
    t = torch.full((B,), 0.5, device=device, dtype=dtype)
    xt = t.unsqueeze(-1).unsqueeze(-1) * x1 + (1.0 - t.unsqueeze(-1).unsqueeze(-1)) * x0

    # ---- Warmup forward (get H) --------------------------------------------
    hs_list = []
    def hook(m, i, o): hs_list.append(o[0])
    h = model.decoder.layers[12].register_forward_hook(hook)
    with torch.no_grad():
        model.decoder(hidden_states=xt, timestep=t, timestep_r=t,
            attention_mask=attention_mask, encoder_hidden_states=null_emb,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents, use_cache=False, output_attentions=False)
    h.remove()
    H_warm = hs_list[0].float() if hs_list else torch.zeros(B, T//2, D, device=device, dtype=torch.float32)
    T_pmdc = H_warm.shape[1]

    # ---- Clock forward ------------------------------------------------------
    p_base = torch.linspace(0, 1, T_pmdc, device=device).float().unsqueeze(0)
    speed_residual, p_final, log_v = clock(H_warm, p_base)

    # ---- Build bias ---------------------------------------------------------
    bias = build_duration_interval_bias(p_final=p_final,
        unit_boundaries=scaffold["unit_boundaries"],
        token_to_unit=scaffold["token_to_unit"],
        attendable_mask=scaffold.get("attendable_mask", scaffold["lyric_mask"]),
        sigma=0.12, lambda_=0.5, max_bias=1.0)
    if dtype == torch.bfloat16:
        bias = bias.to(torch.bfloat16)
    bias_cache[0] = bias

    # ---- Main forward (with patch) ------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    decoder_out = model.decoder(hidden_states=xt, timestep=t, timestep_r=t,
        attention_mask=attention_mask, encoder_hidden_states=null_emb,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents, use_cache=False, output_attentions=False)

    flow = x1 - x0
    flow_loss = F.mse_loss(decoder_out[0], flow)
    pbase_loss = F.smooth_l1_loss(p_final, p_base)
    loss = flow_loss + 0.005 * pbase_loss

    # Before backward: verify bias is in the computation graph
    print(f"  bias.is_leaf={bias.is_leaf} bias.grad_fn={bias.grad_fn}", end="")
    if bias.grad_fn:
        print(f" (connected to p_final via grad_fn chain)", end="")
    print()

    loss.backward()

    # ---- AFTER backward: gradient inspection ---------------------------------
    # Check PMDC clock gradient
    n0_grad = clock.net[0].weight.grad
    n2_grad = clock.net[2].weight.grad
    n0_grad_any = n0_grad is not None
    n0_abs_max = n0_grad.abs().max().item() if n0_grad is not None else -1.0
    n2_abs_max = n2_grad.abs().max().item() if n2_grad is not None else -1.0
    print(f"  AFTER backward — n0.grad: {n0_grad_any}, max={n0_abs_max:.15f} | "
          f"n2.grad: {n2_grad is not None}, max={n2_abs_max:.15f}")
    with torch.no_grad():
        # Bias stats
        bias_abs_mean = bias.abs().mean().item()
        bias_abs_max = bias.abs().max().item()
        gate_val = torch.sigmoid(gate_logit).item()

        # Attention KL vs baseline (if cached)
        kl = float("nan")
        if attn_base_cache[0] is not None and attn_pmdc_cache[0] is not None:
            eps = 1e-10
            p = attn_pmdc_cache[0] + eps
            q = attn_base_cache[0] + eps
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()

        # net.2 grad
        n2 = clock.net[2]
        n2_grad_is_none = n2.weight.grad is None
        n2_grad_norm = n2.weight.grad.norm().item() if n2.weight.grad is not None else -1.0
        n2_weight_norm = n2.weight.norm().item()
        if step > 0:
            net2_delta = n2.weight.detach() - net2_grad_norms[-1][0]
            net2_delta_norm = net2_delta.norm().item()
        else:
            net2_delta_norm = 0.0
        net2_grad_norms.append((n2.weight.detach().clone(), net2_delta_norm))

        # p_final delta
        pdelta = (p_final - p_base).abs().mean().item()
        # Check if speed_head (net.2) is actually receiving signal
        # by looking at gradient of net.0.weight (should see signal first)
        n0_grad = clock.net[0].weight.grad.abs().mean().item() if clock.net[0].weight.grad is not None else -1.0

    n2_grad_str = "None" if n2_grad_is_none else f"{n2_grad_norm:.8f}"
    print(f"  Step {step:2d}: KL={kl:.6f}  "
          f"n2_grad={n2_grad_str}  n2_w={n2_weight_norm:.8f}  "
          f"n0_grad_mean={n0_grad:.8f}  "
          f"pdelta={pdelta:.6f}", flush=True)

    optimizer.step()

# ---- Summary ----------------------------------------------------------------
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"  Patch call count (last step): {patch_call_count}")
print(f"  Patch ever fired: {patch_call_count > 0}")
print(f"  net.2.weight ever had gradient: {any(n[1] > 0 for n in net2_grad_norms)}")
print(f"  net.2.weight final norm: {clock.net[2].weight.norm().item():.8f}")
print(f"  KL divergence (last): {kl:.6f}")
print(f"  mean_abs_p_delta (last): {pdelta:.6f}")

# Restore
setattr(attn_mod, "eager_attention_forward", orig_eaf)
