#!/usr/bin/env python3
"""Minimal test: does TSM get gradients under gradient checkpointing?"""
import sys, os, math
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ.setdefault('ACESTEP_OFFLINE', '1')
os.environ.setdefault('ACESTEP_MINIMAL_COMPONENTS', '1')

import torch, torch.nn as nn, torch.nn.functional as F
from acestep.handler import AceStepHandler
from acestep.phase_memory import TransportRetrievalAdapter
from acestep.modules.transported_structural_memory import TransportedStructuralMemory

DEV = 'cuda'
# Load a SMALLER test with synthetic data to keep VRAM low

print("[1] Loading model...")
dt = AceStepHandler()
dt.initialize_service(project_root='/root/autodl-tmp/Ace-Step1.5',
    config_path='acestep-v15-sft', device=DEV,
    use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model; model.eval()
D = model.config.hidden_size
print(f"    model loaded, D={D}")

# Create TSM
tsm = TransportedStructuralMemory(model_dim=D, memory_dim=256,
    num_heads=4, ffn_dim=512, slot_layers=1).to(DEV).float()
# Non-zero init for gradient test
nn.init.normal_(tsm.output_proj.weight, std=1e-4)
print(f"    TSM output_proj init norm: {tsm.output_proj.weight.norm().item():.6e}")

# Create adapter (scaffold_only, no PM)
adapt = TransportRetrievalAdapter(
    hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
    sinkhorn_iters=5, transport_sigma=0.18, scoring_mode='scaffold_only',
    use_pm_gate=False, gate_hidden_dim=128,
    write_alpha_init=0.001, write_alpha_max=0.01, out_proj_init_std=0.01,
).to(DEV).float()

# Freeze backbone + adapter
for p in model.parameters(): p.requires_grad = False
for p in adapt.parameters(): p.requires_grad = False

B, T_lat = 1, 10  # tiny for speed
eh = torch.randn(B, 100, D, device=DEV, dtype=model.dtype)
ctx_lat = torch.zeros(B, T_lat, 128, device=DEV, dtype=model.dtype)
am = torch.ones(B, T_lat, device=DEV, dtype=model.dtype)
t_t = torch.full((B,), 0.5, device=DEV, dtype=model.dtype)

# Fake unit tensors
U = 5
uth = torch.randn(B, U, D, device=DEV, dtype=torch.float32)
ca = torch.linspace(0, 1, U, device=DEV).unsqueeze(0).expand(B, -1)
mu = torch.ones(B, U, device=DEV) / U
uil = torch.ones(B, U, device=DEV, dtype=torch.bool)
usid = torch.zeros(B, U, dtype=torch.long, device=DEV)

for gc in [True, False]:
    print(f"\n{'='*60}")
    print(f"Gradient checkpointing: {gc}")
    print(f"{'='*60}")

    # Configure checkpointing
    if gc:
        if hasattr(model.decoder, 'gradient_checkpointing_enable'):
            model.decoder.gradient_checkpointing_enable()
        else:
            model.decoder.gradient_checkpointing = True
    else:
        if hasattr(model.decoder, 'gradient_checkpointing_disable'):
            model.decoder.gradient_checkpointing_disable()
        else:
            model.decoder.gradient_checkpointing = False

    torch.manual_seed(42)
    x0 = torch.randn(B, T_lat, 64, device=DEV, dtype=model.dtype)
    x1 = torch.randn_like(x0)
    flow = x1 - x0
    t_val = 0.3
    t_exp = torch.full((B, 1, 1), t_val, device=DEV, dtype=model.dtype)
    xt = t_exp * x1 + (1 - t_exp) * x0
    xt.requires_grad_(True)

    # Pre_hook: compute transport + TSM, modify layer 12 input
    def _pre_hook(module, inputs):
        H = inputs[0]
        T_h = H.shape[1]
        pa = torch.linspace(0, 1, T_h, device=DEV, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        delta_h, Pi, _ = adapt(
            H.float(), eh.float(), None, pa,
            uth.float().expand(B, -1, -1),
            ca.float().expand(B, -1),
            mu.float().expand(B, -1),
            unit_section_id=usid.expand(B, -1),
            unit_is_lyric=uil.expand(B, -1),
        )

        tsm_out, tsm_d = tsm(
            H.float() + delta_h, coupling=Pi.detach(),
            condition_mask=uil.expand(B, -1),
            detach_coupling=True, enable_slot_mixer=False,
        )

        out_hidden = H.float() + delta_h + tsm_out
        return (out_hidden.to(dtype=H.dtype),) + inputs[1:]

    h = model.decoder.layers[12].register_forward_pre_hook(_pre_hook)
    try:
        d_out = model.decoder(
            hidden_states=xt, timestep=t_t, timestep_r=t_t,
            attention_mask=am, encoder_hidden_states=eh,
            encoder_attention_mask=None, context_latents=ctx_lat,
        )
    finally:
        h.remove()

    loss = F.mse_loss(d_out[0], flow)
    print(f"    loss={loss.item():.6f}  requires_grad={loss.requires_grad}")

    if loss.requires_grad:
        loss.backward()
    else:
        print(f"    SKIP: loss has no grad_fn")
        continue

    # Check TSM grad norms
    for name, p in tsm.named_parameters():
        if p.grad is not None:
            g_norm = p.grad.norm().item()
            if g_norm > 0:
                print(f"    {name}: grad={g_norm:.4e}")
            else:
                print(f"    {name}: grad=0")
        else:
            print(f"    {name}: grad=None")

    # Zero grads for next test
    for p in tsm.parameters():
        if p.grad is not None: p.grad.zero_()
    for p in adapt.parameters():
        if p.grad is not None: p.grad.zero_()

print("\nDone")
