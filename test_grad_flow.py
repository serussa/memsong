"""Quick test: does gradient checkpointing preserve kl_loss gradients?"""
import os
os.environ["ACESTEP_LOCAL_MODEL_CODE"] = "1"

import torch
import torch.nn.functional as F
from acestep.phase_memory import PhaseMemory
from acestep.models.sft.modeling_acestep_v15_base import AceStepDiTLayer, AceStepConfig

# Create a single DiT layer with PhaseMemory
config = AceStepConfig(
    hidden_size=2048,
    intermediate_size=8192,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=128,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    attention_bias=False,
    use_sliding_window=False,
    sliding_window=4096,
    layer_types=["full_attention"],
    num_hidden_layers=24,
)

layer = AceStepDiTLayer(config, layer_idx=12, use_cross_attention=True, use_phase_memory=True)
layer = layer.to("cuda", dtype=torch.bfloat16)
layer.train()
layer.gradient_checkpointing = True
layer._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

pm = layer.phase_memory
bn = pm.bias_net[-1]
print(f"bias_net[-1] weights (first 3): {bn.weight[0,:3].detach().cpu().tolist()}")
print(f"bias_net[-1] bias: {bn.bias.detach().cpu().tolist()}")

B, T = 2, 100
hidden = torch.randn(B, T, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
cos = torch.cos(torch.randn(B, T, 128, device="cuda", dtype=torch.bfloat16))
sin = torch.sin(torch.randn(B, T, 128, device="cuda", dtype=torch.bfloat16))
enc_hidden = torch.randn(B, 50, 2048, device="cuda", dtype=torch.bfloat16)
enc_mask = torch.ones(B, 1, T, 50, device="cuda", dtype=torch.bfloat16)
temb = torch.randn(B, 1, 2048, device="cuda", dtype=torch.bfloat16)
step = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
pos_ids = torch.arange(T, device="cuda").unsqueeze(0)
pos_emb = (cos, sin)
beat_phase = torch.rand(B, T, device="cuda", dtype=torch.bfloat16)

# Forward with gradient checkpointing
print("\nForward with gradient_checkpointing=True...")
out = layer(
    hidden, pos_emb, temb, step,
    attention_mask=enc_mask,
    position_ids=pos_ids,
    encoder_hidden_states=enc_hidden,
    encoder_attention_mask=enc_mask,
    beat_phase=beat_phase,
)

print(f"Output tuple length: {len(out)}")
print(f"out[0] shape: {out[0].shape}")

# The kl_loss is at index 1 (second element)
kl_loss = out[1] if len(out) >= 2 else None
print(f"out[1] (kl_loss): {kl_loss}")
if isinstance(kl_loss, torch.Tensor):
    print(f"  value: {kl_loss.item():.6f}")
    print(f"  grad_fn: {kl_loss.grad_fn}")
    print(f"  requires_grad: {kl_loss.requires_grad}")

# Backward
print("\nBackward...")
loss = out[0].mean() + kl_loss.mean() if isinstance(kl_loss, torch.Tensor) else out[0].mean()
loss.backward()

print(f"bias_net[-1].weight.grad[:3]: {bn.weight.grad[0,:3].detach().cpu().tolist() if bn.weight.grad is not None else 'NO GRAD'}")
print(f"bias_net[-1].bias.grad: {bn.bias.grad.detach().cpu().tolist() if bn.bias.grad is not None else 'NO GRAD'}")
print(f"bias_net[0].weight.grad[0,:3]: {layer.phase_memory.bias_net[0].weight.grad[0,:3].detach().cpu().tolist() if layer.phase_memory.bias_net[0].weight.grad is not None else 'NO GRAD'}")

print("\nDONE")
