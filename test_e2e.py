"""End-to-end test: gradient flow through checkpointed DiT layer with PhaseMemory."""
import os
os.environ["ACESTEP_LOCAL_MODEL_CODE"] = "1"
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

# Import needed modules
from acestep.phase_memory import PhaseMemory
from transformers.modeling_layers import GradientCheckpointingLayer

# Create a minimal DiT-like layer with PhaseMemory
class MiniDiTLayer(GradientCheckpointingLayer):
    def __init__(self, hidden_size=512, mem_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.phase_memory = PhaseMemory(dim=hidden_size, mem_dim=mem_dim)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, step, beat_phase=None, **kwargs):
        hidden_states, kl_loss = self.phase_memory(hidden_states, step, beat_phase=beat_phase)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear(hidden_states)
        return (hidden_states, kl_loss)

# Test setup
B, T, D = 2, 50, 512
layer = MiniDiTLayer(hidden_size=D)
layer = layer.to("cuda", dtype=torch.bfloat16)
layer.train()
layer.gradient_checkpointing = True
layer._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

pm = layer.phase_memory
print(f"bias_net[-1].weight[:3]: {pm.bias_net[-1].weight[0,:3].detach().cpu().tolist()}")
print(f"bias_net[-1].bias: {pm.bias_net[-1].bias.detach().cpu().tolist()}")

h = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
step = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
bp = torch.rand(B, T, device="cuda", dtype=torch.bfloat16)

# Forward with gradient checkpointing
print("\nForward with checkpoint...")
out = layer(h, step, beat_phase=bp)
print(f"Output tuple length: {len(out)}")
print(f"out[0].shape: {out[0].shape}")
print(f"out[1]: {out[1]}")
if isinstance(out[1], torch.Tensor):
    print(f"  kl_loss = {out[1].item():.6f}")
    print(f"  grad_fn = {out[1].grad_fn}")
else:
    print(f"  kl_loss type = {type(out[1])}")

# Create a combined loss and backward
loss = out[0].mean()
if isinstance(out[1], torch.Tensor):
    loss = loss + out[1].mean()
print(f"\nLoss: {loss.item():.6f}")
loss.backward()

print(f"\nGradients after backward:")
print(f"  bias_net[-1].weight.grad[:3]: {pm.bias_net[-1].weight.grad[0,:3].detach().cpu().tolist() if pm.bias_net[-1].weight.grad is not None else 'NO GRAD'}")
print(f"  bias_net[-1].bias.grad: {pm.bias_net[-1].bias.grad.detach().cpu().tolist() if pm.bias_net[-1].bias.grad is not None else 'NO GRAD'}")
print(f"  bias_net[0].weight.grad[:3]: {pm.bias_net[0].weight.grad[0,:3].detach().cpu().tolist() if pm.bias_net[0].weight.grad is not None else 'NO GRAD'}")

print("\n=== TEST 2: Without checkpointing ===")
layer.gradient_checkpointing = False
# Reset grads
for p in pm.parameters():
    p.grad = None

h2 = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out2 = layer(h2, step, beat_phase=bp)
loss2 = out2[0].mean() + (out2[1].mean() if isinstance(out2[1], torch.Tensor) else 0)
loss2.backward()
print(f"  bias_net[-1].weight.grad[:3]: {pm.bias_net[-1].weight.grad[0,:3].detach().cpu().tolist() if pm.bias_net[-1].weight.grad is not None else 'NO GRAD'}")
print(f"  bias_net[-1].bias.grad: {pm.bias_net[-1].bias.grad.detach().cpu().tolist() if pm.bias_net[-1].bias.grad is not None else 'NO GRAD'}")

print("\nDONE")
