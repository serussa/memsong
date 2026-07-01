"""Minimal test: does kl_loss produce gradients on bias_net?"""
import torch
import torch.nn.functional as F
from acestep.phase_memory import PhaseMemory

# PhaseMemory with default dims (hidden=2048, mem_dim=128)
pm = PhaseMemory(dim=2048, mem_dim=128).to("cuda", dtype=torch.bfloat16)
pm.train()

# Check bias_net[-1] weights are non-zero
bn = pm.bias_net[-1]
print(f"bias_net[-1].weight[:3]: {bn.weight[0,:3].detach().cpu().tolist()}")
print(f"bias_net[-1].bias: {bn.bias.detach().cpu().tolist()}")

# Create dummy inputs
B, T = 2, 100
h = torch.randn(B, T, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
step = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
bp = torch.rand(B, T, device="cuda", dtype=torch.bfloat16)  # beat_phase [B, T]

# Forward
out, kl = pm(h, step, beat_phase=bp)
print(f"\nkl_loss: {kl} (type={type(kl).__name__})")
if isinstance(kl, torch.Tensor):
    print(f"kl.grad_fn: {kl.grad_fn}")
    print(f"kl.requires_grad: {kl.requires_grad}")

# Check if bias_net[-1].weight has the `_bias_net_output` marker
print(f"\n_bias_net_output attr: {getattr(bn, '_bias_net_output', False)}")

# Now do a backward step
loss = kl.mean() + F.mse_loss(out, torch.zeros_like(out))
loss.backward()

print(f"\nAfter backward:")
print(f"bias_net[-1].weight.grad[:3]: {bn.weight.grad[0,:3].detach().cpu().tolist() if bn.weight.grad is not None else 'NO GRAD'}")
print(f"bias_net[-1].bias.grad: {bn.bias.grad.detach().cpu().tolist() if bn.bias.grad is not None else 'NO GRAD'}")
print(f"bias_net[0].weight.grad[:3]: {pm.bias_net[0].weight.grad[0,:3].detach().cpu().tolist() if pm.bias_net[0].weight.grad is not None else 'NO GRAD'}")

# Test without beat_phase (should return kl=None)
out2, kl2 = pm(h, step, beat_phase=None)
print(f"\nWithout beat_phase: kl={kl2}")
