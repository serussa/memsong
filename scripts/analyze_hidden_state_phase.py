#!/usr/bin/env python3
"""
ACE-Step DiT Hidden-State Phase Analysis — Baseline (no PhaseMemory)

Analyses the phase dynamics of the hidden-state projection through a 
random complex projection (like PhaseMemory.proj_real/proj_imag) applied 
to the hidden states at a selected DiT layer.

This provides a "pseudo-PhaseMemory" baseline to compare against the 
trained PhaseMemory model.

Usage:
    python scripts/analyze_hidden_state_phase.py \
        --model-root /root/autodl-tmp/Ace-Step1.5 \
        --output-dir /root/ACE-Step-1.5/output/hidden_state_phase_analysis
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ACE-Step hidden-state phase dynamics analysis (baseline)")
parser.add_argument("--model-root", type=str,
                    default="/root/autodl-tmp/Ace-Step1.5",
                    help="DiT checkpoint directory")
parser.add_argument("--config", type=str, default="acestep-v15-sft",
                    help="ACE-Step config name")
parser.add_argument("--output-dir", type=str,
                    default="/root/ACE-Step-1.5/output/hidden_state_phase_analysis",
                    help="Output directory for plots and .pt")
parser.add_argument("--infer-steps", type=int, default=25,
                    help="Number of denoising steps")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--target-layer", type=int, default=12,
                    help="Which DiT layer to hook (default: 12, same as PhaseMemory)")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ── Env ────────────────────────────────────────────────────────────────────
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acestep.handler import AceStepHandler

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load DiT model
# ────────────────────────────────────────────────────────────────────────────
print("[1/3] Loading DiT model ...")
handler = AceStepHandler()
status, ok = handler.initialize_service(
    project_root=args.model_root,
    config_path=args.config,
    device=args.device,
    use_flash_attention=False,
    compile_model=False,
    offload_to_cpu=False,
)
if not ok:
    raise RuntimeError(f"Service init failed: {status}")
print(f"  Model loaded: {type(handler.model).__name__}")

model = handler.model
device = handler.device
hidden_size = model.config.hidden_size
mem_dim = 2048  # match PhaseMemory mem_dim for fair comparison

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Build a random complex projection (same shape as PhaseMemory)
# ────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

proj_real = nn.Linear(hidden_size, mem_dim).to(device).to(model.dtype)
proj_imag = nn.Linear(hidden_size, mem_dim).to(device).to(model.dtype)

# No output projection needed — we only record the projected phase.
print(f"  Created random complex projection: {hidden_size} → {mem_dim}")

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Hook the target DiT layer to record hidden-state phase
# ────────────────────────────────────────────────────────────────────────────
target_layer = model.decoder.layers[args.target_layer]
phase_history: list[torch.Tensor] = []

def _record_hidden_phase(mod, inp, out):
    """Record torch.angle of the projected hidden state after this layer."""
    # out is a tuple: (hidden_states, ...)
    h = out[0] if isinstance(out, tuple) else out           # [B, T, D]
    h_mean = h.mean(dim=1)                                   # [B, D]
    zr = proj_real(h_mean)                                   # [B, M]
    zi = proj_imag(h_mean)                                   # [B, M]
    z = torch.complex(zr.float(), zi.float())
    phase_history.append(torch.angle(z).squeeze(0).cpu())    # [M]

handle = target_layer.register_forward_hook(_record_hidden_phase)

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Run a short denoising pass to collect phase states
# ────────────────────────────────────────────────────────────────────────────
print(f"[2/3] Running {args.infer_steps}-step denoising pass via layer {args.target_layer} ...")

dtype = model.dtype
B = 1
seq_len = 1024
hidden_dim = model.config.audio_acoustic_hidden_dim
context_dim = hidden_dim * 2

noise = torch.randn(B, seq_len, hidden_dim, device=device, dtype=dtype)
context = torch.randn(B, seq_len, context_dim, device=device, dtype=dtype)
null_emb = model.null_condition_emb.expand(B, 512, -1).to(device).to(dtype)

timesteps = torch.linspace(1.0, 0.0, args.infer_steps + 1, device=device, dtype=dtype)
latents = noise.clone()

model.eval()
with torch.no_grad():
    for i in range(args.infer_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        _ = model.decoder(
            hidden_states=latents,
            timestep=t.expand(B),
            timestep_r=t_next.expand(B),
            attention_mask=None,
            encoder_hidden_states=null_emb,
            encoder_attention_mask=None,
            context_latents=context,
        )

handle.remove()
print(f"  Collected {len(phase_history)} phase snapshots")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Save phase_history.pt and generate plots
# ────────────────────────────────────────────────────────────────────────────
print("[3/3] Saving and plotting ...")

phase = torch.stack(phase_history).float()      # [T, M]
T, M = phase.shape

torch.save(phase, output_dir / "phase_history.pt")
print(f"  Saved: {output_dir / 'phase_history.pt'}  shape=({T}, {M})")

# ── Recurrence Heatmap ─────────────────────────────────────────────────────
def recurrence_matrix(theta):
    d = theta[:, None, :] - theta[None, :, :]
    return torch.cos(d).mean(dim=-1).numpy()

rec = recurrence_matrix(phase)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(rec, cmap="magma", origin="lower", aspect="auto")
plt.colorbar(im, ax=ax, label="Mean cos phase diff")
ax.set_title(f"Phase Recurrence Heatmap (Original Model, Layer {args.target_layer})")
ax.set_xlabel("Time Step"); ax.set_ylabel("Time Step")
plt.tight_layout(); plt.savefig(output_dir / "recurrence_heatmap.png", dpi=200); plt.close()

# ── Phase Trajectories ─────────────────────────────────────────────────────
nplot = min(4, M)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < nplot:
        th = phase[:, i].numpy()
        ax.plot(np.cos(th), np.sin(th), linewidth=1.0)
        ax.scatter([np.cos(th[0])], [np.sin(th[0])], s=10, c="cyan", label="start")
        ax.set_title(f"Dim {i} (Original)"); ax.set_aspect("equal", "box")
        ax.set_xlabel("cos(theta)"); ax.set_ylabel("sin(theta)")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")
plt.tight_layout(); plt.savefig(output_dir / "phase_trajectory.png", dpi=200); plt.close()

# ── Temporal Recurrence Curve ──────────────────────────────────────────────
curve = np.zeros(T - 1, dtype=np.float32)
for d in range(1, T):
    curve[d - 1] = torch.cos(phase[:-d] - phase[d:]).mean().item()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, T), curve, linewidth=1.5, label="Original model")
ax.set_title(f"Temporal Recurrence Curve (Original Model, Layer {args.target_layer})")
ax.set_xlabel("Temporal distance (delta)"); ax.set_ylabel("Average recurrence")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(output_dir / "recurrence_curve.png", dpi=200); plt.close()

print(f"\nDone.  All outputs in: {output_dir}")
print(f"  phase_history.pt                ({T}, {M})")
print(f"  recurrence_heatmap.png")
print(f"  phase_trajectory.png")
print(f"  recurrence_curve.png")
