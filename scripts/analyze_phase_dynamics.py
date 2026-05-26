#!/usr/bin/env python3
"""
Phase Memory Recurrence Dynamics Analysis — Full Pipeline

1. Loads the DiT model + trained PhaseMemory weights.
2. Hooks PhaseMemory during denoising to record phase angles.
3. Saves phase_history.pt and generates three plots.

Usage:
    python scripts/analyze_phase_dynamics.py \
        --model-root /root/autodl-tmp/Ace-Step1.5 \
        --pm-dir /root/autodl-tmp/new_checkpoints/checkpoints/epoch_30_loss_0.8443 \
        --output-dir /root/ACE-Step-1.5/output/phase_memory_analysis
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Analyze PhaseMemory recurrence dynamics")
parser.add_argument("--model-root", type=str,
                    default="/root/autodl-tmp/Ace-Step1.5",
                    help="DiT checkpoint directory")
parser.add_argument("--pm-dir", type=str,
                    default="/root/autodl-tmp/new_checkpoints/checkpoints/epoch_30_loss_0.8443",
                    help="PhaseMemory weights directory")
parser.add_argument("--config", type=str, default="acestep-v15-sft",
                    help="ACE-Step config name")
parser.add_argument("--output-dir", type=str,
                    default="/root/ACE-Step-1.5/output/phase_memory_analysis",
                    help="Output directory for plots and .pt")
parser.add_argument("--infer-steps", type=int, default=25,
                    help="Number of denoising steps (fewer = faster analysis)")
parser.add_argument("--device", type=str, default="cuda")
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
from acestep.training.phase_memory_checkpoint import load_phase_memory_weights

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load DiT via AceStepHandler (handles remote code, config, etc.)
# ────────────────────────────────────────────────────────────────────────────
print("[1/4] Loading DiT model via AceStepHandler ...")
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

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Load PhaseMemory weights
# ────────────────────────────────────────────────────────────────────────────
print("[2/4] Loading PhaseMemory weights ...")
load_phase_memory_weights(model, args.pm_dir)

# Find the PhaseMemory module
pm_module = None
pm_path = ""
for name, mod in model.named_modules():
    if getattr(mod, "use_phase_memory", False) and hasattr(mod, "phase_memory"):
        pm_module = mod.phase_memory
        pm_path = name
        break
if pm_module is None:
    raise RuntimeError("No PhaseMemory module found!")
print(f"  Found: {pm_path}.phase_memory  (mem_dim={pm_module.proj_real.out_features})")

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Hook PhaseMemory forward to record torch.angle(z) after each call
# ────────────────────────────────────────────────────────────────────────────
phase_history: list[torch.Tensor] = []

def _record_phase(mod, inp, out):
    """Called after PhaseMemory.forward — record angle of complex state."""
    if mod.z_real is not None and mod.z_imag is not None:
        z = torch.complex(mod.z_real.float().detach(), mod.z_imag.float().detach())
        phase_history.append(torch.angle(z).squeeze(0).cpu())  # squeeze batch dim

handle = pm_module.register_forward_hook(_record_phase)

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Run a short denoising pass to collect phase states
# ────────────────────────────────────────────────────────────────────────────
print(f"[3/4] Running {args.infer_steps}-step denoising pass ...")

dtype = model.dtype
B = 1
seq_len = 1024  # arbitrary; adjust to your model's latent length
hidden_dim = model.config.audio_acoustic_hidden_dim
context_dim = hidden_dim * 2  # src_latents + chunk_masks concatenated

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
print("[4/4] Saving and plotting ...")

phase = torch.stack(phase_history).float()  # [T, M]
T, M = phase.shape

torch.save(phase, output_dir / "phase_history.pt")
print(f"  Saved: {output_dir / 'phase_history.pt'}  shape=({T}, {M})")

# ── Recurrence Heatmap ─────────────────────────────────────────────────────
def recurrence_matrix(theta: torch.Tensor) -> np.ndarray:
    d = theta[:, None, :] - theta[None, :, :]          # [T, T, M]
    return torch.cos(d).mean(dim=-1).numpy()            # [T, T]

rec = recurrence_matrix(phase)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(rec, cmap="magma", origin="lower", aspect="auto")
plt.colorbar(im, ax=ax, label="Mean cos phase diff")
ax.set_title("Phase Recurrence Heatmap")
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
        ax.set_title(f"Dim {i}"); ax.set_aspect("equal", "box")
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
ax.plot(np.arange(1, T), curve, linewidth=1.5)
ax.set_title("Temporal Recurrence Curve")
ax.set_xlabel("Temporal distance (delta)"); ax.set_ylabel("Average recurrence")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(output_dir / "recurrence_curve.png", dpi=200); plt.close()

print(f"\nDone.  All outputs in: {output_dir}")
print(f"  phase_history.pt        ({T}, {M})")
print(f"  recurrence_heatmap.png")
print(f"  phase_trajectory.png")
print(f"  recurrence_curve.png")
