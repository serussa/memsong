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
import math
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
parser.add_argument("--num-steps", type=int, default=None,
                    help="Alias for --infer-steps (preferred for experiments)")
parser.add_argument("--timestep-sampling", type=str, default="linear",
                    choices=["linear", "quadratic", "log"],
                    help="Timestep sampling strategy")
parser.add_argument("--proj-alpha", type=float, default=1.0,
                    help="Projection blend factor for PhaseMemory (1.0=full, 0.0=ablation)")
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
proj_r = getattr(pm_module, "proj_real", None) or getattr(pm_module, "proj_r", None)
proj_dim = proj_r.out_features if proj_r is not None else "?"
print(f"  Found: {pm_path}.phase_memory  (mem_dim={proj_dim})")

def _identity_projection(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Project by identity/slice/pad to match out_dim."""
    in_dim = x.shape[-1]
    if in_dim == out_dim:
        return x
    if in_dim > out_dim:
        return x[..., :out_dim]
    pad = x.new_zeros(*x.shape[:-1], out_dim - in_dim)
    return torch.cat([x, pad], dim=-1)


def _wrap_projection(proj_module: torch.nn.Module, alpha: float) -> None:
    """Blend projection output with identity mapping for ablation/weakening."""
    if alpha >= 0.999:
        return

    original_forward = proj_module.forward

    def blended_forward(x: torch.Tensor) -> torch.Tensor:
        proj = original_forward(x)
        if alpha <= 0.0:
            return _identity_projection(x, proj.shape[-1])
        ident = _identity_projection(x, proj.shape[-1])
        return alpha * proj + (1.0 - alpha) * ident

    proj_module.forward = blended_forward


def _extract_phase_state(mod: torch.nn.Module) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Get phase state tensors with support for legacy attribute names."""
    z_real = getattr(mod, "z_real", None)
    z_imag = getattr(mod, "z_imag", None)
    if z_real is None or z_imag is None:
        z_real = getattr(mod, "z_r", None)
        z_imag = getattr(mod, "z_i", None)
    return z_real, z_imag


def _get_projection_modules(
    mod: torch.nn.Module,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    """Resolve projection modules for PhaseMemory across naming variants."""
    proj_r = getattr(mod, "proj_real", None) or getattr(mod, "proj_r", None)
    proj_i = getattr(mod, "proj_imag", None) or getattr(mod, "proj_i", None)
    return proj_r, proj_i


def _build_timesteps(num_steps: int, sampling: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a [num_steps+1] timestep schedule from 1.0 to 0.0."""
    base = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if sampling == "quadratic":
        base = base**2
    elif sampling == "log":
        log_base = 100.0
        base = torch.log1p(base * (log_base - 1.0)) / math.log(log_base)
    return 1.0 - base


proj_r, proj_i = _get_projection_modules(pm_module)
if proj_r is not None and proj_i is not None:
    _wrap_projection(proj_r, args.proj_alpha)
    _wrap_projection(proj_i, args.proj_alpha)
else:
    print("  Projection modules not found; skipping proj-alpha blending")

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Hook PhaseMemory forward to record torch.angle(z) after each call
# ────────────────────────────────────────────────────────────────────────────
phase_history: list[torch.Tensor] = []
injection_norm_history: list[torch.Tensor] = []
state_norm_history: list[torch.Tensor] = []

def _record_phase(mod, inp, out):
    """Called after PhaseMemory.forward — record angle of complex state."""
    z_real, z_imag = _extract_phase_state(mod)
    if z_real is not None and z_imag is not None:
        z = torch.complex(z_real.float().detach(), z_imag.float().detach())
        phase_history.append(torch.angle(z).squeeze(0).cpu())  # squeeze batch dim
        state_norm = torch.sqrt(z_real.float().pow(2) + z_imag.float().pow(2)).mean()
        state_norm_history.append(state_norm.cpu())

    if inp and out is not None:
        injected = out - inp[0]
        inj_norm = injected.float().norm(dim=-1).mean()
        injection_norm_history.append(inj_norm.cpu())

handle = pm_module.register_forward_hook(_record_phase)

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Run a short denoising pass to collect phase states
# ────────────────────────────────────────────────────────────────────────────
num_steps = args.num_steps or args.infer_steps
print(
    f"[3/4] Running {num_steps}-step denoising pass "
    f"(sampling={args.timestep_sampling}, proj_alpha={args.proj_alpha}) ..."
)

dtype = model.dtype
B = 1
seq_len = 1024  # arbitrary; adjust to your model's latent length
hidden_dim = model.config.audio_acoustic_hidden_dim
context_dim = hidden_dim * 2  # src_latents + chunk_masks concatenated

noise = torch.randn(B, seq_len, hidden_dim, device=device, dtype=dtype)
context = torch.randn(B, seq_len, context_dim, device=device, dtype=dtype)
null_emb = model.null_condition_emb.expand(B, 512, -1).to(device).to(dtype)

timesteps = _build_timesteps(num_steps, args.timestep_sampling, device, dtype)
latents = noise.clone()

model.eval()
with torch.no_grad():
    for i in range(num_steps):
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

if injection_norm_history:
    injection_norm = torch.stack(injection_norm_history).float().numpy()
    torch.save(torch.tensor(injection_norm), output_dir / "phase_injection_norm.pt")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(injection_norm.shape[0]), injection_norm, linewidth=1.5)
    ax.set_title("Phase Injection Norm")
    ax.set_xlabel("Time step")
    ax.set_ylabel("||injected||")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_dir / "phase_injection_norm.png", dpi=200); plt.close()

if state_norm_history:
    state_norm = torch.stack(state_norm_history).float().numpy()
    torch.save(torch.tensor(state_norm), output_dir / "phase_state_norm.pt")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(state_norm.shape[0]), state_norm, linewidth=1.5)
    ax.set_title("Phase State Norm")
    ax.set_xlabel("Time step")
    ax.set_ylabel("||state||")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_dir / "phase_state_norm.png", dpi=200); plt.close()

    if state_norm.shape[0] > 1:
        delta = np.abs(np.diff(state_norm))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, state_norm.shape[0]), delta, linewidth=1.5)
        ax.set_title("Phase State Stepwise Change")
        ax.set_xlabel("Time step")
        ax.set_ylabel("|\u0394||state|||")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(output_dir / "phase_state_delta.png", dpi=200); plt.close()

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
