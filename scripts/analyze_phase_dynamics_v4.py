#!/usr/bin/env python3
"""
Phase Memory Recurrence Dynamics Analysis — V4 (Tiny Controlled)

Loads the DiT model + V4 PhaseMemory weights (128-dim, bounded ω, input gate),
hooks PhaseMemory during denoising to record phase angles + gate values,
saves phase_history.pt and generates plots.

Usage:
    python scripts/analyze_phase_dynamics_v4.py \
        --model-root /root/autodl-tmp/Ace-Step1.5 \
        --pm-dir /root/autodl-tmp/new_gt_checkpoints/checkpoints/epoch_5_loss_0.8448 \
        --output-dir /root/ACE-Step-1.5/output/phase_memory_analysis
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Analyze PhaseMemory v4 recurrence dynamics")
parser.add_argument("--model-root", type=str,
                    default="/root/autodl-tmp/Ace-Step1.5",
                    help="DiT checkpoint directory")
parser.add_argument("--pm-dir", type=str,
                    default="/root/autodl-tmp/new_gt_checkpoints/checkpoints/epoch_5_loss_0.8448",
                    help="PhaseMemory V4 weights directory")
parser.add_argument("--config", type=str, default="acestep-v15-sft",
                    help="ACE-Step config name")
parser.add_argument("--output-dir", type=str,
                    default="/root/ACE-Step-1.5/output/phase_memory_analysis",
                    help="Output directory for plots and .pt")
parser.add_argument("--infer-steps", type=int, default=25,
                    help="Number of denoising steps (fewer = faster analysis)")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

DEFAULT_LYRICS = """[Verse] 爱总忽然退潮 心慌乱触礁  沉没在深海里 看海面闪耀 但回忆像水草 紧紧的缠绕 梦才温热眼角 就冰冷掉 努力越过风暴 向着未来飘 我们才会遇到 感动的拥抱 你总是能知道 我的坚强剩多少 [Pre-chorus] 给我最刚好的依靠 [Chorus] 你手心的太阳 只轻放在我背上 委屈就能笑着落泪 被释放 在手心的太阳 黑暗里特别明亮 让远路好像 是一种分享 而不是漫长 [Inst] 让眼睛看不到 嫉妒的燃烧 [Verse] 让耳朵听不到 谎言的吵闹 再没有人相信 爱能永恒那一秒 我们正坚定的微笑 [Chorus] 你手心的太阳 有种安定的力量 就算世界再乱我也 不心慌 我手心的太阳 或许只像个月亮 却用所有爱 为你投射我最暖的光芒 你手心的太阳 只轻放在我背上 [Bridge] 委屈就能笑着落泪 被释放 在手心的太阳 黑暗里特别明亮 让远路好像 是一种分享 你手心的太阳 有种安定的力量 [Chorus] 就算世界再乱我也 不心慌 我手心的太阳 或许只像个月亮 却用所有爱 为你投射我 最暖的光芒
"""

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
# Step 1: Load DiT via AceStepHandler
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
# Step 2: Load V4 PhaseMemory weights
# ────────────────────────────────────────────────────────────────────────────
print("[2/4] Loading PhaseMemory V4 weights ...")
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
# Step 3: Hook PhaseMemory forward to record phase + gate
# ────────────────────────────────────────────────────────────────────────────
phase_history: list[torch.Tensor] = []
gate_history: list[float] = []
norm_history: list[float] = []

def _record_phase(mod, inp, out):
    """Called after PhaseMemory.forward — record phase angle, gate, and |z| norm."""
    if mod.z_real is not None and mod.z_imag is not None:
        zr = mod.z_real.float().detach()
        zi = mod.z_imag.float().detach()
        z = torch.complex(zr, zi)
        phase_history.append(torch.angle(z).squeeze(0).cpu())
        norm_history.append(torch.abs(z).mean().item())
    if hasattr(mod, "last_g") and mod.last_g is not None:
        gate_history.append(mod.last_g.float().mean().item())

handle = pm_module.register_forward_hook(_record_phase)

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Run a short denoising pass to collect phase states
# ────────────────────────────────────────────────────────────────────────────
print(f"[3/4] Running {args.infer_steps}-step denoising pass ...")

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
print(f"  Collected {len(phase_history)} phase snapshots, {len(gate_history)} gate snapshots, {len(norm_history)} norm snapshots")

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Save phase_history.pt and generate plots
# ────────────────────────────────────────────────────────────────────────────
print("[4/4] Saving and plotting ...")

phase = torch.stack(phase_history).float()  # [T, M]
T, M = phase.shape

torch.save(phase, output_dir / "phase_history.pt")
print(f"  Saved: {output_dir / 'phase_history.pt'}  shape=({T}, {M})")

# ── Gate evolution curve with lyrics section overlay ───────────────────────
if gate_history:
    # Parse lyrics sections from DEFAULT_LYRICS
    import re
    sections = list(re.finditer(r"\[(Verse|Pre-chorus|Chorus|Inst|Bridge)\]", DEFAULT_LYRICS))
    section_spans = [(m.group(1), m.start()) for m in sections]
    # Normalise section positions to [0, 1] for overlay
    total_len = len(DEFAULT_LYRICS)
    section_norm = [(label, pos / total_len) for (label, pos) in section_spans]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    xs = range(len(gate_history))
    ax.plot(xs, gate_history, "o-", markersize=4, linewidth=1.5, label="gate")
    ax.axhline(y=0.047, color="gray", linestyle="--", alpha=0.5, label="init (≈0.047)")
    
    # Map each lyrics section to nearest denoising step
    colours = {"Verse": "tab:blue", "Pre-chorus": "tab:orange", "Chorus": "tab:red",
               "Inst": "tab:purple", "Bridge": "tab:green"}
    for label, frac in section_norm:
        step = int(round(frac * (len(gate_history) - 1))) if gate_history else 0
        ax.axvline(x=step, color=colours.get(label, "k"), linestyle="--", alpha=0.6, linewidth=1)
        ax.text(step, ax.get_ylim()[1] * 0.95, label, rotation=90, fontsize=7,
                color=colours.get(label, "k"), va="top", ha="right")
    
    ax.set_title("Input Gate Evolution (V4) vs Lyrics Structure")
    ax.set_xlabel("Denoising Step"); ax.set_ylabel("Gate mean")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(output_dir / "gate_evolution.png", dpi=200); plt.close()
    print(f"  Gate range: [{min(gate_history):.4f}, {max(gate_history):.4f}]")

# ── Norm evolution: |z| over denoising steps (collapse detector) ──────────
if norm_history:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(norm_history)), norm_history, "o-", markersize=4, linewidth=1.5, color="tab:red")
    ax.set_title("Phase Norm |z| Evolution (V4)  — COLLAPSE DETECTOR")
    ax.set_xlabel("Denoising Step"); ax.set_ylabel("Mean |z|")
    ax.grid(True, alpha=0.3)
    # Mark exponential danger threshold
    if norm_history[0] > 0:
        ax.axhline(y=norm_history[0] * 10, color="orange", linestyle="--", alpha=0.5, label="10× initial")
        ax.axhline(y=norm_history[0] * 100, color="red", linestyle="--", alpha=0.5, label="100× initial")
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(output_dir / "norm_evolution.png", dpi=200); plt.close()
    print(f"  Norm range: [{min(norm_history):.4f}, {max(norm_history):.4f}]  "
          f"(ratio: {max(norm_history) / max(norm_history[0], 1e-8):.1f}×)")

# ── Recurrence Heatmap ─────────────────────────────────────────────────────
def recurrence_matrix(theta):
    d = theta[:, None, :] - theta[None, :, :]
    return torch.cos(d).mean(dim=-1).numpy()

rec = recurrence_matrix(phase)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(rec, cmap="magma", origin="lower", aspect="auto")
plt.colorbar(im, ax=ax, label="Mean cos phase diff")
ax.set_title("Phase Recurrence Heatmap (V4 — Tiny Controlled)")
ax.set_xlabel("Time Step"); ax.set_ylabel("Time Step")
plt.tight_layout(); plt.savefig(output_dir / "recurrence_heatmap_v4.png", dpi=200); plt.close()

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
plt.tight_layout(); plt.savefig(output_dir / "phase_trajectory_v4.png", dpi=200); plt.close()

# ── Temporal Recurrence Curve ──────────────────────────────────────────────
curve = np.zeros(T - 1, dtype=np.float32)
for d in range(1, T):
    curve[d - 1] = torch.cos(phase[:-d] - phase[d:]).mean().item()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, T), curve, linewidth=1.5)
ax.set_title("Temporal Recurrence Curve (V4)")
ax.set_xlabel("Temporal distance (delta)"); ax.set_ylabel("Average recurrence")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(output_dir / "recurrence_curve_v4.png", dpi=200); plt.close()

print(f"\nDone.  All outputs in: {output_dir}")
print(f"  phase_history.pt           ({T}, {M})")
print(f"  gate_evolution.png")
print(f"  norm_evolution.png")
print(f"  recurrence_heatmap_v4.png")
print(f"  phase_trajectory_v4.png")
print(f"  recurrence_curve_v4.png")
