#!/usr/bin/env python3
"""Analyze PhaseMemory recurrence for the latest trained weights.

This script:
1) Loads the DiT model via AceStepHandler.
2) Loads PhaseMemory weights from a checkpoint directory.
3) Runs a short denoising pass and records phase angles.
4) Saves phase_history.pt and three diagnostic plots.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PhaseMemory recurrence dynamics (latest)")
    parser.add_argument(
        "--model-root",
        type=str,
        default="/root/autodl-tmp/Ace-Step1.5",
        help="DiT checkpoint directory",
    )
    parser.add_argument(
        "--pm-dir",
        type=str,
        default="/root/autodl-tmp/newest_checkpoints/checkpoints/epoch_20_loss_0.8765",
        help="PhaseMemory weights directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="acestep-v15-sft",
        help="ACE-Step config name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/ACE-Step-1.5/output/phase_memory_analysis_latest",
        help="Output directory for plots and .pt",
    )
    parser.add_argument(
        "--infer-steps",
        type=int,
        default=25,
        help="Number of denoising steps (fewer = faster analysis)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Latent sequence length for the diagnostic run",
    )
    return parser.parse_args()


def _find_phase_memory_module(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    """Locate the PhaseMemory module attached to the DiT model."""
    for name, mod in model.named_modules():
        if hasattr(mod, "phase_memory"):
            return name + ".phase_memory", mod.phase_memory
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "PhaseMemory":
            return name, mod
    raise RuntimeError("No PhaseMemory module found in the model.")


def _get_phase_state(mod: torch.nn.Module) -> Optional[torch.Tensor]:
    """Return complex phase state as a tensor, or None if unavailable."""
    if hasattr(mod, "z_r") and hasattr(mod, "z_i") and mod.z_r is not None and mod.z_i is not None:
        return torch.complex(mod.z_r.float().detach(), mod.z_i.float().detach())
    if hasattr(mod, "z_real") and hasattr(mod, "z_imag") and mod.z_real is not None and mod.z_imag is not None:
        return torch.complex(mod.z_real.float().detach(), mod.z_imag.float().detach())
    return None


def _recurrence_matrix(theta: torch.Tensor) -> np.ndarray:
    d = theta[:, None, :] - theta[None, :, :]
    return torch.cos(d).mean(dim=-1).cpu().numpy()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Offline + minimal components
    os.environ["ACESTEP_OFFLINE"] = "1"
    os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
    sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from acestep.handler import AceStepHandler
    from acestep.training.phase_memory_checkpoint import load_phase_memory_weights

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

    print("[2/4] Loading PhaseMemory weights ...")
    load_phase_memory_weights(model, args.pm_dir)

    pm_path, pm_module = _find_phase_memory_module(model)
    mem_dim = None
    if hasattr(pm_module, "proj_r") and hasattr(pm_module.proj_r, "out_features"):
        mem_dim = pm_module.proj_r.out_features
    print(f"  Found: {pm_path}  (mem_dim={mem_dim})")

    phase_history: list[torch.Tensor] = []

    def _record_phase(mod: torch.nn.Module, _inp: Iterable, _out: torch.Tensor) -> None:
        z = _get_phase_state(mod)
        if z is not None:
            phase_history.append(torch.angle(z).squeeze(0).cpu())

    hook = pm_module.register_forward_hook(_record_phase)

    print(f"[3/4] Running {args.infer_steps}-step denoising pass ...")
    dtype = model.dtype
    bsz = 1
    seq_len = args.seq_len
    hidden_dim = model.config.audio_acoustic_hidden_dim
    context_dim = hidden_dim * 2

    noise = torch.randn(bsz, seq_len, hidden_dim, device=device, dtype=dtype)
    context = torch.randn(bsz, seq_len, context_dim, device=device, dtype=dtype)
    null_emb = model.null_condition_emb.expand(bsz, 512, -1).to(device).to(dtype)

    timesteps = torch.linspace(1.0, 0.0, args.infer_steps + 1, device=device, dtype=dtype)
    latents = noise.clone()

    model.eval()
    with torch.no_grad():
        for i in range(args.infer_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            _ = model.decoder(
                hidden_states=latents,
                timestep=t.expand(bsz),
                timestep_r=t_next.expand(bsz),
                attention_mask=None,
                encoder_hidden_states=null_emb,
                encoder_attention_mask=None,
                context_latents=context,
            )

    hook.remove()
    print(f"  Collected {len(phase_history)} phase snapshots")

    print("[4/4] Saving and plotting ...")
    if not phase_history:
        raise RuntimeError("No phase snapshots recorded. Check PhaseMemory attachment.")

    phase = torch.stack(phase_history).float()
    if phase.ndim == 3:
        # Collapse token dimension for a single phase trajectory per step.
        phase = phase.mean(dim=1)
    elif phase.ndim != 2:
        raise ValueError(f"Unexpected phase tensor shape: {tuple(phase.shape)}")

    steps, mem = phase.shape

    torch.save(phase, output_dir / "phase_history.pt")
    print(f"  Saved: {output_dir / 'phase_history.pt'}  shape=({steps}, {mem})")

    # Recurrence heatmap
    rec = _recurrence_matrix(phase)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rec, cmap="magma", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean cos phase diff")
    ax.set_title("Phase Recurrence Heatmap")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Time Step")
    plt.tight_layout()
    plt.savefig(output_dir / "recurrence_heatmap.png", dpi=200)
    plt.close()

    # Phase trajectories
    nplot = min(4, mem)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < nplot:
            th = phase[:, i].numpy()
            ax.plot(np.cos(th), np.sin(th), linewidth=1.0)
            ax.scatter([np.cos(th[0])], [np.sin(th[0])], s=10, c="cyan", label="start")
            ax.set_title(f"Dim {i}")
            ax.set_aspect("equal", "box")
            ax.set_xlabel("cos(theta)")
            ax.set_ylabel("sin(theta)")
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "phase_trajectory.png", dpi=200)
    plt.close()

    # Temporal recurrence curve
    curve = np.zeros(steps - 1, dtype=np.float32)
    for d in range(1, steps):
        curve[d - 1] = torch.cos(phase[:-d] - phase[d:]).mean().item()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, steps), curve, linewidth=1.5)
    ax.set_title("Temporal Recurrence Curve")
    ax.set_xlabel("Temporal distance (delta)")
    ax.set_ylabel("Average recurrence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recurrence_curve.png", dpi=200)
    plt.close()

    print(f"\nDone. All outputs in: {output_dir}")
    print("  phase_history.pt")
    print("  recurrence_heatmap.png")
    print("  phase_trajectory.png")
    print("  recurrence_curve.png")


if __name__ == "__main__":
    main()
