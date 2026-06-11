#!/usr/bin/env python3
"""PhaseMemory experiment runner: injection energy + latent norm curves."""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="PhaseMemory ablation experiments")
    parser.add_argument(
        "--model-root",
        type=str,
        default="/root/autodl-tmp/Ace-Step1.5",
        help="DiT checkpoint directory",
    )
    parser.add_argument(
        "--pm-dir",
        type=str,
        default="/root/autodl-tmp/new_checkpoints/checkpoints/epoch_30_loss_0.8443",
        help="PhaseMemory weights directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/ACE-Step-1.5/output/phase_memory_experiments",
        help="Output directory for plots and .pt",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_hidden_states(outputs: object) -> torch.Tensor:
    """Extract hidden states from decoder output."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (list, tuple)) and outputs:
        if isinstance(outputs[0], torch.Tensor):
            return outputs[0]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states
    raise RuntimeError("Unable to extract hidden states from decoder output")


def _make_plot(
    out_path: Path,
    xs: Iterable[int],
    series: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data in series:
        ax.plot(xs, data, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ACESTEP_OFFLINE"] = "1"
    os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
    sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))

    from acestep.handler import AceStepHandler
    from acestep.phase_memory import reset_phase_memory, set_phase_memory_scale
    from acestep.training.phase_memory_checkpoint import load_phase_memory_weights

    print("[1/3] Loading model...")
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=args.model_root,
        config_path="acestep-v15-sft",
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"Service init failed: {status}")

    model = handler.model
    device = handler.device

    print("[2/3] Loading PhaseMemory weights...")
    load_phase_memory_weights(model, args.pm_dir)

    pm_module = None
    for name, mod in model.named_modules():
        if getattr(mod, "use_phase_memory", False) and hasattr(mod, "phase_memory"):
            pm_module = mod.phase_memory
            break
    if pm_module is None:
        raise RuntimeError("No PhaseMemory module found")

    dtype = model.dtype
    torch.manual_seed(0)
    B = 1
    seq_len = 1024
    hidden_dim = model.config.audio_acoustic_hidden_dim
    context_dim = hidden_dim * 2
    noise = torch.randn(B, seq_len, hidden_dim, device=device, dtype=dtype)
    context = torch.randn(B, seq_len, context_dim, device=device, dtype=dtype)
    null_emb = model.null_condition_emb.expand(B, 512, -1).to(device).to(dtype)
    timesteps = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device, dtype=dtype)

    def run_pass(scale: float) -> dict[str, np.ndarray]:
        set_phase_memory_scale(model, scale)
        reset_phase_memory(model)

        injection_norm = np.zeros(args.num_steps, dtype=np.float32)
        state_norm = np.zeros(args.num_steps, dtype=np.float32)
        latent_norm = np.zeros(args.num_steps, dtype=np.float32)
        current_step = {"idx": -1}

        def _hook(mod, inp, out):
            idx = current_step["idx"]
            if idx < 0:
                return
            if inp and out is not None:
                injected = out - inp[0]
                injection_norm[idx] = injected.float().norm(dim=-1).mean().item()
            z_real = getattr(mod, "z_real", None)
            z_imag = getattr(mod, "z_imag", None)
            if z_real is not None and z_imag is not None:
                state = torch.sqrt(z_real.float().pow(2) + z_imag.float().pow(2)).mean()
                state_norm[idx] = state.item()

        handle = pm_module.register_forward_hook(_hook)

        latents = noise.clone()
        model.eval()
        with torch.no_grad():
            for i in range(args.num_steps):
                current_step["idx"] = i
                t = timesteps[i]
                t_next = timesteps[i + 1]
                outputs = model.decoder(
                    hidden_states=latents,
                    timestep=t.expand(B),
                    timestep_r=t_next.expand(B),
                    attention_mask=None,
                    encoder_hidden_states=null_emb,
                    encoder_attention_mask=None,
                    context_latents=context,
                )
                latents = _extract_hidden_states(outputs)
                latent_norm[i] = latents.float().norm(dim=-1).mean().item()

        handle.remove()
        return {
            "injection_norm": injection_norm,
            "state_norm": state_norm,
            "latent_norm": latent_norm,
        }

    print("[3/3] Running baseline and full passes...")
    baseline = run_pass(0.0)
    full = run_pass(1.0)

    torch.save(baseline, output_dir / "baseline_stats.pt")
    torch.save(full, output_dir / "full_stats.pt")

    steps = np.arange(args.num_steps)
    _make_plot(
        output_dir / "figure1_injection_energy.png",
        steps,
        [("pm_scale=0.0", baseline["injection_norm"]), ("pm_scale=1.0", full["injection_norm"])],
        "Injection Energy vs Step",
        "||h_out - h_in||",
    )
    _make_plot(
        output_dir / "figure2_latent_norm.png",
        steps,
        [("pm_scale=0.0", baseline["latent_norm"]), ("pm_scale=1.0", full["latent_norm"])],
        "Latent Norm vs Step",
        "||hidden_states||",
    )
    _make_plot(
        output_dir / "phase_state_norm.png",
        steps,
        [("pm_scale=0.0", baseline["state_norm"]), ("pm_scale=1.0", full["state_norm"])],
        "Phase State Norm vs Step",
        "||state||",
    )

    print(f"Done. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
