#!/usr/bin/env python3
"""
φ Intervention Test
====================
Causal test: does φ change → generation structure change?

Three conditions:
  (1) natural  — baseline (normal PhaseMemory)
  (2) perturb  — random phase rotation at each step (entropy injection)
  (3) frozen   — phase state freezes after first step (no temporal accumulation)

Metrics on both φ trajectories and generated latents.

Usage:
    # Quick test (CPU-compatible diagnostic mode, random init)
    ACESTEP_LOCAL_MODEL_CODE=1 python scripts/phi_intervention_test.py --quick

    # Full test with real model
    ACESTEP_LOCAL_MODEL_CODE=1 python scripts/phi_intervention_test.py \
        --model-root /path/to/model \
        --checkpoint-dir /path/to/phase_memory_ckpt \
        --seed 42 --perturb-scale 0.15 --num-steps 50
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Generation-oriented latent metrics (on final latents [T, D])
# ──────────────────────────────────────────────────────────────────────

def compute_latent_metrics(latents: torch.Tensor, short_lags=5, long_start=10, long_end=25) -> dict:
    """
    Compute generation structure metrics from audio latents [T, D].

    Returns:
        dict with keys: frame_cosine_sim, repetition_autocorr, self_sim_entropy,
                        pca_eff_rank, longrange_corr, boundary_changes
    """
    T, D = latents.shape
    if T < 3:
        return {}

    # Normalize per frame
    normed = latents / (latents.norm(dim=-1, keepdim=True) + 1e-8)

    # Frame-to-frame cosine similarity (mean over time)
    cos_sim = (normed[:-1] * normed[1:]).sum(dim=-1).mean().item()

    # Autocorrelation across multiple lags (repetition measure)
    max_lag = min(short_lags + 10, T // 2)
    autocorrs = []
    for lag in range(1, max_lag + 1):
        ac = (normed[:-lag] * normed[lag:]).sum(dim=-1).mean().item()
        autocorrs.append(ac)

    # Short-range repetition (lags 1-5)
    short_ac = float(np.mean(autocorrs[:min(short_lags, len(autocorrs))])) if autocorrs else float('nan')
    # Long-range coherence (lags 10-25 or available)
    lags_avail = len(autocorrs)
    if lags_avail >= long_start:
        long_ac = float(np.mean(autocorrs[long_start - 1:min(long_end, lags_avail)]))
    else:
        long_ac = float('nan')

    # Self-similarity matrix entropy → structure score
    S = latents @ latents.T  # [T, T]
    S_norm = (S - S.min()) / (S.max() - S.min() + 1e-10)
    p = S_norm.flatten().clamp(min=1e-10)
    p = p / p.sum()
    entropy = float(-(p * p.log()).sum())
    norm_entropy = entropy / math.log(T * T)

    # PCA effective rank → dimensionality / mode collapse
    X = latents.numpy()
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var_exp = (s ** 2) / (s ** 2).sum()
        cumvar = np.cumsum(var_exp)
        eff_rank = int((cumvar < 0.95).sum()) + 1
        top3_var = float(var_exp[:3].sum()) if len(var_exp) >= 3 else 1.0
    except np.linalg.LinAlgError:
        eff_rank = float('nan')
        top3_var = float('nan')

    # Change-point detection: count large frame-to-frame jumps
    frame_deltas = (latents[1:] - latents[:-1]).norm(dim=-1)
    threshold = frame_deltas.mean() + 1.0 * frame_deltas.std()
    n_boundaries = int((frame_deltas > threshold).sum().item())

    return {
        "latent_frame_cosine_sim": round(cos_sim, 4),
        "latent_short_autocorr_l1-{}".format(short_lags): round(short_ac, 4),
        "latent_long_autocorr_l{}-{}".format(long_start, long_end): round(long_ac, 4) if not math.isnan(long_ac) else None,
        "latent_self_sim_entropy": round(norm_entropy, 4),
        "latent_pca_eff_rank_95": eff_rank if not math.isnan(eff_rank) else None,
        "latent_pca_top3_var": round(top3_var, 4),
        "latent_n_boundary_changes": n_boundaries,
        "latent_total_var": float(latents.var().item()),
    }


# ──────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────

def run_experiment(
    model,
    pm_module: torch.nn.Module,
    num_steps: int = 50,
    seq_len: int = 750,
    seed: int = 42,
    perturb_scale: float = 0.15,
    verbose: bool = True,
) -> dict:
    """
    Run all three intervention conditions and collect φ + latents.

    Args:
        model: Loaded AceStepDiTModel.
        pm_module: The PhaseMemory instance inside model.
        num_steps: Diffusion steps.
        seq_len: Sequence length (frames).
        seed: Random seed for reproducibility.
        perturb_scale: Noise level for perturb condition.
        verbose: Print progress.

    Returns:
        dict with keys "natural", "perturb", "frozen", each containing:
          - phi: [T_steps, S_tokens, D_mem] unwrapped
          - latents: [S_tokens, D_audio] final denoised
    """
    # Import here to avoid circular dependencies at module level
    from acestep.phase_memory import reset_phase_memory
    from collect_phi_across_regimes import PhaseMemoryHook, _make_conditioning

    interventions = ["natural", "perturb", "frozen"]
    device = next(model.parameters()).device
    model_dtype = model.dtype

    # Build conditioning once (shared across all interventions)
    cond = _make_conditioning(model, seq_len, device, model_dtype, seed)

    print("  Preparing conditioning...", end=" ", flush=True)
    with torch.no_grad():
        enc_hs, enc_am, context_latents = model.prepare_condition(
            text_hidden_states=cond["text_hidden_states"],
            text_attention_mask=cond["text_attention_mask"],
            lyric_hidden_states=cond["lyric_hidden_states"],
            lyric_attention_mask=cond["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=cond["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=cond["refer_audio_order_mask"],
            hidden_states=cond["src_latents"],
            attention_mask=cond["attention_mask"],
            silence_latent=cond["silence_latent"],
            src_latents=cond["src_latents"],
            chunk_masks=cond["chunk_masks"],
            is_covers=cond["is_covers"],
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )
    print("done")

    # Timesteps
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=model_dtype)
    noise = model.prepare_noise(context_latents, seed)

    results = {}
    model.eval()

    for intervention in interventions:
        if verbose:
            print(f"\n  ─── {intervention.upper():8s} ───")

        reset_phase_memory(model)
        pm_module.intervention = intervention
        pm_module.perturb_noise_scale = perturb_scale

        xt = noise.clone()
        hook = PhaseMemoryHook(pm_module)

        start_t = time.time()
        with torch.no_grad():
            with hook:
                for i in range(num_steps):
                    t_curr = t[i]
                    t_next = t[i + 1]

                    t_tensor = t_curr.expand(1)
                    t_next_tensor = t_next.expand(1)

                    decoder_outputs = model.decoder(
                        hidden_states=xt,
                        timestep=t_tensor,
                        timestep_r=t_next_tensor,
                        attention_mask=cond["attention_mask"],
                        encoder_hidden_states=enc_hs,
                        encoder_attention_mask=enc_am,
                        context_latents=context_latents,
                        use_cache=True,
                        beat_phase=None,
                    )
                    vt = decoder_outputs[0]
                    dt = (t_curr - t_next).unsqueeze(-1).unsqueeze(-1)
                    xt = xt - vt * dt

        elapsed = time.time() - start_t

        phi = hook.get_phi(unwrap=True)  # [T_steps, S_tokens, D_mem]
        latents = xt[0].float().cpu()  # [S_tokens, 64]
        z_r = pm_module.z_r[0].float().cpu() if pm_module.z_r is not None else None

        results[intervention] = {
            "phi": phi,
            "latents": latents,
            "z_r": z_r,
            "time_sec": round(elapsed, 1),
        }

        if verbose:
            phi_shape = list(phi.shape)
            lat_shape = list(latents.shape)
            print(f"    φ {phi_shape}, latents {lat_shape}, z_r norm={z_r.norm().item():.3f}, {elapsed:.1f}s")

    return results


def compute_all_metrics(results: dict) -> dict:
    """Compute φ metrics + latent metrics for all conditions."""
    from phi_analysis_metrics import (
        structural_metrics_per_token,
        temporal_structure_metrics_per_token,
    )

    metrics = {}
    for name, data in results.items():
        phi = data["phi"]  # [T_steps, S, D] or [T_steps, D]
        latents = data["latents"]  # [S, D_audio]

        m = {}

        # φ structural metrics
        if phi.ndim == 3 and phi.shape[1] > 1:
            # Per-token aggregation
            struct = structural_metrics_per_token(phi)
            temporal = temporal_structure_metrics_per_token(phi)
        else:
            # 2D
            from phi_analysis_metrics import structural_metrics, temporal_structure_metrics
            struct = structural_metrics(phi)
            temporal = temporal_structure_metrics(phi)

        m.update(struct)
        m.update(temporal)

        # φ PCA
        phi_2d = phi.reshape(phi.shape[0], -1)  # flatten tokens x dims
        from phi_analysis_metrics import pca_effective_rank
        pca = pca_effective_rank(phi_2d)
        m["phi_pca_eff_rank"] = pca["pca_effective_rank"]
        m["phi_pca_top3_var"] = round(pca["pca_top3_var_explained"], 4)

        # Latent generation metrics
        lat_m = compute_latent_metrics(latents, short_lags=5, long_start=10, long_end=min(25, latents.shape[0] - 1))
        m.update(lat_m)

        metrics[name] = m

    return metrics


def print_comparison_table(metrics: dict) -> None:
    """Print formatted markdown comparison table."""
    all_keys = set()
    for m in metrics.values():
        all_keys.update(m.keys())

    # Filter out _std keys (secondary aggregation stats)
    primary_keys = sorted(k for k in all_keys if not k.endswith("_std"))

    # Separate into categories
    phi_keys = sorted(k for k in primary_keys if not k.startswith("latent_"))
    latent_keys = sorted(k for k in primary_keys if k.startswith("latent_"))

    names = list(metrics.keys())

    def fmt(v):
        if v is None:
            return "─"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def print_section(title, keys):
        if not keys:
            return
        print(f"\n  {title}")
        print(f"  {'─' * 70}")
        for k in keys:
            row = [k]
            for n in names:
                row.append(fmt(metrics[n].get(k, "─")))
            # Skip identical rows (no difference across conditions)
            vals = [row[i] for i in range(1, len(row))]
            if len(set(vals)) == 1:
                continue
            print(f"  {row[0]:35s}  " + "  ".join(f"{v:>10s}" for v in vals))
        if not any(True for _ in keys):
            print("  (no differences)")

    print_section("φ Trajectory Metrics", phi_keys)
    print_section("Generation Latent Metrics", latent_keys)


def save_results(results: dict, metrics: dict, out_dir: Path, args) -> None:
    """Save φ tensors, latents, and metrics to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in results:
        cond_dir = out_dir / name
        cond_dir.mkdir(exist_ok=True)
        torch.save(results[name]["phi"], cond_dir / "phi.pt")
        torch.save(results[name]["latents"], cond_dir / "latents.pt")
        if results[name]["z_r"] is not None:
            torch.save(results[name]["z_r"], cond_dir / "z_r.pt")

    # Save metrics as JSON
    metrics_serializable = {}
    for cond, m in metrics.items():
        metrics_serializable[cond] = {k: v for k, v in m.items() if not isinstance(v, (np.ndarray, torch.Tensor))}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_serializable, indent=2))

    # Save args
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n  Results saved to {out_dir}")


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_model_quick(device="cpu"):
    """
    Quick diagnostic mode: create a minimal model with random weights.
    Useful for testing the experiment harness without loading the real model.
    """
    print("  [QUICK MODE] Creating minimal random-weight model for testing...")
    from acestep.phase_memory import PhaseMemory

    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.config = type("Config", (), {
                "audio_acoustic_hidden_dim": 64,
                "text_hidden_dim": 1024,
            })()

            # Minimal decoder that works with PhaseMemory argument pattern
            class MiniDecoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.phase_memory = PhaseMemory(dim=512, mem_dim=128)
                    self.proj = torch.nn.Linear(64, 512)
                    self.out = torch.nn.Linear(512, 64)
                    self.dtype = torch.float32

                def forward(self, hidden_states, timestep, timestep_r,
                            attention_mask, encoder_hidden_states,
                            encoder_attention_mask, context_latents,
                            use_cache=True, past_key_values=None,
                            beat_phase=None):
                    h = self.proj(hidden_states)
                    h, _ = self.phase_memory(h, timestep, beat_phase=beat_phase)
                    out = self.out(h)
                    if use_cache:
                        return (out, None)
                    return (out,)

            self.decoder = MiniDecoder()

            def prepare_condition(self, **kwargs):
                B = kwargs["hidden_states"].shape[0]
                device = kwargs["hidden_states"].device
                dtype = kwargs["hidden_states"].dtype
                return (
                    torch.randn(B, 50, 1024, device=device, dtype=dtype),
                    torch.ones(B, 50, device=device, dtype=dtype),
                    kwargs["hidden_states"],
                )
            self.prepare_condition = prepare_condition.__get__(self)

            def prepare_noise(self, context_latents, seed):
                torch.manual_seed(seed or 42)
                return torch.randn_like(context_latents)
            self.prepare_noise = prepare_noise.__get__(self)

    model = MiniModel().to(device)
    pm_module = model.decoder.phase_memory
    print(f"  Model on {device}, PhaseMemory mem_dim={pm_module.mem_dim}")
    return model, pm_module


def load_model_real(model_root: str, checkpoint_dir: str = None, device="cuda"):
    """
    Load the real ACE-Step model and PhaseMemory weights.

    Uses AceStepHandler to load the DiT model, then loads PhaseMemory
    weights from the checkpoint directory.
    """
    print(f"  Loading model from {model_root}...")
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=model_root,
        config_path="acestep-v15-sft",
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(f"Model loading failed: {status}")

    model = handler.model
    model.eval()

    # Find PhaseMemory module
    pm_module = None
    for mod in model.modules():
        if hasattr(mod, "intervention"):
            pm_module = mod
            break

    if pm_module is None:
        raise RuntimeError("No PhaseMemory module found in model")

    # Load PhaseMemory weights if checkpoint directory provided
    if checkpoint_dir:
        from acestep.training.phase_memory_checkpoint import load_phase_memory_weights
        ckpt_dir = Path(checkpoint_dir)
        if ckpt_dir.exists():
            # Find the latest checkpoint (most recent epoch subdirectory)
            checkpoints = sorted(ckpt_dir.glob("epoch_*"),
                                 key=lambda p: int(p.name.split("_")[1]))
            if checkpoints:
                latest = checkpoints[-1]
                print(f"  Loading PhaseMemory weights from {latest}")
                load_phase_memory_weights(model, str(latest))
            else:
                print(f"  No epoch checkpoints found in {ckpt_dir}, loading from root")
                load_phase_memory_weights(model, str(ckpt_dir))
        else:
            print(f"  Checkpoint dir {ckpt_dir} not found, using random PhaseMemory weights")
    else:
        print(f"  No checkpoint specified, using random PhaseMemory weights")

    print(f"  Model loaded: {type(model).__name__}, PhaseMemory mem_dim={pm_module.mem_dim}")
    return model, pm_module


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="φ Intervention Test")
    parser.add_argument("--model-root", default="/root/autodl-tmp/Ace-Step1.5",
                        help="Path to model root directory")
    parser.add_argument("--checkpoint-dir", default="/root/autodl-tmp/lyrics_checkpoints/checkpoints",
                        help="Path to PhaseMemory checkpoint directory")
    parser.add_argument("--device", default="auto",
                        help="Device (auto / cuda / cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--perturb-scale", type=float, default=0.15,
                        help="Phase perturbation noise scale (radians)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--seq-len", type=int, default=750,
                        help="Sequence length (frames, 25Hz → 30s)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with random weights (CPU-compatible)")
    parser.add_argument("--output-dir", default="output/phi_intervention_test",
                        help="Output directory for results")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    if args.quick:
        model, pm_module = load_model_quick(device=args.device)
    else:
        model, pm_module = load_model_real(args.model_root, args.checkpoint_dir, device=args.device)

    # ── Run experiment ──
    print(f"\n[2/4] Running intervention test (steps={args.num_steps}, seq_len={args.seq_len}, "
          f"perturb_scale={args.perturb_scale})...")
    results = run_experiment(
        model=model,
        pm_module=pm_module,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        seed=args.seed,
        perturb_scale=args.perturb_scale,
        verbose=True,
    )

    # ── Compute metrics ──
    print(f"\n[3/4] Computing metrics...")
    metrics = compute_all_metrics(results)

    # ── Print comparison ──
    print(f"\n[4/4] Results")
    print("=" * 72)
    print_comparison_table(metrics)

    # ── Save ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, metrics, out_dir, args)

    # ── Summary interpretation ──
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("""
  Natural baseline → expected φ: monotonic increase, high short-range coherence,
                      moderate long-range decay.

  Perturbed φ → if φ changes affect structure:
      ↓ monotonicity, ↓ coherence (especially long-range),
      ↑ frame repetition (model confused about position)

  Frozen φ → if temporal accumulation matters:
      ↓ structure entropy (collapse toward repetitive patterns),
      ↓ boundary changes (fewer segment transitions),
      ↑ latent autocorrelation (more self-similar = repetitive)
""")

    print(f"\nResults saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
