#!/usr/bin/env python3
"""
Baseline Hidden Dynamics Probe for ACE-Step DiT.

Hooks into a specified decoder layer and captures hidden states at each
diffusion step, then analyzes the latent trajectory via PCA, velocity,
curvature, frequency structure, and effective rank.

Usage:
    python scripts/baseline_hidden_probe.py [--layer 12] [--prompt "..."] [--steps 50]

This is NOT a PhaseMemory experiment. Purpose: determine whether baseline
DiT hidden trajectories already exhibit smooth latent evolution, turning
points, oscillatory structure, and low-dimensional temporal dynamics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

# Now safe to import handler and inference modules
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Baseline Hidden Dynamics Probe")
parser.add_argument("--layer", type=int, default=12, help="Decoder layer index to hook (default: 12)")
parser.add_argument("--prompt", type=str, default="pop, female vocal, piano, emotional, catchy melody",
                    help="Text prompt for music generation")
parser.add_argument("--lyrics", type=str, default="",
                    help="Lyrics for generation (empty = instrumental)")
parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps (default: 50)")
parser.add_argument("--guidance", type=float, default=7.0, help="CFG guidance scale (default: 7.0)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--duration", type=int, default=30, help="Target duration in seconds (default: 30)")
parser.add_argument("--output_dir", type=str, default="output/baseline_hidden_probe",
                    help="Output directory (default: output/baseline_hidden_probe)")
parser.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices for comparison, e.g. '6,12,18,24'")
parser.add_argument("--no-lm", action="store_true", default=True,
                    help="Skip 5Hz LM (faster, default: True)")
parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parse layer comparison list
layer_comparison = None
if args.layers is not None:
    layer_comparison = [int(x) for x in args.layers.split(",")]


# ---------------------------------------------------------------------------
# Hook: collect hidden states at each diffusion step
# ---------------------------------------------------------------------------
class HiddenStateCollector:
    """Forward hook that collects hidden states at each diffusion step."""

    def __init__(self):
        self.states = []  # list of [1, D] tensors

    def __call__(self, module, input, output):
        # output is (hidden_states, pm_kl_loss)
        hs = output[0]  # [B, T, D]
        # If CFG doubled batch, take conditional half
        if hs.shape[0] > 1:
            hs = hs[:1]
        # Mean over sequence dim → [1, D]
        x_t = hs.mean(dim=1)
        self.states.append(x_t.detach().cpu())

    def get_trajectory(self):
        """Return numpy array X of shape [num_steps, D]."""
        if len(self.states) == 0:
            return np.array([])
        return torch.cat(self.states, dim=0).float().numpy()

    def reset(self):
        self.states = []


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def analyze_pca(X, output_dir, tag=""):
    """PCA trajectory analysis."""
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(X_pca)))
    for i in range(len(X_pca) - 1):
        plt.plot(X_pca[i:i+2, 0], X_pca[i:i+2, 1], color=colors[i], alpha=0.7, linewidth=1.5)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(X_pca)), cmap="viridis", s=30, zorder=5)
    for i in range(len(X_pca)):
        plt.annotate(str(i), (X_pca[i, 0], X_pca[i, 1]), fontsize=6, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")

    sc = plt.scatter(X_pca[0:1, 0], X_pca[0:1, 1], c=[0], cmap="viridis", s=0)  # invisible proxy
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.title(f"Latent Trajectory PCA (Layer {tag or args.layer})")
    plt.colorbar(sc, label="Diffusion Step")
    plt.tight_layout()
    plt.savefig(output_dir / f"trajectory_pca{tag}.png", dpi=150)
    plt.close()

    np.save(output_dir / f"trajectory_pca{tag}.npy", X_pca)

    return {"pca_explained_variance": pca.explained_variance_ratio_.tolist()}


def analyze_velocity(X, output_dir, tag=""):
    """Velocity: L2 norm of consecutive differences."""
    v = np.linalg.norm(np.diff(X, axis=0), axis=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(v, color="coral", linewidth=1.5)
    plt.xlabel("Diffusion Step")
    plt.ylabel("Velocity (L2)")
    plt.title(f"Hidden State Velocity (Layer {tag or args.layer})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"velocity{tag}.png", dpi=150)
    plt.close()

    np.save(output_dir / f"velocity{tag}.npy", v)

    peak_indices = np.argsort(v)[-5:][::-1]
    return {
        "velocity_mean": float(np.mean(v)),
        "velocity_std": float(np.std(v)),
        "velocity_max": float(np.max(v)),
        "velocity_peak_steps": [int(i) for i in peak_indices],
    }


def analyze_curvature(X, output_dir, tag=""):
    """Curvature: 1 - cosθ between consecutive direction changes."""
    d1 = X[1:-1] - X[:-2]
    d2 = X[2:] - X[1:-1]

    dot = np.sum(d1 * d2, axis=1)
    norm = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-8
    kappa = 1.0 - (dot / norm)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(kappa, color="mediumseagreen", linewidth=1.5)
    plt.xlabel("Diffusion Step")
    plt.ylabel("Curvature (1 - cosθ)")
    plt.title(f"Hidden State Curvature (Layer {tag or args.layer})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"curvature{tag}.png", dpi=150)
    plt.close()

    np.save(output_dir / f"curvature{tag}.npy", kappa)

    peak_indices = np.argsort(kappa)[-5:][::-1]
    return {
        "curvature_mean": float(np.mean(kappa)),
        "curvature_std": float(np.std(kappa)),
        "curvature_peak_steps": [int(i) for i in peak_indices],
    }


def analyze_fft(X, output_dir, tag=""):
    """Frequency analysis via FFT on PC1 trajectory."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).squeeze(-1)

    n = len(pc1)
    fft_vals = np.fft.rfft(pc1)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n)

    # Skip DC
    power_ac = power[1:]
    freqs_ac = freqs[1:]

    top_idx = np.argsort(power_ac)[-5:][::-1]
    total_power = np.sum(power_ac)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.stem(freqs_ac, power_ac, basefmt=" ", markerfmt=".")
    plt.xlabel("Frequency (cycles/step)")
    plt.ylabel("Power")
    plt.title(f"FFT Power Spectrum of PC1 (Layer {tag or args.layer})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"fft_power{tag}.png", dpi=150)
    plt.close()

    np.save(output_dir / f"fft_power{tag}.npy", power)

    return {
        "dominant_fft_bins": freqs_ac[top_idx].tolist(),
        "dominant_fft_power_ratios": (power_ac[top_idx] / total_power).tolist(),
    }


def analyze_effective_rank(X):
    """Effective rank from singular value entropy."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] > 1:
        C = Xc.T @ Xc / (Xc.shape[0] - 1)
        s = np.linalg.svd(C, compute_uv=False)
    else:
        s = np.array([1.0])

    p = s / (s.sum() + 1e-10)
    H = -np.sum(p * np.log(p + 1e-10))
    return {"effective_rank": float(np.exp(H)), "hidden_dim": X.shape[1]}


def run_analyses(X, output_dir, tag=""):
    """Run all analyses on a single trajectory."""
    print(f"  Running analyses (shape: {X.shape})...")
    report = {}
    report.update(analyze_pca(X, output_dir, tag=tag))
    report.update(analyze_velocity(X, output_dir, tag=tag))
    report.update(analyze_curvature(X, output_dir, tag=tag))
    report.update(analyze_fft(X, output_dir, tag=tag))
    report.update(analyze_effective_rank(X))
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Baseline Hidden Dynamics Probe for ACE-Step DiT")
    print("=" * 70)

    layers_to_analyze = [args.layer]
    if layer_comparison is not None:
        layers_to_analyze = list(set(layers_to_analyze + layer_comparison))
        layers_to_analyze.sort()
    print(f"Layers to analyze: {layers_to_analyze}")

    # ========== 1. Initialize handler ==========
    print("\n[1/4] Initializing DiT handler...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler() if not args.no_lm else None

    dit_status, dit_success = dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not dit_success:
        print(f"DiT initialization failed: {dit_status}")
        sys.exit(1)

    model = dit_handler.model
    print(f"  Model: {type(model).__name__}, hidden_size={model.config.hidden_size}, layers={model.config.num_hidden_layers}")

    # Disable PhaseMemory on all layers (clean baseline — no custom memory)
    for layer_mod in model.decoder.layers:
        if getattr(layer_mod, "use_phase_memory", False):
            layer_mod.use_phase_memory = False
    print("  PhaseMemory disabled on all layers (clean baseline)")

    # ========== 2. Register hooks ==========
    print("\n[2/4] Registering forward hooks...")
    collectors = {}
    handles = []
    for layer_idx in layers_to_analyze:
        if hasattr(model.decoder, "layers") and layer_idx < len(model.decoder.layers):
            collector = HiddenStateCollector()
            handle = model.decoder.layers[layer_idx].register_forward_hook(collector)
            collectors[layer_idx] = collector
            handles.append(handle)
            print(f"  Hook on decoder.layers[{layer_idx}]")
        else:
            print(f"  WARNING: decoder.layers[{layer_idx}] not found")

    # ========== 3. Initialize 5Hz LM (optional) ==========
    if llm_handler is not None:
        print("\n[3/4] Initializing 5Hz LM...")
        llm_success = llm_handler.initialize(
            checkpoint_dir=str(MODEL_ROOT),
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="pt",
            device=args.device,
        )
        if not llm_success:
            print("  WARNING: 5Hz LM init failed, falling back to no-LM")
            llm_handler = None
        else:
            print("  5Hz LM initialized")

    # ========== 4. Generate ==========
    print(f"\n[4/4] Generating music ({args.steps} steps)...")
    print(f"  Prompt: {args.prompt[:80]}")

    params = GenerationParams(
        task_type="text2music",
        caption=args.prompt,
        lyrics=args.lyrics if args.lyrics else "[Instrumental]",
        instrumental=not bool(args.lyrics),
        duration=args.duration,
        inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        thinking=not args.no_lm,
        use_cot_caption=False,
        use_cot_metas=False,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3",
        use_random_seed=False,
    )

    try:
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=str(OUTPUT_DIR),
        )
        if result.success:
            print("  Generation succeeded")
        else:
            print(f"  Generation status: {result.status_message}")
    except Exception as e:
        print(f"  Generation exception (will use partial data): {e}")
        import traceback
        traceback.print_exc()

    # ========== 5. Analyze ==========
    print("\n" + "=" * 70)
    print("Analyzing hidden state trajectories...")
    print("=" * 70)

    all_reports = {}
    for layer_idx in layers_to_analyze:
        collector = collectors.get(layer_idx)
        if collector is None or len(collector.states) == 0:
            print(f"\n  Layer {layer_idx}: no states collected")
            continue

        X = collector.get_trajectory()
        print(f"\n  Layer {layer_idx}: {X.shape[0]} states, dim={X.shape[1]}")

        tag = f"_layer{layer_idx}"
        report = run_analyses(X, OUTPUT_DIR, tag=tag)
        all_reports[str(layer_idx)] = report

        print(f"    Effective rank: {report['effective_rank']:.2f}")
        print(f"    Velocity: μ={report['velocity_mean']:.4f} σ={report['velocity_std']:.4f}")
        print(f"    Curvature: μ={report['curvature_mean']:.4f} σ={report['curvature_std']:.4f}")
        print(f"    PCA var: {report['pca_explained_variance']}")
        print(f"    Top FFT bins: {report['dominant_fft_bins'][:3]}")

    # Save primary report (default tag, no layer number)
    if str(args.layer) in all_reports:
        primary = all_reports[str(args.layer)]
        with open(OUTPUT_DIR / "report.json", "w") as f:
            json.dump(primary, f, indent=2)
        print(f"\n  Primary report → {OUTPUT_DIR / 'report.json'}")

    # Save comparison report
    if len(all_reports) > 1:
        comp_path = OUTPUT_DIR / "report_comparison.json"
        with open(comp_path, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"  Comparison report → {comp_path}")

        # Print summary table
        print("\n" + "-" * 55)
        print("Layer Comparison")
        print("-" * 55)
        print(f"{'Layer':>6}  {'EffRank':>8}  {'Vel μ':>8}  {'Curv μ':>8}  {'PC1%':>7}")
        for lid, rep in sorted(all_reports.items(), key=lambda x: int(x[0])):
            print(f"{lid:>6}  {rep['effective_rank']:>8.2f}  {rep['velocity_mean']:>8.4f}  "
                  f"{rep['curvature_mean']:>8.4f}  {rep['pca_explained_variance'][0]:>6.1%}")

    # ========== 6. Cleanup ==========
    for h in handles:
        h.remove()

    print("\nDone. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
