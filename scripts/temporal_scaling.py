#!/usr/bin/env python3
"""Temporal scaling experiment: velocity minimum across T=30 and T=50, 4 prompts × 3 seeds."""

import sys, os, warnings, shutil, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from scipy import signal

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output/temporal_scaling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = {
    "ballad_m": "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
    "rock": "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
    "edm": "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
    "pop_f": "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
}
SHORT_LABELS = {"ballad_m": "Ballad", "rock": "Rock", "edm": "EDM", "pop_f": "Vocal Pop"}

SEEDS = [42, 123, 999]
STEPS = [30, 50]

TRAJ_MAIN = Path("output/dynamics_experiment/trajectories")
TRAJ_TEMP = Path("output/ts_stability/temporal_trajs")
TRAJ_CACHE = OUTPUT_DIR / "trajectories"
TRAJ_CACHE.mkdir(exist_ok=True)

# ===================================================================
# STEP 1: populate cache — copy existing T=30 from main experiment
# ===================================================================
print("=" * 60)
print("Temporal Scaling: Velocity Minimum across T=30, T=50")
print("=" * 60)

print("\n[1/4] Populating trajectory cache...")
for pname in PROMPTS:
    for seed in SEEDS:
        # T=30: from dynamics_experiment (step 30 default)
        src = TRAJ_MAIN / f"traj_{pname}_s{seed}.npy"
        dst = TRAJ_CACHE / f"traj_{pname}_T30_s{seed}.npy"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Copied T30 {pname} s{seed}")

        # T=50: from temporal_trajs if exists
        src = TRAJ_TEMP / f"traj_{pname}_T50_s{seed}.npy"
        dst = TRAJ_CACHE / f"traj_{pname}_T50_s{seed}.npy"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Copied T50 {pname} s{seed}")

# ===================================================================
# STEP 2: generate missing trajectories (rock & pop_f T=50)
# ===================================================================
print("\n[2/4] Generating missing trajectories...")
missing = []
for pname in PROMPTS:
    for seed in SEEDS:
        for n_steps in STEPS:
            dst = TRAJ_CACHE / f"traj_{pname}_T{n_steps}_s{seed}.npy"
            if not dst.exists():
                missing.append((pname, n_steps, seed))

if missing:
    print(f"  Missing {len(missing)} trajectories: {missing}")
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    import torch

    class Collector:
        def __init__(self):
            self.states = []
        def __call__(self, mod, inp, out):
            hs = out[0]
            if hs.shape[0] > 1: hs = hs[:1]
            self.states.append(hs.mean(dim=1).detach().cpu())
        def get_traj(self):
            if not self.states: return None
            return torch.cat(self.states, dim=0).float().numpy()
        def reset(self):
            self.states = []

    dit_handler = AceStepHandler()
    dit_status, dit_success = dit_handler.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
        device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    if not dit_success:
        print(f"  Model load failed: {dit_status}")
        sys.exit(1)
    for layer_mod in dit_handler.model.decoder.layers:
        if getattr(layer_mod, "use_phase_memory", False):
            layer_mod.use_phase_memory = False
    collector = Collector()
    handle = dit_handler.model.decoder.layers[12].register_forward_hook(collector)

    for pname, n_steps, seed in missing:
        collector.reset()
        try:
            params = GenerationParams(caption=PROMPTS[pname], lyrics="[Instrumental]",
                instrumental=True, duration=30, inference_steps=n_steps,
                guidance_scale=5.0, seed=seed, thinking=False)
            config = GenerationConfig(batch_size=1, audio_format="mp3", use_random_seed=False)
            result = generate_music(dit_handler, None, params, config)
            X = collector.get_traj()
            if X is not None and len(X) >= 4:
                dst = TRAJ_CACHE / f"traj_{pname}_T{n_steps}_s{seed}.npy"
                np.save(dst, X)
                print(f"  Generated {pname} T={n_steps} s{seed} ({len(X)} steps)")
            else:
                print(f"  FAILED {pname} T={n_steps} s{seed}")
        except Exception as e:
            print(f"  ERROR {pname} T={n_steps} s{seed}: {e}")
    handle.remove()
else:
    print("  All 24 trajectories cached, none missing.")

# ===================================================================
# STEP 3: compute velocity minimum for each run
# ===================================================================
print("\n[3/4] Computing velocity minimum...")
results = {}
for pname in PROMPTS:
    results[pname] = {}
    for n_steps in STEPS:
        results[pname][n_steps] = {}
        for seed in SEEDS:
            fpath = TRAJ_CACHE / f"traj_{pname}_T{n_steps}_s{seed}.npy"
            if not fpath.exists():
                print(f"  MISSING {fpath}")
                continue
            X = np.load(fpath)
            v = norm(np.diff(X, axis=0), axis=1)
            # Smooth
            if len(v) >= 5:
                w = min(5, len(v) - (1 - len(v) % 2))
                v_smooth = signal.savgol_filter(v, w, 2)
            else:
                v_smooth = v
            v_min_step = int(np.argmin(v_smooth))
            results[pname][n_steps][seed] = {
                "t_star": v_min_step,
                "t_star_norm": v_min_step / n_steps,
                "v_min": float(v[v_min_step]),
                "steps": len(X),
            }
            print(f"  {SHORT_LABELS[pname]:>10} T={n_steps} s{seed}: t*={v_min_step:2d} ({v_min_step/n_steps:.3f})")

# ===================================================================
# STEP 4: summary + plots
# ===================================================================
print("\n[4/4] Summary + plots:")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {"ballad_m": "#1f77b4", "rock": "#ff7f0e", "edm": "#2ca02c", "pop_f": "#d62728"}

# Left: t* vs T
ax = axes[0]
for pname in PROMPTS:
    means, stds = [], []
    for ns in [30, 50]:
        vals = [results[pname][ns][s]["t_star"] for s in SEEDS if s in results[pname][ns]]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    ax.errorbar([30, 50], means, yerr=stds, fmt="-o", color=colors[pname],
                label=SHORT_LABELS[pname], linewidth=2, capsize=4)
ax.plot([20, 55], [6.6, 16.5], "--", color="gray", alpha=0.3, label="t*/T=0.33")
ax.set_xlabel("Diffusion Steps (T)")
ax.set_ylabel("t* (velocity minimum)")
ax.set_title("Absolute t* Scaling")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Right: t*/T ratio
ax = axes[1]
for pname in PROMPTS:
    ratios = []
    for ns in [30, 50]:
        vals = [results[pname][ns][s]["t_star_norm"] for s in SEEDS if s in results[pname][ns]]
        ratios.append(np.mean(vals))
    ax.plot([30, 50], ratios, "-o", color=colors[pname], label=SHORT_LABELS[pname], linewidth=2)
    # Print constancy
    constancy = np.std(ratios)
    print(f"  {SHORT_LABELS[pname]:>10}: t*/T constancy σ={constancy:.4f} ({ratios[0]:.3f} → {ratios[1]:.3f})")
ax.axhline(0.33, color="gray", linestyle="--", alpha=0.3, label="0.33 baseline")
ax.set_xlabel("Diffusion Steps (T)")
ax.set_ylabel("t* / T")
ax.set_title("Normalized t* Scaling")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Velocity Minimum as Dynamical Phase Transition", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temporal_scaling.png", dpi=150)
plt.close()
print(f"\n  Plot: {OUTPUT_DIR / 'temporal_scaling.png'}")

# Build report
report = {"config": {"prompts": list(PROMPTS.keys()), "seeds": SEEDS, "steps": STEPS}, "results": results}
with open(OUTPUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"  Report: {OUTPUT_DIR / 'report.json'}")
print("=" * 60)
print("DONE")
