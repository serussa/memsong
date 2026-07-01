#!/usr/bin/env python3
"""
Comprehensive Beat Alignment Diagnosis for ACE-Step 1.5.

Probes ALL decoder layers for hidden dynamics, PhaseMemory internals,
and cross-attention patterns. Designed to identify root causes of:
- 抢拍/落拍 (rushing/dragging beats)
- Lyrics sung early or repeated

Usage:
    ACESTEP_LOCAL_MODEL_CODE=1 python scripts/diagnose_beat_alignment.py \
        [--quick] [--steps 10] [--duration 10]
"""

import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
CHECKPOINT_DIR = Path("/root/autodl-tmp/lyrics_checkpoints/checkpoints")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.training.phase_memory_checkpoint import load_phase_memory_weights


# ========================================================================
# Layer-wise Hidden State Collector
# ========================================================================
class MultiLayerCollector:
    """Hooks into multiple decoder layers and collects hidden states at each diffusion step."""

    def __init__(self, model, layer_indices=None):
        self.states = defaultdict(list)           # layer_idx → list of [B, T, D]
        self.attentions = defaultdict(list)       # layer_idx → list of attention weights
        self.pm_internals = defaultdict(list)     # "field" → list of values
        self.hooks = []
        self.layer_indices = layer_indices or list(range(24))

        # Hook every DiT layer
        for idx, layer in enumerate(model.decoder.layers):
            if idx not in self.layer_indices:
                continue
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

        # Hook PhaseMemory specifically for internals
        for name, mod in model.named_modules():
            if 'phase_memory' in name.lower() and isinstance(mod, torch.nn.Module):
                pm_hook = mod.register_forward_hook(self._make_pm_hook(name))
                self.hooks.append(pm_hook)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[0] > 1:  # CFG doubled batch → take conditional half
                hs = hs[:1]
            self.states[layer_idx].append(hs.detach().cpu())
        return hook

    def _make_pm_hook(self, name):
        def hook(module, input, output):
            # Collect internal state from PhaseMemory buffers
            internal = {}
            for buf_name in ['z_r', 'z_i', 'traj', 'anchor']:
                buf = getattr(module, buf_name, None)
                if buf is not None:
                    internal[buf_name] = buf.detach().cpu()
            # Collect kl_loss
            if hasattr(module, 'kl_loss') and module.kl_loss is not None:
                internal['kl_loss'] = module.kl_loss.detach().cpu()
            # Collect bias_net parameters at this step
            if hasattr(module, 'bias_net'):
                # We can't easily get intermediate values without another hook
                pass
            self.pm_internals[name].append(internal)
        return hook

    def get_layer_trajectories(self):
        """Return dict: layer_idx → numpy array [num_steps, T, D] (squeezed)"""
        result = {}
        for idx, states in self.states.items():
            if states:
                arr = torch.stack(states, dim=0).float().numpy()  # [steps, B, T, D]
                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = arr.squeeze(1)  # [steps, T, D]
                result[idx] = arr
        return result

    def reset(self):
        self.states.clear()
        self.attentions.clear()
        self.pm_internals.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ========================================================================
# PhaseMemory Internal State Collector (inside forward pass)
# ========================================================================
class PhaseMemoryProbe:
    """Monkey-patches PhaseMemory.forward to record all internal variables."""

    def __init__(self, model):
        self.records = defaultdict(list)
        self._original_forward = None
        self._patched = False
        for name, mod in model.named_modules():
            if isinstance(mod, type(model)) and hasattr(mod, 'decoder'):
                continue
            if 'PhaseMemory' in type(mod).__name__:
                self._patch(mod)

    def _patch(self, pm_module):
        self._patched = True
        original_forward = pm_module.forward

        def patched_forward(h, diffusion_step, beat_phase=None):
            B, T, D = h.shape

            # Complex projection
            zr = pm_module.proj_r(h)
            zi = pm_module.proj_i(h)
            norm = torch.sqrt(zr**2 + zi**2 + 1e-6)
            zr = zr / norm
            zi = zi / norm

            # Init memory
            if pm_module.z_r is None or pm_module.z_r.shape[:2] != (B, T):
                pm_module.z_r = zr.detach()
                pm_module.z_i = zi.detach()
                pm_module.traj = None

            # Entropy gate
            step_ratio = diffusion_step.to(zr.dtype).view(B, 1, 1) / 1000.0
            if step_ratio.shape[1] != T:
                step_ratio = step_ratio.expand(B, T, 1)
            energy = (zr**2 + zi**2).mean(dim=-1, keepdim=True)
            drift = (zr - pm_module.z_r).abs().mean(dim=-1, keepdim=True)
            ctrl = torch.cat([step_ratio, energy, drift], dim=-1)
            ctrl = ctrl.to(pm_module.entropy_net[0].weight.dtype)
            alpha = torch.sigmoid(pm_module.entropy_net(ctrl))

            # Adaptive anchor
            anchor_input = torch.cat([pm_module.z_r, pm_module.z_i], dim=-1)
            anchor_update = pm_module.anchor_net(anchor_input).mean(dim=(0, 1), keepdim=True)
            pm_module.anchor = 0.995 * pm_module.anchor + 0.005 * anchor_update.detach()
            anchor = pm_module.anchor.expand(B, T, -1)

            # Phase dynamics (omega)
            omega_in = torch.cat([h, zr, zi, anchor], dim=-1)
            omega = math.pi * torch.tanh(pm_module.omega(omega_in))

            c = torch.cos(omega)
            s = torch.sin(omega)
            zr_rot = zr * c - zi * s
            zi_rot = zr * s + zi * c

            # Update
            zr_new = zr_rot + alpha * (zr + anchor)
            zi_new = zi_rot + alpha * (zi + anchor)

            # --- RECORD BEFORE BIAS ---
            self.records['step_ratio'].append(step_ratio[0, 0, 0].item())
            self.records['alpha'].append(alpha[0, 0, 0].item())
            self.records['omega'].append(omega[0, 0, 0].item())
            self.records['energy'].append(energy[0, 0, 0].item())
            self.records['drift'].append(drift[0, 0, 0].item())

            # Beat alignment bias
            mu_val, log_var_val, delta_val = None, None, None
            if beat_phase is not None:
                zr_mean = zr_new.mean(dim=-1, keepdim=True)
                zi_mean = zi_new.mean(dim=-1, keepdim=True)
                bp = beat_phase.unsqueeze(-1) if beat_phase.dim() == 2 else beat_phase
                if bp.shape[1] != T:
                    bp = bp.permute(0, 2, 1)
                    bp = F.interpolate(bp, size=T, mode='linear', align_corners=False)
                    bp = bp.permute(0, 2, 1)
                bias_in = torch.cat([zr_mean, zi_mean, step_ratio, bp], dim=-1)
                bias_out = pm_module.bias_net(bias_in)
                mu = bias_out[..., 0:1]
                log_var = bias_out[..., 1:2]
                mu_val = mu[0, 0, 0].item()
                log_var_val = log_var[0, 0, 0].item()

                if pm_module.training:
                    eps = torch.randn_like(mu)
                    raw_delta = mu + torch.exp(0.5 * log_var) * eps
                else:
                    raw_delta = mu
                delta = pm_module.max_delta * torch.tanh(raw_delta)
                delta_val = delta[0, 0, 0].item()

                # --- RECORD BIAS ---
                self.records['bias_mu'].append(mu_val)
                self.records['bias_log_var'].append(log_var_val)
                self.records['bias_delta'].append(delta_val)
                self.records['beat_phase_input'].append(bp[0, 0, 0].item() if bp.shape[-1] >= 1 else 0)

                cos_d = torch.cos(delta)
                sin_d = torch.sin(delta)
                zr_orig, zi_orig = zr_new, zi_new
                zr_new = zr_orig * cos_d - zi_orig * sin_d
                zi_new = zr_orig * sin_d + zi_orig * cos_d

            # Trajectory memory
            traj = torch.cat([zr_new, zi_new], dim=-1)
            if pm_module.traj is None:
                pm_module.traj = traj.detach()
            if pm_module.intervention != "frozen":
                pm_module.traj = 0.9 * pm_module.traj + 0.1 * traj.detach()
            zr_new = zr_new + 0.1 * pm_module.traj[..., :pm_module.mem_dim]
            zi_new = zi_new + 0.1 * pm_module.traj[..., pm_module.mem_dim:]

            # Stabilize
            scale = torch.sqrt(zr_new**2 + zi_new**2 + 1.0)
            zr_new = zr_new / scale
            zi_new = zi_new / scale

            # Persist
            if pm_module.intervention != "frozen":
                pm_module.z_r = zr_new.detach()
                pm_module.z_i = zi_new.detach()

            z = torch.cat([zr_new, zi_new], dim=-1)
            out = h + pm_module.phase_scale * pm_module.out(z)
            return out, None

        pm_module.forward = patched_forward

    def get_results(self):
        return dict(self.records)

    def cleanup(self):
        pass  # forward was replaced, but that's fine for a script


# ========================================================================
# Analysis Functions
# ========================================================================

def compute_trajectory_metrics(trajectory, label=""):
    """Compute metrics from trajectory with shape [steps, ...]."""
    traj = np.asarray(trajectory)
    # Flatten/aggregate down to [steps, D]
    while traj.ndim > 2:
        traj = traj.mean(axis=1)  # keep reducing

    num_steps, D = traj.shape
    if num_steps < 3:
        return {}

    metrics = {}

    # 1. Frame-to-frame cosine similarity (smoothness)
    norms = np.linalg.norm(traj, axis=-1, keepdims=True) + 1e-8
    traj_normed = traj / norms
    cos_sims = np.sum(traj_normed[:-1] * traj_normed[1:], axis=-1)
    metrics['mean_cos_sim'] = float(cos_sims.mean())
    metrics['std_cos_sim'] = float(cos_sims.std())
    metrics['min_cos_sim'] = float(cos_sims.min())
    metrics['max_cos_sim'] = float(cos_sims.max())

    # 2. Step-wise velocity magnitude
    velocities = np.diff(traj, axis=0)
    vel_mags = np.linalg.norm(velocities, axis=-1)
    metrics['mean_velocity'] = float(vel_mags.mean())
    metrics['max_velocity'] = float(vel_mags.max())
    metrics['std_velocity'] = float(vel_mags.std())

    # 3. Acceleration magnitude (2nd derivative)
    if num_steps > 3:
        accels = np.diff(velocities, axis=0)
        metrics['mean_acceleration'] = float(np.mean(np.abs(accels)))
        metrics['max_acceleration'] = float(np.max(np.abs(accels)))

    # 4. PCA effective rank (dimensionality)
    try:
        X = traj - traj.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        var_exp = (s ** 2) / (s ** 2).sum()
        cumvar = np.cumsum(var_exp)
        metrics['pca_eff_rank_95'] = int((cumvar < 0.95).sum()) + 1
        metrics['pca_eff_rank_99'] = int((cumvar < 0.99).sum()) + 1
        metrics['top1_var'] = float(var_exp[0]) if len(var_exp) > 0 else 0
        metrics['top3_var'] = float(var_exp[:3].sum()) if len(var_exp) >= 3 else float(var_exp.sum())
    except np.linalg.LinAlgError:
        pass

    # 5. FFT frequency analysis (detect periodic structure)
    if num_steps > 10:
        fft_vals = np.fft.rfft(traj, axis=0)
        fft_mags = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(num_steps)
        dominant_freq_idx = np.argmax(fft_mags.sum(axis=-1))
        metrics['dominant_freq'] = float(freqs[dominant_freq_idx])
        # Power spectrum entropy (uniform = noise, peaked = periodic)
        power = fft_mags.sum(axis=-1)
        power_norm = power / power.sum()
        power_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10)) / np.log(len(power_norm))
        metrics['power_entropy_norm'] = float(power_entropy)

    # 6. Autocorrelation structure
    autocorrs = []
    for lag in range(1, min(20, num_steps // 2)):
        ac = np.mean(np.sum(traj_normed[:-lag] * traj_normed[lag:], axis=-1))
        autocorrs.append(ac)
    if autocorrs:
        metrics['short_lag_ac'] = float(np.mean(autocorrs[:5]))  # lags 1-5
        metrics['long_lag_ac'] = float(np.mean(autocorrs[5:])) if len(autocorrs) > 5 else float('nan')
        metrics['max_autocorr_lag'] = int(np.argmax(autocorrs)) + 1

    # 7. Change point detection (large jumps)
    jump_threshold = np.percentile(vel_mags, 90) if len(vel_mags) > 5 else np.mean(vel_mags) * 2
    change_points = np.where(vel_mags > jump_threshold)[0]
    metrics['num_change_points'] = int(len(change_points))
    metrics['change_point_density'] = float(len(change_points) / num_steps)

    return metrics


def analyze_cross_attention(model, collector, steps=10):
    """Analyze cross-attention patterns between audio and lyrics."""
    results = {}
    for layer_idx in sorted(collector.attentions.keys()):
        attn_list = collector.attentions[layer_idx]
        if not attn_list:
            continue
        # Average attention weights over diffusion steps
        attn_stacked = torch.stack(attn_list, dim=0)  # [steps, B, heads, T_audio, T_lyric]
        attn_mean = attn_stacked.mean(dim=(0, 2))  # [B, T_audio, T_lyric]
        results[layer_idx] = {
            'mean_attn': attn_mean[0].numpy(),
            'attn_entropy': float(
                -(attn_mean[0] * torch.log(attn_mean[0].clamp(min=1e-10))).sum(dim=-1).mean()
            ),
        }
    return results


def report_metrics(metrics, title="", prefix=""):
    """Pretty-print metrics dict."""
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"  {prefix}{key}: {val:.6f}")
        elif isinstance(val, int):
            lines.append(f"  {prefix}{key}: {val}")
        elif isinstance(val, np.float64):
            lines.append(f"  {prefix}{key}: {val:.6f}")
        elif isinstance(val, np.int64):
            lines.append(f"  {prefix}{key}: {val}")
        else:
            lines.append(f"  {prefix}{key}: {val}")
    print("\n".join(lines))


# ========================================================================
# Main Experiment
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description="Beat Alignment Diagnosis")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 5 steps, 10s duration")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--load-ckpt", action="store_true", help="Load PhaseMemory checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/diagnose_beat")
    args = parser.parse_args()

    infer_steps = args.steps or (5 if args.quick else 20)
    duration = args.duration or (10 if args.quick else 30)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  BEAT ALIGNMENT DIAGNOSIS")
    print(f"  Steps={infer_steps}, Duration={duration}s, Seed={args.seed}")
    print(f"{'='*60}")

    # ---- Init models ----
    print("\n[1] Initializing models...")
    dit_handler = AceStepHandler()
    dit_status, dit_success = dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not dit_success:
        print(f"FAILED: {dit_status}")
        sys.exit(1)
    dit_handler.model.eval()

    # Optionally load PhaseMemory checkpoint
    if args.load_ckpt:
        ckpts = sorted(CHECKPOINT_DIR.glob("epoch_*"),
                       key=lambda p: int(p.name.split("_")[1]))
        if ckpts:
            print(f"Loading PhaseMemory checkpoint: {ckpts[-1].name}")
            load_phase_memory_weights(dit_handler.model, str(ckpts[-1]))

    # ---- Hook probes ----
    print("\n[2] Installing probes...")
    # Hook ALL layers for hidden state collection
    all_layers = list(range(24))
    collector = MultiLayerCollector(dit_handler.model, layer_indices=all_layers)

    # Patch PhaseMemory for internal state recording
    pm_probe = PhaseMemoryProbe(dit_handler.model)

    # ---- Run generation ----
    print(f"\n[3] Generating {duration}s with {infer_steps} steps...")
    params = GenerationParams(
        task_type="text2music",
        caption="pop, female vocal, piano, emotional, catchy melody",
        lyrics="""[Verse 1] 春风轻轻吹过 花开在路的两侧
        我走在人群中 心里有首歌
        那些年少梦想 还在前方闪烁
        一步一步向前 不怕路曲折
        [Chorus] 梦在前方 路在脚下
        就算风雨再大 也不会害怕
        梦在前方 勇敢去闯
        让每一刻都闪亮
        """,
        instrumental=False,
        bpm=90,
        duration=duration,
        inference_steps=infer_steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        thinking=True,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False)

    t0 = time.time()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=LLMHandler(),
        params=params,
        config=config,
        save_dir=str(OUTPUT_DIR),
    )
    gen_time = time.time() - t0
    print(f"Generation completed in {gen_time:.1f}s")

    # ---- Collect probe data ----
    print("\n[4] Analyzing hidden state trajectories...")
    layer_trajs = collector.get_layer_trajectories()
    pm_data = pm_probe.get_results()

    # ---- Layer-wise analysis ----
    all_metrics = {}
    for layer_idx in sorted(layer_trajs.keys()):
        traj = layer_trajs[layer_idx]
        metrics = compute_trajectory_metrics(traj, label=f"Layer {layer_idx}")
        all_metrics[layer_idx] = metrics
        if args.quick and layer_idx not in [0, 12, 23]:
            continue
        print(f"\n  --- Layer {layer_idx} ({traj.shape[0]} steps, {traj.shape[2]} dim) ---")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    # ---- Layer Comparison Summary ----
    print(f"\n{'='*60}")
    print(f"  LAYER COMPARISON SUMMARY")
    print(f"{'='*60}")

    # Track metrics across layers
    metrics_to_track = ['mean_cos_sim', 'mean_velocity', 'pca_eff_rank_95',
                        'power_entropy_norm', 'dominant_freq', 'num_change_points',
                        'short_lag_ac', 'max_autocorr_lag']

    header = f"  {'Layer':>6} | " + " | ".join(f"{m:>18}" for m in metrics_to_track)
    print(header)
    print("  " + "-" * len(header))
    for layer_idx in sorted(all_metrics.keys()):
        m = all_metrics[layer_idx]
        vals = []
        for k in metrics_to_track:
            v = m.get(k, float('nan'))
            if isinstance(v, float):
                vals.append(f"{v:>18.6f}" if not math.isnan(v) else f"{'N/A':>18}")
            else:
                vals.append(f"{v:>18}")
        print(f"  {layer_idx:>6} | " + " | ".join(vals))

    # ---- PhaseMemory Internal Analysis ----
    print(f"\n{'='*60}")
    print(f"  PHASEMEMORY INTERNAL STATE")
    print(f"{'='*60}")

    if pm_data:
        for key, vals in pm_data.items():
            arr = np.array(vals)
            if len(arr) > 0:
                print(f"  {key:20s}: mean={arr.mean():.6f}, std={arr.std():.6f}, "
                      f"min={arr.min():.6f}, max={arr.max():.6f}, "
                      f"start={arr[0]:.6f}, end={arr[-1]:.6f}")
                # Save to file
                np.save(OUTPUT_DIR / f"pm_{key}.npy", arr)
    else:
        print("  (No PhaseMemory data collected)")

    # ---- Save hidden state trajectories ----
    print(f"\n[5] Saving data to {OUTPUT_DIR}...")
    for layer_idx, traj in layer_trajs.items():
        # Save mean over sequence dim for PCA
        traj_mean = traj.mean(axis=1)  # [steps, D]
        np.save(OUTPUT_DIR / f"layer_{layer_idx:02d}_traj.npy", traj_mean)
        # Save full tensor
        np.save(OUTPUT_DIR / f"layer_{layer_idx:02d}_full.npy", traj)

    # Save all metrics as JSON
    metrics_serializable = {}
    for layer_idx, m in all_metrics.items():
        metrics_serializable[str(layer_idx)] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in m.items()
        }
    with open(OUTPUT_DIR / "layer_metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Diagnostics saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
