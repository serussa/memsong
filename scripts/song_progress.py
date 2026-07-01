#!/usr/bin/env python3
"""
Song Progress Experiment — v2 (memory efficient).

Hooks decoder.layers[12] and saves only mean-pooled trajectories
[steps, D] instead of full [steps, T, D]. Also computes region-level
(early/mid/late) features at collection time.

4 songs × 3 seeds × 2 modes (instrumental, lyrics) = 24 runs.
"""

import json, os, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy import signal
import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

OUTPUT_DIR = Path("output/song_progress")
TRAJ_DIR = OUTPUT_DIR / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

# 4 songs with diverse lyric structures
SONGS = [
    {"name": "simple",
     "caption": "ballad, pop, female vocal, piano, strings, melancholic, emotional, 141 bpm, A minor",
     "lyrics_file": "094a2d4ceb1d5a36bf9d1cc17b9173dd73ddd2c3_1407.lyrics.txt"},
    {"name": "medium",
     "caption": "pop, female vocal, piano, synthesizer, drums, melancholic, romantic, 161 bpm, G major",
     "lyrics_file": "3b7a6a0aa174a6e84106078de00c1dba15e9f9ce_1700.lyrics.txt"},
    {"name": "complex",
     "caption": "rap, hip hop, male vocal, drums, bass, synthesizer, energetic, 141 bpm, D major",
     "lyrics_file": "02146eebf9c7af14bfdf7fd4235bc060d648b8ed_984.lyrics.txt"},
    {"name": "repetitive",
     "caption": "pop, female vocal, piano, drums, bass, emotional, 120 bpm, D major",
     "lyrics_file": "3fcb749666691c3cd973f27f4dd3332ffd322d02_999.lyrics.txt"},
]
LYRICS_DIR = Path("/root/autodl-tmp/musicdata/audios")
SEEDS = [42, 123, 999]
STEPS = 50
GUIDANCE = 5.0


class Collector:
    """Collects per-step hidden states. Saves mean-pooled + region-pooled."""

    def __init__(self, n_regions=5):
        self.T = None
        self.means = []        # [steps, D] — full seq mean
        self.regions = None    # [n_regions, steps, D] — region means
        self.n_regions = n_regions

    def __call__(self, mod, inp, out):
        hs = out[0]  # [B, T, D]
        if hs.shape[0] > 1:
            hs = hs[:1]  # take conditional half under CFG
        B, T_, D = hs.shape
        T = int(T_)
        if self.T is None:
            self.T = T
            bounds = [(int(i / self.n_regions * T), int((i + 1) / self.n_regions * T))
                      for i in range(self.n_regions)]
            self.region_bounds = bounds
            self.regions = [[] for _ in range(self.n_regions)]
        bounds = self.region_bounds

        # Full mean
        self.means.append(hs.mean(dim=1).squeeze(0).detach().cpu())  # [D]

        # Region means
        hs_cpu = hs[0].detach().cpu()  # [T, D]
        for ri, (s, e) in enumerate(bounds):
            chunk = hs_cpu[s:e]  # [sub_T, D]
            self.regions[ri].append(chunk.mean(dim=0))  # [D]

    def get(self):
        """Returns dict of trajectories {name: [steps, D]}."""
        if not self.means:
            return None
        result = {"full": torch.stack(self.means, dim=0).float().numpy()}  # [steps, D]
        for ri in range(self.n_regions):
            if self.regions[ri]:
                result[f"region_{ri}"] = torch.stack(self.regions[ri], dim=0).float().numpy()
        return result

    def reset(self):
        self.T = None
        self.means = []
        self.regions = None
        self.region_bounds = None


def compute_metrics(X_dict):
    """Compute dynamics metrics for each trajectory in X_dict."""
    results = {}
    for name, X in X_dict.items():
        if not isinstance(X, np.ndarray):
            continue
        if X.ndim != 2:
            continue
        steps, D = X.shape
        v = norm(np.diff(X, axis=0), axis=1)
        w = min(5, len(v) - (1 - len(v) % 2))
        v_smooth = signal.savgol_filter(v, w, 2) if len(v) >= 5 else v
        v_min_step = int(np.argmin(v_smooth))

        vc = 99
        if steps >= 3:
            d1 = X[1:-1] - X[:-2]
            d2 = X[2:] - X[1:-1]
            dot = np.sum(d1 * d2, axis=1)
            nrm = norm(d1, axis=1) * norm(d2, axis=1) + 1e-10
            cos_sim = np.clip(dot / nrm, -1.0, 1.0)
            curvature = 1.0 - cos_sim
            c_peak = int(np.argmax(curvature)) if len(curvature) > 0 else 0
            vc = abs(v_min_step - c_peak)

        Xc = X - X.mean(axis=0, keepdims=True)
        C = Xc.T @ Xc / (steps - 1)
        s = np.linalg.svd(C, compute_uv=False)
        p = s / (s.sum() + 1e-10)
        H = -np.sum(p * np.log(p + 1e-10))

        results[name] = {
            "v_norm": [float(x) for x in v],
            "v_min_step": int(v_min_step),
            "v_min_norm": float(v_min_step / steps),
            "effective_rank": float(np.exp(H)),
            "vc_alignment": vc,
        }
    return results


def main():
    print("=" * 70)
    print("  SONG PROGRESS EXPERIMENT")
    print("  Impact of lyrics on latent dynamics (layer 12)")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model...")
    dit_handler = AceStepHandler()
    dit_status, dit_success = dit_handler.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
        device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    if not dit_success: print("FAILED"); sys.exit(1)
    for lm in dit_handler.model.decoder.layers:
        if getattr(lm, "use_phase_memory", False): lm.use_phase_memory = False

    collector = Collector(n_regions=5)
    handle = dit_handler.model.decoder.layers[12].register_forward_hook(collector)

    # Generate
    total = len(SONGS) * len(SEEDS) * 2
    done = 0
    all_metrics = {}
    print(f"\n[2/3] Generating {total} runs...")

    for song in SONGS:
        name = song["name"]
        lyrics_path = LYRICS_DIR / song["lyrics_file"]
        lyrics_text = open(lyrics_path).read().strip()

        for seed in SEEDS:
            for mode, lyrics_val in [("instrumental", "[Instrumental]"), ("lyrics", lyrics_text)]:
                cache_file = TRAJ_DIR / f"traj_{name}_{mode}_s{seed}.npy"
                if cache_file.exists():
                    X_dict = np.load(cache_file, allow_pickle=True).item()
                else:
                    collector.reset()
                    try:
                        params = GenerationParams(
                            caption=song["caption"], lyrics=lyrics_val,
                            instrumental=(mode == "instrumental"), duration=30,
                            inference_steps=STEPS, guidance_scale=GUIDANCE, seed=seed,
                            thinking=False, use_cot_caption=False, use_cot_metas=False)
                        config = GenerationConfig(batch_size=1, audio_format="mp3",
                                                   use_random_seed=False)
                        generate_music(dit_handler, None, params, config)
                        X_dict = collector.get()
                        if X_dict is None:
                            print(f"    FAIL {name} {mode} s{seed}")
                            continue
                        np.save(cache_file, X_dict, allow_pickle=True)
                    except Exception as e:
                        print(f"    ERROR {name} {mode} s{seed}: {e}")
                        continue

                metrics = compute_metrics(X_dict)
                all_metrics[f"{name}_{mode}_s{seed}"] = metrics
                done += 1
                f_er = metrics.get("full", {}).get("effective_rank", 0)
                f_t = metrics.get("full", {}).get("v_min_norm", 0)
                region_ers = ", ".join(
                    f"{k[-7:]}={v['effective_rank']:.2f}"
                    for k, v in sorted(metrics.items()) if k.startswith("region"))
                print(f"    {name:>10} {mode:>12} s{seed:>3}: "
                      f"ER={f_er:.2f} t*={f_t:.3f} | {region_ers} [{done}/{total}]")

    handle.remove()

    # ===================== SUMMARY =====================
    print(f"\n[3/3] Summary ({done} runs):")

    print(f"\n  {'Song':>12} {'Mode':>14} {'ER':>6} {'t*/T':>7} {'V-C':>5} "
          f"{'R0_ER':>7} {'R1_ER':>7} {'R2_ER':>7} {'R3_ER':>7} {'R4_ER':>7}")
    print("  " + "-" * 85)

    for song in SONGS:
        name = song["name"]
        for mode in ["instrumental", "lyrics"]:
            ers, ts, vcs = [], [], []
            region_ers = defaultdict(list)
            for seed in SEEDS:
                key = f"{name}_{mode}_s{seed}"
                m = all_metrics.get(key, {}).get("full", {})
                if m:
                    ers.append(m["effective_rank"])
                    ts.append(m["v_min_norm"])
                    vcs.append(m["vc_alignment"])
                for ri in range(5):
                    rk = f"region_{ri}"
                    rm = all_metrics.get(key, {}).get(rk, {})
                    if rm:
                        region_ers[ri].append(rm.get("effective_rank", 0))

            if not ers:
                continue
            re_str = " ".join(f"{np.mean(region_ers[ri]):.2f}" if region_ers[ri] else "   -"
                              for ri in range(5))
            print(f"  {name:>12} {mode:>14} {np.mean(ers):>6.2f} {np.mean(ts):>7.3f} "
                  f"{np.mean(vcs):>5.0f}  {re_str}")

    # Lyrics vs instrumental delta
    print(f"\n  Δ (Lyrics − Instrumental):")
    print(f"  {'Song':>12} {'ΔER':>8} {'Δt*/T':>9}")
    print("  " + "-" * 31)
    for song in SONGS:
        name = song["name"]
        lyr_er = np.mean([all_metrics.get(f"{name}_lyrics_s{s}", {}).get("full", {}).get("effective_rank", 0)
                          for s in SEEDS])
        ins_er = np.mean([all_metrics.get(f"{name}_instrumental_s{s}", {}).get("full", {}).get("effective_rank", 0)
                          for s in SEEDS])
        lyr_t = np.mean([all_metrics.get(f"{name}_lyrics_s{s}", {}).get("full", {}).get("v_min_norm", 0)
                         for s in SEEDS])
        ins_t = np.mean([all_metrics.get(f"{name}_instrumental_s{s}", {}).get("full", {}).get("v_min_norm", 0)
                         for s in SEEDS])
        if lyr_er and ins_er:
            print(f"  {name:>12} {lyr_er - ins_er:>+8.2f} {lyr_t - ins_t:>+9.3f}")

    # ===================== PLOTS =====================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Velocity profiles: lyrics vs instrumental per song
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for si, song in enumerate(SONGS):
        name = song["name"]
        for mi, mode in enumerate(["instrumental", "lyrics"]):
            ax = axes[si, mi]
            for seed in SEEDS:
                key = f"{name}_{mode}_s{seed}"
                m = all_metrics.get(key, {}).get("full", {})
                if m:
                    v = m["v_norm"]
                    ax.plot(v, alpha=0.6, linewidth=0.8)
            ax.set_title(f"{name} — {mode}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Velocity")
            ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "velocity_profiles.png", dpi=150)
    plt.close()
    print(f"\n  [plot] velocity_profiles.png")

    # 2. Region effective rank: lyrics vs instrumental
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(5)
    bw = 0.3
    for mi, mode in enumerate(["instrumental", "lyrics"]):
        for si, song in enumerate(SONGS):
            name = song["name"]
            vals = []
            for ri in range(5):
                key = f"{name}_{mode}_s{SEEDS[0]}"
                rv = all_metrics.get(key, {}).get(f"region_{ri}", {}).get("effective_rank", 0)
                vals.append(rv)
            offset = (si - 1.5) * bw
            ax.bar(x + offset, vals, bw, alpha=0.7, label=f"{name}_{mode}" if si == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels(["Region 0\n(early)", "1", "2", "3", "4\n(late)"])
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank by Sequence Position (first seed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "region_er.png", dpi=150)
    plt.close()
    print(f"  [plot] region_er.png")

    # Save report
    report = {
        "config": {"songs": [s["name"] for s in SONGS], "seeds": SEEDS, "steps": STEPS},
        "per_run": {k: {sk: sv for sk, sv in v.items()}
                    for k, v in all_metrics.items()},
    }
    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {OUTPUT_DIR / 'report.json'}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
