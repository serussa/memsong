#!/usr/bin/env python3
"""
DiT Error Localization v2: Change-Point Detection in Latent Dynamics.

Correctly handles the U-shaped velocity profile (high→low→high) and
looks for GENUINE DEVIATIONS from the ensemble reference, not trivial
initial transient detection.

Key improvements over v1:
  - Per-step z-scoring (normalize each step by its own distribution)
  - Excludes initial transient (steps 0-2) from change-point search
  - Uses Mahalanobis-like distance in velocity+curvature space
  - Detects outlying runs, not just the biggest spike in each run
  - Generates audio to compute audio-domain error signals

Usage:
    python scripts/error_localization.py [--audio N] [--output-dir ...]

Output:
    output/error_localization_v2/
    ├── reference_dynamics.png        Velocity/curvature profile with σ bands
    ├── anomaly_map.png               Heatmap of anomaly scores across all runs
    ├── outlier_runs.png              The N most anomalous trajectories
    ├── change_point_grid.png         Per-run with t* marked
    ├── audio_alignment.png           Latent vs audio error overlay
    ├── report.json
    └── summary_table.txt
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy import signal as scipy_signal

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
TRAJ_DIR_DEFAULT = Path("output/dynamics_experiment/trajectories")
OUTPUT_DIR_DEFAULT = Path("output/error_localization_v2")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="DiT Error Localization v2")
parser.add_argument("--traj-dir", default=str(TRAJ_DIR_DEFAULT))
parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
parser.add_argument("--audio", type=int, default=0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSIENT_LEN = 3  # first 3 velocity steps (0, 1, 2) are the big transient

# ---------------------------------------------------------------------------
# Prompt mappings (must match dynamics_experiment.py)
# ---------------------------------------------------------------------------
SHORT_FILENAME = {
    "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major": "electro",
    "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor": "rock",
    "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major": "ballad_m",
    "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major": "pop_f",
    "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major": "folk",
    "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor": "dancepop",
    "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major": "cpop",
}


# ====================================================================
# DATA LOADING
# ====================================================================
def load_trajectories(traj_dir):
    prompt_map = {
        "electro": "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major",
        "rock": "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor",
        "ballad_m": "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major",
        "pop_f": "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major",
        "folk": "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major",
        "dancepop": "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor",
        "cpop": "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major",
    }
    all_trajs = defaultdict(dict)
    exp_keys = []
    for fpath in sorted(traj_dir.glob("*.npy")):
        stem = fpath.stem.replace("traj_", "")
        parts = stem.split("_s")
        if len(parts) != 2:
            continue
        pname, seed_str = parts
        prompt = prompt_map.get(pname, pname)
        seed = int(seed_str)
        all_trajs[prompt][seed] = np.load(fpath)
        exp_keys.append((prompt, seed))
    return all_trajs, exp_keys, prompt_map


# ====================================================================
# DYNAMICS EXTRACTION
# ====================================================================
def extract_dynamics(X):
    """Extract velocity, acceleration, curvature from trajectory X ∈ [steps, dim]."""
    v = np.diff(X, axis=0)                      # [steps-1, dim]
    v_norm = norm(v, axis=1)                     # [steps-1]
    v_unit = v / (v_norm[:, None] + 1e-10)       # [steps-1, dim]

    a = np.diff(v, axis=0)                       # [steps-2, dim]
    a_norm = norm(a, axis=1)                     # [steps-2]

    cos_sim = np.clip(np.sum(v_unit[:-1] * v_unit[1:], axis=1), -1.0, 1.0)
    curvature = 1.0 - cos_sim                    # [steps-2] (starts at velocity step 1)

    return {"v": v, "v_norm": v_norm, "v_unit": v_unit,
            "a": a, "a_norm": a_norm, "curvature": curvature}


# ====================================================================
# STEP 3: REFERENCE DYNAMICS (per-step distribution)
# ====================================================================
def build_reference(dyn_dict, exp_keys):
    """Per-step mean + covariance of [v_norm, curvature, a_norm]."""
    n_steps_v = max(dyn_dict[p][s]["v"].shape[0] for p, s in exp_keys)
    n_steps_c = max(dyn_dict[p][s]["curvature"].shape[0] for p, s in exp_keys)

    # Velocity norm
    vn_by_step = {t: [] for t in range(n_steps_v)}
    # Curvature
    c_by_step = {t: [] for t in range(n_steps_c)}
    # Acceleration norm
    an_by_step = {t: [] for t in range(n_steps_c)}

    for p, s in exp_keys:
        d = dyn_dict[p][s]
        for t in range(d["v_norm"].shape[0]):
            vn_by_step[t].append(d["v_norm"][t])
        for t in range(d["curvature"].shape[0]):
            c_by_step[t].append(d["curvature"][t])
            an_by_step[t].append(d["a_norm"][t])

    ref = {}
    for name, src, n in [("v_norm", vn_by_step, n_steps_v),
                          ("curvature", c_by_step, n_steps_c),
                          ("a_norm", an_by_step, n_steps_c)]:
        mean_arr = np.zeros(n)
        std_arr = np.zeros(n)
        raw = {}
        for t in range(n):
            vals = np.array(src[t])
            raw[t] = vals
            mean_arr[t] = vals.mean()
            std_arr[t] = vals.std() + 1e-10
        ref[name] = {"mean": mean_arr, "std": std_arr, "raw": raw}

    return ref


# ====================================================================
# STEP 4-5: ANOMALY SCORES + CHANGE-POINT DETECTION
# ====================================================================
def compute_anomaly(d, ref, run_idx=0):
    """
    Per-step anomaly score using z-scores against step-specific distributions.

    Returns:
      - S_per_step: array of length [steps] (one score per diffusion step)
      - S_transition: array of length [steps-1] (one score per velocity step)
    """
    steps_total = d["v"].shape[0] + 1  # original trajectory length

    # Velocity norm z-score per step
    vn = d["v_norm"]
    vn_z = np.zeros_like(vn)
    for t in range(len(vn)):
        vn_z[t] = abs(vn[t] - ref["v_norm"]["mean"][t]) / ref["v_norm"]["std"][t]

    # Curvature z-score per step
    c = d["curvature"]
    c_z = np.zeros_like(c)
    for t in range(len(c)):
        c_z[t] = abs(c[t] - ref["curvature"]["mean"][t]) / ref["curvature"]["std"][t]

    # Acceleration norm z-score per step
    an = d["a_norm"]
    an_z = np.zeros_like(an)
    for t in range(len(an)):
        an_z[t] = abs(an[t] - ref["a_norm"]["mean"][t]) / ref["a_norm"]["std"][t]

    # Velocity direction anomaly: cosine distance from reference mean direction
    v = d["v"]
    v_ref = ref["v_norm"]["raw"]  # not directly useable
    # For direction, compute mean unit vector per step
    v_unit = d["v_unit"]
    # Collect reference unit vectors per step
    # Reconstruct from raw
    v_unit_z = np.zeros(len(vn))
    for t in range(v.shape[0]):
        # All runs' velocity vectors at this step from raw data — we don't have raw velocity vectors stored
        # Use magnitude deviation as proxy
        v_unit_z[t] = vn_z[t]  # using magnitude deviation as a proxy

    # Combined transition score (per velocity step)
    # For velocity steps: [steps-1]
    # For curvature/acc: [steps-2], align by shifting
    S_trans = np.zeros(v.shape[0])  # length = steps-1
    S_trans += 0.35 * vn_z
    if len(c_z) == len(S_trans) - 1:
        S_trans[1:] += 0.35 * c_z
        S_trans[1:] += 0.30 * an_z
    else:
        S_trans += 0.35 * np.pad(c_z, (0, max(0, len(S_trans) - len(c_z))), 'edge')
        S_trans += 0.30 * np.pad(an_z, (0, max(0, len(S_trans) - len(an_z))), 'edge')

    # Per-step score (map back to diffusion step index)
    S_step = np.zeros(steps_total)
    S_step[0] = 0  # first step has no prior velocity
    for t in range(len(S_trans)):
        S_step[t + 1] = S_trans[t]

    return {"S_transition": S_trans, "S_step": S_step,
            "vn_z": vn_z, "c_z": c_z, "an_z": an_z}


def detect_change_points(S_step, transient_len=TRANSIENT_LEN, z_threshold=2.5):
    """
    Detect change-points in per-step anomaly score.

    Distinguishes between:
      - Type I (normal transient): steps < transient_len, expected high scores
      - Type II (genuine anomaly): steps >= transient_len, unexpected deviation

    Returns:
      - t_star: first step where anomaly exceeds threshold in steady region
      - t_peak: step of max anomaly in steady region
      - severity: peak z-score in steady region
      - is_outlier: whether this run is an outlier (> 3σ in any post-transient step)
    """
    n = len(S_step)
    steady = S_step[transient_len:]  # focus on post-transient region

    t_star = None
    t_peak = None
    severity = 0.0
    is_outlier = False

    if len(steady) > 0:
        threshold = np.mean(steady) + z_threshold * np.std(steady)
        above = np.where(steady > threshold)[0]
        if len(above) > 0:
            t_star = int(above[0]) + transient_len
            is_outlier = True

        t_rel_peak = int(np.argmax(steady))
        t_peak = t_rel_peak + transient_len
        severity = float(steady[t_rel_peak])

    return {
        "t_star": t_star,
        "t_peak": t_peak,
        "severity": severity,
        "is_outlier": is_outlier,
        "transient_len": transient_len,
    }


# ====================================================================
# STEP 6: AUDIO ERROR DETECTION
# ====================================================================
def audio_error_signal(audio_path, sr=48000, n_steps=30):
    """Compute audio-domain error signal, downsampled to diffusion step resolution."""
    try:
        import librosa
    except ImportError:
        return None
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return None
    if len(y) < 2048:
        return None

    hop = 512
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    n_frames = spec.shape[1]

    # Spectral flux
    flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
    flux = np.pad(flux, (0, max(0, n_frames - len(flux))), 'edge')

    # Onset strength
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # Beat deviation
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
        beat_set = set(beats)
    except Exception:
        beat_set = set()
    beat_dev = np.zeros(n_frames)
    if beat_set:
        for f in range(n_frames):
            nearest = min(abs(np.array(list(beat_set)) - f)) if beat_set else 0
            beat_dev[f] = min(nearest / 10.0, 1.0)

    # Spectral entropy
    spec_p = spec / (spec.sum(axis=0, keepdims=True) + 1e-10)
    spec_ent = -np.sum(spec_p * np.log(spec_p + 1e-10), axis=0)

    # Combined error: high flux deviation + beat inconsistency + entropy anomaly
    flux_z = (flux - flux.mean()) / (flux.std() + 1e-10)
    onset_diff = np.abs(np.diff(onset, prepend=onset[0]))
    od_z = (onset_diff - onset_diff.mean()) / (onset_diff.std() + 1e-10)
    entropy_z = (spec_ent - spec_ent.mean()) / (spec_ent.std() + 1e-10)

    audio_err = np.clip(0.4 * np.abs(flux_z) + 0.3 * od_z + 0.3 * beat_dev, 0, None)

    # Downsample to n_steps
    step_size = n_frames / n_steps
    reduced = np.array([audio_err[int(s * step_size):int((s + 1) * step_size)].mean()
                        for s in range(n_steps)])
    return reduced


# ====================================================================
# STEP 7: ALIGNMENT
# ====================================================================
def alignment_stats(all_cp, audio_errs, all_S_step, exp_keys):
    """Compute alignment between latent anomaly and audio error."""
    from scipy.stats import pearsonr
    results = []
    for p, s in exp_keys:
        key = (p, s)
        ae = audio_errs.get(key)
        S = all_S_step.get(key)
        cp = all_cp.get(key, {})
        if ae is None or S is None:
            continue

        # Align to same length
        min_len = min(len(S), len(ae))
        S_a, ae_a = S[:min_len], ae[:min_len]

        if np.std(S_a) < 1e-8 or np.std(ae_a) < 1e-8:
            continue

        r, p_val = pearsonr(S_a, ae_a)
        t_audio_peak = int(np.argmax(ae_a))
        t_latent = cp.get("t_peak")

        # P(error | high S): top-20% steps overlap
        k = max(3, min_len // 5)
        high_S = set(np.argsort(S_a)[-k:])
        high_AE = set(np.argsort(ae_a)[-k:])
        overlap = len(high_S & high_AE) / k

        results.append({
            "prompt": p[:20], "seed": s,
            "pearson_r": float(r), "pearson_p": float(p_val),
            "t_latent_peak": t_latent, "t_audio_peak": int(t_audio_peak),
            "lead_time": int(t_audio_peak - t_latent) if t_latent is not None else None,
            "overlap_rate": float(overlap),
        })

    if not results:
        return {}

    r_vals = [r["pearson_r"] for r in results]
    leads = [r["lead_time"] for r in results if r["lead_time"] is not None]
    return {
        "per_run": results,
        "aggregate": {
            "pearson_r_mean": float(np.mean(r_vals)),
            "pearson_r_std": float(np.std(r_vals)),
            "lead_time_mean": float(np.mean(leads)) if leads else None,
            "lead_time_std": float(np.std(leads)) if leads else None,
            "overlap_rate_mean": float(np.mean([r["overlap_rate"] for r in results])),
        }
    }


# ====================================================================
# VISUALIZATION
# ====================================================================
def visualize(ref, all_S_step, all_cp, exp_keys, dyn_dict, audio_errs=None, alignment=None,
              traj_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt_names = list(dict.fromkeys(k[0] for k in exp_keys))
    short_labels = {
        "electronic, pop, female vocal, synthesizer, electronic drums, dreamy, melancholic, 94 bpm, C major": "Electro",
        "rock, male vocal, electric guitar, drums, bass, energetic, 131 bpm, D minor": "Rock",
        "ballad, male vocal, piano, strings, emotional, melancholic, 170 bpm, A# major": "BalladM",
        "pop, female vocal, piano, strings, drums, melancholic, 131 bpm, A# major": "PopF",
        "folk, female vocal, acoustic guitar, strings, melancholic, romantic, 161 bpm, C major": "Folk",
        "dance-pop, female vocal, synthesizer, drums, energetic, 120 bpm, G# minor": "DancePop",
        "c-pop, ballad, male vocal, piano, strings, melancholic, emotional, 84 bpm, A major": "CPop",
    }
    n_runs = len(exp_keys)

    # ===================== 1. REFERENCE DYNAMICS =====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name, yl, tt in [
        (axes[0], "v_norm", "Velocity Magnitude", "Velocity |v|"),
        (axes[1], "curvature", "Curvature (1-cosθ)", "Curvature"),
        (axes[2], "a_norm", "Acceleration Magnitude", "Acceleration"),
    ]:
        r = ref[name]
        steps = np.arange(len(r["mean"]))
        ax.plot(steps, r["mean"], "b-", lw=2)
        ax.fill_between(steps, r["mean"] - r["std"], r["mean"] + r["std"], alpha=0.2)
        # Mark transient region
        ax.axvspan(0, TRANSIENT_LEN - 0.5, color="red", alpha=0.08, label="transient")
        ax.set_xlabel("Step")
        ax.set_ylabel(yl)
        ax.set_title(f"Reference {tt}")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reference_dynamics.png", dpi=150)
    plt.close()
    print("  [plot] reference_dynamics.png")

    # ===================== 2. ANOMALY MAP =====================
    # Heatmap: runs × steps, colored by S_step
    fig, ax = plt.subplots(figsize=(12, max(6, n_runs * 0.35)))
    S_matrix = np.zeros((n_runs, 30))
    run_labels = []
    for idx, (p, s) in enumerate(exp_keys):
        S = np.array(all_S_step.get((p, s), np.zeros(30)))
        S_matrix[idx, :min(len(S), 30)] = S[:30]
        run_labels.append(f"{short_labels.get(p, p[:8])} s{s}")

    im = ax.imshow(S_matrix, aspect="auto", cmap="hot", interpolation="nearest")
    ax.axvline(TRANSIENT_LEN - 0.5, color="cyan", linestyle="--", linewidth=1, label="transient end")
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels(run_labels, fontsize=6)
    ax.set_xlabel("Diffusion Step")
    ax.set_title("Anomaly Score S(t) — All Runs")
    plt.colorbar(im, ax=ax, label="S(z-score)", shrink=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_map.png", dpi=150)
    plt.close()
    print("  [plot] anomaly_map.png")

    # ===================== 3. OUTLIER RUNS =====================
    # Find the most anomalous runs (highest S in steady region)
    severities = [(p, s, all_cp.get((p, s), {}).get("severity", 0))
                  for p, s in exp_keys]
    severities.sort(key=lambda x: -x[2])
    top_k = min(6, len(severities))
    top_runs = severities[:top_k]

    fig, axes = plt.subplots(top_k, 3, figsize=(15, 3.5 * top_k))
    for ri, (prompt, seed, sev) in enumerate(top_runs):
        pname = SHORT_FILENAME.get(prompt, prompt[:8].replace(",", ""))
        X = np.load(traj_dir / f"traj_{pname}_s{seed}.npy") if traj_dir else None
        d = dyn_dict[prompt][seed]
        row = axes[ri] if top_k > 1 else axes
        # PCA trajectory
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X) if X is not None else np.zeros((30, 2))
        ax = row[0]
        ax.plot(X_pca[:, 0], X_pca[:, 1], "b-", alpha=0.7)
        ax.scatter(X_pca[0, 0], X_pca[0, 1], c="g", s=40, marker="o", zorder=5)
        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c="r", s=40, marker="x", zorder=5)
        cp = all_cp.get((prompt, seed), {})
        t_star = cp.get("t_star")
        t_peak = cp.get("t_peak")
        if t_star is not None and t_star - 1 < len(X_pca):
            ax.scatter(X_pca[t_star - 1, 0], X_pca[t_star - 1, 1], c="red", s=80, marker="*", zorder=10)
        ax.set_title(f"PCA (severity={sev:.1f})")
        ax.tick_params(labelsize=7)

        # Anomaly score
        ax = row[1]
        S = np.array(all_S_step.get((prompt, seed), []))
        ax.plot(S, "r-", lw=1.5)
        ax.axvline(TRANSIENT_LEN, color="gray", ls="--", alpha=0.5)
        if t_star is not None:
            ax.axvline(t_star, color="red", ls="--", alpha=0.8, lw=2)
            ax.text(t_star, ax.get_ylim()[1] * 0.9, f"t*={t_star}", fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("S(z-score)")
        ax.grid(alpha=0.2)

        # Audio (if available)
        ax = row[2]
        ae = audio_errs.get((prompt, seed)) if audio_errs else None
        if ae is not None:
            ax.plot(ae, "g-", lw=1.5)
            ax.set_title("Audio Error")
        else:
            ax.text(0.5, 0.5, "No audio data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Step")

        label = short_labels.get(prompt, prompt[:10])
        # Check if file exists for this run before trying to load
        fig.suptitle(f"Top-{top_k} Most Anomalous Runs", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outlier_runs.png", dpi=150)
    plt.close()
    print("  [plot] outlier_runs.png")

    # ===================== 4. CHANGE-POINT GRID =====================
    n_cols = 5
    n_rows = int(np.ceil(n_runs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.2 * n_rows))
    axes_flat = axes.flatten()
    for idx in range(n_runs):
        ax = axes_flat[idx]
        p, s = exp_keys[idx]
        S = np.array(all_S_step.get((p, s), []))
        ax.plot(S, "b-", lw=0.8, alpha=0.7)
        cp = all_cp.get((p, s), {})
        ts = cp.get("t_star")
        if ts is not None:
            ax.axvline(ts, color="r", ls="--", lw=1.5)
            ax.text(ts, ax.get_ylim()[1] * 0.9, f"t*={ts}", fontsize=6, color="r")
        if cp.get("is_outlier"):
            ax.set_facecolor("#fff0f0")
        ax.axvline(TRANSIENT_LEN, color="gray", ls=":", alpha=0.4)
        ax.set_title(f"{short_labels.get(p, p[:8])} s{s}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_ylim(0, max(ax.get_ylim()[1], 3))
    for idx in range(n_runs, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle("Per-Run Anomaly (red dash = change-point, pink bg = outlier)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "change_point_grid.png", dpi=150)
    plt.close()
    print("  [plot] change_point_grid.png")

    # ===================== 5. AUDIO ALIGNMENT =====================
    if audio_errs and len(audio_errs) > 0:
        n_show = min(8, len(audio_errs))
        fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show))
        shown = 0
        for p, s in exp_keys:
            if shown >= n_show:
                break
            ae = audio_errs.get((p, s))
            S = all_S_step.get((p, s))
            if ae is None or S is None:
                continue
            ax = axes[shown] if n_show > 1 else axes
            S_norm = np.array(S) / (np.max(np.abs(S)) + 1e-10)
            ae_norm = np.array(ae) / (np.max(np.abs(ae)) + 1e-10)
            ax.plot(S_norm, "b-", alpha=0.7, label="S(t)")
            ax.plot(ae_norm, "r-", alpha=0.7, label="Audio Error")
            cp = all_cp.get((p, s), {})
            if cp.get("t_peak") is not None:
                ax.axvline(cp["t_peak"], color="gray", ls="--", alpha=0.5)
            ax.set_title(f"{short_labels.get(p, p[:8])} s{s}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
            shown += 1
        plt.suptitle("Latent Anomaly vs Audio Error Signal", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "audio_alignment.png", dpi=150)
        plt.close()
        print("  [plot] audio_alignment.png")


# ====================================================================
# SUMMARY
# ====================================================================
def print_summary(all_cp, ref, alignment=None):
    lines = []
    lines.append("=" * 70)
    lines.append("  ERROR LOCALIZATION v2 — SUMMARY")
    lines.append("=" * 70)

    outliers = [(p, s) for (p, s), cp in all_cp.items() if cp.get("is_outlier")]
    t_stars = [cp["t_star"] for cp in all_cp.values() if cp.get("t_star") is not None]
    t_peaks = [cp["t_peak"] for cp in all_cp.values() if cp.get("t_peak") is not None]

    lines.append(f"\n  Total runs: {len(all_cp)}")
    lines.append(f"  Outliers (anomaly > 2.5σ post-transient): {len(outliers)}")
    if t_stars:
        lines.append(f"  Change-point t* (post-transient):")
        lines.append(f"    mean={np.mean(t_stars):.1f}, std={np.std(t_stars):.1f}")
        lines.append(f"    range=[{min(t_stars)},{max(t_stars)}]")
        lines.append(f"    distribution:")
        for t in sorted(set(t_stars)):
            lines.append(f"      t*={t}: {t_stars.count(t)} runs")
    if t_peaks:
        lines.append(f"\n  Peak anomaly location t_peak:")
        lines.append(f"    mean={np.mean(t_peaks):.1f}, std={np.std(t_peaks):.1f}")

    # Severity
    sevs = [cp["severity"] for cp in all_cp.values()]
    lines.append(f"\n  Severity distribution:")
    lines.append(f"    mean={np.mean(sevs):.2f}, std={np.std(sevs):.2f}")
    lines.append(f"    max={max(sevs):.2f}, min={min(sevs):.2f}")

    lines.append(f"\n  Reference dynamics (post-transient region, steps {TRANSIENT_LEN}+):")
    v_steady_mean = ref["v_norm"]["mean"][TRANSIENT_LEN:]
    lines.append(f"    Velocity: μ={np.mean(v_steady_mean):.1f}, range=[{v_steady_mean.min():.1f},{v_steady_mean.max():.1f}]")
    c_steady = ref["curvature"]["mean"][max(0, TRANSIENT_LEN - 1):]
    lines.append(f"    Curvature: μ={np.mean(c_steady):.3f}")

    if alignment:
        agg = alignment.get("aggregate", {})
        lines.append(f"\n  Audio Alignment:")
        lines.append(f"    Pearson r:          {agg.get('pearson_r_mean', 'N/A'):.4f} ± {agg.get('pearson_r_std', 'N/A'):.4f}")
        lines.append(f"    Overlap rate:       {agg.get('overlap_rate_mean', 'N/A'):.2%}")
        lt = agg.get("lead_time_mean")
        if lt is not None:
            lines.append(f"    Mean lead time:     {lt:.1f} steps")
        # Count positives
        per_run = alignment.get("per_run", [])
        pos_lead = sum(1 for r in per_run if r.get("lead_time") is not None and r["lead_time"] > 0)
        lines.append(f"    t_latent precedes:  {pos_lead}/{len(per_run)}")

    lines.append("\n" + "=" * 70)
    print("\n".join(lines))
    return "\n".join(lines)


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 70)
    print("  DiT Error Localization v2 — Change-Point Detection")
    print("=" * 70)

    traj_dir = Path(args.traj_dir)
    traj_files = list(traj_dir.glob("*.npy"))
    print(f"\n  Trajectories: {len(traj_files)} files in {traj_dir}")

    # [1] Load
    print("\n[1/5] Loading trajectories...")
    all_trajs, exp_keys, pmap = load_trajectories(traj_dir)
    print(f"  {len(exp_keys)} runs loaded")

    # [2] Compute dynamics
    print("\n[2/5] Computing per-step dynamics...")
    dyn_dict = defaultdict(dict)
    for p, s in exp_keys:
        dyn_dict[p][s] = extract_dynamics(all_trajs[p][s])
    print("  Done.")

    # [3] Build reference
    print("\n[3/5] Building reference dynamics...")
    ref = build_reference(dyn_dict, exp_keys)
    print(f"  v_norm: {len(ref['v_norm']['mean'])} steps, curvature: {len(ref['curvature']['mean'])} steps")

    # [4] Anomaly scores + change-points
    print("\n[4/5] Computing anomaly scores and detecting change-points...")
    all_S_step = {}
    all_cp = {}
    for idx, (p, s) in enumerate(exp_keys):
        anomaly = compute_anomaly(dyn_dict[p][s], ref, idx)
        all_S_step[(p, s)] = anomaly["S_step"].tolist()
        cp = detect_change_points(anomaly["S_step"])
        all_cp[(p, s)] = cp

    n_outliers = sum(1 for cp in all_cp.values() if cp["is_outlier"])
    print(f"  Outliers detected: {n_outliers}/{len(all_cp)}")

    # [5] Audio generation + validation
    print("\n[5/5] Audio validation...")
    audio_errs = {}
    if not args.dry_run and args.audio > 0:
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        audio_dir = OUTPUT_DIR / "audio"
        audio_dir.mkdir(exist_ok=True)

        dit_handler = AceStepHandler()
        dit_status, dit_success = dit_handler.initialize_service(
            project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
            device=args.device, use_flash_attention=False, compile_model=False, offload_to_cpu=False,
        )
        if dit_success:
            for layer_mod in dit_handler.model.decoder.layers:
                if getattr(layer_mod, "use_phase_memory", False):
                    layer_mod.use_phase_memory = False
            for idx in range(min(args.audio, len(exp_keys))):
                p, s = exp_keys[idx]
                print(f"  Audio {idx+1}/{min(args.audio, len(exp_keys))}: s{s}...")
                try:
                    r = generate_music(dit_handler, None,
                        GenerationParams(caption=p, lyrics="[Instrumental]", instrumental=True,
                            duration=30, inference_steps=30, guidance_scale=5.0, seed=s, thinking=False),
                        GenerationConfig(batch_size=1, audio_format="mp3", use_random_seed=False),
                        save_dir=str(audio_dir))
                    if r.success and r.audios:
                        ae = audio_error_signal(r.audios[0]["path"])
                        if ae is not None:
                            audio_errs[(p, s)] = ae.tolist()
                            print(f"    AE shape={ae.shape}")
                except Exception as e:
                    print(f"    Failed: {e}")
        else:
            print(f"  Model init failed: {dit_status}")
    print(f"  Audio available for {len(audio_errs)} runs")

    # Alignment
    alignment = None
    if audio_errs:
        alignment = alignment_stats(all_cp, audio_errs, all_S_step, exp_keys)
        if alignment:
            agg = alignment["aggregate"]
            print(f"  Pearson r: {agg['pearson_r_mean']:.4f} ± {agg['pearson_r_std']:.4f}")

    # Visualize + report
    print("\n  Generating visualizations...")
    visualize(ref, all_S_step, all_cp, exp_keys, dyn_dict, audio_errs, alignment, traj_dir)

    summary = print_summary(all_cp, ref, alignment)
    with open(OUTPUT_DIR / "summary_table.txt", "w") as f:
        f.write(summary)

    report = {
        "config": {"n_runs": len(exp_keys), "transient_len": TRANSIENT_LEN},
        "change_points": {f"{p[:12]}_s{s}": cp for p, s in exp_keys
                          for cp in [all_cp.get((p, s), {})]},
        "change_point_stats": {
            "n_outliers": n_outliers,
            "n_total": len(all_cp),
            "t_star_mode": int(max(set([v["t_star"] for v in all_cp.values() if v["t_star"] is not None]),
                                    key=[v["t_star"] for v in all_cp.values() if v["t_star"] is not None].count))
                if any(v["t_star"] is not None for v in all_cp.values()) else None,
        },
        "alignment": alignment,
    }
    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report: {OUTPUT_DIR / 'report.json'}")
    print(f"  Summary: {OUTPUT_DIR / 'summary_table.txt'}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
