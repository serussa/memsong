#!/usr/bin/env python3
"""
Layer-12 hidden-state dynamics analysis using dataset samples.

Picks samples from musicdata (with captions + lyrics), generates via handler,
hooks layer-12, saves audio + metrics, then waits for quality labels.
"""

import argparse
import json
import os
import re
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
import soundfile as sf


parser = argparse.ArgumentParser()
parser.add_argument("--model-root", default="/root/autodl-tmp/Ace-Step1.5")
parser.add_argument("--config", default="acestep-v15-sft")
parser.add_argument("--output-dir", default="/root/ACE-Step-1.5/output/layer12_dynamics_analysis")
parser.add_argument("--n-samples", type=int, default=6)
parser.add_argument("--infer-steps", type=int, default=25)
parser.add_argument("--device", default="cpu")
parser.add_argument("--layer-idx", type=int, default=12)
parser.add_argument("--seed-offset", type=int, default=100)
parser.add_argument("--duration", type=float, default=15.0)
parser.add_argument("--labels", type=str, default=None)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
audio_dir = output_dir / "audio"
audio_dir.mkdir(exist_ok=True)

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
from acestep.handler import AceStepHandler

DATA_DIR = Path("/root/autodl-tmp/musicdata/audios")
FEATURE_NAMES = [
    "v_min", "t_vmin", "rank_mid", "rank_var",
    "trajectory_length", "early_divergence_slope",
]


def parse_caption(caption: str):
    """Extract bpm and key from dataset caption format."""
    bpm = None
    key = ""
    bpm_m = re.search(r'(\d+)\s*bpm', caption, re.IGNORECASE)
    if bpm_m:
        bpm = int(bpm_m.group(1))
    key_m = re.search(r'([A-G][#b]?\s+(major|minor))', caption, re.IGNORECASE)
    if key_m:
        key = key_m.group(1)
    return bpm, key


def load_dataset_samples(n: int):
    """Load n random samples from musicdata that have captions + lyrics."""
    all_samples = []
    for f in os.listdir(str(DATA_DIR)):
        if f.endswith(".mp3"):
            base = f.replace(".mp3", "")
            cap_file = DATA_DIR / f"{base}.caption.txt"
            lyr_file = DATA_DIR / f"{base}.lyrics.txt"
            if cap_file.exists() and lyr_file.exists():
                caption = cap_file.read_text().strip()
                lyrics = lyr_file.read_text().strip()
                if len(lyrics) > 20:
                    all_samples.append((base, caption, lyrics))
    random.seed(42)
    random.shuffle(all_samples)
    return all_samples[:n]


def compute_effective_rank(X: np.ndarray) -> int:
    if X.shape[0] < 2 or X.shape[1] < 2:
        return 1
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumsum, 0.90) + 1)


def compute_dynamics_metrics(h_t: torch.Tensor):
    T, B, S, D = h_t.shape
    h_np = h_t.cpu().numpy().reshape(T, -1, D)
    h_flat_mean = h_t.mean(dim=(1, 2))
    diffs = h_t[1:] - h_t[:-1]
    velocities = torch.norm(diffs, dim=-1).mean(dim=(1, 2))
    v_min = float(velocities.min())
    t_vmin = float(velocities.argmin().item() / max(T - 1, 1))
    mid_start = int(0.3 * T)
    mid_end = max(int(0.5 * T), mid_start + 2)
    h_mid = h_np[mid_start:mid_end].reshape(-1, D)
    rank_mid = compute_effective_rank(h_mid)
    window = max(3, T // 5)
    ranks = []
    for start in range(0, T - window + 1):
        seg = h_np[start:start + window].reshape(-1, D)
        if seg.shape[0] >= 2 and np.var(seg) > 1e-12:
            ranks.append(compute_effective_rank(seg))
    rank_var = float(np.var(ranks)) if len(ranks) > 1 else 0.0
    trajectory_length = float(velocities.sum())
    h_0 = h_flat_mean[0:1]
    divergences = torch.norm(h_flat_mean - h_0, dim=-1)
    early_end = int(0.3 * T)
    if early_end >= 2:
        t_norm = np.arange(early_end) / T
        slope = np.polyfit(t_norm, divergences[:early_end].cpu().numpy(), 1)[0]
        early_divergence_slope = float(slope)
    else:
        early_divergence_slope = 0.0
    return {
        "v_min": v_min, "t_vmin": t_vmin,
        "rank_mid": int(rank_mid), "rank_var": rank_var,
        "trajectory_length": trajectory_length,
        "early_divergence_slope": early_divergence_slope,
        "feature_vector": [
            v_min, t_vmin, int(rank_mid), rank_var,
            trajectory_length, early_divergence_slope,
        ],
    }


def run_analysis(features_arr, labels, feature_names):
    good_mask = np.array(labels) == 0
    bad_mask = np.array(labels) == 1
    good_idx = np.where(good_mask)[0]
    bad_idx = np.where(bad_mask)[0]
    print(f"\n  GOOD: {len(good_idx)}  BAD: {len(bad_idx)}")
    if len(good_idx) < 1 or len(bad_idx) < 1:
        return {"conclusion": "insufficient data"}
    good_feats = features_arr[good_idx]
    bad_feats = features_arr[bad_idx]
    print(f"\n  A. Feature Distribution: GOOD vs BAD")
    print(f"  {'-'*88}")
    print(f"  {'Feature':<25} {'GOOD mean':<12} {'BAD mean':<12} {'diff':<12} {'GOOD var':<12} {'BAD var':<12}")
    print(f"  {'-'*88}")
    diffs = []
    for j, name in enumerate(feature_names):
        gm = float(good_feats[:, j].mean())
        gv = float(good_feats[:, j].var())
        bm = float(bad_feats[:, j].mean())
        bv = float(bad_feats[:, j].var())
        d = gm - bm
        diffs.append(abs(d))
        print(f"  {name:<25} {gm:<12.4f} {bm:<12.4f} {d:<12.4f} {gv:<12.4f} {bv:<12.4f}")
    print(f"\n  B. Feature Separability")
    ranked = sorted(zip(feature_names, diffs), key=lambda x: -x[1])
    for rank, (name, d) in enumerate(ranked, 1):
        print(f"     {rank}. {name:<25} |diff|={d:.4f}")
    print(f"\n  C. Early divergence:")
    ed_idx = feature_names.index("early_divergence_slope")
    ed_gm = float(good_feats[:, ed_idx].mean())
    ed_bm = float(bad_feats[:, ed_idx].mean())
    ed_diff = abs(ed_gm - ed_bm)
    print(f"     |diff|={ed_diff:.4f} (good={ed_gm:.4f} bad={ed_bm:.4f})")
    print(f"     -> {'separates' if ed_diff > 0.1 * max(abs(ed_gm), abs(ed_bm), 1e-8) else 'does NOT separate'}")
    print(f"\n  D. Conclusion")
    n_sig = sum(1 for j in range(len(feature_names))
                if diffs[j] > 0.10 * max(np.abs(features_arr).mean(axis=0)[j], 1e-8))
    if n_sig >= 2:
        conclusion = "layer-12 dynamics predictive of failure"
    else:
        conclusion = "layer-12 dynamics not predictive of failure"
    print(f"     Significant diffs: {n_sig}/{len(feature_names)}")
    print(f"     -> {conclusion}")
    return {
        "feature_names": feature_names,
        "good_mean": good_feats.mean(axis=0).tolist(),
        "bad_mean": bad_feats.mean(axis=0).tolist(),
        "good_var": good_feats.var(axis=0).tolist(),
        "bad_var": bad_feats.var(axis=0).tolist(),
        "diff_mean": (good_feats.mean(axis=0) - bad_feats.mean(axis=0)).tolist(),
        "feature_ranking": [(n, float(d)) for n, d in ranked],
        "conclusion": conclusion,
    }


def main():
    print("=" * 70)
    print("  Layer-12 Dynamics Analysis (Dataset Samples with Lyrics)")
    print("=" * 70)

    data_path = output_dir / "layer12_data.pt"
    label_path = output_dir / "labels.json"

    if args.labels:
        with open(args.labels) as f:
            labels_dict = json.load(f)
        saved = torch.load(data_path, weights_only=False)
        all_features = saved["features"]
        all_metadata = saved["metadata"]
        audio_paths = saved.get("audio_paths", [])
        features_arr = np.array(all_features, dtype=float)
        n = len(all_features)
        labels = [int(labels_dict[str(i)]) for i in range(n)]
        print(f"\n  Labels: {labels}")
        analysis = run_analysis(features_arr, labels, FEATURE_NAMES)
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n  Analysis: {output_dir / 'analysis.json'}")
        print(f"\n  CONCLUSION: {analysis['conclusion']}")
        return

    # Load dataset samples
    samples = load_dataset_samples(args.n_samples)

    print(f"\n[1/4] Loading model ...")
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=args.model_root, config_path=args.config,
        device=args.device, use_flash_attention=False,
        compile_model=False, offload_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"Init failed: {status}")

    model = handler.model
    device = handler.device
    sample_rate = getattr(handler, 'sample_rate', 48000)

    if handler.vae is None:
        print("  ERROR: VAE not loaded.")
        sys.exit(1)

    all_features = []
    all_metadata = []
    audio_paths = []
    all_h_t = []

    print(f"\n[2/4] Generating {len(samples)} samples from dataset ...")

    for idx, (base, caption, lyrics) in enumerate(samples):
        seed = args.seed_offset + idx
        bpm, key = parse_caption(caption)

        print(f"\n  --- Sample {idx+1}/{len(samples)} ---")
        print(f"  Caption: {caption[:80]}...")
        print(f"  Lyrics: {lyrics[:60].replace(chr(10), ' ')}...")
        print(f"  BPM: {bpm}, Key: {key}, Seed: {seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Hook layer 12
        h_history = []
        target_layer = model.decoder.layers[args.layer_idx]
        def make_recorder(history):
            def _record(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                history.append(h.detach().cpu())
            return _record
        handle = target_layer.register_forward_hook(make_recorder(h_history))

        # Generate music with full params (including lyrics, bpm, key)
        t0 = time.time()
        result = handler.generate_music(
            captions=caption,
            lyrics=lyrics,
            bpm=bpm,
            key_scale=key,
            inference_steps=args.infer_steps,
            seed=seed,
            use_random_seed=False,
            guidance_scale=1.0,
            audio_duration=args.duration,
            infer_method="ode",
            batch_size=1,
        )
        elapsed = time.time() - t0
        handle.remove()

        # Extract h_t from denoising steps
        T_exp = args.infer_steps
        h_denoise = h_history[-T_exp:] if len(h_history) >= T_exp else h_history
        h_t = torch.stack(h_denoise, dim=0)
        print(f"  h_t: {tuple(h_t.shape)}  ({elapsed:.0f}s)")

        # Save audio
        success = result.get("success", False)
        audios = result.get("audios", [])
        if success and audios:
            wav_tensor = audios[0]["tensor"]
            sr = audios[0]["sample_rate"]
            wav_np = wav_tensor[0].cpu().numpy() if wav_tensor.dim() == 2 else wav_tensor.cpu().numpy()
            audio_path = audio_dir / f"sample_{idx}.wav"
            sf.write(str(audio_path), wav_np, int(sr))
            audio_paths.append(str(audio_path))
            print(f"  Audio: {audio_path.name} ({len(wav_np)/sr:.1f}s)")
        else:
            err = result.get("error", "unknown")
            print(f"  Generation FAILED: {err[:100]}")
            audio_paths.append(None)

        # Metrics
        if h_t.shape[0] >= 3:
            metrics = compute_dynamics_metrics(h_t)
            all_features.append(metrics["feature_vector"])
            print(f"  v_min={metrics['v_min']:.2f}  t_vmin={metrics['t_vmin']:.3f}  "
                  f"rank_mid={metrics['rank_mid']}  rank_var={metrics['rank_var']:.2f}  "
                  f"traj_len={metrics['trajectory_length']:.0f}  "
                  f"early_div={metrics['early_divergence_slope']:.0f}")
        else:
            all_features.append([0.0] * 6)

        all_metadata.append({
            "sample_idx": idx, "caption": caption, "lyrics": lyrics[:100],
            "seed": seed, "bpm": bpm, "key": key, "file": base,
        })
        all_h_t.append(h_t)

    # Save
    torch.save({
        "h_t": all_h_t, "features": all_features,
        "feature_names": FEATURE_NAMES, "metadata": all_metadata,
        "audio_paths": audio_paths,
    }, data_path)

    features_arr = np.array(all_features, dtype=float)
    n = len(all_features)

    results = {
        "feature_names": FEATURE_NAMES,
        "features": [list(f) for f in all_features],
        "metadata": all_metadata,
        "audio_paths": [str(p) if p else None for p in audio_paths],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print table
    print("\n" + "=" * 70)
    print("  FEATURE TABLE + AUDIO")
    print("=" * 70)
    print(f"  {'Samp':<6} {'v_min':<10} {'t_vmin':<8} {'rank_mid':<9} "
          f"{'rank_var':<9} {'traj_len':<10} {'early_div':<11} audio")
    for i in range(n):
        fv = features_arr[i]
        ap = Path(audio_paths[i]).name if audio_paths[i] else "N/A"
        print(f"  {i:<6} {fv[0]:<10.2f} {fv[1]:<8.3f} {int(fv[2]):<9} "
              f"{fv[3]:<9.2f} {fv[4]:<10.0f} {fv[5]:<11.0f} {ap}")
    print(f"\n  All audio: {audio_dir}/\n")

    if label_path.exists():
        with open(label_path) as f:
            labels_dict = json.load(f)
        labels = [int(labels_dict.get(str(i), -1)) for i in range(n)]
        missing = [i for i, l in enumerate(labels) if l == -1]
        if not missing:
            analysis = run_analysis(features_arr, labels, FEATURE_NAMES)
            with open(output_dir / "analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\n  CONCLUSION: {analysis['conclusion']}")
            return

    print(f"\n  == READY FOR LABELLING ==")
    print(f"  Audio: {audio_dir}/")
    for i in range(n):
        print(f"    {i}: {all_metadata[i]['caption'][:60]}... -> {Path(audio_paths[i]).name}")
    print("\n  Reply with JSON: {\"0\": 0, \"1\": 1, ...}")


if __name__ == "__main__":
    main()
