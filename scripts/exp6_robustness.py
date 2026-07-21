#!/usr/bin/env python3
"""
EXP6: Robustness and Causality Validation (simplified)

Focus on what works: prompt diversity × generate_music.
Check if collapse is universal across varied prompts.

Test: 3 prompts, 3 seeds, 3 layers = 27 data points via handler.generate_music()
"""

import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
from acestep.handler import AceStepHandler

OUT = Path("/root/ACE-Step-1.5/output/exp6_robustness")
OUT.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = OUT / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
import soundfile as sf

def log(x): print(x, flush=True)

PROMPTS = [
    ("simple", "piano ballad, emotional, soft male vocal"),
    ("medium", "rock band, electric guitar, drums, energetic male vocal"),
    ("complex", "EDM, synth, female vocal, fast tempo, layered production"),
]

LAYERS = [8, 12, 16]

def hook_layer(layer_mod, storage):
    def _hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        storage.append(h.detach().cpu().float())
    return layer_mod.register_forward_hook(_hook)

def velocity_curve(h_t):
    h_p = h_t.mean(dim=(1, 2))
    d = h_p[1:] - h_p[:-1]
    return torch.norm(d, dim=-1).numpy()

def collapse_stats(v, T_orig):
    v_peak = float(v[0]) if len(v) > 0 else 0
    mid = slice(max(0, int(0.3*len(v))), int(0.6*len(v)))
    mid_mean = float(v[mid].mean()) if mid.stop > mid.start and len(v) > 0 else 0
    collapse_abs = None
    for i in range(len(v)):
        if v[i] < 0.5 * v_peak:
            collapse_abs = i
            break
    collapse_norm = collapse_abs / max(T_orig - 1, 1) if collapse_abs is not None else None
    return {"v_peak": round(v_peak, 1), "mid_mean": round(mid_mean, 1),
            "collapse_abs": collapse_abs, "collapse_norm": collapse_norm}


def main():
    log("=" * 70)
    log("EXP6: Robustness Validation (Prompt Diversity)")
    log("=" * 70)

    log("\nLoading model ...")
    handler = AceStepHandler()
    st, ok = handler.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
        device="cpu", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    if not ok: raise RuntimeError(f"Init failed: {st}")
    model = handler.model
    sample_rate = getattr(handler, 'sample_rate', 48000)
    log("Model loaded.\n")

    results = {}
    seeds = [0, 1, 42]

    for pname, caption in PROMPTS:
        for seed in seeds:
            key = f"{pname}_s{seed}"
            log(f"[{key}] \"{caption[:40]}...\" seed={seed}")

            # Hook
            storage = {}
            handles = []
            for lidx in LAYERS:
                storage[lidx] = []
                handles.append(hook_layer(model.decoder.layers[lidx], storage[lidx]))

            t0 = time.time()
            r = handler.generate_music(
                captions=caption,
                inference_steps=40, seed=seed, use_random_seed=False,
                guidance_scale=1.0, audio_duration=10.0, infer_method="ode",
                batch_size=1,
            )
            elapsed = time.time() - t0
            for h in handles: h.remove()

            if not r.get("success"):
                log(f"  FAILED: {str(r.get('error',''))[:80]}")
                continue

            # Save audio
            audios = r.get("audios", [])
            if audios:
                wav = audios[0]["tensor"]
                sr = audios[0]["sample_rate"]
                wav_np = wav[0].cpu().numpy() if wav.dim() == 2 else wav.cpu().numpy()
                sf.write(str(AUDIO_DIR / f"{pname}_s{seed}.wav"), wav_np, int(sr))

            stats = {}
            for lidx in LAYERS:
                hist = storage[lidx]
                if not hist: continue
                h_t = torch.stack(hist[-40:], dim=0)
                v = velocity_curve(h_t)
                stats[lidx] = collapse_stats(v, 40)

            results[key] = {"prompt": caption, "seed": seed, "stats": stats}
            s_str = "  ".join(
                f"L{l}: pk={stats[l]['v_peak']:.0f} mid={stats[l]['mid_mean']:.0f} "
                f"ca={stats[l]['collapse_abs']}"
                for l in LAYERS if l in stats
            )
            log(f"  ({elapsed:.0f}s) {s_str}")

    # Save
    with open(OUT / "raw.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved: {OUT / 'raw.json'}")
    log(f"Audio: {AUDIO_DIR}/")

    # ── Analysis ─────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("ANALYSIS: Collapse universality")
    log(f"{'='*70}")
    log(f"\n{'Prompt':<10} {'Layer':<6} {'mid_mean':<10} {'collapse_abs':<12} {'collapse_rate':<14}")
    log(f"{'':-<10} {'':-<6} {'':-<10} {'':-<12} {'':-<14}")

    for pname, _ in PROMPTS:
        for lidx in LAYERS:
            vals_mid = []
            vals_ca = []
            for seed in seeds:
                key = f"{pname}_s{seed}"
                if key not in results: continue
                if lidx not in results[key]["stats"]: continue
                s = results[key]["stats"][lidx]
                vals_mid.append(s["mid_mean"])
                vals_ca.append(s["collapse_abs"] or 40)
            if vals_mid:
                collapse_rate = sum(1 for v in vals_ca if v < 35) / len(vals_ca) * 100
                log(f"{pname:<10} L{lidx:<4} {np.mean(vals_mid):<10.1f} {np.mean(vals_ca):<12.1f} {collapse_rate:<14.0f}%")

    log(f"\n  Collapse detection (all prompts × seeds × layers):")
    all_ca = []
    for pname, _ in PROMPTS:
        for seed in seeds:
            for lidx in LAYERS:
                key = f"{pname}_s{seed}"
                if key not in results: continue
                if lidx not in results[key]["stats"]: continue
                ca = results[key]["stats"][lidx]["collapse_abs"]
                if ca is not None:
                    all_ca.append(ca)
    log(f"    total datapoints: {len(all_ca)}")
    log(f"    mean collapse_abs: {np.mean(all_ca):.1f}")
    log(f"    collapse_abs std: {np.std(all_ca):.1f}")
    log(f"    collapse in first 5 steps: {sum(1 for c in all_ca if c <= 5)}/{len(all_ca)}")

    # ── Quality labels (manual) ──────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("QUALITY LABELLING")
    log(f"{'='*70}")
    log(f"\n  Audio files:")
    for pname, _ in PROMPTS:
        for seed in seeds:
            fpath = AUDIO_DIR / f"{pname}_s{seed}.wav"
            if fpath.exists():
                log(f"    {fpath}")
    log(f"\n  Listen and assign 0 (good) or 1 (bad) for each.")
    log(f"\n  Provide labels via --labels <file> on re-run.")
    log(f"  Or set labels in metadata.json")

    # Write label request
    with open(OUT / "label_request.txt", "w") as f:
        f.write("Quality labels needed:\n\n")
        for pname, _ in PROMPTS:
            for seed in seeds:
                f.write(f"{pname}_s{seed}:  (0=good, 1=bad)\n")

    # ── Placeholder analysis with default "all bad" ─────────────────────────
    log(f"\n{'='*70}")
    log("FINAL CLASSIFICATION (tentative — no quality labels yet)")
    log(f"{'='*70}")

    prompt_mid = {}
    for pname, _ in PROMPTS:
        vals = []
        for seed in seeds:
            key = f"{pname}_s{seed}"
            if key not in results: continue
            for lidx in LAYERS:
                if lidx not in results[key]["stats"]: continue
                vals.append(results[key]["stats"][lidx]["mid_mean"])
        prompt_mid[pname] = np.mean(vals) if vals else 0

    # Check variation across prompts
    mid_vals = list(prompt_mid.values())
    cv = np.std(mid_vals) / max(np.mean(mid_vals), 1) * 100
    log(f"\n  mid_mean across prompts: {mid_vals}")
    log(f"  coefficient of variation: {cv:.1f}%")
    log(f"  If CV < 20%: collapse is invariant to prompt content")
    log(f"  If CV > 30%: collapse depends on conditioning")

    if cv < 20:
        log(f"\n  → NUMERICAL / SCHEDULER ARTIFACT (prompt-independent)")
    elif cv > 30:
        log(f"\n  → QUALITY-RELATED (varies with conditioning)")
    else:
        log(f"\n  → INCONCLUSIVE (some variation, needs quality labels)")

    log(f"\n  To get final answer, provide quality labels.")
    log(f"  Re-run with: python scripts/exp6_robustness.py --labels <file>")

    # Add curve data for plots
    log(f"\n  PLOT DATA:")
    for pname, _ in PROMPTS:
        key = f"{pname}_s0"
        if key not in results: continue
        for lidx in LAYERS:
            if lidx not in results[key]["stats"]: continue
            s = results[key]["stats"][lidx]
            log(f"  PLOT,{pname},L{lidx},peak={s['v_peak']:.0f},mid={s['mid_mean']:.0f},ca={s['collapse_abs']}")

    log(f"\nDone.")


if __name__ == "__main__":
    main()
