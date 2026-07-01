#!/usr/bin/env python3
"""
EXP A: Step Rescaling / Effective Dt Normalization Test

Compares hidden-state velocity dynamics between:
- Condition A (baseline):  40-step normal diffusion
- Condition B (rescaled):  Same 40-step trajectory, velocity computed every 2 steps
  (effective dt = 2/40 = 1/20 per velocity measurement)

Both conditions use handler.generate_music() with identical inputs.
Only the measurement resolution differs.

Hypothesis:
- If collapse is INTRINSIC:  mid_mean stays low regardless of measurement resolution
- If collapse is DISCRETIZATION:  effective 20-step velocity shows recovery
"""

import json, os, sys, time, re, random
from pathlib import Path
import numpy as np
import torch

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
from acestep.handler import AceStepHandler

DATA_DIR = Path("/root/autodl-tmp/musicdata/audios")
OUT = Path("/root/ACE-Step-1.5/output/step_rescaling_exp")
OUT.mkdir(parents=True, exist_ok=True)

def log(x): print(x, flush=True)

def get_sample():
    base = "125f6d8d53ee079f7f3f7e6753509b587924b048_965"
    cap = (DATA_DIR / f"{base}.caption.txt").read_text().strip()
    lyr = (DATA_DIR / f"{base}.lyrics.txt").read_text().strip()
    return base, cap, lyr

def parse_caption(caption):
    bpm, key = None, ""
    m = re.search(r'(\d+)\s*bpm', caption, re.I)
    if m: bpm = int(m.group(1))
    m = re.search(r'([A-G][#b]?\s+(major|minor))', caption, re.I)
    if m: key = m.group(1)
    return bpm, key

def hook_layer(layer_mod, storage):
    def _hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        storage.append(h.detach().cpu().float())
    return layer_mod.register_forward_hook(_hook)

def velocity_curve(h_t):
    """Standard consecutive-step velocity: v_t = ||h_{t+1} - h_t||"""
    h_p = h_t.mean(dim=(1, 2))
    d = h_p[1:] - h_p[:-1]
    return torch.norm(d, dim=-1).numpy()

def velocity_curve_skip(h_t, k=2):
    """
    Step-skipped velocity: v_t = ||h_{t+k} - h_t|| / k
    Measures effective dynamics at k-step resolution.
    """
    h_p = h_t.mean(dim=(1, 2))
    T = h_p.shape[0]
    # Aligned: for each t where t+k < T, compute velocity
    # This gives us (T - k) velocity measurements at effective dt = k * original_dt
    d = h_p[k:] - h_p[:-k]
    # Normalize by k to get per-step velocity equivalent
    return (torch.norm(d, dim=-1) / k).numpy()

def collapse_stats(v, T_orig):
    v_peak = float(v[0]) if len(v) > 0 else 0
    v_min = float(v.min()) if len(v) > 0 else 0
    t_vmin = float(v.argmin()) / max(T_orig - 1, 1) if len(v) > 0 else 0
    mid = slice(max(0, int(0.3*len(v))), int(0.6*len(v)))
    mid_mean = float(v[mid].mean()) if mid.stop > mid.start and len(v) > 0 else 0
    collapse_t = None
    for i in range(len(v)):
        if v[i] < 0.5 * v_peak:
            collapse_t = i / max(T_orig - 1, 1)
            break
    return {"v_peak": round(v_peak,1), "v_min": round(v_min,1),
            "t_vmin": round(t_vmin,3), "mid_mean": round(mid_mean,1),
            "collapse_t": collapse_t,
            "v_late_avg": round(float(v[-3:].mean()),1) if len(v) >= 3 else 0,
            "curve": [round(float(x),1) for x in v]}


def run_40step_gen(handler, caption, lyrics, bpm, key, seed):
    """Run 40-step generate_music, return per-layer h_t."""
    model = handler.model
    storage = {}
    handles = []
    layers = [8, 12, 16]
    for lidx in layers:
        storage[lidx] = []
        handles.append(hook_layer(model.decoder.layers[lidx], storage[lidx]))

    t0 = time.time()
    r = handler.generate_music(
        captions=caption, lyrics=lyrics, bpm=bpm, key_scale=key,
        inference_steps=40, seed=seed, use_random_seed=False,
        guidance_scale=1.0, audio_duration=10.0, infer_method="ode",
        batch_size=1,
    )
    elapsed = time.time() - t0
    for h in handles: h.remove()

    if not r.get("success"):
        log(f"  FAILED: {str(r.get('error','unknown'))[:80]}")
        return None

    # Extract h_t for each layer (last 40 recordings = denoising loop)
    result = {}
    for lidx in layers:
        hist = storage[lidx]
        if not hist: continue
        h_t = torch.stack(hist[-40:], dim=0)  # [40, 1, S, D]
        result[lidx] = h_t
    return result


def main():
    log("=" * 65)
    log("EXP A: Step Rescaling / Effective Dt Normalization Test")
    log("=" * 65)

    log("\nLoading model ...")
    handler = AceStepHandler()
    st, ok = handler.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
        device="cpu", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    if not ok: raise RuntimeError(f"Init failed: {st}")
    log("Model loaded.\n")

    base, caption, lyrics = get_sample()
    bpm, key = parse_caption(caption)
    log(f"Sample: {caption}")

    results = {}
    seeds = [0, 1, 42]
    LAYERS = [8, 12, 16]

    for seed in seeds:
        log(f"\n{'='*65}")
        log(f"Seed {seed}: 40-step generation")
        log(f"{'='*65}")

        h_t_dict = run_40step_gen(handler, caption, lyrics, bpm, key, seed)
        if h_t_dict is None: continue

        # Condition A: consecutive-step velocity (40 -> 39 measurements)
        log(f"\n  Condition A: consecutive-step velocity (40 steps, full resolution)")
        stats_a = {}
        for lidx in LAYERS:
            if lidx not in h_t_dict: continue
            h_t = h_t_dict[lidx]
            v = velocity_curve(h_t)  # 39 values
            stats_a[lidx] = collapse_stats(v, 40)
            log(f"    L{lidx:2d}: peak={stats_a[lidx]['v_peak']:.0f} "
                f"t_vmin={stats_a[lidx]['t_vmin']:.3f} mid={stats_a[lidx]['mid_mean']:.0f} "
                f"collapse={stats_a[lidx]['collapse_t']}")
        results[f"seed{seed}_A"] = stats_a

        # Condition B: step-skipped velocity (40 -> 20 measurements, dt doubled)
        log(f"\n  Condition B: skip-2 velocity (effective 20 steps, dt = 2/40)")
        stats_b = {}
        for lidx in LAYERS:
            if lidx not in h_t_dict: continue
            h_t = h_t_dict[lidx]
            v = velocity_curve_skip(h_t, k=2)  # 38 values, per-step normalized
            stats_b[lidx] = collapse_stats(v, 40)
            log(f"    L{lidx:2d}: peak={stats_b[lidx]['v_peak']:.0f} "
                f"t_vmin={stats_b[lidx]['t_vmin']:.3f} mid={stats_b[lidx]['mid_mean']:.0f} "
                f"collapse={stats_b[lidx]['collapse_t']}")
        results[f"seed{seed}_B"] = stats_b

        # Additional: 20-step reference (what does actual 20-step trajectory look like?)
        # We can also check: for h_t from 40 steps, if we subsample every 2nd, velocity at 2x
        log(f"\n  Condition C (extra): subsample every 2nd state, then consecutive velocity")
        stats_c = {}
        for lidx in LAYERS:
            if lidx not in h_t_dict: continue
            h_t = h_t_dict[lidx]  # [40, ...]
            h_t_sub = h_t[::2]    # [20, ...]
            v = velocity_curve(h_t_sub)  # 19 values
            stats_c[lidx] = collapse_stats(v, 20)
            log(f"    L{lidx:2d}: peak={stats_c[lidx]['v_peak']:.0f} "
                f"t_vmin={stats_c[lidx]['t_vmin']:.3f} mid={stats_c[lidx]['mid_mean']:.0f} "
                f"collapse={stats_c[lidx]['collapse_t']}")
        results[f"seed{seed}_C"] = stats_c

    # Save
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved: {OUT / 'results.json'}")

    # ── Comparison table ────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("COMPARISON TABLE: A (consecutive 40) vs B (skip-2) vs C (subsample 20)")
    log("=" * 65)
    log(f"{'Seed':<6} {'Cond':<6} {'Layer':<6} {'v_peak':<8} {'t_vmin':<8} {'mid_mean':<8} {'collapse_t':<10} {'late_avg':<8}")
    log(f"{'':-<6} {'':-<6} {'':-<6} {'':-<8} {'':-<8} {'':-<8} {'':-<10} {'':-<8}")

    for seed in seeds:
        for cond in ["A", "B", "C"]:
            key = f"seed{seed}_{cond}"
            if key not in results: continue
            for lidx in sorted(results[key].keys()):
                s = results[key][lidx]
                ct = "None" if s['collapse_t'] is None else f"{s['collapse_t']:.3f}"
                log(f"{seed:<6} {cond:<6} L{lidx:<4} {s['v_peak']:<8.0f} {s['t_vmin']:<8.3f} "
                    f"{s['mid_mean']:<8.1f} {ct:<10} {s['v_late_avg']:<8.0f}")

    # ── Delta analysis (B vs A, C vs A) ────────────────────────────────────────
    log("\n" + "=" * 65)
    log("DELTA vs Condition A (baseline consecutive 40-step)")
    log("=" * 65)
    log(f"{'Seed':<6} {'Cond':<6} {'Layer':<6} {'Δmid_mean':<10} {'Δcollapse_t':<12} {'Δt_vmin':<10}")
    log(f"{'':-<6} {'':-<6} {'':-<6} {'':-<10} {'':-<12} {'':-<10}")

    for seed in seeds:
        for cond in ["B", "C"]:
            ak = f"seed{seed}_A"
            ck = f"seed{seed}_{cond}"
            if ak not in results or ck not in results: continue
            for lidx in sorted(results[ak].keys()):
                if lidx not in results[ck]: continue
                sa = results[ak][lidx]
                sc = results[ck][lidx]
                d_mid = sc['mid_mean'] - sa['mid_mean']
                d_tvm = sc['t_vmin'] - sa['t_vmin']
                ca = sa['collapse_t'] or 1.0
                cc = sc['collapse_t'] or 1.0
                d_col = cc - ca
                log(f"{seed:<6} {cond:<6} L{lidx:<4} {d_mid:<+10.1f} {d_col:<+12.3f} {d_tvm:<+10.3f}")

    # ── Key question ────────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("KEY QUESTION: Does step-skipping (B) or subsampling (C) remove collapse?")
    log("=" * 65)

    # Check: in Condition B, does the collapse_t disappear or shift?
    b_collapse_free = all(
        results.get(f"seed{s}_B", {}).get(str(l), {}).get("collapse_t") is None
        for s in seeds for l in LAYERS
        if f"seed{s}_B" in results and str(l) in results[f"seed{s}_B"]
    )

    # Check: in Condition B vs A, does mid_mean increase?
    deltas_mid = []
    for s in seeds:
        ak = f"seed{s}_A"
        bk = f"seed{s}_B"
        if ak not in results or bk not in results: continue
        for l in sorted(results[ak].keys()):
            if l not in results[bk]: continue
            deltas_mid.append(results[bk][l]['mid_mean'] - results[ak][l]['mid_mean'])

    avg_delta_mid = np.mean(deltas_mid) if deltas_mid else 0
    pct_recovery = sum(1 for d in deltas_mid if d > 0) / max(len(deltas_mid), 1) * 100

    log(f"\n  B vs A: avg Δmid_mean = {avg_delta_mid:.1f}  "
        f"(positive = skip-2 reduces collapse)")
    log(f"  B collapse-free: {b_collapse_free}")
    log(f"  % of layer-seeds where mid_mean INCREASED in B: {pct_recovery:.0f}%")

    log(f"\n  Raw curves for plotting (seed=0):")
    log(f"  cond,layer,step,v")
    for cond in ["A", "B"]:
        key = f"seed0_{cond}"
        if key not in results: continue
        for lidx in sorted(results[key].keys()):
            s = results[key][lidx]
            curve = s.get("curve", [])
            for si, sv in enumerate(curve):
                log(f"  PLOT,{cond},L{lidx},{si},{sv}")

    # ── FINAL DECISION ─────────────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("FINAL DECISION")
    log("=" * 65)

    # Heuristic:
    # If skip-2 velocity (B) has systematically HIGHER mid_mean than consecutive (A)
    #   → the collapse is an artifact of step-by-step measurement noise
    # If skip-2 velocity (B) has SIMILAR or LOWER mid_mean than consecutive (A)
    #   → the collapse reflects genuine trajectory stagnation

    if avg_delta_mid > 15:
        decision = "DISCRETIZATION ARTIFACT (step size driven)"
        evidence = (f"Step-skipping raises mid-phase velocity by {avg_delta_mid:.0f} points "
                    f"on average. When measured at effective dt=2/40, the trajectory shows "
                    f"substantially more mid-phase dynamics than consecutive-step measurement suggests.")
    elif avg_delta_mid < -15:
        decision = "INTRINSIC LATENT DYNAMICS"
        evidence = (f"Step-skipping does NOT raise mid-phase velocity (Δ={avg_delta_mid:.0f}). "
                    f"The stagnation persists even at coarser measurement resolution, "
                    f"confirming it is a genuine trajectory feature.")
    else:
        decision = "INCONCLUSIVE"
        evidence = (f"Δmid_mean = {avg_delta_mid:.1f} is within noise range. "
                    f"Cannot confidently distinguish discretization from intrinsic dynamics.")

    log(f"\n  Decision: {decision}")
    log(f"  Evidence: {evidence}")
    log(f"\n  Raw data: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
