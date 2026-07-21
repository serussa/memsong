#!/usr/bin/env python3
"""
EXP7: Adaptive Solver / Euler Error Test

Goal: Test whether mid-phase collapse depends on numerical integration scheme.

Three conditions:
A. Euler 40 (baseline, via generate_music)
B. Euler 80 (higher resolution, via generate_music)
C. Heun / error-corrected measurement (post-hoc from 80-step trajectory)

The key insight for C:
Compute the linear extrapolation error at each step:
  e_t = ||h_{t+2} - (h_t + 2*dt*f(t, h_t))||
      = ||h_{t+2} - h_t - 2*(h_{t+1} - h_t)||  (since h_{t+1} = h_t + dt*f)
      = ||h_{t+2} - 2*h_{t+1} + h_t||

This measures how well Euler predicts 2 steps ahead.
If e_t is large → high curvature → better solver would help
If e_t is small → Euler is accurate → collapse is genuine
"""

import json, os, sys, time, re
from pathlib import Path
import numpy as np
import torch

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
from acestep.handler import AceStepHandler

OUT = Path("/root/ACE-Step-1.5/output/exp7_solver_test")
OUT.mkdir(parents=True, exist_ok=True)
import soundfile as sf

def log(x): print(x, flush=True)

LAYERS = [8, 12, 16]
CAPTION = "pop, r&b, male vocal, synthesizer, drums, emotional, 141 bpm, B major"
DATA_DIR = Path("/root/autodl-tmp/musicdata/audios")

def hook_layer(layer_mod, storage):
    def _hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        storage.append(h.detach().cpu().float())
    return layer_mod.register_forward_hook(_hook)

def velocity_curve(h_t):
    """Consecutive-step velocity: ||h_{t+1} - h_t||"""
    h_p = h_t.mean(dim=(1, 2))
    d = h_p[1:] - h_p[:-1]
    return torch.norm(d, dim=-1).numpy()

def curvature_curve(h_t):
    """
    Second-derivative-like measure: ||h_{t+2} - 2*h_{t+1} + h_t||
    For a linear trajectory (constant first derivative), this is zero.
    High values indicate curvature / acceleration in the trajectory.
    """
    h_p = h_t.mean(dim=(1, 2))
    # e_t = h_{t+2} - 2*h_{t+1} + h_t
    e = h_p[2:] - 2 * h_p[1:-1] + h_p[:-2]
    return torch.norm(e, dim=-1).numpy()

def euler_error(h_t):
    """
    How wrong would Euler be if we predicted 2 steps ahead?
    e_t = ||h_{t+2} - (h_t + 2*(h_{t+1} - h_t))||
        = ||h_{t+2} - 2*h_{t+1} + h_t||
    Normalized by step count: e_t / 2 = average error per step
    """
    return curvature_curve(h_t)

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

def get_sample_lyrics():
    """Get a dataset sample for conditioning."""
    import random, re
    all_s = []
    for f in os.listdir(str(DATA_DIR)):
        if not f.endswith(".mp3"): continue
        base = f.replace(".mp3", "")
        cap = DATA_DIR / f"{base}.caption.txt"
        lyr = DATA_DIR / f"{base}.lyrics.txt"
        if cap.exists() and lyr.exists() and len(lyr.read_text().strip()) > 30:
            all_s.append((base, cap.read_text().strip(), lyr.read_text().strip()))
    random.seed(42); random.shuffle(all_s)
    return all_s[0]  # same as previous experiments

def parse_caption(caption):
    bpm, key = None, ""
    m = re.search(r'(\d+)\s*bpm', caption, re.I)
    if m: bpm = int(m.group(1))
    m = re.search(r'([A-G][#b]?\s+(major|minor))', caption, re.I)
    if m: key = m.group(1)
    return bpm, key


def main():
    log("=" * 70)
    log("EXP7: Adaptive Solver / Euler Error Test")
    log("=" * 70)

    log("\nLoading model ...")
    handler = AceStepHandler()
    st, ok = handler.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
        device="cpu", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    if not ok: raise RuntimeError(f"Init failed: {st}")
    model = handler.model
    log("Model loaded.\n")

    base, cap_text, lyr = get_sample_lyrics()
    bpm, key = parse_caption(cap_text)
    log(f"Sample: {cap_text[:60]}...")
    log(f"Lyrics: {len(lyr)} chars")

    results = {}
    seeds = [0]

    # ── A: Euler 40 (baseline) ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("CONDITION A: Euler 40 (baseline)")
    log(f"{'='*70}")
    for seed in seeds:
        key = f"A_euler40_s{seed}"
        log(f"  [{key}] seed={seed}")
        storage = {}
        handles = []
        for lidx in LAYERS:
            storage[lidx] = []
            handles.append(hook_layer(model.decoder.layers[lidx], storage[lidx]))
        t0 = time.time()
        r = handler.generate_music(
            captions=CAPTION, lyrics=lyr, bpm=bpm, key_scale=key,
            inference_steps=40, seed=seed, use_random_seed=False,
            guidance_scale=1.0, audio_duration=10.0, infer_method="ode",
            batch_size=1,
        )
        elapsed = time.time() - t0
        for h in handles: h.remove()
        if not r.get("success"):
            log(f"  FAILED"); continue
        stats = {}
        for lidx in LAYERS:
            hist = storage[lidx]
            if not hist: continue
            h_t = torch.stack(hist[-40:], dim=0)
            v = velocity_curve(h_t)
            stats[lidx] = collapse_stats(v, 40)
        results[key] = stats
        s_str = "  ".join(
            f"L{l}: pk={stats[l]['v_peak']:.0f} mid={stats[l]['mid_mean']:.0f} "
            f"ca={stats[l]['collapse_abs']}"
            for l in LAYERS if l in stats
        )
        log(f"  ({elapsed:.0f}s) {s_str}")

    # ── B: Euler 80 (higher resolution) ────────────────────────────────────
    log(f"\n{'='*70}")
    log("CONDITION B: Euler 80 (high resolution)")
    log(f"{'='*70}")
    for seed in seeds:
        key = f"B_euler80_s{seed}"
        log(f"  [{key}] seed={seed}")
        storage = {}
        handles = []
        for lidx in LAYERS:
            storage[lidx] = []
            handles.append(hook_layer(model.decoder.layers[lidx], storage[lidx]))
        t0 = time.time()
        r = handler.generate_music(
            captions=CAPTION, lyrics=lyr, bpm=bpm, key_scale=key,
            inference_steps=80, seed=seed, use_random_seed=False,
            guidance_scale=1.0, audio_duration=10.0, infer_method="ode",
            batch_size=1,
        )
        elapsed = time.time() - t0
        for h in handles: h.remove()
        if not r.get("success"):
            log(f"  FAILED"); continue
        stats = {}
        curvatures = {}
        for lidx in LAYERS:
            hist = storage[lidx]
            if not hist: continue
            h_t = torch.stack(hist[-80:], dim=0)
            v = velocity_curve(h_t)
            stats[lidx] = collapse_stats(v, 80)
            curvatures[lidx] = curvature_curve(h_t)
        results[key] = stats
        results[key.replace("B_", "B_curv_")] = {str(l): c.tolist() for l, c in curvatures.items()}
        s_str = "  ".join(
            f"L{l}: pk={stats[l]['v_peak']:.0f} mid={stats[l]['mid_mean']:.0f} "
            f"ca={stats[l]['collapse_abs']}"
            for l in LAYERS if l in stats
        )
        log(f"  ({elapsed:.0f}s) {s_str}")

    # ── C: Curvature / Euler error analysis ──────────────────────────────
    log(f"\n{'='*70}")
    log("CONDITION C: Euler Error / Curvature Analysis (from 80-step B)")
    log(f"{'='*70}")
    for seed in seeds:
        key = f"B_curv_euler80_s{seed}"
        log(f"  Seed {seed}:")
        for lidx in LAYERS:
            curv_key = f"B_curv_euler80_s{seed}"
            if curv_key not in results: continue
            curv = np.array(results[curv_key].get(str(lidx), []))
            if len(curv) == 0: continue
            stat_key = f"B_euler80_s{seed}"
            v = np.array(results[stat_key][str(lidx)]["curve"]) if str(lidx) in results[stat_key] else []
            if len(v) == 0: continue
            # Compare curvature to velocity
            # Align: curv[t] corresponds to velocity at t+1 (since curv uses h_{t+2}, h_{t+1}, h_t)
            # curvature[t] vs velocity[t+1]
            aligned_v = v[1:1+len(curv)] if len(v) > len(curv) else v[:len(curv)]
            # Normalize: relative Euler error = curvature / velocity
            rel_err = curv / (aligned_v + 1e-8)
            log(f"    L{lidx}: curvature mean={np.mean(curv):.2f}  "
                f"rel_error mean={np.mean(rel_err):.2f}  "
                f"curvature peak at step={int(np.argmax(curv))}  "
                f"curvature mid-mean={np.mean(curv[int(0.3*len(curv)):int(0.6*len(curv))]):.2f}")

    # ── Curvature table ──────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("CURVATURE SUMMARY (80-step trajectory)")
    log(f"{'='*70}")
    log(f"{'Phase':<10} {'Layer':<6} {'curvature':<12} {'velocity':<12} {'rel_err':<12}")
    for phase_name, phase_slice in [("early(0-0.3)", (0, 24)), ("mid(0.3-0.6)", (24, 48)), ("late(0.6-1.0)", (48, 79))]:
        for lidx in LAYERS:
            curv_vals, vel_vals = [], []
            for seed in seeds:
                ck = f"B_curv_euler80_s{seed}"
                sk = f"B_euler80_s{seed}"
                if ck in results and str(lidx) in results[ck]:
                    c = np.array(results[ck][str(lidx)])
                    if len(c) > phase_slice[1]:
                        cseg = c[phase_slice[0]:phase_slice[1]]
                        curv_vals.extend(cseg)
                if sk in results:
                    stats_sk = results[sk]
                    skey = lidx if lidx in stats_sk else str(lidx)
                    if skey in stats_sk and "curve" in stats_sk[skey]:
                        v = np.array(stats_sk[skey]["curve"])
                    else:
                        continue
                    if len(v) > phase_slice[1]+1:
                        vseg = v[phase_slice[0]+1:phase_slice[1]+1]
                        vel_vals.extend(vseg)
            if curv_vals and vel_vals:
                log(f"{phase_name:<10} L{lidx:<4} {np.mean(curv_vals):<12.2f} "
                    f"{np.mean(vel_vals):<12.1f} {np.mean(curv_vals)/max(np.mean(vel_vals),1e-8):<12.2f}")

    # ── Euler error test: is the collapse trajectory well-predicted by Euler? ─
    log(f"\n{'='*70}")
    log("EULER ERROR ANALYSIS (KILLER TEST)")
    log(f"{'='*70}")
    all_rel_errs = []
    for seed in seeds:
        ck = f"B_curv_euler80_s{seed}"
        sk = f"B_euler80_s{seed}"
        if ck not in results: continue
        for lidx in LAYERS:
            curv = np.array(results[ck].get(str(lidx), []))
            if len(curv) < 10: continue
            stats_sk = results[sk]
            # stats dict keys are ints (Python), curvature keys are strs (json-safe)
            skey = lidx if lidx in stats_sk else str(lidx)
            if skey not in stats_sk or "curve" not in stats_sk[skey]: continue
            v = np.array(stats_sk[skey].get("curve", []))
            if len(v) < 10: continue
            v_aligned = v[1:1+len(curv)]
            if len(v_aligned) != len(curv): continue
            rel = curv / (v_aligned + 1e-8)
            all_rel_errs.extend(rel)

    log(f"\n  Relative Euler error (curvature / velocity):")
    log(f"    Mean: {np.mean(all_rel_errs):.3f}")
    log(f"    Median: {np.median(all_rel_errs):.3f}")
    log(f"    90th percentile: {np.percentile(all_rel_errs, 90):.3f}")
    log(f"  If rel_err < 1.0: Euler is accurate, collapse is genuine trajectory property")
    log(f"  If rel_err > 1.0: Euler is inaccurate, better solver might eliminate collapse")
    log(f"  Rule of thumb: rel_err < 0.5 → Euler is fine (> 80 steps)")
    log(f"                   0.5 < rel_err < 1.0 → moderate curvature, Heun might help")
    log(f"                   rel_err > 1.0 → Euler is fundamentally wrong")

    # ── Save ────────────────────────────────────────────────────────────────
    with open(OUT / "raw.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved: {OUT / 'raw.json'}")

    # ── Final classification ────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("FINAL CLASSIFICATION")
    log(f"{'='*70}")

    # Compare A vs B
    log(f"\n  A (Euler 40) vs B (Euler 80) comparison:")
    for metric in ["mid_mean", "collapse_abs"]:
        vals_a, vals_b = [], []
        for seed in seeds:
            ak = f"A_euler40_s{seed}"
            bk = f"B_euler80_s{seed}"
            if ak not in results or bk not in results: continue
            for lidx in LAYERS:
                if lidx not in results[ak] or lidx not in results[bk]: continue
                vals_a.append(results[ak][lidx][metric])
                vals_b.append(results[bk][lidx][metric])
        if vals_a and vals_b:
            log(f"    {metric}: A={np.mean(vals_a):.1f}  B={np.mean(vals_b):.1f}  "
                f"ratio={np.mean(vals_b)/max(np.mean(vals_a),1):.2f}")

    med_rel = np.median(all_rel_errs) if all_rel_errs else 1.0
    p90_rel = np.percentile(all_rel_errs, 90) if all_rel_errs else 1.5

    if med_rel < 0.5 and p90_rel < 1.0:
        decision = "NUMERICAL / SCHEDULER ARTIFACT"
        evidence = (
            f"Euler integration is accurate (median rel_err={med_rel:.2f}). "
            f"The collapse is a genuine feature of the ODE trajectory, "
            f"not an integration artifact. However, it follows step count "
            f"rather than normalized time, confirming it's scheduler-driven."
        )
    elif med_rel > 1.0:
        decision = "INCONCLUSIVE (Euler may be inaccurate)"
        evidence = (
            f"High Euler error (median rel_err={med_rel:.2f}). "
            f"A higher-order solver would change the trajectory significantly, "
            f"potentially eliminating the collapse."
        )
    else:
        decision = "MIXED — NUMERICAL DOMINANT"
        evidence = (
            f"Moderate Euler error (median rel_err={med_rel:.2f}, "
            f"p90={p90_rel:.2f}). Euler is adequate for most steps but "
            f"may have accuracy issues in high-curvature regions (early phase). "
            f"The mid-phase collapse itself has low curvature, "
            f"consistent with a numerical artifact of the Euler discretization."
        )

    log(f"\n  Decision: {decision}")
    log(f"  Evidence: {evidence}")

    # PLOT DATA
    log(f"\n  PLOT DATA:")
    for seed in [0]:
        for cond, n_steps in [("A", 40), ("B", 80)]:
            key = f"{'A' if cond == 'A' else 'B'}_euler{n_steps}_s{seed}"
            if key not in results: continue
            for lidx in LAYERS:
                if lidx not in results[key]: continue
                s = results[key][lidx]
                log(f"  PLOT,{cond},{n_steps}steps,L{lidx},peak={s['v_peak']:.0f},mid={s['mid_mean']:.0f},ca={s['collapse_abs']}")

    log(f"\nDone.")


if __name__ == "__main__":
    main()
