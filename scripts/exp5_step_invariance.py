#!/usr/bin/env python3
"""
EXP5: Step Invariance / Resolution Sensitivity Test

Three conditions: 20, 40, 80 inference steps.
Tests whether mid-phase velocity collapse is intrinsic (normalised time invariant)
or discretization-driven (absolute step dependent).
"""

import json, os, sys, time, re
from pathlib import Path
import numpy as np
import torch

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
from acestep.handler import AceStepHandler

DATA_DIR = Path("/root/autodl-tmp/musicdata/audios")
OUT = Path("/root/ACE-Step-1.5/output/exp5_step_invariance")
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
    h_p = h_t.mean(dim=(1, 2))
    d = h_p[1:] - h_p[:-1]
    return torch.norm(d, dim=-1).numpy()

def collapse_stats(v, T_orig):
    v_peak = float(v[0]) if len(v) > 0 else 0
    v_min = float(v.min()) if len(v) > 0 else 0
    # absolute step of minimum
    t_vmin_abs = int(v.argmin()) if len(v) > 0 else 0
    t_vmin_norm = t_vmin_abs / max(T_orig - 1, 1)
    mid = slice(max(0, int(0.3 * len(v))), int(0.6 * len(v)))
    mid_mean = float(v[mid].mean()) if mid.stop > mid.start and len(v) > 0 else 0
    collapse_abs = None
    for i in range(len(v)):
        if v[i] < 0.5 * v_peak:
            collapse_abs = i
            break
    collapse_norm = collapse_abs / max(T_orig - 1, 1) if collapse_abs is not None else None
    return {
        "v_peak": round(v_peak, 1),
        "v_min": round(v_min, 1),
        "t_vmin_abs": int(t_vmin_abs),
        "t_vmin_norm": round(t_vmin_norm, 4),
        "mid_mean": round(mid_mean, 1),
        "collapse_abs": collapse_abs,
        "collapse_norm": collapse_norm,
        "v_late_avg": round(float(v[-3:].mean()), 1) if len(v) >= 3 else 0,
        "curve": [round(float(x), 1) for x in v],
    }


def run_n_steps(handler, caption, lyrics, bpm, key, seed, n_steps, layers=None):
    """Run generate_music with n_steps, return per-layer h_t."""
    model = handler.model
    layers = layers or [12]
    storage = {}
    handles = []
    for lidx in layers:
        storage[lidx] = []
        handles.append(hook_layer(model.decoder.layers[lidx], storage[lidx]))

    t0 = time.time()
    r = handler.generate_music(
        captions=caption, lyrics=lyrics, bpm=bpm, key_scale=key,
        inference_steps=n_steps, seed=seed, use_random_seed=False,
        guidance_scale=1.0, audio_duration=10.0, infer_method="ode",
        batch_size=1,
    )
    elapsed = time.time() - t0
    for h in handles: h.remove()

    if not r.get("success"):
        log(f"  FAILED: {str(r.get('error','unknown'))[:80]}")
        return None

    result = {}
    for lidx in layers:
        hist = storage[lidx]
        if not hist: continue
        h_t = torch.stack(hist[-n_steps:], dim=0)
        result[lidx] = h_t
    return result, elapsed


def main():
    log("=" * 70)
    log("EXP5: Step Invariance / Resolution Sensitivity Test")
    log("=" * 70)

    log("\nLoading model ...")
    handler = AceStepHandler()
    st, ok = handler.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
        device="cpu", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    if not ok: raise RuntimeError(f"Init failed: {st}")
    log("Model loaded.\n")

    base, caption, lyrics = get_sample()
    bpm, key = parse_caption(caption)
    log(f"Sample: {caption}\n")

    all_results = {}
    seeds = [0, 1, 42]
    step_configs = [20, 40, 80]
    LAYERS = [8, 12, 16]

    for steps in step_configs:
        for seed in seeds:
            key = f"s{steps}_seed{seed}"
            log(f"[{key}] Running {steps} steps, seed={seed} ...")
            out = run_n_steps(handler, caption, lyrics, bpm, key, seed,
                              n_steps=steps, layers=LAYERS)
            if out is None:
                continue
            h_dict, elapsed = out
            log(f"  Done ({elapsed:.0f}s)")

            stats = {}
            for lidx in LAYERS:
                if lidx not in h_dict: continue
                h_t = h_dict[lidx]
                v = velocity_curve(h_t)
                stats[lidx] = collapse_stats(v, steps)
                s = stats[lidx]
                cn_str = f"{s['collapse_norm']:.3f}" if s['collapse_norm'] is not None else "None"
                log(f"  L{lidx:2d}: peak={s['v_peak']:.0f} "
                    f"collapse_abs={s['collapse_abs']} collapse_norm={cn_str} "
                    f"mid={s['mid_mean']:.0f} t_vmin_abs={s['t_vmin_abs']}")
            all_results[key] = stats
            log("")

    # Save raw
    with open(OUT / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Saved: {OUT / 'raw.json'}\n")

    # ── TABLE ────────────────────────────────────────────────────────────────
    log("=" * 70)
    log("RESULTS TABLE (averaged across layers and seeds)")
    log("=" * 70)
    log(f"{'Steps':<8} {'Seed':<6} {'v_peak':<8} {'mid_mean':<8} "
        f"{'collapse_abs':<12} {'collapse_norm':<13} {'t_vmin_abs':<10} {'t_vmin_norm':<12}")
    log(f"{'':-<8} {'':-<6} {'':-<8} {'':-<8} {'':-<12} {'':-<13} {'':-<10} {'':-<12}")

    for steps in step_configs:
        for seed in seeds:
            key = f"s{steps}_seed{seed}"
            if key not in all_results: continue
            vals = []
            for lidx in LAYERS:
                if lidx not in all_results[key]: continue
                s = all_results[key][lidx]
                vals.append(s)
            if not vals: continue
            # average across layers
            avg_peak = np.mean([v["v_peak"] for v in vals])
            avg_mid = np.mean([v["mid_mean"] for v in vals])
            # collapse_abs: use mean, handle None as max_steps
            col_abs = [v["collapse_abs"] or steps for v in vals]
            avg_col_abs = np.mean(col_abs)
            avg_col_norm = np.mean([v["collapse_norm"] or 1.0 for v in vals])
            avg_tvm_abs = np.mean([v["t_vmin_abs"] for v in vals])
            avg_tvm_norm = np.mean([v["t_vmin_norm"] for v in vals])
            log(f"{steps:<8} {seed:<6} {avg_peak:<8.0f} {avg_mid:<8.1f} "
                f"{avg_col_abs:<12.1f} {avg_col_norm:<13.4f} {avg_tvm_abs:<10.1f} {avg_tvm_norm:<12.4f}")

    # ── COLLAPSE_NORM INVARIANCE ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("COLLAPSE NORMALISED TIME INVARIANCE TEST")
    log(f"{'='*70}")
    log(f"{'Steps':<8} {'collapse_norm (mean)':<20} {'collapse_norm (std)':<20} {'t_vmin_norm (mean)':<20}")
    log(f"{'':-<8} {'':-<20} {'':-<20} {'':-<20}")

    for steps in step_configs:
        col_norms = []
        tvm_norms = []
        for seed in seeds:
            key = f"s{steps}_seed{seed}"
            if key not in all_results: continue
            for lidx in LAYERS:
                if lidx not in all_results[key]: continue
                s = all_results[key][lidx]
                col_norms.append(s["collapse_norm"] or 1.0)
                tvm_norms.append(s["t_vmin_norm"])
        if col_norms:
            log(f"{steps:<8} {np.mean(col_norms):<20.4f} {np.std(col_norms):<20.4f} {np.mean(tvm_norms):<20.4f}")

    # ── DECISION ─────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DECISION")
    log(f"{'='*70}")

    # Gather collapse_norm per config
    norm_by_steps = {}
    for steps in step_configs:
        vals = []
        for seed in seeds:
            key = f"s{steps}_seed{seed}"
            if key not in all_results: continue
            for lidx in LAYERS:
                if lidx not in all_results[key]: continue
                vals.append(all_results[key][lidx]["collapse_norm"] or 1.0)
        norm_by_steps[steps] = vals

    # Test if collapse_norm is invariant across 20/40/80
    if len(norm_by_steps) == 3:
        m20 = np.mean(norm_by_steps[20])
        m40 = np.mean(norm_by_steps[40])
        m80 = np.mean(norm_by_steps[80])
        s20 = np.std(norm_by_steps[20])
        s40 = np.std(norm_by_steps[40])
        s80 = np.std(norm_by_steps[80])

        log(f"\n  collapse_norm mean:  20={m20:.3f}  40={m40:.3f}  80={m80:.3f}")
        log(f"  collapse_norm std:   20={s20:.3f}  40={s40:.3f}  80={s80:.3f}")

        # Check if all within 2 std of each other
        all_means = [m20, m40, m80]
        mean_of_means = np.mean(all_means)
        max_dev = max(abs(m - mean_of_means) for m in all_means)

        log(f"\n  Mean of means: {mean_of_means:.4f}")
        log(f"  Max deviation from grand mean: {max_dev:.4f}")

        # Raw t_vmin_abs trend
        log(f"\n  ABSOLUTE step trends:")
        for steps in step_configs:
            abs_vals = []
            for seed in seeds:
                key = f"s{steps}_seed{seed}"
                if key not in all_results: continue
                for lidx in LAYERS:
                    if lidx not in all_results[key]: continue
                    abs_vals.append(all_results[key][lidx]["collapse_abs"] or steps)
            log(f"    {steps} steps: collapse_abs mean={np.mean(abs_vals):.1f}  "
                f"t_vmin_abs mean={np.mean([all_results.get(f's{steps}_seed{s}',{}).get(l,{}).get('t_vmin_abs',0) for s in seeds for l in LAYERS]):.1f}")

        if max_dev < 0.05:
            decision = "INTRINSIC DYNAMICS (step-invariant)"
            evidence = (
                f"collapse_norm/T is stable across 20/40/80 steps: "
                f"means {m20:.3f}, {m40:.3f}, {m80:.3f} within {max_dev:.3f} of each other "
                f"in normalized time."
            )
        elif max_dev > 0.15:
            decision = "DISCRETIZATION / SCHEDULE EFFECT (step-dependent)"
            evidence = (
                f"collapse_norm shifts systematically with step count: "
                f"{m20:.3f} -> {m40:.3f} -> {m80:.3f}, deviation {max_dev:.3f} from grand mean."
            )
        else:
            decision = "MIXED / INCONCLUSIVE"
            evidence = (
                f"collapse_norm partially stable but drifts with step count: "
                f"{m20:.3f} -> {m40:.3f} -> {m80:.3f}, deviation {max_dev:.3f}."
            )

        log(f"\n  Decision: {decision}")
        log(f"  Evidence: {evidence}")

    # ── PLOT DATA ────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("PLOT DATA (seed=0, layer=12, all 3 conditions)")
    log(f"{'='*70}")
    log(f"step,t_norm,cond,v")
    for steps in step_configs:
        key = f"s{steps}_seed0"
        if key not in all_results or "12" not in all_results[key]: continue
        curve = all_results[key]["12"]["curve"]
        for si, sv in enumerate(curve):
            t_norm = si / max(steps - 1, 1)
            log(f"{si},{t_norm:.4f},{steps}steps,{sv}")

    log(f"\nDone. All data: {OUT/'raw.json'}")


if __name__ == "__main__":
    main()
