#!/usr/bin/env python3
"""Minimal controlled experiments for mid-phase bottleneck attribution."""

import json, os, sys, time, re, random
from pathlib import Path
import numpy as np
import torch

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
sys.path.insert(0, str(Path("/root/ACE-Step-1.5")))
from acestep.handler import AceStepHandler

DATA_DIR = Path("/root/autodl-tmp/musicdata/audios")
OUT = Path("/root/ACE-Step-1.5/output/controlled_experiments")
OUT.mkdir(parents=True, exist_ok=True)

def log(x):
    print(x, flush=True)

def pick_sample():
    all_s = []
    for f in os.listdir(str(DATA_DIR)):
        if not f.endswith(".mp3"): continue
        base = f.replace(".mp3", "")
        cap = DATA_DIR / f"{base}.caption.txt"
        lyr = DATA_DIR / f"{base}.lyrics.txt"
        if cap.exists() and lyr.exists():
            c = cap.read_text().strip()
            l = lyr.read_text().strip()
            if len(l) > 30: all_s.append((base, c, l))
    random.seed(42); random.shuffle(all_s)
    return all_s[0]

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
    T1 = len(v)
    v_min = float(v.min())
    t_vmin = float(v.argmin()) / max(T_orig - 1, 1)
    mid = slice(max(0, int(0.3*T1)), int(0.6*T1))
    mid_mean = float(v[mid].mean()) if mid.stop > mid.start else 0
    v_max = v[0]
    collapse_t = None
    for i in range(len(v)):
        if v[i] < 0.5 * v_max:
            collapse_t = i / max(T_orig - 1, 1)
            break
    return {"v_min": round(v_min,1), "t_vmin": round(t_vmin,3),
            "mid_mean": round(mid_mean,1), "collapse_t": collapse_t,
            "v_peak": round(float(v_max),1), "v_late_avg": round(float(v[-3:].mean()),1)}

def run(handler, caption, lyrics, bpm, key, seed,
        steps=25, guidance=1.0, layers=None):
    model = handler.model
    storage = {}
    handles = []
    for lidx in (layers or [12]):
        storage[lidx] = []
        handles.append(hook_layer(model.decoder.layers[lidx], storage[lidx]))
    t0 = time.time()
    r = handler.generate_music(
        captions=caption, lyrics=lyrics, bpm=bpm, key_scale=key,
        inference_steps=steps, seed=seed, use_random_seed=False,
        guidance_scale=guidance, audio_duration=10.0, infer_method="ode",
        batch_size=1,
    )
    elapsed = time.time() - t0
    for h in handles: h.remove()
    if not r.get("success"):
        log(f"  FAILED: {str(r.get('error','unknown'))[:80]}")
        return None
    stats = {}
    for lidx, hist in storage.items():
        if not hist: continue
        h_t = torch.stack(hist[-steps:], dim=0)
        v = velocity_curve(h_t)
        stats[lidx] = collapse_stats(v, steps)
        log(f"    L{lidx:2d}: peak={stats[lidx]['v_peak']:.0f} "
            f"t_vmin={stats[lidx]['t_vmin']:.3f} mid={stats[lidx]['mid_mean']:.0f} "
            f"collapse={stats[lidx]['collapse_t']} "
            f"v_late={stats[lidx]['v_late_avg']:.0f}")
    return stats

# ── Load model ──────────────────────────────────────────────────────────────
log("Loading model ...")
handler = AceStepHandler()
st, ok = handler.initialize_service(
    project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
    device="cpu", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
if not ok: raise RuntimeError(f"Init failed: {st}")
log("Model loaded.\n")

# Pick one sample
base, caption, lyrics = pick_sample()
bpm, key = parse_caption(caption)
log(f"Sample: {caption}")
log(f"Lyrics len: {len(lyrics)} chars")
log("")

all_data = {}

# ── EXP 1: Multi-layer (8, 12, 16) — 1 call ────────────────────────────────
log("="*60)
log("EXP 1: layers 8, 12, 16 (single generation)")
log("="*60)
s1 = run(handler, caption, lyrics, bpm, key, seed=500, steps=25,
         guidance=1.0, layers=[8, 12, 16])
if s1: all_data["exp1"] = s1
log("")

# ── EXP 2: CFG sensitivity (guidance=1.0 vs 3.0) — 2 calls ────────────────
log("="*60)
log("EXP 2: CFG 1.0 vs 3.0 (layer 12)")
log("="*60)
s2_g1 = run(handler, caption, lyrics, bpm, key, seed=600, steps=25,
            guidance=1.0, layers=[12])
s2_g3 = run(handler, caption, lyrics, bpm, key, seed=601, steps=25,
            guidance=3.0, layers=[12])
if s2_g1 and s2_g3:
    all_data["exp2_cfg1"] = s2_g1
    all_data["exp2_cfg3"] = s2_g3
log("")

# ── EXP 3: Step scaling (20 vs 40 steps) — 2 calls ─────────────────────────
log("="*60)
log("EXP 3: steps 20 vs 40 (layers 8, 12, 16)")
log("="*60)
s3_20 = run(handler, caption, lyrics, bpm, key, seed=700, steps=20,
            guidance=1.0, layers=[8, 12, 16])
s3_40 = run(handler, caption, lyrics, bpm, key, seed=701, steps=40,
            guidance=1.0, layers=[8, 12, 16])
if s3_20 and s3_40:
    all_data["exp3_20"] = s3_20
    all_data["exp3_40"] = s3_40
log("")

# ── Save ────────────────────────────────────────────────────────────────────
with open(OUT / "raw.json", "w") as f:
    json.dump(all_data, f, indent=2)

# ── Decisions ───────────────────────────────────────────────────────────────
log("\n" + "="*60)
log("DECISION TABLE")
log("="*60)

# 1. Layer consistency
if "exp1" in all_data and len(all_data["exp1"]) >= 2:
    t8 = all_data["exp1"][8]["t_vmin"]
    t12 = all_data["exp1"][12]["t_vmin"]
    t16 = all_data["exp1"][16]["t_vmin"]
    log(f"\n1. Layer consistency:")
    log(f"   L8  t_vmin={t8:.3f}")
    log(f"   L12 t_vmin={t12:.3f}")
    log(f"   L16 t_vmin={t16:.3f}")
    sync = max(abs(t8-t12), abs(t12-t16), abs(t8-t16)) < 0.1
    log(f"   {'SYNCHRONOUS' if sync else 'NOT SYNCHRONOUS'}")
    log(f"   Evidence: all layers collapse at t_vmin={np.mean([t8,t12,t16]):.3f} "
        f"(max spread {max(abs(t8-t12),abs(t12-t16),abs(t8-t16)):.3f})")

# 2. CFG sensitivity
if "exp2_cfg1" in all_data and "exp2_cfg3" in all_data:
    cfg1 = all_data["exp2_cfg1"][12]["mid_mean"]
    cfg3 = all_data["exp2_cfg3"][12]["mid_mean"]
    d_cfg = abs(cfg1 - cfg3)
    log(f"\n2. CFG sensitivity:")
    log(f"   guidance=1.0 mid_mean={cfg1:.1f}")
    log(f"   guidance=3.0 mid_mean={cfg3:.1f}")
    log(f"   delta={d_cfg:.1f} {'(HIGH)' if d_cfg > 50 else '(LOW)'}")
    log(f"   Collapse timing: cfg1={all_data['exp2_cfg1'][12]['collapse_t']}, "
        f"cfg3={all_data['exp2_cfg3'][12]['collapse_t']}")

# 3. Step scaling
if "exp3_20" in all_data and "exp3_40" in all_data:
    log(f"\n3. Step scaling invariance:")
    for nm in ["exp3_20", "exp3_40"]:
        steps_n = 20 if "20" in nm else 40
        log(f"   {steps_n} steps:")
        for lidx in sorted(all_data[nm].keys()):
            s = all_data[nm][lidx]
            log(f"     L{lidx}: t_vmin={s['t_vmin']:.3f} collapse={s['collapse_t']} "
                f"mid={s['mid_mean']:.0f} late={s['v_late_avg']:.0f}")

# 4. Final
log(f"\n4. Final attribution:")
if "exp1" in all_data:
    t_vals = [all_data["exp1"][l]["t_vmin"] for l in sorted(all_data["exp1"].keys())]
    sync = max(t_vals) - min(t_vals) < 0.1
    if sync and (d_cfg if 'd_cfg' in dir() else 0) < 50:
        log(f"   LATENT SYSTEM — multi-layer sync collapse, CFG-insensitive")
    elif sync:
        log(f"   ATTENTION / CONDITIONING — CFG moves collapse")
    else:
        log(f"   LAYER-LOCAL — layers not synchronized")
else:
    log(f"   PARTIAL DATA — insufficient for conclusion")

log(f"\nAll data: {OUT / 'raw.json'}")
log("="*60)
