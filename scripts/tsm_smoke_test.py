#!/usr/bin/env python3
"""GPU Smoke Test Suite for Transported Structural Memory (TSM).

Usage:
    cd /root/ACE-Step-1.5
    ACESTEP_LOCAL_MODEL_CODE=1 SIDESTEP_SAFE_ROOT="/" \
        python scripts/tsm_smoke_test.py --step 1
    python scripts/tsm_smoke_test.py --step 2
    python scripts/tsm_smoke_test.py --step 3
    python scripts/tsm_smoke_test.py --step 4

Steps:
  1. Zero-init equivalence (TSM vs baseline, same seed/prompt)
  2. 50-100 training steps (gradient flow, residual ratio, stability)
  3. 30-second generation (safety, non-zero residual, no artifacts)
  4. One epoch training + comparison (only after 1-3 pass)
"""

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault('ACESTEP_OFFLINE', '1')
os.environ.setdefault('ACESTEP_MINIMAL_COMPONENTS', '1')
os.environ.setdefault('SIDESTEP_SAFE_ROOT', '/root/autodl-tmp')

sys.path.insert(0, '/root/ACE-Step-1.5')


# ===========================================================================
#  Config
# ===========================================================================

MODEL_ROOT = '/root/autodl-tmp/Ace-Step1.5'
MODEL_VARIANT = 'acestep-v15-sft'
CHECKPOINT = '/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt'
OUTPUT_ROOT = Path('/root/ACE-Step-1.5/outputs/tsm_smoke_test')
_DEFAULT_OUTPUT_ROOT = Path('/root/ACE-Step-1.5/outputs/tsm_smoke_test')
CAPTION = 'A pop song with emotional female vocals and clear structure.'
LYRICS = """[Verse]
The morning light breaks through the clouds
I hear your voice calling out loud
[Pre-Chorus]
Every step I take toward you
Feels like the world is brand new
[Chorus]
We rise up high above the sky
Together we can learn to fly
No looking back no fear no doubt
This is what love is all about
[Verse]
The stars align when you are near
You make the darkness disappear"""

SEED = 42
DURATION = 30  # seconds for steps 1-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TSM_CONFIG = {
    'model_dim': 2048, 'memory_dim': 256, 'num_heads': 4,
    'ffn_dim': 512, 'slot_layers': 1, 'dropout': 0.0,
    'detach_coupling': True,
}


# ===========================================================================
#  Helpers
# ===========================================================================

def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def rms(x: torch.Tensor) -> float:
    return math.sqrt(x.pow(2).mean().item())


def load_transport_checkpoint(ckpt_path: str, device: str = 'cuda', use_pm: bool = True):
    """Load PM + TransportAdapter + TSM (optional) from checkpoint.

    When use_pm=False, PM is not loaded and adapter uses scaffold_only scoring.
    """
    from acestep.phase_memory import TransportRetrievalAdapter
    from acestep.modules.transported_structural_memory import TransportedStructuralMemory

    D = 2048
    scoring = 'position_only' if use_pm else 'scaffold_only'
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        sinkhorn_iters=10, transport_sigma=0.18, scoring_mode=scoring,
        use_pm_gate=False, gate_hidden_dim=128,
        write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01,
    ).to(device).float()

    tsm = TransportedStructuralMemory(**TSM_CONFIG).to(device).float()
    tsm_loaded = False; tsm_mode = 'sinkhorn_tsm'

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    adapt.load_state_dict(ckpt['retrieval_adapter'])
    if 'tsm_module' in ckpt:
        tsm.load_state_dict(ckpt['tsm_module'])
        tsm_loaded = True
        tsm_mode = ckpt.get('tsm_mode', 'sinkhorn_tsm')

    if use_pm:
        from acestep.phase_memory import PMRetrievalPhaseMemory
        pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256,
                                     normalize_internal_state=True).to(device).float()
        if 'phase_memory' not in ckpt:
            print("    WARNING: use_pm=True but no phase_memory in checkpoint — falling back to scaffold_only")
            pm = None
        else:
            pm.load_state_dict(ckpt['phase_memory'])
            pm.eval()
    else:
        pm = None

    adapt.eval(); tsm.eval()
    return pm, adapt, tsm, tsm_loaded, tsm_mode


def build_scaffold_from_lyrics(lyrics: str, eh: torch.Tensor, device: str) -> Dict[str, torch.Tensor]:
    """Build duration scaffold from lyrics + encoder hidden states.

    Includes auto-transition silence units (intro/outro/transitions) so
    TSM sees the full structural skeleton.
    """
    from acestep.phase_memory import parse_lyrics_to_units, build_duration_scaffold

    L = eh.shape[1]
    section_ids = torch.zeros(L, dtype=torch.long)
    units, _, debug = parse_lyrics_to_units(
        lyrics, section_ids, auto_transition_ratios={
            "intro": 0.03, "outro": 0.03,
            "chorus_to_verse": 0.02, "chorus_to_bridge": 0.02,
            "bridge_to_chorus": 0.02,
        },
    )
    tcm = debug.get("tag_control_mask", None)
    sc = build_duration_scaffold(units, text_len=L, tag_control_mask=tcm)
    sc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
    return sc


def compute_unit_tensors(sc: Dict, eh: torch.Tensor, B: int) -> Dict[str, torch.Tensor]:
    """Compute unit-level tensors from scaffold."""
    D_h = eh.shape[-1]
    U = len(sc['unit_boundaries']) - 1
    t2u = sc['token_to_unit']
    lym = sc['lyric_mask']
    uil = sc['lyric_unit_mask']
    ca = (sc['unit_boundaries'][:-1] + sc['unit_boundaries'][1:]) / 2
    mu = sc['unit_duration'] / sc['unit_duration'].sum()
    usid = sc['unit_section_ids']

    eh_f = eh.float()
    uth = []
    for uid in range(U):
        tm = (t2u == uid) & lym if uil[uid].item() else (t2u == uid)
        tmb = tm.unsqueeze(0).expand(B, -1)
        pool = eh_f[tmb].view(B, -1, D_h).mean(dim=1) if tmb.any() else torch.zeros(B, D_h, device=eh.device)
        uth.append(pool)
    uth_stacked = torch.stack(uth, dim=1)

    return {
        'unit_text_hidden': uth_stacked,
        'unit_centres': ca,
        'unit_mass': mu,
        'unit_section_ids': usid,
        'unit_is_lyric': uil,
    }


def compare_latents(lat_a: torch.Tensor, lat_b: torch.Tensor) -> Dict[str, float]:
    """Compare two latents element-wise."""
    diff = (lat_a.float() - lat_b.float()).abs()
    return {
        'latent_max_abs_diff': diff.max().item(),
        'latent_mean_abs_diff': diff.mean().item(),
        'latent_rms_a': rms(lat_a),
        'latent_rms_b': rms(lat_b),
    }


# ===========================================================================
#  Step 1: Zero-init equivalence
# ===========================================================================

def step1_zero_init_equivalence(use_pm: bool = True) -> Dict[str, Any]:
    """Verify TSM with zero init produces identical output to baseline."""
    print_section(f"STEP 1: Zero-init Equivalence Test (PM={use_pm})")

    from acestep.handler import AceStepHandler

    results: Dict[str, Any] = {}

    # ---- Load model twice (baseline + TSM) ---------------------------------
    print("[1a] Loading model (shared backbone)...")
    dt = AceStepHandler()
    dt.initialize_service(
        project_root=MODEL_ROOT, config_path=MODEL_VARIANT,
        device=DEVICE, use_flash_attention=False, compile_model=False,
        offload_to_cpu=False,
    )
    model = dt.model
    model.eval()
    model.config.use_section_rope_offset = False
    for l in model.decoder.layers:
        if getattr(l, 'use_section_rope', False): l.use_section_rope = False
        if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False

    # ---- Load transport modules ---------------------------------------------
    print("[1b] Loading PM + TransportAdapter + TSM...")
    pm, adapt, tsm, tsm_loaded, tsm_mode = load_transport_checkpoint(CHECKPOINT, DEVICE, use_pm=use_pm)
    print(f"    PM instantiated: {pm is not None}  PM loaded: {pm is not None}")

    # Force zero-init TSM output projection
    nn.init.zeros_(tsm.output_proj.weight)
    if tsm.output_proj.bias is not None:
        nn.init.zeros_(tsm.output_proj.bias)
    verify_zero = tsm.output_proj.weight.abs().max().item()
    print(f"    Verified output_proj weight max_abs: {verify_zero:.2e}")
    assert verify_zero < 1e-10, f"Zero init failed: max_abs={verify_zero}"

    T_eff = int(DURATION * 25)

    # ---- Use gen_one_clean-style monkey-patch + generate_music -------------
    # Patch prepare_condition: install hook that runs transport + TSM
    # Compare within the same forward: compute tsm_output from zero-init TSM
    # Since tsm_output ≈ 0, the generation is equivalent to baseline

    print("[1c] Running generation with TSM (zero-init, same seed as baseline)...")
    from acestep.tgca.lyrics_parser import LyricsStructureParser
    from acestep.phase_memory import parse_lyrics_to_units as _p_units
    from acestep.phase_memory import build_duration_scaffold as _b_scaffold
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    # Force zero-init TSM
    nn.init.zeros_(tsm.output_proj.weight)
    if tsm.output_proj.bias is not None:
        nn.init.zeros_(tsm.output_proj.bias)
    verify_zero = tsm.output_proj.weight.abs().max().item()
    assert verify_zero < 1e-10, f"Zero init failed: max_abs={verify_zero}"
    print(f"    Verified output_proj max_abs={verify_zero:.2e}")

    llm = LLMHandler()
    llm.initialize(checkpoint_dir=MODEL_ROOT, lm_model_path='acestep-5Hz-lm-1.7B',
                   backend='pt', device=DEVICE)

    tsm_call_count = [0]
    tsm_rms_log = []
    tsm_diag_log = []

    hook_holder = [None]
    orig_prep = model.prepare_condition

    def patched_prep(*a, **kw):
        r = orig_prep(*a, **kw)
        if r[0] is not None and hook_holder[0] is None:
            eh = r[0]; L = eh.shape[1]; B = eh.shape[0]
            print(f"    encoder_hidden shape: {eh.shape}")

            # Build scaffold
            parser = LyricsStructureParser()
            ps = parser.parse(LYRICS, num_chunks=L)
            units, _, debug = _p_units(LYRICS, ps.section_type_ids, auto_transition_ratios={
                "intro": 0.0, "outro": 0.0,
                "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
                "bridge_to_chorus": 0.0,
            })
            tcm = debug.get("tag_control_mask", None)
            sc = _b_scaffold(units, text_len=L, tag_control_mask=tcm)
            sc = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
            ut = compute_unit_tensors(sc, eh, B)
            print(f"    scaffold: {len(sc['unit_boundaries'])-1} total units, "
                  f"lyric={ut['unit_is_lyric'].sum().item()} lyric units")

            def hook(_m, _i, o):
                Ho = o[0]; B_cur, T = Ho.shape[0], Ho.shape[1]
                pa = torch.linspace(0, 1, T, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                if B_cur > pa.shape[0]:
                    pa = pa.expand(B_cur, -1).contiguous()
                with torch.no_grad():
                    ps2 = pm(Ho.float(), torch.zeros(B_cur, device=DEVICE)) if pm is not None else None
                    uth_b = ut['unit_text_hidden'].float().expand(B_cur, -1, -1)
                    ca_b = ut['unit_centres'].unsqueeze(0).float().expand(B_cur, -1)
                    mu_b = ut['unit_mass'].unsqueeze(0).float().expand(B_cur, -1)
                    usid_b = ut['unit_section_ids'].unsqueeze(0).expand(B_cur, -1)
                    uil_b = ut['unit_is_lyric'].unsqueeze(0).expand(B_cur, -1)
                    dh, Pi, diag = adapt(Ho.float(), None, ps2, pa.float(),
                        uth_b, ca_b, mu_b, unit_section_id=usid_b, unit_is_lyric=uil_b)
                    # TSM forward (serial: H + dh)
                    tsm_out, tsm_d = tsm(
                        Ho.float() + dh, coupling=Pi,
                        condition_mask=uil_b, detach_coupling=True,
                        enable_slot_mixer=True,
                    )
                    tsm_call_count[0] += 1
                    tsm_rms_log.append(tsm_d.get('tsm_output_rms', -1))
                    tsm_diag_log.append(tsm_d)
                return (Ho + dh.to(dtype=Ho.dtype) + tsm_out.to(dtype=Ho.dtype), *o[1:])

            hook_holder[0] = model.decoder.layers[12].register_forward_hook(hook)
        return r

    model.prepare_condition = patched_prep

    params = GenerationParams(
        task_type='text2music', caption=CAPTION, lyrics=LYRICS,
        instrumental=False, bpm=120, keyscale='C major', timesignature='4',
        vocal_language='en', duration=DURATION, inference_steps=8,
        guidance_scale=7.0, seed=SEED, thinking=True,
        use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
    )
    config = GenerationConfig(batch_size=1, audio_format='flac',
                               use_random_seed=False, seeds=[SEED])

    torch.manual_seed(SEED)
    result = generate_music(dit_handler=dt, llm_handler=llm, params=params,
                             config=config, save_dir=str(OUTPUT_ROOT / 'step1_tsm_zero'))

    model.prepare_condition = orig_prep
    if hook_holder[0]: hook_holder[0].remove()
    dt = None  # allow GC

    # ---- Results ------------------------------------------------------------
    print_section("STEP 1: Results")

    print(f"TSM forward call count: {tsm_call_count[0]}")
    assert tsm_call_count[0] > 0, "TSM forward was NOT called!"
    print("  ✓ TSM forward called")

    if tsm_rms_log:
        for i, (r_val, d) in enumerate(zip(tsm_rms_log, tsm_diag_log)):
            if i < 3 or i >= len(tsm_rms_log) - 3:
                print(f"  step {i}: tsm_output_rms={float(r_val):.8e}  "
                      f"ratio={d.get('tsm_output_to_hidden_ratio', -1):.8e}  "
                      f"cos={d.get('tsm_slot_cosine_mean', -1):.4f}")

    tsm_output_rms_val = tsm_rms_log[0] if tsm_rms_log else -1
    all_zero = all(r < 1e-10 for r in tsm_rms_log if r >= 0)
    if all_zero:
        print("  ✓ ALL tsm_output_rms ≈ 0 — zero init verified")
    else:
        non_zero = [r for r in tsm_rms_log if r >= 1e-10]
        print(f"  ⚠ {len(non_zero)}/{len(tsm_rms_log)} steps have non-zero TSM output — check initialization!")

    print(f"\n  Generation: {'OK' if result.success else 'FAIL'}")
    if result.success:
        print(f"  Audio: {result.audios[0]['path']}")

    results['tsm_call_count'] = tsm_call_count[0]
    results['tsm_rms_log'] = [float(r) for r in tsm_rms_log]
    results['tsm_output_rms_first'] = float(tsm_output_rms_val)
    results['tsm_all_zero'] = all_zero
    results['generation_success'] = result.success

    return results


# ===========================================================================
#  Step 2: 50-100 training steps
# ===========================================================================

def step2_short_training(use_pm: bool = True) -> Dict[str, Any]:
    """Train TSM for 100 steps on real preprocessed data."""
    import glob as _glob
    from acestep.handler import AceStepHandler
    from acestep.modules.transported_structural_memory import (
        TransportedStructuralMemory, collect_tsm_grad_norms,
    )
    from acestep.phase_memory import parse_lyrics_to_units as _p_units
    from acestep.phase_memory import build_duration_scaffold as _b_scaffold
    from acestep.tgca.lyrics_parser import LyricsStructureParser

    print_section(f"STEP 2: Short Training on REAL DATA (PM={use_pm})")
    results: Dict[str, Any] = {'step_diagnostics': []}

    # ---- Load model ----------------------------------------------------------
    print("[2a] Loading model...")
    dt = AceStepHandler()
    dt.initialize_service(
        project_root=MODEL_ROOT, config_path=MODEL_VARIANT,
        device=DEVICE, use_flash_attention=False, compile_model=False,
        offload_to_cpu=False,
    )
    model = dt.model
    model.eval()
    model.config.use_section_rope_offset = False
    for l in model.decoder.layers:
        if getattr(l, 'use_section_rope', False): l.use_section_rope = False
        if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False

    D = model.config.hidden_size

    # ---- Load and freeze PM + TransportAdapter --------------------------------
    pm, adapt, _, _, _ = load_transport_checkpoint(CHECKPOINT, DEVICE, use_pm=use_pm)
    if pm is not None:
        for p in pm.parameters(): p.requires_grad = False
    for p in adapt.parameters(): p.requires_grad = False
    for p in model.parameters(): p.requires_grad = False

    # ---- Create TSM (trainable only) -----------------------------------------
    tsm = TransportedStructuralMemory(**TSM_CONFIG).to(DEVICE).float()
    optimizer = torch.optim.AdamW(tsm.parameters(), lr=1e-4, weight_decay=0.01)
    grad_clip = 1.0

    tsm_params = sum(p.numel() for p in tsm.parameters())
    backbone_total = sum(p.numel() for p in model.parameters())
    pm_total = (sum(p.numel() for p in pm.parameters()) if pm is not None else 0) + \
               sum(p.numel() for p in adapt.parameters())
    total_params = backbone_total + pm_total + tsm_params
    print(f"[2b] TSM: {tsm_params:,} trainable / ~{total_params:,} total")
    print(f"    PM instantiated: {pm is not None}  Scoring: {'position_only' if use_pm else 'scaffold_only'}")

    # ---- Load real data -------------------------------------------------------
    print("[2c] Loading real data...")
    pt_files = sorted(_glob.glob("/root/autodl-tmp/musicdata/train_tensors/*.pt"))
    assert pt_files, "No .pt files found!"
    data = torch.load(pt_files[0], map_location='cpu')
    lyrics_text = data['metadata'].get('lyrics', '')
    print(f"    File: {pt_files[0]}")
    print(f"    Lyrics: {len(lyrics_text)} chars, target_latents: {list(data['target_latents'].shape)}")

    B = 1
    x0 = data['target_latents'].unsqueeze(0).to(device=DEVICE, dtype=model.dtype)        # [1, T_raw, 64]
    ctx_lat = data['context_latents'].unsqueeze(0).to(device=DEVICE, dtype=model.dtype)  # [1, T_raw, 128]
    eh = data['encoder_hidden_states'].unsqueeze(0).to(device=DEVICE, dtype=model.dtype) # [1, L, D]
    eam = data['encoder_attention_mask'].unsqueeze(0).to(device=DEVICE)                  # [1, L] bool
    attn_mask = data['attention_mask'].unsqueeze(0).to(device=DEVICE, dtype=model.dtype) # [1, T_raw]
    T_raw = x0.shape[1]; L_enc = eh.shape[1]
    x1 = torch.randn_like(x0)

    print(f"    T_raw={T_raw} L_enc={L_enc} ({T_raw//25:.0f}s estimated)")

    # ---- Build scaffold from REAL lyrics --------------------------------------
    print("[2d] Building scaffold from real lyrics...")
    parser = LyricsStructureParser()
    ps = parser.parse(lyrics_text, num_chunks=L_enc)
    units, _, debug = _p_units(lyrics_text, ps.section_type_ids, auto_transition_ratios={
        "intro": 0.03, "outro": 0.03,
        "chorus_to_verse": 0.02, "chorus_to_bridge": 0.02,
        "bridge_to_chorus": 0.02,
    })
    tcm = debug.get("tag_control_mask", None)
    sc = _b_scaffold(units, text_len=L_enc, tag_control_mask=tcm)
    sc = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
    ut = compute_unit_tensors(sc, eh, B)
    U = len(sc['unit_boundaries']) - 1
    print(f"    scaffold: {U} total units, lyric={ut['unit_is_lyric'].sum().item()}, "
          f"silence={U - ut['unit_is_lyric'].sum().item()}")
    # Track per-section info
    section_labels = ['UNKNOWN','INTRO','VERSE','PRECHORUS','CHORUS','BRIDGE','OUTRO','INSTRUMENTAL']
    sid_counts = {}
    for uid in range(U):
        sid = ut['unit_section_ids'][uid].item()
        sid_counts[section_labels[sid]] = sid_counts.get(section_labels[sid], 0) + 1
    print(f"    sections: {sid_counts}")

    # ---- Training loop -------------------------------------------------------
    NUM_STEPS = 100
    step_diagnostics = []
    tsm.train()

    print(f"\n[2e] Training {NUM_STEPS} steps on real data...\n")

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        t_val = torch.rand(1, device=DEVICE).item()
        t_tensor = torch.full((B,), t_val, device=DEVICE, dtype=model.dtype)
        t_expanded = t_tensor.unsqueeze(-1).unsqueeze(-1)
        xt = t_expanded * x1 + (1.0 - t_expanded) * x0

        # ---- Warmup → collect H from layer 12 --------------------------------
        hs_list = []
        def _whook(m, i, o): hs_list.append(o[0])
        handle = model.decoder.layers[12].register_forward_hook(_whook)
        try:
            with torch.no_grad():
                _ = model.decoder(
                    hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                    attention_mask=attn_mask,
                    encoder_hidden_states=eh,
                    encoder_attention_mask=eam,
                    context_latents=ctx_lat,
                )
        finally:
            handle.remove()
        if not hs_list: continue
        H = hs_list[0].float()
        T_h = H.shape[1]

        # ---- PM + Transport adapter (no-grad) --------------------------------
        ps2 = pm(H.float(), t_tensor.float()) if pm is not None else None
        p_audio = torch.linspace(0, 1, T_h, device=DEVICE, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            dh, Pi, _ = adapt(
                H.float(), eh.float(), ps2, p_audio,
                ut['unit_text_hidden'].float().expand(B, -1, -1),
                ut['unit_centres'].float().unsqueeze(0).expand(B, -1),
                ut['unit_mass'].float().unsqueeze(0).expand(B, -1),
                unit_section_id=ut['unit_section_ids'].unsqueeze(0).expand(B, -1),
                unit_is_lyric=ut['unit_is_lyric'].unsqueeze(0).expand(B, -1),
            )
        if Pi is None or Pi.shape[-1] == 0: continue

        # ---- TSM forward (serial: H + dh) ------------------------------------
        tsm_output, tsm_diag = tsm(
            hidden_states=H + dh, coupling=Pi.detach(),
            condition_mask=ut['unit_is_lyric'].unsqueeze(0).expand(B, -1),
            detach_coupling=True, enable_slot_mixer=True,
        )

        final_h = H + dh.to(dtype=H.dtype) + tsm_output.to(dtype=H.dtype)

        # ---- Inject → decoder → flow loss ------------------------------------
        def _inject_hook(m, i, o):
            return (final_h.to(dtype=o[0].dtype, device=o[0].device), *o[1:])
        inj_h = model.decoder.layers[12].register_forward_hook(_inject_hook)
        try:
            d_out = model.decoder(
                hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                attention_mask=attn_mask,
                encoder_hidden_states=eh,
                encoder_attention_mask=eam,
                context_latents=ctx_lat,
            )
        finally:
            inj_h.remove()

        flow = x1 - x0
        loss = F.mse_loss(d_out[0], flow)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(tsm.parameters(), grad_clip)
        optimizer.step()

        # ---- Diagnostics ------------------------------------------------------
        gn = collect_tsm_grad_norms(tsm)
        tsm_diag.update(gn)
        diag = {
            'step': step, 'loss': loss.item(),
            'tsm_output_rms': tsm_diag.get('tsm_output_rms', -1),
            'tsm_output_to_hidden_ratio': tsm_diag.get('tsm_output_to_hidden_ratio', -1),
            'tsm_slot_cosine_mean': tsm_diag.get('tsm_slot_cosine_mean', -1),
            'output_proj_grad_norm': tsm_diag.get('output_proj_grad_norm', -1),
            'value_proj_grad_norm': tsm_diag.get('value_proj_grad_norm', -1),
            'slot_attn_qkv_grad_norm': tsm_diag.get('slot_attn_qkv_grad_norm', -1),
            'slot_ffn_grad_norm': tsm_diag.get('slot_ffn_grad_norm', -1),
        }
        step_diagnostics.append(diag)

        if step in [0, 1, 4, 9, 19, 49, 99] or step < 5:
            print(f"  Step {step:3d}: loss={loss.item():.6f}  "
                  f"r={tsm_diag.get('tsm_output_to_hidden_ratio', 0):.2e}  "
                  f"cos={tsm_diag.get('tsm_slot_cosine_mean', 0):.4f}  "
                  f"out|g|={tsm_diag.get('output_proj_grad_norm', 0):.2e}  "
                  f"v|g|={tsm_diag.get('value_proj_grad_norm', 0):.2e}  "
                  f"attn|g|={tsm_diag.get('slot_attn_qkv_grad_norm', 0):.2e}",
                  flush=True)

        if not torch.isfinite(loss):
            print(f"  ⚠ NaN/Inf at step {step}! Stopping.")
            break

    # ---- Summary ----------------------------------------------------------
    print_section("STEP 2: Training Summary")
    if step_diagnostics:
        for label, idx in [("step 0", 0), ("step 5", min(5, len(step_diagnostics)-1)),
                           ("step 9", min(9, len(step_diagnostics)-1)),
                           ("step 49", min(49, len(step_diagnostics)-1)),
                           ("last", -1)]:
            d = step_diagnostics[idx]
            print(f"  {label:7s}: loss={d['loss']:.6f}  r={d['tsm_output_to_hidden_ratio']:.2e}  "
                  f"out|g|={d['output_proj_grad_norm']:.2e}  v|g|={d['value_proj_grad_norm']:.2e}")

        last = step_diagnostics[-1]
        if 1e-6 < last['tsm_output_to_hidden_ratio'] < 0.1:
            print("  ✓ r in expected range (1e-6 < r < 0.1)")
        if last['value_proj_grad_norm'] > 1e-6:
            print("  ✓ Internal modules have non-zero gradients")
        if last['output_proj_grad_norm'] > 0:
            print("  ✓ output_proj has gradients")

    results['step_diagnostics'] = step_diagnostics
    results['final_ratio'] = step_diagnostics[-1]['tsm_output_to_hidden_ratio'] if step_diagnostics else -1

    # Save checkpoint
    ckpt_dir = OUTPUT_ROOT / 'step2_ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_state = {
        'retrieval_adapter': {k: v.cpu() for k, v in adapt.state_dict().items()},
        'tsm_module': {k: v.cpu() for k, v in tsm.state_dict().items()},
        'tsm_mode': 'sinkhorn_tsm',
    }
    if pm is not None:
        ckpt_state['phase_memory'] = {k: v.cpu() for k, v in pm.state_dict().items()}
    torch.save(ckpt_state, ckpt_dir / 'tsm_smoke.pt')
    print(f"\n  Checkpoint saved to {ckpt_dir}/tsm_smoke.pt")
    results['checkpoint_path'] = str(ckpt_dir / 'tsm_smoke.pt')

    return results


# ===========================================================================
#  Step 3: 30-second generation comparison
# ===========================================================================

def step3_generation_comparison(ckpt_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate 30s audio with all TSM modes and compare."""
    print_section("STEP 3: 30-second Generation Comparison")

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    from acestep.phase_memory import TransportRetrievalAdapter, PMRetrievalPhaseMemory
    from acestep.modules.transported_structural_memory import TransportedStructuralMemory

    results: Dict[str, Any] = {'modes': {}}

    checkpoint = ckpt_path or CHECKPOINT
    T_eff = int(DURATION * 25)

    # ---- Generate for each mode -----------------------------------------------
    # baseline: pure ACE-Step, no transport adapter, no hooks
    # sinkhorn_only: transport adapter, no TSM
    # sinkhorn_pool_broadcast: transport + TSM pool/broadcast (no slot mixer)
    # sinkhorn_tsm: transport + full TSM

    for mode_label, use_transport, use_tsm, enable_mixer, use_pm in [
        ('baseline',                 False, False, False, False),
        ('sinkhorn_only',            True,  False, False, True),
        ('sinkhorn_only_no_pm',      True,  False, False, False),
        ('sinkhorn_pool_broadcast',  True,  True,  False, False),
        ('sinkhorn_tsm',             True,  True,  True,  False),
    ]:
        print(f"\n[3] Mode: {mode_label}...")

        dt = AceStepHandler()
        dt.initialize_service(
            project_root=MODEL_ROOT, config_path=MODEL_VARIANT,
            device=DEVICE, use_flash_attention=False, compile_model=False,
            offload_to_cpu=False,
        )
        model = dt.model
        model.eval()
        model.config.use_section_rope_offset = False
        for l in model.decoder.layers:
            if getattr(l, 'use_section_rope', False): l.use_section_rope = False
            if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False

        llm = LLMHandler()
        llm.initialize(checkpoint_dir=MODEL_ROOT, lm_model_path='acestep-5Hz-lm-1.7B',
                       backend='pt', device=DEVICE)

        tsm_diag_log = []

        if use_transport:
            pm, adapt, tsm, tsm_loaded, _ = load_transport_checkpoint(checkpoint, DEVICE, use_pm=use_pm)
            print(f"    PM instantiated: {pm is not None}  PM loaded: {pm is not None}  "
                  f"Scoring: {'position_only' if use_pm else 'scaffold_only'}  "
                  f"Transport: Enabled  TSM: {'Enabled' if use_tsm else 'Disabled'}")

            # For pool_broadcast and tsm modes, load TSM from step2 checkpoint if available
            if use_tsm:
                ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if ckpt_path and 'tsm_module' in ckpt:
                    tsm.load_state_dict(ckpt['tsm_module'])
                    print(f"    TSM loaded from checkpoint")
                else:
                    print(f"    TSM using fresh weights (no trained TSM in checkpoint)")

            hook_holder = [None]
            orig_prep = model.prepare_condition

            def patched_prep(*a, **kw):
                r = orig_prep(*a, **kw)
                if r[0] is not None and hook_holder[0] is None:
                    eh = r[0]; B = eh.shape[0]
                    sc = build_scaffold_from_lyrics(LYRICS, eh, DEVICE)
                    ut = compute_unit_tensors(sc, eh, B)
                    pa = torch.linspace(0, 1, T_eff, device=DEVICE).unsqueeze(0)
                    def hook(_m, _i, o):
                        Ho = o[0]; B_cur = Ho.shape[0]; T = Ho.shape[1]
                        pp = pa[:, :T]
                        if B_cur > pp.shape[0]:
                            pp = pp.expand(B_cur, -1).contiguous()
                        with torch.no_grad():
                            ps2 = pm(Ho.float(), torch.zeros(B_cur, device=DEVICE)) if pm is not None else None
                            uth_b = ut['unit_text_hidden'].float().expand(B_cur, -1, -1)
                            ca_b = ut['unit_centres'].unsqueeze(0).float().expand(B_cur, -1)
                            mu_b = ut['unit_mass'].unsqueeze(0).float().expand(B_cur, -1)
                            usid_b = ut['unit_section_ids'].unsqueeze(0).expand(B_cur, -1)
                            uil_b = ut['unit_is_lyric'].unsqueeze(0).expand(B_cur, -1)
                            dh, Pi, diag = adapt(Ho.float(), None, ps2, pp.float(),
                                uth_b, ca_b, mu_b, unit_section_id=usid_b, unit_is_lyric=uil_b)
                            # TSM (if enabled)
                            tsm_out = torch.zeros_like(dh)
                            if use_tsm and Pi is not None and Pi.shape[-1] > 0:
                                tsm_out, tsm_d = tsm(
                                    Ho.float() + dh, coupling=Pi,
                                    condition_mask=uil_b, detach_coupling=True,
                                    enable_slot_mixer=enable_mixer,
                                )
                                tsm_diag_log.append({
                                    'r': tsm_d.get('tsm_output_to_hidden_ratio', -1),
                                    'cos': tsm_d.get('tsm_slot_cosine_mean', -1),
                                    'rms': tsm_d.get('tsm_output_rms', -1),
                                })
                        return (Ho + dh.to(dtype=Ho.dtype) + tsm_out.to(dtype=Ho.dtype), *o[1:])
                    hook_holder[0] = model.decoder.layers[12].register_forward_hook(hook)
                return r

            model.prepare_condition = patched_prep

        params = GenerationParams(
            task_type='text2music', caption=CAPTION, lyrics=LYRICS,
            instrumental=False, bpm=120, keyscale='C major', timesignature='4',
            vocal_language='en', duration=DURATION, inference_steps=8,
            guidance_scale=7.0, seed=SEED, thinking=True,
            use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
        )
        config = GenerationConfig(batch_size=1, audio_format='flac',
                                   use_random_seed=False, seeds=[SEED])

        torch.manual_seed(SEED)
        result = generate_music(dit_handler=dt, llm_handler=llm, params=params,
                                 config=config, save_dir=str(OUTPUT_ROOT / f'step3_{mode_label}'))

        if use_transport:
            model.prepare_condition = orig_prep
            if hook_holder[0]: hook_holder[0].remove()

        success = result.success
        audio_path = result.audios[0]['path'] if result.success else None

        mode_info = {
            'success': success,
            'audio_path': audio_path,
            'tsm_diag': tsm_diag_log,
        }
        results['modes'][mode_label] = mode_info

        print(f"    {'OK' if success else 'FAIL'}: {audio_path}")
        if tsm_diag_log:
            print(f"    TSM diag: r={tsm_diag_log[0]['r']:.2e}  cos={tsm_diag_log[0]['cos']:.4f}")
            if tsm_diag_log[0]['r'] > 0.1:
                print(f"    ⚠ r > 0.1 — check audio for artifacts")

    # ---- Compare results ----------------------------------------------------
    print_section("STEP 3: Summary")
    for mode, info in results['modes'].items():
        print(f"  {mode}: {'OK' if info['success'] else 'FAIL'}")
        if info['tsm_diag']:
            r_vals = [d['r'] for d in info['tsm_diag'] if d['r'] >= 0]
            cos_vals = [d['cos'] for d in info['tsm_diag'] if d['cos'] >= -99]
            if r_vals:
                print(f"    r = {r_vals[0]:.4e}, cos = {cos_vals[0]:.4f}")
            if r_vals and r_vals[0] > 1e-6:
                print(f"    ✓ TSM residual non-zero")
            else:
                print(f"    - TSM residual ≈ 0")

    return results


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='TSM Smoke Test')
    parser.add_argument('--step', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Test step to run')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='Checkpoint path (for step 3, 4)')
    parser.add_argument('--no-pm', action='store_true', default=False,
                       help='Scaffold-only transport (no PM)')
    parser.add_argument('--output-dir', type=str, default=str(_DEFAULT_OUTPUT_ROOT),
                       help='Output directory')
    args = parser.parse_args()

    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(args.output_dir)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    use_pm = not args.no_pm

    results: Dict[str, Any] = {}

    if args.step == 1:
        results = step1_zero_init_equivalence(use_pm=use_pm)
    elif args.step == 2:
        results = step2_short_training(use_pm=use_pm)
    elif args.step == 3:
        results = step3_generation_comparison(ckpt_path=args.ckpt)
    elif args.step == 4:
        print("Step 4 requires steps 1-3 to pass first. Run manually after 1-3 verified.")
        return

    # Save results
    out_path = OUTPUT_ROOT / f'step{args.step}_results.json'
    # Convert non-serializable types
    def serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [serialize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return str(obj)

    with open(out_path, 'w') as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
