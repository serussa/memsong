#!/root/miniconda3/envs/musicgen/bin/python
import sys, os
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
os.environ['SIDESTEP_SAFE_ROOT'] = '/root/autodl-tmp'
import torch, pickle
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Parse input: support both old 7-tuple and new dict
_input = pickle.load(sys.stdin.buffer)
if isinstance(_input, dict):
    METHOD = _input.get('method', 'baseline')
    CKPT = _input.get('ckpt', '')
    CAPTION = _input.get('caption', '')
    LYRICS = _input.get('lyrics', '')
    OUT_DIR = _input.get('out_dir', '.')
    SEED = _input.get('seed', 42)
    DURATION = _input.get('duration', 30)
    REALLOCATION_LAMBDA = _input.get('reallocation_lambda', None)
    LATE_ONLY_REALLOCATION = _input.get('late_only_reallocation', False)
    STEP_GATED_REALLOCATION = _input.get('step_gated_reallocation', False)
    STEP_GATED_START = _input.get('step_gated_start', 0.25)
    STEP_GATED_END = _input.get('step_gated_end', 0.65)
    REALLOCATION_LAM_MIN = _input.get('reallocation_lam_min', None)
    TSM_GAIN = _input.get('tsm_gain', 1.0)
    TASK_TYPE = _input.get('task_type', 'text2music')
    USE_LM = _input.get('use_lm', True)
    LM_DIAGNOSE = _input.get('lm_diagnose', False)
    SRC_AUDIO = _input.get('src_audio', None)
    REPAINTING_START = _input.get('repainting_start', 0.0)
    REPAINTING_END = _input.get('repainting_end', -1)
    REPAINT_MODE = _input.get('repaint_mode', 'balanced')
    REPAINT_STRENGTH = _input.get('repaint_strength', 0.5)
else:
    # Legacy tuple
    METHOD, CKPT, CAPTION, LYRICS, OUT_DIR, SEED, DURATION = _input[:7]
    REALLOCATION_LAMBDA = _input[7] if len(_input) > 7 else None
    LATE_ONLY_REALLOCATION = _input[8] if len(_input) > 8 else False
    STEP_GATED_REALLOCATION = _input[9] if len(_input) > 9 else False
    STEP_GATED_START = _input[10] if len(_input) > 10 else 0.25
    STEP_GATED_END = _input[11] if len(_input) > 11 else 0.65
    REALLOCATION_LAM_MIN = _input[12] if len(_input) > 12 else None
    TSM_GAIN = _input[13] if len(_input) > 13 else 1.0
    TASK_TYPE = _input[14] if len(_input) > 14 else 'text2music'
    SRC_AUDIO = _input[15] if len(_input) > 15 else None
    REPAINTING_START = _input[16] if len(_input) > 16 else 0.0
    REPAINTING_END = _input[17] if len(_input) > 17 else -1

dt = AceStepHandler()
dt.initialize_service(project_root='/root/autodl-tmp/Ace-Step1.5', config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model.eval(); device = next(model.parameters()).device
model.config.use_section_rope_offset = False
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False
llm = LLMHandler()
llm.initialize(checkpoint_dir='/root/autodl-tmp/Ace-Step1.5/checkpoints', lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda')

# --- Shared scaffolding setup for both TSM and reallocation ---
needs_scaffold = (METHOD != 'baseline') or (REALLOCATION_LAMBDA is not None and REALLOCATION_LAMBDA > 0)

if needs_scaffold:
    from acestep.phase_memory import parse_lyrics_to_units, build_duration_scaffold
    from acestep.tgca.lyrics_parser import LyricsStructureParser
    # Setup structure holder (populated in patched prepare_condition)
    scaffold_holder = {}
    hook_holder = [None] if METHOD != 'baseline' else [None]
    orig = model.prepare_condition

    def patched(*a, **kw):
        r = orig(*a, **kw)
        if r[0] is not None and hook_holder[0] is None:
            eh = r[0]; L = eh.shape[1]; B, D_h = eh.shape[0], eh.shape[-1]
            clean_lyrics = LYRICS
            import re as _re
            clean_lyrics = _re.sub(r'\[[^\]]+\]\n*', '', clean_lyrics).strip()
            parser = LyricsStructureParser(); ps = parser.parse(clean_lyrics, num_chunks=L)
            units, _, debug = parse_lyrics_to_units(clean_lyrics, ps.section_type_ids, auto_transition_ratios={})
            sc = build_duration_scaffold(units, text_len=L, tag_control_mask=debug.get('tag_control_mask'))
            sc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
            U = len(sc['unit_boundaries']) - 1
            ca = (sc['unit_boundaries'][:-1] + sc['unit_boundaries'][1:]) / 2
            mu = sc['unit_duration'] / sc['unit_duration'].sum()
            usid = sc['unit_section_ids']; uil = sc['lyric_unit_mask']; t2u = sc['token_to_unit']; lym = sc['lyric_mask']
            eh_f = eh.float(); uth = []
            for uid in range(U):
                tm = (t2u == uid) & lym if uil[uid].item() else (t2u == uid)
                tmb = tm.unsqueeze(0).expand(B, -1)
                uth.append(eh_f[tmb].view(B, -1, D_h).mean(dim=1) if tmb.any() else torch.zeros(B, D_h, device=device))
            try:
                uth = torch.stack(uth, dim=1)
            except RuntimeError:
                for i, t in enumerate(uth):
                    print(f"DEBUG uth[{i}]: shape={t.shape} dtype={t.dtype}", file=sys.stderr)
                raise
            scaffold_holder.update(scaffold=sc, t2u=t2u, lym=lym, uil=uil, uth=uth,
                                   ca=ca, mu=mu, usid=usid, B=B, L=L, D_h=D_h, U=U)

            if METHOD != 'baseline':
                # Setup TSM / transport + hook
                from acestep.phase_memory import TransportRetrievalAdapter
                from acestep.modules.transported_structural_memory import TransportedStructuralMemory
                D = model.config.hidden_size
                use_pm = (METHOD == 'transport_only')
                scoring = 'position_only' if use_pm else 'scaffold_only'
                adapt = TransportRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
                    transport_mode='sinkhorn', sinkhorn_iters=10, transport_sigma=0.18,
                    transport_qk_scale=0, scoring_mode=scoring, use_pm_gate=False, gate_hidden_dim=128,
                    write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01).to(device).float()
                tsm = TransportedStructuralMemory(model_dim=D, memory_dim=256, num_heads=4,
                    ffn_dim=512, slot_layers=1, dropout=0.0, detach_coupling=True).to(device).float()
                tsm_loaded = False
                ckpt_data = torch.load(CKPT, map_location='cpu', weights_only=True)
                adapt.load_state_dict(ckpt_data['retrieval_adapter'], strict=False)
                if 'tsm_module' in ckpt_data:
                    tsm.load_state_dict(ckpt_data['tsm_module'], strict=False)
                    tsm_loaded = True; tsm_mode = ckpt_data.get('tsm_mode', 'sinkhorn_tsm')
                if use_pm:
                    from acestep.phase_memory import PMRetrievalPhaseMemory
                    pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256,
                        normalize_internal_state=True).to(device).float()
                    pm.load_state_dict(ckpt_data['phase_memory'], strict=False); pm.eval()
                else:
                    pm = None
                adapt.eval(); tsm.eval()

                def hook(_m, _i, o):
                    Ho = o[0]; Bt, T = Ho.shape[0], Ho.shape[1]
                    # Build position encoding from actual latent frames T (not heuristic duration)
                    pp = torch.linspace(0, 1, T, device=Ho.device, dtype=torch.float32).unsqueeze(0)
                    if Bt > pp.shape[0]: pp = pp.expand(Bt, -1).contiguous()
                    s = scaffold_holder
                    uth_b = s['uth'].expand(Bt, -1, -1) if Bt > s['B'] else s['uth']
                    ca_b = s['ca'].unsqueeze(0).expand(Bt, -1) if Bt > 1 else s['ca'].unsqueeze(0)
                    mu_b = s['mu'].unsqueeze(0).expand(Bt, -1) if Bt > 1 else s['mu'].unsqueeze(0)
                    usid_b = s['usid'].unsqueeze(0).expand(Bt, -1) if Bt > 1 else s['usid'].unsqueeze(0)
                    uil_b = s['uil'].unsqueeze(0).expand(Bt, -1) if Bt > 1 else s['uil'].unsqueeze(0)
                    with torch.no_grad():
                        if pm is not None:
                            ps2 = pm(Ho.float(), torch.zeros(Bt, device=device))
                        else:
                            ps2 = torch.zeros(Bt, T, 256, device=device, dtype=torch.float32)
                        dh, Pi, diag = adapt(Ho.float(), None, ps2, pp.float(), uth_b.float(),
                            ca_b.float(), mu_b.float(), usid_b, unit_is_lyric=uil_b)
                        tsm_out = torch.zeros_like(dh)
                        if tsm_loaded and Pi is not None and Pi.shape[-1] > 0:
                            tsm_out, _ = tsm(Ho.float() + dh, coupling=Pi,
                                condition_mask=uil_b, detach_coupling=True,
                                enable_slot_mixer=(tsm_mode != 'sinkhorn_pool_broadcast'))
                    # TSM residual gain (print diagnostics once per inference)
                    if not hasattr(hook, '_tsm_done'):
                        hook._tsm_done = False
                    tsm_gain_local = TSM_GAIN
                    if tsm_loaded and not hook._tsm_done and tsm_out.abs().sum() > 0:
                        hook._tsm_done = True
                        tsm_rms = tsm_out.float().norm().item() / (tsm_out.numel() ** 0.5)
                        h_rms = Ho.float().norm().item() / (Ho.numel() ** 0.5)
                        print(f"[tsm_gain={tsm_gain_local}] delta_rms={tsm_rms:.6f} hidden_rms={h_rms:.6f} "
                              f"ratio={tsm_rms/(h_rms+1e-12):.4f}", flush=True)
                    return (Ho + dh.to(dtype=Ho.dtype) + (tsm_gain_local * tsm_out).to(dtype=Ho.dtype), *o[1:])
                hook_holder[0] = model.decoder.layers[12].register_forward_hook(hook)

            # ----- Sinkhorn-guided marginal reallocation (inference-only, no TSM needed) -----
            if REALLOCATION_LAMBDA is not None and REALLOCATION_LAMBDA > 0:
                from acestep.modules.marginal_reallocation import apply_reallocation_to_model
                _realloc_lam = 0 if LATE_ONLY_REALLOCATION else REALLOCATION_LAMBDA
                _realloc_late_max = REALLOCATION_LAMBDA if LATE_ONLY_REALLOCATION else None
                _realloc_rd = apply_reallocation_to_model(
                    model, device, scaffold_holder['t2u'], scaffold_holder['lym'],
                    scaffold_holder['uil'], scaffold_holder['uth'],
                    lam=_realloc_lam, layers=(8, 16),
                    late_only_lambda_max=_realloc_late_max,
                    step_gated=STEP_GATED_REALLOCATION,
                    step_start=STEP_GATED_START, step_end=STEP_GATED_END,
                    num_steps=50,  # inference_steps hardcoded below
                    lam_min=REALLOCATION_LAM_MIN,
                )
                model._realloc_rd = _realloc_rd
                model._realloc_rd = _realloc_rd

        return r
    model.prepare_condition = patched

has_cjk = any('一' <= c <= '鿿' or '㐀' <= c <= '䶿' for c in LYRICS)
vocal_lang = 'zh' if has_cjk else 'en'
params_kw = dict(
    task_type=TASK_TYPE, caption=CAPTION, lyrics=LYRICS,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language=vocal_lang, duration=DURATION, inference_steps=50, guidance_scale=7.0,
    seed=SEED,
    thinking=USE_LM, use_cot_metas=USE_LM, use_cot_caption=USE_LM, lm_temperature=0.75)
if TASK_TYPE in ('repaint', 'cover', 'extract'):
    params_kw['thinking'] = False
    params_kw['use_cot_metas'] = False
    params_kw['use_cot_caption'] = False
    params_kw['repainting_start'] = REPAINTING_START
    params_kw['repainting_end'] = REPAINTING_END
    params_kw['repaint_mode'] = REPAINT_MODE
    params_kw['repaint_strength'] = REPAINT_STRENGTH
    params_kw['src_audio'] = SRC_AUDIO
params = GenerationParams(**params_kw)
config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[SEED])

# ---- LM Diagnostic: check if audio_codes contain recoverable lyrics ----
_lm_diag_codes = None
if LM_DIAGNOSE and USE_LM:
    print("[lm_diag] Getting audio codes from LLM...", flush=True)
    _lm_res = llm.generate_with_stop_condition(
        caption=CAPTION, lyrics=LYRICS,
        infer_type="llm_dit",
        temperature=0.75, cfg_scale=2.0,
        target_duration=float(DURATION),
        use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
        use_constrained_decoding=True,
        batch_size=1, seeds=[SEED],
    )
    _raw = _lm_res.get("audio_codes", "")
    if isinstance(_raw, list) and len(_raw) > 0:
        _lm_diag_codes = _raw[0]
    elif isinstance(_raw, str) and _raw.strip():
        _lm_diag_codes = _raw

    if _lm_diag_codes:
        n_codes = len(_lm_diag_codes.split("<|audio_code_")) - 1
        print(f"[lm_diag] Got {n_codes} audio codes. Re-inserting, skipping internal LM.", flush=True)
        params_kw["audio_codes"] = _lm_diag_codes
        params_kw["thinking"] = False
        params_kw["use_cot_metas"] = False
        params_kw["use_cot_caption"] = False
        params = GenerationParams(**params_kw)
    else:
        print("[lm_diag] WARNING: no audio codes from LLM", flush=True)

result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir=OUT_DIR)

# Print diagnostics if reallocation was applied
_rd = getattr(model, '_realloc_rd', None)
if _rd is not None:
    from acestep.modules.marginal_reallocation import print_diagnostics
    print_diagnostics(_rd)
    from acestep.modules.marginal_reallocation import restore_eager_attention
    restore_eager_attention()

print('OK:' + (result.audios[0]['path'] if result.audios else 'no_audio'))

# ---- LM Diagnostic: reverse transcription ----
if LM_DIAGNOSE and _lm_diag_codes:
    import difflib
    _rev_prompt = (
        "Below is the audio semantic code sequence of a song. "
        "Read these codes and output the exact lyrics being sung.\n\n"
        "Audio codes:\n" + _lm_diag_codes + "\n\nLyrics:"
    )
    _chat = [{"role": "system", "content": "You transcribe lyrics from audio codes."},
             {"role": "user", "content": _rev_prompt}]
    _rev_input = llm.llm_tokenizer.apply_chat_template(
        _chat, tokenize=False, add_generation_prompt=True)
    try:
        _rev_ids = llm.llm_tokenizer(_rev_input, return_tensors="pt").to(llm.device)
        _rev_out = llm.llm.generate(
            **_rev_ids, max_new_tokens=512, temperature=0.1, do_sample=False,
            pad_token_id=llm.llm_tokenizer.eos_token_id)
        _rev_text = llm.llm_tokenizer.decode(
            _rev_out[0][_rev_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        _rev_text = ""
        print(f"[lm_diag] Reverse transcription failed: {e}", flush=True)

    import re as _re2
    _clean_lyrics = _re2.sub(r'\[[^\]]+\]\n*', '', LYRICS).strip()
    if _rev_text:
        _matcher = difflib.SequenceMatcher(None, _clean_lyrics, _rev_text)
        _S = _I = _D = 0
        for _tag, _i1, _i2, _j1, _j2 in _matcher.get_opcodes():
            if _tag == 'replace': _S += max(_i2 - _i1, _j2 - _j1)
            elif _tag == 'delete': _D += (_i2 - _i1)
            elif _tag == 'insert': _I += (_j2 - _j1)
        _N = len(_clean_lyrics)
        _cer = (_S + _D + _I) / max(_N, 1)
        print(f"\n[lm_diag] {'='*50}", flush=True)
        print(f"[lm_diag] Codes--text CER: {_cer:.4f}  (S={_S} D={_D} I={_I} N={_N})", flush=True)
        print(f"[lm_diag] Recovered: {_rev_text[:200]}", flush=True)
        print(f"[lm_diag] Original:  {_clean_lyrics[:200]}", flush=True)
        print(f"[lm_diag] {'='*50}", flush=True)
    else:
        print(f"[lm_diag] No reverse transcription output", flush=True)
