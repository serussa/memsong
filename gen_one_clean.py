#!/usr/bin/env python3
"""Single generation: python gen_one_clean.py <baseline|transport_only> <out_dir> <caption> <lyrics>"""
import sys, os, json, pickle, subprocess
from pathlib import Path

sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
os.environ['SIDESTEP_SAFE_ROOT'] = '/root/autodl-tmp'

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

args = sys.argv[1:]
method, out_dir, caption, lyrics = args[0], args[1], args[2], args[3]
seed = int(args[4]); duration = int(args[5])
CKPT = args[6] if len(args) > 6 else '/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt'

MODEL_ROOT = '/root/autodl-tmp/Ace-Step1.5'

dt = AceStepHandler()
dt.initialize_service(project_root=MODEL_ROOT, config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model.eval(); device = next(model.parameters()).device
model.config.use_section_rope_offset = False
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False
llm = LLMHandler()
llm.initialize(checkpoint_dir=MODEL_ROOT, lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda')

if method == 'transport_only':
    from acestep.phase_memory import TransportRetrievalAdapter, PMRetrievalPhaseMemory, parse_lyrics_to_units, build_duration_scaffold
    from acestep.tgca.lyrics_parser import LyricsStructureParser
    D = model.config.hidden_size
    pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True).to(device).float()
    adapt = TransportRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        sinkhorn_iters=10, transport_sigma=0.18, scoring_mode='position_only',
        use_pm_gate=False, gate_hidden_dim=128, write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01).to(device).float()
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
    pm.load_state_dict(ckpt['phase_memory']); adapt.load_state_dict(ckpt['retrieval_adapter'])
    adapt.eval(); pm.eval()
    T_eff = int(duration * 25)
    hook_holder = [None]
    orig = model.prepare_condition
    def patched(*a, **kw):
        r = orig(*a, **kw)
        if r[0] is not None and hook_holder[0] is None:
            eh = r[0]; L = eh.shape[1]; B, D_h = eh.shape[0], eh.shape[-1]
            parser = LyricsStructureParser(); ps = parser.parse(lyrics, num_chunks=L)
            units, _, debug = parse_lyrics_to_units(lyrics, ps.section_type_ids, auto_transition_ratios={})
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
            uth = torch.stack(uth, dim=1); pa = torch.linspace(0, 1, T_eff, device=device).unsqueeze(0)
            def hook(_m, _i, o):
                Ho = o[0]; B, T = Ho.shape[0], Ho.shape[1]; pp = pa[:, :T]
                if B > pp.shape[0]: pp = pp.expand(B, -1).contiguous()
                with torch.no_grad():
                    ps2 = pm(Ho.float(), torch.zeros(B, device=device))
                    uth_b = uth.float().expand(B, -1, -1)
                    ca_b = ca.unsqueeze(0).float().expand(B, -1)
                    mu_b = mu.unsqueeze(0).float().expand(B, -1)
                    usid_b = usid.unsqueeze(0).expand(B, -1)
                    uil_b = uil.unsqueeze(0).expand(B, -1)
                    dh, Pi, diag = adapt(Ho.float(), None, ps2, pp.float(), uth_b, ca_b, mu_b,
                        unit_section_id=usid_b, unit_is_lyric=uil_b)
                return (Ho + dh.to(dtype=Ho.dtype), *o[1:])
            hook_holder[0] = model.decoder.layers[12].register_forward_hook(hook)
        return r
    model.prepare_condition = patched

params = GenerationParams(task_type='text2music', caption=caption, lyrics=lyrics,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language='zh', duration=duration, inference_steps=50, guidance_scale=7.0,
    seed=seed, thinking=True, use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75)
config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[seed])
result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir=out_dir)
if result.success and result.audios:
    print(f'OK:{result.audios[0]["path"]}')
else:
    print(f'FAIL:{result.error}')
    sys.exit(1)
