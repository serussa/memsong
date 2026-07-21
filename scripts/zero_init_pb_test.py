#!/usr/bin/env python3
"""Test: untrained (zero-init) pool_broadcast vs scaffold_only — is training required?"""
import sys, os, json, math
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ.setdefault('ACESTEP_OFFLINE', '1')
os.environ.setdefault('ACESTEP_MINIMAL_COMPONENTS', '1')

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.phase_memory import TransportRetrievalAdapter, parse_lyrics_to_units, build_duration_scaffold
from acestep.modules.transported_structural_memory import TransportedStructuralMemory
from acestep.tgca.lyrics_parser import LyricsStructureParser

MODEL_ROOT = '/root/autodl-tmp/Ace-Step1.5'
DEV = 'cuda'
CKPT = '/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt'
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
OUT = '/root/ACE-Step-1.5/outputs/tsm_smoke_test_no_pm/step3_zero_pb_untrained'

print('=== Untrained pool_broadcast (output_proj = zero) ===')
print(f'PM: False  Scoring: scaffold_only  TSM: pool_broadcast (untrained)')

torch.manual_seed(42)
dt = AceStepHandler()
dt.initialize_service(project_root=MODEL_ROOT, config_path='acestep-v15-sft',
    device=DEV, use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model; model.eval()
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False
D = model.config.hidden_size

adapt = TransportRetrievalAdapter(
    hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
    sinkhorn_iters=10, transport_sigma=0.18, scoring_mode='scaffold_only',
    use_pm_gate=False, gate_hidden_dim=128,
    write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01,
).to(DEV).float()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
adapt.load_state_dict(ckpt['retrieval_adapter']); adapt.eval()

# Fresh TSM, NEVER trained
tsm = TransportedStructuralMemory(
    model_dim=D, memory_dim=256, num_heads=4,
    ffn_dim=512, slot_layers=1, dropout=0.0, detach_coupling=True,
).to(DEV).float()
wmax = tsm.output_proj.weight.abs().max().item()
assert wmax < 1e-10, f'output_proj not zero! max={wmax}'
print(f'output_proj max_abs={wmax:.2e} (verified zero)')
tsm.eval()
tsm_r_log = []

llm = LLMHandler()
llm.initialize(checkpoint_dir=MODEL_ROOT, lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device=DEV)

T_eff = 750
hook_holder = [None]
orig_prep = model.prepare_condition

def patched_prep(*a, **kw):
    r = orig_prep(*a, **kw)
    if r[0] is not None and hook_holder[0] is None:
        eh = r[0]; B = eh.shape[0]; L = eh.shape[1]
        parser = LyricsStructureParser(); ps = parser.parse(LYRICS, num_chunks=L)
        units, _, debug = parse_lyrics_to_units(LYRICS, ps.section_type_ids,
            auto_transition_ratios={'intro':0,'outro':0,'chorus_to_verse':0,'chorus_to_bridge':0,'bridge_to_chorus':0})
        sc = build_duration_scaffold(units, text_len=L, tag_control_mask=debug.get('tag_control_mask'))
        sc = {k: v.to(DEV) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
        U = len(sc['unit_boundaries']) - 1; t2u = sc['token_to_unit']; lym = sc['lyric_mask']
        uil = sc['lyric_unit_mask']; ca = (sc['unit_boundaries'][:-1] + sc['unit_boundaries'][1:]) / 2
        mu = sc['unit_duration'] / sc['unit_duration'].sum(); usid = sc['unit_section_ids']
        eh_f = eh.float(); uth = []
        for uid in range(U):
            tm = (t2u == uid) & lym if uil[uid].item() else (t2u == uid)
            tmb = tm.unsqueeze(0).expand(B, -1)
            uth.append(eh_f[tmb].view(B, -1, D).mean(dim=1) if tmb.any() else torch.zeros(B, D, device=DEV))
        uth = torch.stack(uth, dim=1); pa = torch.linspace(0, 1, T_eff, device=DEV).unsqueeze(0)

        def hook(_m, _i, o):
            Ho = o[0]; B_cur, T = Ho.shape[0], Ho.shape[1]; pp = pa[:, :T]
            if B_cur > pp.shape[0]: pp = pp.expand(B_cur, -1).contiguous()
            with torch.no_grad():
                dh, Pi, _ = adapt(Ho.float(), None, None, pp.float(),
                    uth.float().expand(B_cur, -1, -1),
                    ca.unsqueeze(0).float().expand(B_cur, -1),
                    mu.unsqueeze(0).float().expand(B_cur, -1),
                    unit_section_id=usid.unsqueeze(0).expand(B_cur, -1),
                    unit_is_lyric=uil.unsqueeze(0).expand(B_cur, -1))
                tsm_out, tsm_d = tsm(Ho.float() + dh, coupling=Pi,
                    condition_mask=uil.unsqueeze(0).expand(B_cur, -1),
                    detach_coupling=True, enable_slot_mixer=False)
                tsm_r_log.append({'r': tsm_d.get('tsm_output_to_hidden_ratio', -1),
                                  'rms': tsm_d.get('tsm_output_rms', -1)})
            return (Ho + dh.to(dtype=Ho.dtype) + tsm_out.to(dtype=Ho.dtype), *o[1:])
        hook_holder[0] = model.decoder.layers[12].register_forward_hook(hook)
    return r

model.prepare_condition = patched_prep

params = GenerationParams(
    task_type='text2music', caption='A pop song', lyrics=LYRICS,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language='en', duration=30, inference_steps=8, guidance_scale=7.0,
    seed=42, thinking=True, use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
)
config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[42])
result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir=OUT)

model.prepare_condition = orig_prep
if hook_holder[0]: hook_holder[0].remove()

print(f'Gen: {"OK" if result.success else "FAIL"}')
all_zero = True
for i, d in enumerate(tsm_r_log):
    all_zero = all_zero and (d['rms'] < 1e-10)
    if i < 3 or i >= len(tsm_r_log) - 3:
        print(f'  step {i}: r={d["r"]:.4e}  rms={d["rms"]:.4e}')

print()
if all_zero:
    print('VERDICT: untrained pool_broadcast output IS exactly zero.')
    print('output_proj zero-init guarantees H == H at init.')
    print('Any pool_broadcast benefit REQUIRES training value_proj + output_proj.')
else:
    print('WARNING: output non-zero — check zero init!')

with open('/root/ACE-Step-1.5/outputs/tsm_smoke_test_no_pm/untrained_pb_result.json', 'w') as f:
    json.dump({'all_zero': bool(all_zero), 'tsm_r_log': tsm_r_log}, f, indent=2)
