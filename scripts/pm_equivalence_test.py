#!/usr/bin/env python3
"""Quantitative test: PM vs scaffold-only Sinkhorn equivalence."""
import torch, math, sys, os
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ.setdefault('ACESTEP_OFFLINE', '1')
os.environ.setdefault('ACESTEP_MINIMAL_COMPONENTS', '1')

from acestep.handler import AceStepHandler
from acestep.phase_memory import (
    TransportRetrievalAdapter, PMRetrievalPhaseMemory, log_sinkhorn,
    parse_lyrics_to_units, build_duration_scaffold,
)
from acestep.tgca.lyrics_parser import LyricsStructureParser

MODEL_ROOT = '/root/autodl-tmp/Ace-Step1.5'
DEVICE = 'cuda'
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

print('[1] Loading model...')
dt = AceStepHandler()
dt.initialize_service(
    project_root=MODEL_ROOT, config_path='acestep-v15-sft',
    device=DEVICE, use_flash_attention=False, compile_model=False,
    offload_to_cpu=False,
)
model = dt.model; model.eval()
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False
D = model.config.hidden_size

print('[2] Loading PM + TransportAdapter...')
pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256).to(DEVICE).float()
adapt = TransportRetrievalAdapter(
    hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
    sinkhorn_iters=10, transport_sigma=0.18, scoring_mode='position_only',
    use_pm_gate=False, gate_hidden_dim=128,
    write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01,
).to(DEVICE).float()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
pm.load_state_dict(ckpt['phase_memory'])
adapt.load_state_dict(ckpt['retrieval_adapter'])
pm.eval(); adapt.eval()

print('[3] Getting encoder hidden via full prepare_condition pipeline...')
# Use the generate_music pipeline to get encoder_hidden properly
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

llm = LLMHandler()
llm.initialize(
    checkpoint_dir=MODEL_ROOT, lm_model_path='acestep-5Hz-lm-1.7B',
    backend='pt', device=DEVICE,
)

# Collect encoder_hidden and H from a single generate_music call
captured = {'eh': None, 'H': None, 'T_h': None}
hook_holder = [None]
orig_prep = model.prepare_condition

params = GenerationParams(
    task_type='text2music', caption='test', lyrics=LYRICS,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language='en', duration=30, inference_steps=8,
    guidance_scale=7.0, seed=42, thinking=True,
    use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
)

def patched_prep(*a, **kw):
    r = orig_prep(*a, **kw)
    if r[0] is not None and hook_holder[0] is None:
        captured['eh'] = r[0].to(DEVICE).float()
        # Install layer-12 hook to capture H
        def layer_hook(_m, _i, o):
            captured['H'] = o[0].float().detach()
            captured['T_h'] = captured['H'].shape[1]
        hook_holder[0] = model.decoder.layers[12].register_forward_hook(layer_hook)
    return r

model.prepare_condition = patched_prep

config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[42])
torch.manual_seed(42)
_ = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir='/tmp/pm_test')

model.prepare_condition = orig_prep
if hook_holder[0]: hook_holder[0].remove()

eh = captured['eh']; H_raw = captured['H']; T_h = captured['T_h']
assert eh is not None, "Failed to capture encoder_hidden!"
assert H_raw is not None, "Failed to capture H from layer 12!"
# CFG may double batch size — take first half
B_orig, L = eh.shape[0], eh.shape[1]
if H_raw.shape[0] > B_orig:
    print(f'    CFG detected: H batch {H_raw.shape[0]} -> using first {B_orig}')
    H = H_raw[:B_orig]
else:
    H = H_raw
B = H.shape[0]
print(f'    encoder_hidden: {list(eh.shape)}, H: {list(H.shape)}')

parser = LyricsStructureParser()
ps = parser.parse(LYRICS, num_chunks=L)
units, _, debug = parse_lyrics_to_units(
    LYRICS, ps.section_type_ids,
    auto_transition_ratios={
        'intro': 0.0, 'outro': 0.0,
        'chorus_to_verse': 0.0, 'chorus_to_bridge': 0.0,
        'bridge_to_chorus': 0.0,
    },
)
sc = build_duration_scaffold(units, text_len=L, tag_control_mask=debug.get('tag_control_mask'))
sc = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}

U = len(sc['unit_boundaries']) - 1
t2u = sc['token_to_unit']; lym = sc['lyric_mask']
uil = sc['lyric_unit_mask']
ca = (sc['unit_boundaries'][:-1] + sc['unit_boundaries'][1:]) / 2
mu = sc['unit_duration'] / sc['unit_duration'].sum()
usid = sc['unit_section_ids']

# Compute unit_text_hidden
eh_f = eh.float()
uth = []
for uid in range(U):
    tm = (t2u == uid) & lym if uil[uid].item() else (t2u == uid)
    tmb = tm.unsqueeze(0).expand(B, -1)
    pool = eh_f[tmb].view(B, -1, D).mean(dim=1) if tmb.any() else torch.zeros(B, D, device=DEVICE)
    uth.append(pool)
uth = torch.stack(uth, dim=1).float()
print(f'    scaffold: {U} units, lyric={uil.sum().item()}, T_h={T_h}')

# =========================================================================
# P_old: WITH PM (full adapter forward)
# =========================================================================
print('[4] P_old (with PM)...')
pa = torch.linspace(0, 1, T_h, device=DEVICE, dtype=torch.float32).unsqueeze(0).expand(B, -1)
t_dummy = torch.zeros(B, device=DEVICE)
ps2 = pm(H, t_dummy)
with torch.no_grad():
    dh_old, Pi_old, diag_old = adapt(
        H, eh.float(), ps2, pa,
        uth.float().expand(B, -1, -1),
        ca.unsqueeze(0).float().expand(B, -1),
        mu.unsqueeze(0).float().expand(B, -1),
        unit_section_id=usid.unsqueeze(0).expand(B, -1),
        unit_is_lyric=uil.unsqueeze(0).expand(B, -1),
    )

# =========================================================================
# P_new: scaffold-only (base_logit only, no PM/QK residual)
# =========================================================================
print('[6] P_new (scaffold-only, no PM)...')
K = Pi_old.shape[-1]
nu = torch.full((B, T_h,), 1.0 / T_h, device=DEVICE, dtype=torch.float32)
dist = pa[:, :, None] - ca.float().unsqueeze(0)
base_logit = -(dist / 0.18) ** 2
mu_f = mu.unsqueeze(0).float().expand(B, -1)
Pi_new, _ = log_sinkhorn(base_logit, nu, mu_f, iters=10)
Pi_new[:, :, ~uil] = 0

# delta_h_new: v → ctx → out_proj → RMS calibrate
v = adapt.v_mlp(uth.float().expand(B, -1, -1))
v_lyric = torch.where(
    uil.unsqueeze(0).unsqueeze(-1),
    v, adapt.null_value.expand(B, K, -1),
)
ctx_new = torch.matmul(Pi_new, v_lyric) / nu.unsqueeze(-1).clamp(min=1e-10)
raw_res = adapt.out_proj(ctx_new)
raw_rms = torch.sqrt(raw_res.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
unit_res = raw_res / raw_rms
h_rms = torch.sqrt(H.pow(2).mean(dim=-1, keepdim=True)).detach()
dh_new = adapt.write_alpha * h_rms * unit_res

# =========================================================================
# Internal consistency: recompute P_old from raw logits
# =========================================================================
print('[7] Internal consistency check...')
with torch.no_grad():
    pm_state_n = adapt.pm_state_norm(ps2)
    a_feat = torch.stack([pa, pa ** 2, 1.0 - pa], dim=-1)
    audio_coord = adapt.audio_coord_mlp(a_feat)
    q_in = torch.cat([
        pm_state_n, audio_coord,
        torch.zeros(B, T_h, adapt.time_dim, device=DEVICE, dtype=torch.float32),
    ], dim=-1)
    q = torch.nn.functional.normalize(adapt.q_mlp(q_in), dim=-1, p=2)
    u_feat = torch.stack([ca, ca ** 2, 1.0 - ca], dim=-1)
    unit_coord = adapt.unit_coord_mlp(u_feat.float())
    sec_emb = adapt.section_embedding(usid.long()).unsqueeze(0).expand(B, -1, -1)
    k_in = torch.cat([
        uth.float(), unit_coord.unsqueeze(0).expand(B, -1, -1), sec_emb,
    ], dim=-1)
    k = torch.nn.functional.normalize(adapt.k_mlp(k_in), dim=-1, p=2)
    R = torch.matmul(q, k.transpose(-1, -2))
    L_old_recon = base_logit + 0.01 * R
    Pi_recon, _ = log_sinkhorn(L_old_recon, nu, mu_f, iters=10)
    Pi_recon[:, :, ~uil] = 0

pi_recon_l1 = (Pi_old[0] - Pi_recon[0]).abs().sum() / Pi_old[0].sum().clamp(min=1e-10)
print(f'    Pi_old vs Pi_recon L1: {pi_recon_l1.item():.6e} (should be < 1e-5)')

# =========================================================================
# Comparison
# =========================================================================
print()
print('=' * 60)
print('PM vs SCAFFOLD-ONLY EQUIVALENCE')
print('=' * 60)

pi_diff_abs = (Pi_old[0] - Pi_new[0]).abs()
pi_l1_rel = pi_diff_abs.sum() / Pi_old[0].sum().clamp(min=1e-10)
pi_max_diff = pi_diff_abs.max().item()
old_argmax = Pi_old[0].argmax(dim=-1)
new_argmax = Pi_new[0].argmax(dim=-1)
arg_agree = (old_argmax == new_argmax).float().mean().item()

pi1_pass = "PASS" if pi_l1_rel < 1e-3 else "FAIL"
print(f"Pi relative L1:      {pi_l1_rel.item():.6e}  (< 1e-3? {pi1_pass})")
print(f"Pi max absolute diff: {pi_max_diff:.6e}")
arg_pass = "PASS" if arg_agree > 0.99 else "FAIL"
print(f"Pi argmax agreement:  {arg_agree*100:.2f}%  (> 99%? {arg_pass})")

dh_cos = torch.nn.functional.cosine_similarity(
    dh_old[0].flatten(), dh_new[0].flatten(), dim=0,
).item()
dh_l2_rel = (dh_old[0] - dh_new[0]).norm() / dh_old[0].norm().clamp(min=1e-10)
cos_pass = "PASS" if dh_cos > 0.999 else "FAIL"
l2_pass = "PASS" if dh_l2_rel < 1e-3 else "FAIL"
print(f"delta_h cosine:       {dh_cos:.6f}  (> 0.999? {cos_pass})")
print(f"delta_h rel L2:       {dh_l2_rel.item():.6e}  (< 1e-3? {l2_pass})")

print()
print('--- Supporting stats ---')
print(f'R std: {R.std().item():.6f}')
print(f'base_logit std: {base_logit.std().item():.6f}')
print(f'R/base_logit ratio: {R.std().item() / base_logit.std().item():.6e}')
print(f"Pi_old entropy: {diag_old.get('entropy', -1):.4f}")
print(f'dh_old rms: {dh_old.pow(2).mean().sqrt().item():.6e}')
print(f'dh_new rms: {dh_new.pow(2).mean().sqrt().item():.6e}')

all_pass = (
    pi_l1_rel < 1e-3 and arg_agree > 0.99 and
    dh_cos > 0.999 and dh_l2_rel < 1e-3
)
print()
verdict_str = "PASS -- PM can be removed" if all_pass else "NEEDS REVIEW -- PM matters"
print(f"VERDICT: {verdict_str}")
print()

# Save to file for later reference
import json
result = {
    'pi_l1_rel': float(pi_l1_rel.item()),
    'pi_max_diff': float(pi_max_diff),
    'argmax_agreement': float(arg_agree),
    'dh_cosine': float(dh_cos),
    'dh_l2_rel': float(dh_l2_rel.item()),
    'R_std': float(R.std().item()),
    'base_logit_std': float(base_logit.std().item()),
    'R_base_ratio': float(R.std().item() / base_logit.std().item()),
    'all_pass': bool(all_pass),
}
with open('/root/ACE-Step-1.5/outputs/tsm_smoke_test/pm_equivalence.json', 'w') as f:
    json.dump(result, f, indent=2)
print('Saved to outputs/tsm_smoke_test/pm_equivalence.json')
