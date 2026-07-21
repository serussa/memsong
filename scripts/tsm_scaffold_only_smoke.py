#!/usr/bin/env python3
"""30s smoke test: old PM vs scaffold-only vs scaffold-only+zeroTSM."""
import torch, torch.nn as nn, os, sys, json
os.environ['ACESTEP_OFFLINE']='1'; os.environ['ACESTEP_MINIMAL_COMPONENTS']='1'
sys.path.insert(0,'/root/ACE-Step-1.5')

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.phase_memory import TransportRetrievalAdapter, parse_lyrics_to_units, build_duration_scaffold
from acestep.modules.transported_structural_memory import TransportedStructuralMemory
from acestep.tgca.lyrics_parser import LyricsStructureParser

DEVICE='cuda'; DURATION=30; SEED=42; T_eff=int(DURATION*25)
MODEL_ROOT='/root/autodl-tmp/Ace-Step1.5'
OUT_ROOT='/root/ACE-Step-1.5/outputs/tsm_scaffold_only_smoke'
CKPT_PATH='/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt'
CAPTION='A pop song with emotional female vocals and clear structure.'
LYRICS="""[Verse]
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

os.makedirs(OUT_ROOT,exist_ok=True)

def make_adapter(mode='scaffold_only'):
    return TransportRetrievalAdapter(
        hidden_dim=2048, text_dim=2048, pm_dim=256, d_r=256,
        sinkhorn_iters=10, transport_sigma=0.18,
        scoring_mode=mode,
        use_pm_gate=False, write_alpha_init=0.005, write_alpha_max=0.01,
    ).to(DEVICE).float()

def make_tsm():
    return TransportedStructuralMemory(
        model_dim=2048, memory_dim=256, num_heads=4,
        ffn_dim=512, slot_layers=1, dropout=0.0, detach_coupling=True,
    ).to(DEVICE).float()

def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def run_generation(label, setup_fn):
    """Run one 30s generation. setup_fn receives (model) and returns cleanup_fn."""
    dt=AceStepHandler()
    dt.initialize_service(project_root=MODEL_ROOT,config_path='acestep-v15-sft',
        device=DEVICE,use_flash_attention=False,compile_model=False,offload_to_cpu=False)
    model=dt.model; model.eval()
    model.config.use_section_rope_offset=False
    for l in model.decoder.layers:
        if getattr(l,'use_section_rope',False): l.use_section_rope=False
        if getattr(l,'use_phase_memory',False): l.use_phase_memory=False

    llm=LLMHandler()
    llm.initialize(checkpoint_dir=MODEL_ROOT,lm_model_path='acestep-5Hz-lm-1.7B',backend='pt',device=DEVICE)
    cleanup = setup_fn(model)

    params=GenerationParams(task_type='text2music',caption=CAPTION,lyrics=LYRICS,
        instrumental=False,bpm=120,keyscale='C major',timesignature='4',
        vocal_language='en',duration=DURATION,inference_steps=8,guidance_scale=7.0,
        seed=SEED,thinking=True,use_cot_metas=True,use_cot_caption=True,lm_temperature=0.75)
    config=GenerationConfig(batch_size=1,audio_format='flac',use_random_seed=False,seeds=[SEED])
    torch.manual_seed(SEED)
    result=generate_music(dit_handler=dt,llm_handler=llm,params=params,config=config,
        save_dir=f'{OUT_ROOT}/{label}')
    cleanup()
    return result


# ============================================================
# Mode A: Old PM-based sinkhorn_only
# ============================================================
print("="*60)
print("A: Old PM-based sinkhorn_only")
print("="*60)

from acestep.phase_memory import PMRetrievalPhaseMemory
D=2048
old_pm = PMRetrievalPhaseMemory(dim=D,mem_dim=128,hidden_dim=256).to(DEVICE).float()
old_ad = make_adapter('position_only')
old_ckpt = load_ckpt(CKPT_PATH)
old_pm.load_state_dict(old_ckpt['phase_memory'])
old_ad.load_state_dict(old_ckpt['retrieval_adapter'])
print("PM instantiated: True")
print("PM checkpoint loaded: True")
print("Transport scoring: position_only (PM-based QK)")
print("TSM: Disabled")

hook_a=[None]
orig_a=run_generation.__wrapped__ if hasattr(run_generation,'__wrapped__') else None

def a_setup(model):
    orig_prep=model.prepare_condition
    def patched(*a,**kw):
        r=orig_prep(*a,**kw)
        if r[0] is not None and hook_a[0] is None:
            eh=r[0]; L=eh.shape[1]; B=eh.shape[0]
            parser=LyricsStructureParser(); ps=parser.parse(LYRICS,num_chunks=L)
            units,_,db=parse_lyrics_to_units(LYRICS,ps.section_type_ids,auto_transition_ratios={})
            sc=build_duration_scaffold(units,text_len=L,tag_control_mask=db.get('tag_control_mask'))
            sc={k:v.to(DEVICE) if isinstance(v,torch.Tensor) else v for k,v in sc.items()}
            U=len(sc['unit_boundaries'])-1
            ca=(sc['unit_boundaries'][:-1]+sc['unit_boundaries'][1:])/2
            mu=sc['unit_duration']/sc['unit_duration'].sum()
            us=sc['unit_section_ids']; ui=sc['lyric_unit_mask']; t2=sc['token_to_unit']; ly=sc['lyric_mask']
            eh_f=eh.float(); ut=[]
            for uid in range(U):
                tm=(t2==uid)&ly if ui[uid].item() else (t2==uid)
                tmb=tm.unsqueeze(0).expand(B,-1)
                ut.append(eh_f[tmb].view(B,-1,D).mean(dim=1) if tmb.any() else torch.zeros(B,D,device=DEVICE))
            ut=torch.stack(ut,dim=1); pa=torch.linspace(0,1,T_eff,device=DEVICE).unsqueeze(0)
            def hk(_m,_i,o):
                Ho=o[0]; Bc,Tc=Ho.shape[0],Ho.shape[1]; pp=pa[:,:Tc]
                if Bc>pp.shape[0]: pp=pp.expand(Bc,-1).contiguous()
                with torch.no_grad():
                    ps2=old_pm(Ho.float(),torch.zeros(Bc,device=DEVICE))
                    dh,Pi,_=old_ad(Ho.float(),None,ps2,pp.float(),
                        ut.float().expand(Bc,-1,-1),ca.float().unsqueeze(0).expand(Bc,-1),
                        mu.float().unsqueeze(0).expand(Bc,-1),us.unsqueeze(0).expand(Bc,-1),
                        ui.unsqueeze(0).expand(Bc,-1))
                return (Ho+dh.to(dtype=Ho.dtype),*o[1:])
            hook_a[0]=model.decoder.layers[12].register_forward_hook(hk)
        return r
    model.prepare_condition=patched
    def cleanup():
        model.prepare_condition=orig_prep
        if hook_a[0]: hook_a[0].remove()
    return cleanup

r_a=run_generation('a_pm_sinkhorn_only',a_setup)
print(f"  {'OK' if r_a.success else 'FAIL'}: {r_a.audios[0]['path'] if r_a.success else r_a.error}")

# ============================================================
# Mode B: Scaffold-only sinkhorn (NO PM)
# ============================================================
print("\n"+"="*60)
print("B: Scaffold-only sinkhorn (NO PM)")
print("="*60)

b_ad = make_adapter('scaffold_only')
b_ad.load_state_dict(old_ckpt['retrieval_adapter'])
print("PM instantiated: False")
print("PM checkpoint loaded: False")
print("Transport scoring: scaffold_only")
print("Transport retrieval: Enabled")
print("TSM: Disabled")

hook_b=[None]
def b_setup(model):
    orig_prep=model.prepare_condition
    def patched(*a,**kw):
        r=orig_prep(*a,**kw)
        if r[0] is not None and hook_b[0] is None:
            eh=r[0]; L=eh.shape[1]; B=eh.shape[0]
            parser=LyricsStructureParser(); ps=parser.parse(LYRICS,num_chunks=L)
            units,_,db=parse_lyrics_to_units(LYRICS,ps.section_type_ids,auto_transition_ratios={})
            sc=build_duration_scaffold(units,text_len=L,tag_control_mask=db.get('tag_control_mask'))
            sc={k:v.to(DEVICE) for k,v in sc.items()}
            U=len(sc['unit_boundaries'])-1
            ca=(sc['unit_boundaries'][:-1]+sc['unit_boundaries'][1:])/2
            mu=sc['unit_duration']/sc['unit_duration'].sum()
            us=sc['unit_section_ids']; ui=sc['lyric_unit_mask']; t2=sc['token_to_unit']; ly=sc['lyric_mask']
            eh_f=eh.float(); ut=[]
            for uid in range(U):
                tm=(t2==uid)&ly if ui[uid].item() else (t2==uid)
                tmb=tm.unsqueeze(0).expand(B,-1)
                ut.append(eh_f[tmb].view(B,-1,D).mean(dim=1) if tmb.any() else torch.zeros(B,D,device=DEVICE))
            ut=torch.stack(ut,dim=1); pa=torch.linspace(0,1,T_eff,device=DEVICE).unsqueeze(0)
            def hk(_m,_i,o):
                Ho=o[0]; Bc,Tc=Ho.shape[0],Ho.shape[1]; pp=pa[:,:Tc]
                if Bc>pp.shape[0]: pp=pp.expand(Bc,-1).contiguous()
                with torch.no_grad():
                    dh,Pi,_=b_ad(Ho.float(),None,None,pp.float(),  # pm_state=None
                        ut.float().expand(Bc,-1,-1),ca.float().unsqueeze(0).expand(Bc,-1),
                        mu.float().unsqueeze(0).expand(Bc,-1),us.unsqueeze(0).expand(Bc,-1),ui.unsqueeze(0).expand(Bc,-1))
                return (Ho+dh.to(dtype=Ho.dtype),*o[1:])
            hook_b[0]=model.decoder.layers[12].register_forward_hook(hk)
        return r
    model.prepare_condition=patched
    def cleanup():
        model.prepare_condition=orig_prep
        if hook_b[0]: hook_b[0].remove()
    return cleanup

r_b=run_generation('b_scaffold_only',b_setup)
print(f"  {'OK' if r_b.success else 'FAIL'}: {r_b.audios[0]['path'] if r_b.success else r_b.error}")

# ============================================================
# Mode C: Scaffold-only + zero-init TSM
# ============================================================
print("\n"+"="*60)
print("C: Scaffold-only + zero-init TSM")
print("="*60)

c_ad = make_adapter('scaffold_only')
c_tsm = make_tsm()
# Zero init
nn.init.zeros_(c_tsm.output_proj.weight)
if c_tsm.output_proj.bias is not None: nn.init.zeros_(c_tsm.output_proj.bias)
c_ad.load_state_dict(b_ad.state_dict())
print("PM instantiated: False")
print("PM checkpoint loaded: False")
print("Transport scoring: scaffold_only")
print("Transport retrieval: Enabled")
print("TSM: Enabled (zero-init)")
print(f"Verified output_proj max_abs: {c_tsm.output_proj.weight.abs().max().item():.2e}")

hook_c=[None]; tsm_call=[0]; tsm_r_log=[]
def c_setup(model):
    orig_prep=model.prepare_condition
    def patched(*a,**kw):
        r=orig_prep(*a,**kw)
        if r[0] is not None and hook_c[0] is None:
            eh=r[0]; L=eh.shape[1]; B=eh.shape[0]
            parser=LyricsStructureParser(); ps=parser.parse(LYRICS,num_chunks=L)
            units,_,db=parse_lyrics_to_units(LYRICS,ps.section_type_ids,auto_transition_ratios={})
            sc=build_duration_scaffold(units,text_len=L,tag_control_mask=db.get('tag_control_mask'))
            sc={k:v.to(DEVICE) for k,v in sc.items()}
            U=len(sc['unit_boundaries'])-1
            ca=(sc['unit_boundaries'][:-1]+sc['unit_boundaries'][1:])/2
            mu=sc['unit_duration']/sc['unit_duration'].sum()
            us=sc['unit_section_ids']; ui=sc['lyric_unit_mask']; t2=sc['token_to_unit']; ly=sc['lyric_mask']
            eh_f=eh.float(); ut=[]
            for uid in range(U):
                tm=(t2==uid)&ly if ui[uid].item() else (t2==uid)
                tmb=tm.unsqueeze(0).expand(B,-1)
                ut.append(eh_f[tmb].view(B,-1,D).mean(dim=1) if tmb.any() else torch.zeros(B,D,device=DEVICE))
            ut=torch.stack(ut,dim=1); pa=torch.linspace(0,1,T_eff,device=DEVICE).unsqueeze(0)
            def hk(_m,_i,o):
                Ho=o[0]; Bc,Tc=Ho.shape[0],Ho.shape[1]; pp=pa[:,:Tc]
                if Bc>pp.shape[0]: pp=pp.expand(Bc,-1).contiguous()
                with torch.no_grad():
                    dh,Pi,_=c_ad(Ho.float(),None,None,pp.float(),
                        ut.float().expand(Bc,-1,-1),ca.float().unsqueeze(0).expand(Bc,-1),
                        mu.float().unsqueeze(0).expand(Bc,-1),us.unsqueeze(0).expand(Bc,-1),ui.unsqueeze(0).expand(Bc,-1))
                    tsm_out,t_d=c_tsm(Ho.float()+dh,coupling=Pi,
                        condition_mask=ui.unsqueeze(0).expand(Bc,-1),
                        detach_coupling=True,enable_slot_mixer=True)
                    tsm_call[0]+=1; tsm_r_log.append(t_d.get('tsm_output_rms',-1))
                return (Ho+dh.to(dtype=Ho.dtype)+tsm_out.to(dtype=Ho.dtype),*o[1:])
            hook_c[0]=model.decoder.layers[12].register_forward_hook(hk)
        return r
    model.prepare_condition=patched
    def cleanup():
        model.prepare_condition=orig_prep
        if hook_c[0]: hook_c[0].remove()
    return cleanup

r_c=run_generation('c_scaffold_tsm_zero',c_setup)
print(f"  {'OK' if r_c.success else 'FAIL'}: {r_c.audios[0]['path'] if r_c.success else r_c.error}")

# ============================================================
# Summary
# ============================================================
print("\n"+"="*60)
print("SUMMARY")
print("="*60)
for label,r in [('A: PM sinkhorn_only',r_a),('B: scaffold-only',r_b),('C: scaffold-only+zeroTSM',r_c)]:
    status='OK' if r.success else 'FAIL'
    path=r.audios[0]['path'] if r.success else r.error
    print(f"  {label}: {status}  {path}")

print(f"\nTSM forward calls: {tsm_call[0]}")
if tsm_r_log:
    all_zero=all(v<1e-10 for v in tsm_r_log if v>=0)
    print(f"TSM output RMS: {[f'{v:.2e}' for v in tsm_r_log[:3]]}{'...' if len(tsm_r_log)>3 else ''}")
    print(f"{'✓ ALL zero — zero init verified' if all_zero else '✗ NON-ZERO — check init!'}")

# Save
json.dump({
    'a':{'success':r_a.success,'path':r_a.audios[0]['path'] if r_a.success else None},
    'b':{'success':r_b.success,'path':r_b.audios[0]['path'] if r_b.success else None},
    'c':{'success':r_c.success,'path':r_c.audios[0]['path'] if r_c.success else None},
    'tsm_calls':tsm_call[0],'tsm_r_log':[float(v) for v in tsm_r_log]
},open(f'{OUT_ROOT}/summary.json','w'),indent=2)
print(f"\nSaved: {OUT_ROOT}/summary.json")
