#!/usr/bin/env python3
"""20-step TSM gradient + weight delta test."""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys
os.environ['ACESTEP_OFFLINE']='1'; os.environ['ACESTEP_MINIMAL_COMPONENTS']='1'
sys.path.insert(0,'/root/ACE-Step-1.5')

from acestep.handler import AceStepHandler
from acestep.phase_memory import TransportRetrievalAdapter, PMRetrievalPhaseMemory, parse_lyrics_to_units, build_duration_scaffold
from acestep.modules.transported_structural_memory import TransportedStructuralMemory, collect_tsm_grad_norms
from acestep.tgca.lyrics_parser import LyricsStructureParser

DEVICE='cuda'; D=2048
dt=AceStepHandler()
dt.initialize_service(project_root='/root/autodl-tmp/Ace-Step1.5', config_path='acestep-v15-sft',
    device=DEVICE, use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model=dt.model; model.eval()
for l in model.decoder.layers:
    if getattr(l,'use_section_rope',False): l.use_section_rope=False
    if getattr(l,'use_phase_memory',False): l.use_phase_memory=False

CKPT='/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt'
pm=PMRetrievalPhaseMemory(dim=D,mem_dim=128,hidden_dim=256).to(DEVICE).float()
adapt=TransportRetrievalAdapter(hidden_dim=D,text_dim=D,pm_dim=256,d_r=256,
    sinkhorn_iters=10,transport_sigma=0.18,scoring_mode='position_only',
    use_pm_gate=False,write_alpha_init=0.005,write_alpha_max=0.01,out_proj_init_std=0.01).to(DEVICE).float()
tsm_module=TransportedStructuralMemory(model_dim=D,memory_dim=256,num_heads=4,
    ffn_dim=512,slot_layers=1,dropout=0.0,detach_coupling=True).to(DEVICE).float()
ckpt=torch.load(CKPT,map_location='cpu',weights_only=True)
pm.load_state_dict(ckpt['phase_memory']); adapt.load_state_dict(ckpt['retrieval_adapter'])

for p in model.parameters(): p.requires_grad=False
for p in pm.parameters(): p.requires_grad=False
for p in adapt.parameters(): p.requires_grad=False

tsm_init = {k: v.clone().cpu() for k, v in tsm_module.state_dict().items()}

lyrics_text = "[Verse]\nThe morning light breaks through\ntest lyric line\n\n[Chorus]\nWe rise up high above the sky\nTogether we fly\n"
parser=LyricsStructureParser()
L=100; section_ids=torch.zeros(L, dtype=torch.long)
ps=parser.parse(lyrics_text, num_chunks=L)
units,_,debug=parse_lyrics_to_units(lyrics_text, ps.section_type_ids,
    auto_transition_ratios={'intro':0.0,'outro':0.0,'chorus_to_verse':0.0,'chorus_to_bridge':0.0,'bridge_to_chorus':0.0})
tcm=debug.get('tag_control_mask')
sc=build_duration_scaffold(units, text_len=L, tag_control_mask=tcm)
sc_dev={k:v.to(DEVICE) if isinstance(v,torch.Tensor) else v for k,v in sc.items()}
lm=sc_dev.get('lyric_unit_mask'); lu=torch.where(lm)[0]; K=len(lu)
c_all=(sc_dev['unit_boundaries'][:-1]+sc_dev['unit_boundaries'][1:])/2
c_u=c_all[lu]; mu_u=sc_dev['unit_duration'][lu]; mu_u/=mu_u.sum()
uid_u=sc_dev['unit_section_ids'][lu] if sc_dev.get('unit_section_ids') is not None else torch.zeros(K,dtype=torch.long,device=DEVICE)
uil_u=lm[lu]
t2u=sc_dev['token_to_unit']; lym=sc_dev['lyric_mask']
encoder_hidden=torch.randn(1,L,D,device=DEVICE,dtype=model.dtype)
eh_f=encoder_hidden.float()
uth_list=[]
for uid in lu.tolist():
    tmask=(t2u==uid)&lym; tmask_b=tmask.unsqueeze(0).expand(1,-1)
    uth_list.append(eh_f[tmask_b].view(1,-1,D).mean(dim=1) if tmask_b.any() else torch.zeros(1,D,device=DEVICE,dtype=torch.float32))
uth=torch.stack(uth_list,dim=1)

opt=torch.optim.AdamW(tsm_module.parameters(), lr=1e-4)
bsz=1; T_lat=375
x0=torch.randn(bsz,T_lat,64,device=DEVICE,dtype=model.dtype)
ctx=torch.zeros(bsz,T_lat,128,device=DEVICE,dtype=model.dtype)
amask=torch.ones(bsz,T_lat,device=DEVICE,dtype=model.dtype)

print(f"K={K}  encoder={list(encoder_hidden.shape)}  latent={list(x0.shape)}")
print("20 steps with pre_hook architecture...\n")

for step in range(20):
    x1=torch.randn_like(x0)
    t_val=torch.rand(1,device=DEVICE).item()
    t_tensor=torch.full((bsz,),t_val,device=DEVICE,dtype=model.dtype)
    xt=t_val*x1+(1.0-t_val)*x0
    xt.requires_grad_(True)

    opt.zero_grad()

    def _ph(module,inputs):
        H=inputs[0]; T_h=H.shape[1]
        pa=torch.linspace(0,1,T_h,device=DEVICE,dtype=torch.float32).unsqueeze(0).expand(bsz,-1)
        ps2=pm(H.float(),t_tensor.float())
        dh=torch.zeros_like(H.float()); tsm_out=torch.zeros_like(H.float()); tsm_d={}
        if K>0:
            _,Pi,_=adapt(H.float(),encoder_hidden.float(),ps2,pa,
                uth.float().expand(bsz,-1,-1),c_u.float().unsqueeze(0).expand(bsz,-1),
                mu_u.float().unsqueeze(0).expand(bsz,-1),
                unit_section_id=uid_u.unsqueeze(0).expand(bsz,-1),
                unit_is_lyric=uil_u.unsqueeze(0).expand(bsz,-1))
            if Pi.shape[-1]>0:
                tsm_out,tsm_d=tsm_module(H.float()+dh,coupling=Pi.detach(),
                    condition_mask=uil_u.unsqueeze(0).expand(bsz,-1),
                    detach_coupling=True,enable_slot_mixer=True)
        _ph._tsm_diag=tsm_d
        return (H.float()+dh+tsm_out).to(dtype=H.dtype), *inputs[1:]

    ph=model.decoder.layers[12].register_forward_pre_hook(_ph)
    out=model.decoder(hidden_states=xt,timestep=t_tensor,timestep_r=t_tensor,
        attention_mask=amask,encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=None,context_latents=ctx,use_cache=False)
    ph.remove()

    loss=F.mse_loss(out[0],x1-x0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tsm_module.parameters(),1.0)
    opt.step()

    gn=collect_tsm_grad_norms(tsm_module)
    r_actual=float(_ph._tsm_diag.get('tsm_output_to_hidden_ratio',-1))
    if step in [0,4,9,19]:
        print(f"step {step:2d}: loss={loss.item():.4f}  r={r_actual:.2e}  "
              f"out_proj={gn['output_proj_grad_norm']:.2e}  "
              f"v_proj={gn['value_proj_grad_norm']:.2e}  "
              f"attn={gn['slot_attn_qkv_grad_norm']:.2e}  "
              f"ffn={gn['slot_ffn_grad_norm']:.2e}")

print()
max_delta=0.0
for k in sorted(tsm_init.keys()):
    d=(tsm_module.state_dict()[k].cpu()-tsm_init[k]).abs().max().item()
    if d>max_delta: max_delta=d
    if d>1e-8:
        print(f"  Δ {k}: max_abs={d:.2e}")
print(f"\nMax weight delta: {max_delta:.2e}")
if max_delta > 1e-10:
    print("✓ WEIGHTS ARE UPDATING — gradient flow confirmed")
else:
    print("✗ NO WEIGHT CHANGE — gradient blocked!")
