#!/usr/bin/env python3
"""快捷诊断：hook 注入的 signal 经过后续 12 层后还剩多少。"""
import os, sys, torch
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'

from acestep.handler import AceStepHandler
from acestep.phase_memory import PMRetrievalPhaseMemory, LyricRetrievalAdapter

CKPT = "/root/autodl-tmp/pm_retrieval_1epoch_v3/checkpoints/epoch_2_loss_1.3316/pm_retrieval.pt"
MODEL_ROOT = "/root/autodl-tmp/Ace-Step1.5"
N_LAYERS = 24; LAYER = 12

dt = AceStepHandler()
dt.initialize_service(project_root=MODEL_ROOT, config_path="acestep-v15-sft",
    device="cuda", use_flash_attention=False, compile_model=False)
model = dt.model.eval().cuda()
D = model.config.hidden_size

pm = PMRetrievalPhaseMemory(dim=D).cuda().float().eval()
adapt = LyricRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=64,
    residual_scale=0.1, gamma_init=0.1, use_adapter_scaffold_prior=True).cuda().float().eval()
sd = torch.load(CKPT, map_location="cuda", weights_only=False)
pm.load_state_dict(sd["phase_memory"], strict=False)
adapt.load_state_dict(sd["retrieval_adapter"], strict=False)
print(f"gamma_r={adapt.gamma_r.item():.5f} qk_score_scale={adapt.qk_score_scale.item():.4f}")

# ── Synthetic forward (single step, real-like shapes) ──
# NOTE: decoder takes noisy latents [B, T, 64] + context_latents [B, T, 128]
# After concat → [B, T, 192] → proj_in Conv1d(192→2048, kernel=2, stride=2)
B, T, T_t = 1, 256, 200  # T must be even for proj_in stride=2
model.decoder.eval()
xt = torch.randn(B, T, 64, device="cuda", dtype=torch.bfloat16)
t = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
attn_mask = torch.ones(B, T, device="cuda", dtype=torch.bfloat16)
enc = torch.randn(B, T_t, D, device="cuda", dtype=torch.bfloat16)
enc_attn = torch.ones(B, T_t, device="cuda", dtype=torch.bool)
ctx = torch.zeros(B, T, 128, device="cuda", dtype=torch.bfloat16)

# Collect hidden norms for clean forward
h_clean = {}
def make_capture_clean(d):
    def hook(m, i, o):
        pass  # We'll do this differently
    return hook

# Get layer 12 hidden AND final output in one forward
h12 = None
def cap_h12(m, i, o):
    global h12
    h12 = o[0].float().clone()
h12_handle = model.decoder.layers[LAYER].register_forward_hook(cap_h12)
with torch.no_grad():
    out_clean = model.decoder(
        hidden_states=xt, timestep=t, timestep_r=t,
        attention_mask=attn_mask, encoder_hidden_states=enc,
        encoder_attention_mask=enc_attn, context_latents=ctx,
        use_cache=False,
    )
h12_handle.remove()
hidden_clean_0 = out_clean[0].float()

print(f"\nBaseline:")
print(f"  H_layer12 norm    = {h12.norm(dim=-1).mean().item():.4f}")
print(f"  Final output norm = {hidden_clean_0.norm(dim=-1).mean().item():.4f}")

# ── Forward with PM injection ──
h12_modified = None
ret_res_norm = None
h12_native_T = None

def inject_and_capture(m, i, o):
    global h12_modified, ret_res_norm, h12_native_T
    h = o[0].float()  # [B, T_h, 2048] — T_h = T/2 after proj_in
    B_h, T_h, _ = h.shape
    h12_native_T = T_h

    # Build scaffold aligned to T_h
    p_a = torch.linspace(0, 1, T_h, device=h.device, dtype=torch.float32).unsqueeze(0).expand(B_h, -1)
    c_t = torch.linspace(0, 1, T_t, device=h.device, dtype=torch.float32).unsqueeze(0).expand(B_h, -1)
    sid = torch.zeros(B_h, T_t, dtype=torch.long, device=h.device)
    ttid = torch.zeros(B_h, T_t, dtype=torch.long, device=h.device)
    temb = torch.zeros(B_h, 128, device=h.device, dtype=torch.float32)

    ps = pm(h, None)
    ret_res, _, _ = adapt(
        h, enc.float(), ps, p_a, c_t, sid, ttid, temb,
        use_scaffold_prior=True,
    )
    ret_res_norm_val = ret_res.norm(dim=-1).mean().item()
    h_new = h + adapt.gamma_r * ret_res
    h12_modified = h_new
    ret_res_norm = ret_res_norm_val
    return (h_new.to(dtype=o[0].dtype), *o[1:])

h12_handle2 = model.decoder.layers[LAYER].register_forward_hook(inject_and_capture)
with torch.no_grad():
    out_pm = model.decoder(
        hidden_states=xt, timestep=t, timestep_r=t,
        attention_mask=attn_mask, encoder_hidden_states=enc,
        encoder_attention_mask=enc_attn, context_latents=ctx,
        use_cache=False,
    )
    hidden_pm_out = out_pm[0].float()
h12_handle2.remove()

delta_at_inject = (h12_modified - h12).norm(dim=-1).mean().item()
delta_at_output = (hidden_pm_out - hidden_clean_0).norm(dim=-1).mean().item()

print(f"\nPM-Retrieval Injection:")
print(f"  ret_res_norm         = {ret_res_norm:.6f}")
print(f"  gamma_r              = {adapt.gamma_r.item():.5f}")
print(f"  |Δh| at layer 12     = {delta_at_inject:.6f}")
print(f"  |Δh| in final output = {delta_at_output:.6f}")
print(f"  === Decay ratio      = {delta_at_output/max(delta_at_inject, 1e-10)*100:.2f}% ===")
print(f"  Clean final norm     = {hidden_clean_0.norm(dim=-1).mean().item():.4f}")
print(f"  PM final norm        = {hidden_pm_out.norm(dim=-1).mean().item():.4f}")
print(f"  === Relative change  = {delta_at_output/max(hidden_clean_0.norm(dim=-1).mean().item(), 1e-10)*100:.4f}% ===")

# ── What if we bypass gamma? ──
print(f"\n--- Bypass test: gamma_r=1.0, residual_scale=1.0 ---")
adapt.gamma_r.data.fill_(1.0)
adapt.residual_scale = 1.0

h12_handle3 = model.decoder.layers[LAYER].register_forward_hook(inject_and_capture)
with torch.no_grad():
    out_pm_strong = model.decoder(
        hidden_states=xt, timestep=t, timestep_r=t,
        attention_mask=attn_mask, encoder_hidden_states=enc,
        encoder_attention_mask=enc_attn, context_latents=ctx,
        use_cache=False,
    )
    hidden_pm_strong = out_pm_strong[0].float()
h12_handle3.remove()

delta_strong_inject = (h12_modified - h12).norm(dim=-1).mean().item()
delta_strong_out = (hidden_pm_strong - hidden_clean_0).norm(dim=-1).mean().item()
print(f"  |Δh| at layer 12     = {delta_strong_inject:.4f}")
print(f"  |Δh| in final output = {delta_strong_out:.4f}")
print(f"  Decay ratio          = {delta_strong_out/max(delta_strong_inject,1e-10)*100:.2f}%")
print(f"  Final norm delta     = {hidden_pm_strong.norm(dim=-1).mean().item() - hidden_clean_0.norm(dim=-1).mean().item():.4f}")
