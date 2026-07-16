#!/usr/bin/env python3
"""
v3: 公平对比 Baseline vs Sinkhorn
- Baseline 用 per-row 归一化显示（每帧最大值为1）
- 两者都用相同的 downsample rate 和 colorbar
- 添加数值统计说明
"""

import os, sys, math
from pathlib import Path
import numpy as np
import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.phase_memory import (
    TransportRetrievalAdapter, PMRetrievalPhaseMemory,
    parse_lyrics_to_units, build_duration_scaffold,
)

SAMPLE_ID = "b704dd0b245aa2ebaf6229399ca942a13bc907cb_1754"
DATASET_DIR = Path("/root/autodl-tmp/musicdata/dataset")
TENSOR_DIR = Path("/root/autodl-tmp/musicdata/train_tensors")
OUTPUT_DIR = Path("/root/ACE-Step-1.5")

pt_data = torch.load(str(TENSOR_DIR / f"{SAMPLE_ID}.pt"), map_location="cpu", weights_only=True)
lyrics_text = (DATASET_DIR / f"{SAMPLE_ID}.lyrics.txt").read_text(encoding="utf-8").strip()

handler = AceStepHandler()
handler.initialize_service(
    project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
    device="cuda", use_flash_attention=False, compile_model=False,
)
model = handler.model.eval()
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

model.config.use_section_rope_offset = False
for layer_mod in model.decoder.layers:
    if getattr(layer_mod, "use_section_rope", False): layer_mod.use_section_rope = False
    if getattr(layer_mod, "use_phase_memory", False): layer_mod.use_phase_memory = False
D = model.config.hidden_size

ckpt = torch.load("/root/autodl-tmp/pmctr_v6_active_writer_d256_epoch1/checkpoints/epoch_1_loss_1.2982/pm_retrieval.pt", map_location="cpu", weights_only=True)
pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True).to(device).float()
pm.load_state_dict(ckpt["phase_memory"])
adapt = TransportRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
    transport_mode="sinkhorn", sinkhorn_iters=5, transport_sigma=0.18, transport_qk_scale=1.0,
    write_alpha_init=0.001, write_alpha_max=0.01, out_proj_init_std=0.01).to(device).float()
adapt.load_state_dict(ckpt["retrieval_adapter"])
adapt.eval(); pm.eval()

parser = LyricsStructureParser()
L_raw = pt_data["encoder_hidden_states"].shape[0]
parsed = parser.parse(lyrics_text, num_chunks=L_raw)
section_ids = parsed.section_type_ids
SECTION_NAMES = {0:"UNKNOWN",1:"INTRO",2:"VERSE",3:"PRECHORUS",4:"CHORUS",5:"BRIDGE",6:"OUTRO",7:"INSTR"}

units, _, debug = parse_lyrics_to_units(lyrics_text, section_ids, auto_transition_ratios={})
tag_control_mask = debug.get("tag_control_mask", None)
scaffold = build_duration_scaffold(units, text_len=L_raw, tag_control_mask=tag_control_mask)
token_to_unit = scaffold["token_to_unit"]
lyric_mask = scaffold["lyric_mask"]
unit_boundaries, unit_duration = scaffold["unit_boundaries"], scaffold["unit_duration"]
u_section_ids = scaffold["unit_section_ids"]
lyric_unit_mask = scaffold["lyric_unit_mask"]
U = len(unit_boundaries) - 1
c_all = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
mu_all = unit_duration.clone()
mu_all = mu_all / mu_all.sum()

eh = pt_data["encoder_hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
eh_f = eh.float()
unit_text_hidden_list = []
for uid in range(U):
    is_lyric_u = lyric_unit_mask[uid].item() if lyric_unit_mask is not None else True
    if is_lyric_u: token_mask = (token_to_unit == uid) & lyric_mask
    else: token_mask = token_to_unit == uid
    tmb = token_mask.unsqueeze(0).expand(1, -1)
    if tmb.any(): pooled = eh_f[tmb].view(1, -1, D).mean(dim=1)
    else: pooled = torch.zeros(1, D, device=device, dtype=torch.float32)
    unit_text_hidden_list.append(pooled)
unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)

unit_section_id = u_section_ids.to(device) if u_section_ids is not None else torch.zeros(U, dtype=torch.long, device=device)
unit_is_lyric_local = lyric_unit_mask.to(device) if lyric_unit_mask is not None else torch.ones(U, dtype=torch.bool, device=device)

xt = pt_data["target_latents"].unsqueeze(0).to(device=device, dtype=dtype)
am = pt_data["attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
eh_in = pt_data["encoder_hidden_states"].unsqueeze(0).to(device=device, dtype=dtype)
eam_in = pt_data["encoder_attention_mask"].unsqueeze(0).to(device=device, dtype=dtype)
ctx = pt_data["context_latents"].unsqueeze(0).to(device=device, dtype=dtype)
t_tensor = torch.full((1,), 0.0, device=device, dtype=dtype)

# ── Forward A: Pure baseline ──────────────────────────────────────────
hs_a, ca_a = [], []
def hook_hs(m, i, o): hs_a.append(o[0].detach().float())
def hook_ca(m, i, o):
    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
        ca_a.append(o[1].detach().cpu())
h1 = model.decoder.layers[12].register_forward_hook(hook_hs)
h2 = model.decoder.layers[12].cross_attn.register_forward_hook(hook_ca)
with torch.no_grad():
    model.decoder(hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                  attention_mask=am, encoder_hidden_states=eh_in,
                  encoder_attention_mask=eam_in, context_latents=ctx,
                  use_cache=False, output_attentions=True)
h1.remove(); h2.remove()
H_a, ca_12_a = hs_a[0], torch.stack(ca_a, dim=0).squeeze(1)
T_eff = H_a.shape[1]

# ── Forward B: Sinkhorn ──────────────────────────────────────────────
hs_b = []
h3 = model.decoder.layers[12].register_forward_hook(lambda m,i,o: hs_b.append(o[0].detach().float()))
with torch.no_grad():
    model.decoder(hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
                  attention_mask=am, encoder_hidden_states=eh_in,
                  encoder_attention_mask=eam_in, context_latents=ctx,
                  use_cache=False, output_attentions=False)
h3.remove()

with torch.no_grad():
    pm_state = pm(hs_b[0], t_tensor)
p_audio = torch.linspace(0, 1, T_eff, device=device, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    _, Pi, diag = adapt(hidden_states=hs_b[0], text_hidden=eh_in, pm_state=pm_state,
        p_audio=p_audio, unit_text_hidden=unit_text_hidden,
        unit_c_pos=c_all.to(device).unsqueeze(0), unit_mass=mu_all.to(device).unsqueeze(0),
        unit_section_id=unit_section_id.unsqueeze(0), unit_is_lyric=unit_is_lyric_local.unsqueeze(0))
Pi_np = Pi.squeeze(0).float().cpu().numpy()
K = Pi_np.shape[1]

# ── Visualisation ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

lyric_token_idx = torch.where(lyric_mask)[0]
l_start, l_end = lyric_token_idx[0].item(), lyric_token_idx[-1].item() + 1
c_unit_np = c_all.cpu().numpy()

ds_rate = 10
T_ds = T_eff // ds_rate

# ── Figure 1: Fair comparison with per-row norm ───────────────────────
ca_mean = ca_12_a[0].mean(dim=0).float().numpy()  # [T, L]
ca_lyric = ca_mean[:, l_start:l_end]
L_lyric = l_end - l_start

# Per-row normalize: each timestep's attention sums to 1
# Already true for softmax, but visualisation uses raw values which are very small
# Create a second view: log scale to see structure better

ca_ds = ca_lyric[:T_ds*ds_rate].reshape(T_ds, ds_rate, L_lyric).mean(axis=1)
pi_ds = Pi_np[:T_ds*ds_rate].reshape(T_ds, ds_rate, K).mean(axis=1)

fig, axes = plt.subplots(2, 3, figsize=(22, 10))

# Row 1: linear scale
ax = axes[0,0]
im = ax.imshow(ca_ds, aspect='auto', cmap='inferno', interpolation='nearest', extent=[0, L_lyric, T_ds, 0])
ax.set_title(f"Baseline (linear scale)\nL_lyric={L_lyric} tokens")
ax.set_ylabel("Audio frame")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0,1]
im = ax.imshow(pi_ds, aspect='auto', cmap='inferno', interpolation='nearest', extent=[0, K, T_ds, 0])
ax.set_title(f"Sinkhorn Pi (linear scale)\nK={K} units")
plt.colorbar(im, ax=ax, fraction=0.046)

# Row 2: per-row normalized (show relative focus per frame)
# For baseline: normalize each row to [0,1] so we can see which tokens get MORE relative attention
ca_row_norm = np.zeros_like(ca_ds)
for t in range(T_ds):
    row = ca_ds[t]
    mn, mx = row.min(), row.max()
    ca_row_norm[t] = (row - mn) / (mx - mn + 1e-10)

pi_row_norm = np.zeros_like(pi_ds)
for t in range(T_ds):
    row = pi_ds[t]
    mn, mx = row.min(), row.max()
    pi_row_norm[t] = (row - mn) / (mx - mn + 1e-10)

ax = axes[1,0]
im = ax.imshow(ca_row_norm, aspect='auto', cmap='inferno', interpolation='nearest', extent=[0, L_lyric, T_ds, 0])
ax.set_title(f"Baseline (per-row normalized)\nshows RELATIVE focus per frame")
ax.set_xlabel("Lyric token index")
ax.set_ylabel("Audio frame")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1,1]
im = ax.imshow(pi_row_norm, aspect='auto', cmap='inferno', interpolation='nearest', extent=[0, K, T_ds, 0])
ax.set_title(f"Sinkhorn Pi (per-row normalized)\n")
ax.set_xlabel("Unit index")
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel C: annotation note
ax = axes[0,2]
ax.axis('off')
info_text = (
    f"Notes:\n\n"
    f"Baseline: 2954 frames x 769 tokens\n"
    f"每个frame的attention要分给769个token\n"
    f"平均每个token仅得到 ~0.13%\n"
    f"linear scale下必然显得很暗\n\n"
    f"Sinkhorn: 2954 frames x 29 units\n"
    f"mass集中在少数unit\n"
    f"linear scale下显得更亮\n\n"
    f"per-row归一化可看相对结构\n"
    f"（每帧最关注的token标为1.0）"
)
ax.text(0.1, 0.5, info_text, fontsize=11, va='center', transform=ax.transAxes,
        fontfamily='monospace')

ax = axes[1,2]
ax.axis('off')
# Centroid comparison
time_axis = np.arange(T_eff) / T_eff
token_to_unit_np = token_to_unit.cpu().numpy()
ca_unit = np.zeros((T_eff, K))
for k in range(K):
    mask_k = token_to_unit_np == k
    if mask_k.sum() > 0:
        ca_unit[:, k] = ca_mean[:, mask_k].sum(axis=1)
ca_cent = np.array([(ca_unit[t]*c_unit_np).sum()/(ca_unit[t].sum()+1e-10) for t in range(T_eff)])
pi_cent = np.array([(Pi_np[t]*c_unit_np).sum()/(Pi_np[t].sum()+1e-10) for t in range(T_eff)])
ca_r2 = spearmanr(ca_cent, np.linspace(0,1,T_eff))[0]
pi_r2 = spearmanr(pi_cent, np.linspace(0,1,T_eff))[0]

ax.plot(time_axis, ca_cent, color='steelblue', linewidth=1.0, alpha=0.8, label=f'Baseline (rho={ca_r2:.3f})')
ax.plot(time_axis, pi_cent, color='coral', linewidth=1.2, label=f'Sinkhorn (rho={pi_r2:.3f})')
ax.set_xlabel('Normalized audio time')
ax.set_ylabel('Centroid (unit pos)')
ax.set_title('Centroid Trajectory')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(f"Fair Comparison: Baseline vs Sinkhorn — {SAMPLE_ID}", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_comparison_fair.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_comparison_fair.png")

# ── Figure 2: Per-head with per-row norm ──────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(22, 18))
for h in range(16):
    ax = axes[h // 4, h % 4]
    attn_h = ca_12_a[0, h].float().numpy()[:, l_start:l_end]
    attn_h_ds = attn_h[:T_ds*ds_rate].reshape(T_ds, ds_rate, L_lyric).mean(axis=1)
    # per-row norm
    for t in range(T_ds):
        row = attn_h_ds[t]; mn, mx = row.min(), row.max()
        attn_h_ds[t] = (row - mn) / (mx - mn + 1e-10)
    ax.imshow(attn_h_ds, aspect='auto', cmap='viridis', interpolation='nearest',
              extent=[0, L_lyric, T_ds, 0])
    ax.set_title(f"Head {h}")
fig.suptitle(f"Per-Head Baseline Cross-Attention (per-row normalized)\n{len(SAMPLE_ID)}", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_per_head_row_norm.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_per_head_row_norm.png")

# ── Figure 3: Log-scale comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ca_log = np.log10(ca_ds + 1e-8)
im = axes[0].imshow(ca_log, aspect='auto', cmap='inferno', interpolation='nearest',
                     extent=[0, L_lyric, T_ds, 0])
axes[0].set_title(f"Baseline (log10 scale)")
axes[0].set_xlabel("Lyric token index"); axes[0].set_ylabel("Audio frame")
plt.colorbar(im, ax=axes[0], fraction=0.046, label='log10(attn)')

pi_log = np.log10(pi_ds + 1e-8)
im = axes[1].imshow(pi_log, aspect='auto', cmap='inferno', interpolation='nearest',
                     extent=[0, K, T_ds, 0])
axes[1].set_title(f"Sinkhorn Pi (log10 scale)")
axes[1].set_xlabel("Unit index")
plt.colorbar(im, ax=axes[1], fraction=0.046, label='log10(attn)')

fig.suptitle(f"Log-Scale Comparison — {SAMPLE_ID}", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_comparison_log.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_comparison_log.png")

# ── Summary stats ─────────────────────────────────────────────────────
token_coverage = ca_mean.sum(axis=0)
token_coverage_norm = token_coverage / token_coverage.sum()
per_frame_top1 = ca_mean.max(axis=1)

print(f"\n{'='*60}")
print(f"BASELINE STATS")
print(f"{'='*60}")
print(f"Total tokens: {L_raw}")
n80 = np.searchsorted(np.cumsum(np.sort(token_coverage_norm)[::-1]), 0.8) + 1
n90 = np.searchsorted(np.cumsum(np.sort(token_coverage_norm)[::-1]), 0.9) + 1
print(f"80% attention => {n80} tokens ({n80/L_raw*100:.1f}%)")
print(f"90% attention => {n90} tokens ({n90/L_raw*100:.1f}%)")
print(f"Top-1 per frame: mean={per_frame_top1.mean():.4f}")
print(f"Centroid vs time: rho={ca_r2:.4f}")
print(f"\nSINKHORN STATS")
print(f"Centroid vs time: rho={pi_r2:.4f}")
print(f"row_error={diag.get('row_error',-1):.6f}, col_error={diag.get('col_error',-1):.6f}, entropy={diag.get('entropy',-1):.6f}")
print(f"\nImages saved to {OUTPUT_DIR}")