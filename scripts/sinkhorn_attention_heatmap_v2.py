#!/usr/bin/env python3
"""
修正版：分别用独立 forward 捕获 baseline cross-attention 和 Sinkhorn transport plan Pi。
"""

import os, sys, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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

# ── Load preprocessed data ─────────────────────────────────────────────
pt_path = TENSOR_DIR / f"{SAMPLE_ID}.pt"
pt_data = torch.load(str(pt_path), map_location="cpu", weights_only=True)
target_latents = pt_data["target_latents"]
attention_mask = pt_data["attention_mask"]
encoder_hidden_states = pt_data["encoder_hidden_states"]
encoder_attention_mask = pt_data["encoder_attention_mask"]
context_latents = pt_data["context_latents"]

lyrics_text = (DATASET_DIR / f"{SAMPLE_ID}.lyrics.txt").read_text(encoding="utf-8").strip()
print(f"Lyrics ({len(lyrics_text)} chars)")

# ── Load model ─────────────────────────────────────────────────────────
print("Loading SFT model...")
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
    if getattr(layer_mod, "use_section_rope", False):
        layer_mod.use_section_rope = False
    if getattr(layer_mod, "use_phase_memory", False):
        layer_mod.use_phase_memory = False

D = model.config.hidden_size

# ── Load trained adapter ───────────────────────────────────────────────
ckpt_path = "/root/autodl-tmp/pmctr_v6_active_writer_d256_epoch1/checkpoints/epoch_1_loss_1.2982/pm_retrieval.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256,
                            normalize_internal_state=True).to(device).float()
pm.load_state_dict(checkpoint["phase_memory"])

adapt = TransportRetrievalAdapter(
    hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
    transport_mode="sinkhorn", sinkhorn_iters=5,
    transport_sigma=0.18, transport_qk_scale=1.0,
    write_alpha_init=0.001, write_alpha_max=0.01,
    out_proj_init_std=0.01,
).to(device).float()
adapt.load_state_dict(checkpoint["retrieval_adapter"])
adapt.eval()
pm.eval()
print("Checkpoint loaded ✓")

# ── Parse section_ids ──────────────────────────────────────────────────
parser = LyricsStructureParser()
L_raw = encoder_hidden_states.shape[0]
parsed = parser.parse(lyrics_text, num_chunks=L_raw)
section_ids = parsed.section_type_ids

# ── Parse units + scaffold ─────────────────────────────────────────────
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}
units, _, debug = parse_lyrics_to_units(
    lyrics_text, section_ids,
    auto_transition_ratios={"intro": 0.0, "outro": 0.0,
        "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0, "bridge_to_chorus": 0.0},
)
tag_control_mask = debug.get("tag_control_mask", None)
scaffold = build_duration_scaffold(units, text_len=L_raw, tag_control_mask=tag_control_mask)

token_to_unit = scaffold["token_to_unit"]
lyric_mask = scaffold["lyric_mask"]
unit_boundaries = scaffold["unit_boundaries"]
unit_duration = scaffold["unit_duration"]
u_section_ids = scaffold["unit_section_ids"]
lyric_unit_mask = scaffold["lyric_unit_mask"]

U = len(unit_boundaries) - 1
c_all = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2
mu_all = unit_duration.clone()
mu_all = mu_all / mu_all.sum()

unit_text_hidden_list = []
eh = encoder_hidden_states.unsqueeze(0).to(device=device, dtype=dtype)
eh_f = eh.float()
B = 1
for uid in range(U):
    is_lyric_u = lyric_unit_mask[uid].item() if lyric_unit_mask is not None else True
    if is_lyric_u:
        token_mask = (token_to_unit == uid) & lyric_mask
    else:
        token_mask = token_to_unit == uid
    token_mask_b = token_mask.unsqueeze(0).expand(B, -1)
    if token_mask_b.any():
        pooled = eh_f[token_mask_b].view(B, -1, D).mean(dim=1)
    else:
        pooled = torch.zeros(B, D, device=device, dtype=torch.float32)
    unit_text_hidden_list.append(pooled)
unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)

unit_section_id = u_section_ids.to(device) if u_section_ids is not None else torch.zeros(U, dtype=torch.long, device=device)
unit_is_lyric_local = lyric_unit_mask.to(device) if lyric_unit_mask is not None else torch.ones(U, dtype=torch.bool, device=device)

c_unit = c_all.to(device)
mu = mu_all.to(device)
usid_b = unit_section_id.unsqueeze(0)
unit_is_lyric_b = unit_is_lyric_local.unsqueeze(0)
c_unit_b = c_unit.unsqueeze(0)
mu_b = mu.unsqueeze(0)

# ── Prepare inputs ─────────────────────────────────────────────────────
xt = target_latents.unsqueeze(0).to(device=device, dtype=dtype)
am = attention_mask.unsqueeze(0).to(device=device, dtype=dtype)
eh_in = encoder_hidden_states.unsqueeze(0).to(device=device, dtype=dtype)
eam_in = encoder_attention_mask.unsqueeze(0).to(device=device, dtype=dtype)
ctx = context_latents.unsqueeze(0).to(device=device, dtype=dtype)
t_tensor = torch.full((1,), 0.0, device=device, dtype=dtype)

# =====================================================================
# FORWARD A: 纯 Baseline (无注入, 无 Sinkhorn)
# 同时捕获 layer 12 hidden states & cross-attention weights
# =====================================================================
print("\n=== Forward A: Pure Baseline (capture cross-attn + hidden) ===")

hs_a = []
ca_a = []
def hook_hs_a(m, i, o):
    hs_a.append(o[0].detach().float())
def hook_ca_a(module, input, output):
    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
        ca_a.append(output[1].detach().cpu())

h_hs = model.decoder.layers[12].register_forward_hook(hook_hs_a)
h_ca = model.decoder.layers[12].cross_attn.register_forward_hook(hook_ca_a)

with torch.no_grad():
    _ = model.decoder(
        hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
        attention_mask=am, encoder_hidden_states=eh_in,
        encoder_attention_mask=eam_in, context_latents=ctx,
        use_cache=False, output_attentions=True,
    )
h_hs.remove()
h_ca.remove()

H_a = hs_a[0]  # [1, T_eff, D]
T_eff = H_a.shape[1]
ca_12_a = torch.stack(ca_a, dim=0).squeeze(1)  # [1, H, T, L]

print(f"  H: {H_a.shape}, cross-attn: {ca_12_a.shape}")

# =====================================================================
# FORWARD B: Sinkhorn warmup → PM → adapter → Pi
# 第二次 forward 带 inject hook (只捕获 Pi, 不看 cross-attention)
# =====================================================================
print("\n=== Forward B: Warmup for Sinkhorn ===")

hs_b = []
def hook_hs_b(m, i, o):
    hs_b.append(o[0].detach().float())
h_hs_b = model.decoder.layers[12].register_forward_hook(hook_hs_b)

with torch.no_grad():
    _ = model.decoder(
        hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
        attention_mask=am, encoder_hidden_states=eh_in,
        encoder_attention_mask=eam_in, context_latents=ctx,
        use_cache=False, output_attentions=False,
    )
h_hs_b.remove()

H_b = hs_b[0]

with torch.no_grad():
    pm_state = pm(H_b, t_tensor)

p_audio = torch.linspace(0, 1, T_eff, device=device, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    delta_h, Pi, diag = adapt(
        hidden_states=H_b, text_hidden=eh_in, pm_state=pm_state,
        p_audio=p_audio, unit_text_hidden=unit_text_hidden,
        unit_c_pos=c_unit_b, unit_mass=mu_b,
        unit_section_id=usid_b, unit_is_lyric=unit_is_lyric_b,
    )

Pi_np = Pi.squeeze(0).float().cpu().numpy()  # [T_eff, K=U]
print(f"  Pi: {Pi.shape}")

for k, v in diag.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")

# =====================================================================
# VISUALISATION
# =====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

lyric_token_indices = torch.where(lyric_mask)[0]
lyric_start = lyric_token_indices[0].item()
lyric_end = lyric_token_indices[-1].item() + 1
L_lyric = lyric_end - lyric_start
K = Pi_np.shape[1]

ds_rate = 10
T_ds = T_eff // ds_rate

# ── Figure 1: Three-panel comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))

# Panel A: baseline cross-attn (layer 12, mean over 16 heads)
ca_lyric = ca_12_a[0].mean(dim=0).float().numpy()[:, lyric_start:lyric_end]  # [T, L_lyric]
ca_ds = ca_lyric[:T_ds*ds_rate].reshape(T_ds, ds_rate, L_lyric).mean(axis=1)
im0 = axes[0].imshow(ca_ds, aspect='auto', cmap='inferno', interpolation='nearest',
                      extent=[0, L_lyric, T_ds, 0])
axes[0].set_title(f"Baseline Cross-Attn\n(layer 12, mean 16 heads)")
axes[0].set_xlabel("Lyric token index")
axes[0].set_ylabel("Audio frame")
plt.colorbar(im0, ax=axes[0], fraction=0.046)
# Log scale to show detail
axes[0].set_title(f"Baseline Cross-Attn (mean 16 heads)\nT_eff={T_eff}, L_lyric={L_lyric}")

# Panel B: Sinkhorn Pi
pi_ds = Pi_np[:T_ds*ds_rate].reshape(T_ds, ds_rate, K).mean(axis=1)
im1 = axes[1].imshow(pi_ds, aspect='auto', cmap='inferno', interpolation='nearest',
                      extent=[0, K, T_ds, 0])
axes[1].set_title(f"Sinkhorn Transport Plan Pi\n(K={K} units)")
axes[1].set_xlabel("Lyric unit index")
axes[1].set_ylabel("Audio frame")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Panel C: Sinkhorn Pi with section annotations
unit_labels = [f"{SECTION_NAMES.get(int(unit_section_id[i].item()), '?')[:3]}{i}" for i in range(K)]
im2 = axes[2].imshow(pi_ds, aspect='auto', cmap='inferno', interpolation='nearest',
                      extent=[0, K, T_ds, 0])
for i in range(K+1):
    axes[2].axvline(x=i-0.5, color='white', linewidth=0.3, alpha=0.5)
axes[2].set_title(f"Sinkhorn Pi (annotated)")
axes[2].set_xlabel("Lyric unit")
axes[2].set_ylabel("Audio frame")
yticks = np.arange(0, K, max(1, K // 25))
axes[2].set_xticks(yticks)
axes[2].set_xticklabels([unit_labels[i] for i in yticks], rotation=90, fontsize=5)
plt.colorbar(im2, ax=axes[2], fraction=0.046)

fig.suptitle(f"Baseline vs Sinkhorn Transport — {SAMPLE_ID}\n"
             f"基线: token-level attention (lyric区域={L_lyric}列) | Sinkhorn: unit-level transport (K={K}个units)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_vs_baseline_v2.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_sinkhorn_vs_baseline_v2.png")

# ── Figure 2: Per-head baseline (to compare with original analysis) ────
fig, axes = plt.subplots(4, 4, figsize=(22, 18))
for h in range(16):
    ax = axes[h // 4, h % 4]
    attn_h = ca_12_a[0, h].float().numpy()
    attn_h_lyric = attn_h[:, lyric_start:lyric_end]
    attn_h_ds = attn_h_lyric[:T_ds*ds_rate].reshape(T_ds, ds_rate, L_lyric).mean(axis=1)
    ax.imshow(attn_h_ds, aspect='auto', cmap='viridis', interpolation='nearest',
              extent=[0, L_lyric, T_ds, 0])
    ax.set_title(f"Head {h}")
fig.suptitle(f"Per-Head Baseline Cross-Attention (layer 12)\n{SAMPLE_ID} | Pure forward (no injection)", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_per_head_v2.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_sinkhorn_per_head_v2.png")

# ── Figure 3: Centroid comparison ──────────────────────────────────────
time_axis = np.arange(T_eff) / T_eff

# Baseline: unit-level centroid from token-level attention
ca_mean = ca_12_a[0].mean(dim=0).float().numpy()  # [T, L]
token_to_unit_np = token_to_unit.cpu().numpy()
c_unit_np = c_unit.cpu().numpy()

ca_unit = np.zeros((T_eff, K))
for k in range(K):
    mask_k = token_to_unit_np == k
    if mask_k.sum() > 0:
        ca_unit[:, k] = ca_mean[:, mask_k].sum(axis=1)

ca_centroid = np.array([
    (ca_unit[t] * c_unit_np).sum() / (ca_unit[t].sum() + 1e-10)
    for t in range(T_eff)
])

# Sinkhorn Pi centroid
pi_centroid = np.array([
    (Pi_np[t] * c_unit_np).sum() / (Pi_np[t].sum() + 1e-10)
    for t in range(T_eff)
])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time_axis, ca_centroid, color="steelblue", linewidth=1.0, alpha=0.8, label="Baseline cross-attn")
ax.plot(time_axis, pi_centroid, color="coral", linewidth=1.2, label="Sinkhorn Pi")
for k in range(K):
    ax.axhline(y=c_unit_np[k], color="gray", linewidth=0.3, alpha=0.3)
ax.set_xlabel("Normalized audio time")
ax.set_ylabel("Attention centroid (unit position)")
ax.set_title(f"Centroid: Baseline vs Sinkhorn\n{SAMPLE_ID}")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_centroid_v2.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_sinkhorn_centroid_v2.png")

# ── Figure 4: Marginal validation ─────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [1, 3]})

col_mass = Pi_np.sum(axis=0)
col_mass_norm = col_mass / col_mass.sum()
mu_np = mu.cpu().numpy()

axes[0].bar(np.arange(K)-0.15, col_mass_norm, width=0.3, alpha=0.8, color='coral', label='Sinkhorn col marginal')
axes[0].bar(np.arange(K)+0.15, mu_np, width=0.3, alpha=0.6, color='steelblue', label='Target (mu)')
axes[0].set_ylabel('Mass')
axes[0].set_title('Column Marginal vs Target')
axes[0].legend(fontsize=8)
unit_labels_full = [f"{SECTION_NAMES.get(int(unit_section_id[i].item()), '?')[:3]}{i}" for i in range(K)]
axes[0].set_xticks(np.arange(K))
axes[0].set_xticklabels(unit_labels_full, rotation=90, fontsize=5)

row_mass = Pi_np.sum(axis=1)
expected = 1.0 / T_eff
ds_vis = max(1, T_eff // 1000)
row_ds = row_mass[:T_eff//ds_vis*ds_vis].reshape(-1, ds_vis).mean(axis=1)
time_ds = time_axis[:T_eff//ds_vis*ds_vis].reshape(-1, ds_vis).mean(axis=1)
axes[1].plot(time_ds, row_ds, color='coral', linewidth=1.0, label=f'Row marginal')
axes[1].axhline(y=expected, color='steelblue', linestyle='--', linewidth=0.8, label=f'Expected = {expected:.6f}')
axes[1].set_xlabel('Normalized audio time')
axes[1].set_ylabel('Row sum')
axes[1].set_title('Row Marginal (should be uniform)')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle(f"Sinkhorn Marginal Validation — {SAMPLE_ID}", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_marginals_v2.png", dpi=120, bbox_inches='tight')
plt.close(fig)
print("Saved: attn_sinkhorn_marginals_v2.png")

# ── Quantitative ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"QUANTITATIVE COMPARISON")
print(f"{'='*60}")
ca_spearman = spearmanr(ca_centroid, np.linspace(0, 1, T_eff))[0]
pi_spearman = spearmanr(pi_centroid, np.linspace(0, 1, T_eff))[0]
print(f"Centroid vs time (baseline):          rho={ca_spearman:.4f}")
print(f"Centroid vs time (Sinkhorn Pi):       rho={pi_spearman:.4f}")
print(f"Pi row_error:  {diag.get('row_error', -1):.6f}")
print(f"Pi col_error:  {diag.get('col_error', -1):.6f}")
print(f"Pi entropy:    {diag.get('entropy', -1):.6f}")

# Per-unit coverage
unit_coverage = Pi_np.sum(axis=0)
unit_coverage_norm = unit_coverage / unit_coverage.sum()
print("\nPer-unit Sinkhorn coverage vs target:")
top_k = np.argsort(unit_coverage_norm)[-10:][::-1]
for idx in top_k:
    sid = int(u_section_ids[idx].item())
    sn = SECTION_NAMES.get(sid, "?")
    print(f"  Unit {idx:2d} [{sn:>8}] pos={c_unit_np[idx]:.3f} "
          f"mass_target={mu_np[idx]:.4f} coverage={unit_coverage_norm[idx]:.4f}")

print(f"\nAll figures -> {OUTPUT_DIR}")
