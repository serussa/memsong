#!/usr/bin/env python3
"""
TransportRetrievalAdapter 的 Sinkhorn transport plan Pi 热力图可视化。
与 baseline cross-attention 做对比。
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
    scaffold_progress, log_sinkhorn,
)

# ── IDs ────────────────────────────────────────────────────────────────
SAMPLE_ID = "b704dd0b245aa2ebaf6229399ca942a13bc907cb_1754"
AUDIO_DIR = Path("/root/autodl-tmp/musicdata/audios")
DATASET_DIR = Path("/root/autodl-tmp/musicdata/dataset")
TENSOR_DIR = Path("/root/autodl-tmp/musicdata/train_tensors")
OUTPUT_DIR = Path("/root/ACE-Step-1.5")
os.makedirs(str(OUTPUT_DIR), exist_ok=True)

# ── Load preprocessed data ─────────────────────────────────────────────
pt_path = TENSOR_DIR / f"{SAMPLE_ID}.pt"
pt_data = torch.load(str(pt_path), map_location="cpu", weights_only=True)
target_latents = pt_data["target_latents"]
attention_mask = pt_data["attention_mask"]
encoder_hidden_states = pt_data["encoder_hidden_states"]
encoder_attention_mask = pt_data["encoder_attention_mask"]
context_latents = pt_data["context_latents"]

# ── Load raw lyrics text ───────────────────────────────────────────────
lyrics_path = DATASET_DIR / f"{SAMPLE_ID}.lyrics.txt"
lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
print(f"Lyrics ({len(lyrics_text)} chars) ✓")

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
print(f"  device={device}, dtype={dtype}")

# Disable Section-RoPE + PhaseMemory (baseline decoder)
model.config.use_section_rope_offset = False
for layer_mod in model.decoder.layers:
    if getattr(layer_mod, "use_section_rope", False):
        layer_mod.use_section_rope = False
    if getattr(layer_mod, "use_phase_memory", False):
        layer_mod.use_phase_memory = False

D = model.config.hidden_size  # 2048

# ── Load trained TransportRetrievalAdapter + PM ─────────────────────────
ckpt_path = "/root/autodl-tmp/pmctr_v6_active_writer_d256_epoch1/checkpoints/epoch_1_loss_1.2982/pm_retrieval.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# Build PhaseMemory (PMRetrievalPhaseMemory)
pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256,
                            normalize_internal_state=True).to(device).float()
pm.load_state_dict(checkpoint["phase_memory"])

# Build TransportRetrievalAdapter
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
print(f"Loaded checkpoint ✓")

# ── Parse section_ids ──────────────────────────────────────────────────
parser = LyricsStructureParser()
L_raw = encoder_hidden_states.shape[0]
parsed = parser.parse(lyrics_text, num_chunks=L_raw)
section_ids = parsed.section_type_ids
print(f"Section IDs: {section_ids.shape}")

# ── Parse lyrics to units + build scaffold ─────────────────────────────
SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}
units, _, debug = parse_lyrics_to_units(
    lyrics_text, section_ids,
    auto_transition_ratios={
        "intro": 0.0, "outro": 0.0,
        "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
        "bridge_to_chorus": 0.0,
    },
)
tag_control_mask = debug.get("tag_control_mask", None)
print(f"Parsed {len(units)} lyric units")

# Build scaffold dict — contains unit_boundaries, duration, token_to_unit, etc.
scaffold = build_duration_scaffold(units, text_len=L_raw, tag_control_mask=tag_control_mask)

token_to_unit = scaffold["token_to_unit"]  # [L]
lyric_mask = scaffold["lyric_mask"]
unit_boundaries = scaffold["unit_boundaries"]  # [U+1]
unit_duration = scaffold["unit_duration"]  # [U]
u_section_ids = scaffold.get("unit_section_ids")
lyric_unit_mask = scaffold.get("lyric_unit_mask")  # [U]
control_mask = scaffold.get("control_mask")

U = len(unit_boundaries) - 1  # total units (lyric + silence/control)
print(f"Total units: {U}")

# Unit section IDs
if u_section_ids is not None:
    unit_section_id = u_section_ids.to(device)
else:
    unit_section_id = torch.zeros(U, dtype=torch.long, device=device)

# unit_is_lyric
if lyric_unit_mask is not None:
    unit_is_lyric_local = lyric_unit_mask.to(device)
else:
    unit_is_lyric_local = torch.ones(U, dtype=torch.bool, device=device)

# ── Build unit-level tensors ───────────────────────────────────────────
# Unit centre positions
c_all = (unit_boundaries[:-1] + unit_boundaries[1:]) / 2  # [U]
c_unit = c_all.to(device)

# Unit mass (proportional to duration)
mu_all = unit_duration.clone()
mu_all = mu_all / mu_all.sum()
mu = mu_all.to(device)

# Pool encoder hidden states per unit.
# Must match exactly how fixed_lora_module.py does it (lines 1039-1052).
eh = encoder_hidden_states.unsqueeze(0).to(device=device, dtype=dtype)  # [1, L, D]
eh_f = eh.float()
B = 1

unit_text_hidden_list = []
for uid in range(U):
    is_lyric_u = unit_is_lyric_local[uid].item()
    if is_lyric_u:
        token_mask = (token_to_unit == uid) & lyric_mask
    else:
        token_mask = token_to_unit == uid
    token_mask_b = token_mask.unsqueeze(0).expand(B, -1)  # [1, L]
    if token_mask_b.any():
        pooled = eh_f[token_mask_b].view(B, -1, D).mean(dim=1)
    else:
        pooled = torch.zeros(B, D, device=device, dtype=torch.float32)
    unit_text_hidden_list.append(pooled)
unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)  # [1, U, D]
print(f"unit_text_hidden: {unit_text_hidden.shape}")

# Expand for batch
c_unit_b = c_unit.unsqueeze(0)  # [1, U]
mu_b = mu.unsqueeze(0)  # [1, U]
usid_b = unit_section_id.unsqueeze(0)  # [1, U]
unit_is_lyric_b = unit_is_lyric_local.unsqueeze(0)  # [1, U]

# ── Run warmup forward → collect layer 12 hidden states ────────────────
T_bs = target_latents.shape[0]
xt = target_latents.unsqueeze(0).to(device=device, dtype=dtype)
am = attention_mask.unsqueeze(0).to(device=device, dtype=dtype)
eh_in = encoder_hidden_states.unsqueeze(0).to(device=device, dtype=dtype)
eam_in = encoder_attention_mask.unsqueeze(0).to(device=device, dtype=dtype)
ctx = context_latents.unsqueeze(0).to(device=device, dtype=dtype)
t_tensor = torch.full((1,), 0.0, device=device, dtype=dtype)

hs_list = []
def _warmup_hook(m, i, o):
    hs_list.append(o[0].detach().float())
handle = model.decoder.layers[12].register_forward_hook(_warmup_hook)

print(f"Warmup forward (T={T_bs})...")
with torch.no_grad():
    _ = model.decoder(
        hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
        attention_mask=am, encoder_hidden_states=eh_in,
        encoder_attention_mask=eam_in, context_latents=ctx,
        use_cache=False, output_attentions=False,
    )
handle.remove()

H = hs_list[0]  # [1, T_eff, D]
T_eff = H.shape[1]
print(f"Layer 12 hidden: {H.shape}")

# ── PM forward → pm_state ──────────────────────────────────────────────
with torch.no_grad():
    pm_state = pm(H, t_tensor)
print(f"pm_state: {pm_state.shape}")

# ── Adapter forward → Pi ───────────────────────────────────────────────
p_audio = torch.linspace(0, 1, T_eff, device=device, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    delta_h, Pi, diag = adapt(
        hidden_states=H,
        text_hidden=eh_in,
        pm_state=pm_state,
        p_audio=p_audio,
        unit_text_hidden=unit_text_hidden,
        unit_c_pos=c_unit_b,
        unit_mass=mu_b,
        unit_section_id=usid_b,
        unit_is_lyric=unit_is_lyric_b,
    )
print(f"Pi shape: {Pi.shape}")
for k, v in diag.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")

Pi_np = Pi.squeeze(0).float().cpu().numpy()  # [T, K=U]

# ── Run with cross-attention hooks for comparison ──────────────────────
ca_weights = {}
def make_ca_hook(layer_idx):
    def hook(mod, inp, out):
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            if layer_idx not in ca_weights:
                ca_weights[layer_idx] = []
            ca_weights[layer_idx].append(out[1].detach().cpu())
    return hook

hooks = []
for i, layer in enumerate(model.decoder.layers):
    if hasattr(layer, 'cross_attn') and layer.use_cross_attention:
        h = layer.cross_attn.register_forward_hook(make_ca_hook(i))
        hooks.append(h)

# Inject delta_h into layer 12
def _inject_hook(m, i, o):
    dh = delta_h.to(dtype=o[0].dtype, device=o[0].device)
    return (o[0] + dh, *o[1:])
inj_handle = model.decoder.layers[12].register_forward_hook(_inject_hook)

with torch.no_grad():
    _ = model.decoder(
        hidden_states=xt, timestep=t_tensor, timestep_r=t_tensor,
        attention_mask=am, encoder_hidden_states=eh_in,
        encoder_attention_mask=eam_in, context_latents=ctx,
        use_cache=False, output_attentions=True,
    )

inj_handle.remove()
for h in hooks:
    h.remove()

# ── Visualisation ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Lyric region bounds
lyric_token_indices = torch.where(lyric_mask)[0]
lyric_start = lyric_token_indices[0].item()
lyric_end = lyric_token_indices[-1].item() + 1

# ── Figure 1: Baseline vs Sinkhorn ─────────────────────────────────────
ds_rate = 10
T_ds = T_eff // ds_rate

fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))

# Panel A: Baseline cross-attention mean (layer 12)
ca_12 = ca_weights.get(12, None)
if ca_12 is not None:
    ca_12_t = torch.stack(ca_12, dim=0)  # [1, B, H, T, L]
    ca_12_w = ca_12_t.squeeze(1)  # [1, H, T, L]
    ca_mean = ca_12_w[0].mean(dim=0).float().numpy()  # [T, L]
    ca_lyric = ca_mean[:, lyric_start:lyric_end]
    L_lyric = lyric_end - lyric_start
    if T_eff >= ds_rate:
        ca_ds = ca_lyric[:T_ds*ds_rate].reshape(T_ds, ds_rate, L_lyric).mean(axis=1)
    else:
        ca_ds = ca_lyric
    ax = axes[0]
    im = ax.imshow(ca_ds, aspect='auto', cmap='inferno', interpolation='nearest',
                   extent=[0, L_lyric, T_ds if T_eff >= ds_rate else T_eff, 0])
    ax.set_title(f"Baseline Cross-Attn\n(layer 12, mean 16 heads)")
    ax.set_xlabel("Lyric token index")
    ax.set_ylabel("Audio frame (ds={}x)".format(ds_rate))
    plt.colorbar(im, ax=ax, fraction=0.046)

# Panel B: Sinkhorn transport plan Pi
K = Pi_np.shape[1]
if T_eff >= ds_rate:
    pi_ds = Pi_np[:T_ds*ds_rate].reshape(T_ds, ds_rate, K).mean(axis=1)
else:
    pi_ds = Pi_np
im = axes[1].imshow(pi_ds, aspect='auto', cmap='inferno', interpolation='nearest',
                    extent=[0, K, T_ds if T_eff >= ds_rate else T_eff, 0])
axes[1].set_title(f"Sinkhorn Transport Plan Pi\n(K={K} units)")
axes[1].set_xlabel("Lyric unit index")
axes[1].set_ylabel("Audio frame (ds={}x)".format(ds_rate))
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Panel C: Sinkhorn Pi with unit annotations
# Map unit index to section label
unit_labels = [f"{SECTION_NAMES.get(int(unit_section_id[i].item()), '?')[:3]}{i}" for i in range(K)]
im = axes[2].imshow(pi_ds, aspect='auto', cmap='inferno', interpolation='nearest',
                    extent=[0, K, T_ds if T_eff >= ds_rate else T_eff, 0])
for i in range(K+1):
    axes[2].axvline(x=i-0.5, color='white', linewidth=0.3, alpha=0.5)
axes[2].set_title(f"Sinkhorn Pi (annotated units)")
axes[2].set_xlabel("Lyric unit")
axes[2].set_ylabel("Audio frame (ds={}x)".format(ds_rate))
yticks = np.arange(0, K, max(1, K // 25))
axes[2].set_xticks(yticks)
axes[2].set_xticklabels([unit_labels[i] for i in yticks], rotation=90, fontsize=5)
plt.colorbar(im, ax=axes[2], fraction=0.046)

fig.suptitle(f"Sinkhorn Transport vs Baseline Cross-Attention\n{SAMPLE_ID}", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_vs_baseline.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: attn_sinkhorn_vs_baseline.png")

# ── Figure 2: Per-head cross-attention ─────────────────────────────────
if ca_12 is not None:
    L_lyric = lyric_end - lyric_start
    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    for h in range(16):
        ax = axes[h // 4, h % 4]
        attn_h = ca_12_w[0, h].float().numpy()
        attn_h_lyric = attn_h[:, lyric_start:lyric_end]
        if T_eff >= ds_rate:
            attn_h_ds = attn_h_lyric[:T_ds*ds_rate].reshape(T_ds, ds_rate, L_lyric).mean(axis=1)
        else:
            attn_h_ds = attn_h_lyric
        ax.imshow(attn_h_ds, aspect='auto', cmap='viridis', interpolation='nearest',
                  extent=[0, L_lyric, T_ds if T_eff >= ds_rate else T_eff, 0])
        ax.set_title(f"Head {h}")
    fig.suptitle(f"Per-Head Cross-Attention (layer 12)\n{SAMPLE_ID}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attn_sinkhorn_per_head_baseline.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: attn_sinkhorn_per_head_baseline.png")

# ── Figure 3: Centroid comparison ──────────────────────────────────────
time_axis = np.arange(T_eff) / T_eff

# Centroid from baseline cross-attention (token-level → unit-level)
if ca_12 is not None:
    ca_mean = ca_12_w[0].mean(dim=0).float().numpy()  # [T, L]
    # Map token attention to unit level
    ca_unit = np.zeros((T_eff, K))
    token_to_unit_np = token_to_unit.cpu().numpy()
    for t in range(T_eff):
        for k in range(K):
            mask_k = token_to_unit_np == k
            if mask_k.sum() > 0:
                ca_unit[t, k] = ca_mean[t, mask_k].sum()
    c_unit_np = c_unit.cpu().numpy()
    ca_centroid = np.array([
        (ca_unit[t] * c_unit_np).sum() / (ca_unit[t].sum() + 1e-10)
        for t in range(T_eff)
    ])
else:
    ca_centroid = None

# Centroid from Sinkhorn Pi
pi_centroid = np.array([
    (Pi_np[t] * c_unit_np).sum() / (Pi_np[t].sum() + 1e-10)
    for t in range(T_eff)
])

fig, ax = plt.subplots(figsize=(14, 5))
if ca_centroid is not None:
    ax.plot(time_axis, ca_centroid, color="steelblue", linewidth=1.0, alpha=0.8, label="Baseline cross-attn")
ax.plot(time_axis, pi_centroid, color="coral", linewidth=1.2, label="Sinkhorn Pi")
for k in range(K):
    ax.axhline(y=c_unit_np[k], color="gray", linewidth=0.3, alpha=0.3)
ax.set_xlabel("Normalized audio time")
ax.set_ylabel("Attention centroid (unit position)")
ax.set_title(f"Attention Centroid: Baseline vs Sinkhorn\n{SAMPLE_ID}")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_centroid.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: attn_sinkhorn_centroid.png")

# ── Figure 4: Sinkhorn marginals ───────────────────────────────────────
# Instead of a pixel-based heatmap, show row/col marginals as bar/line plots
fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [1, 3]})

# Top: column marginal (total mass per unit) vs target mass
col_mass = Pi_np.sum(axis=0)  # [K]
col_mass_norm = col_mass / col_mass.sum()
mu_np = mu.cpu().numpy()

axes[0].bar(np.arange(K)-0.15, col_mass_norm, width=0.3, alpha=0.8, color='coral', label='Sinkhorn col marginal')
axes[0].bar(np.arange(K)+0.15, mu_np, width=0.3, alpha=0.6, color='steelblue', label='Target (mu)')
axes[0].set_ylabel('Mass')
axes[0].set_title('Column Marginal vs Target (unit mass)')
axes[0].legend(fontsize=8)
axes[0].set_xticks(np.arange(K))
unit_labels = [f"{SECTION_NAMES.get(int(unit_section_id[i].item()), '?')[:3]}{i}" for i in range(K)]
axes[0].set_xticklabels(unit_labels, rotation=90, fontsize=5)

# Bottom: row marginal (should be ~uniform)
row_mass = Pi_np.sum(axis=1)  # [T]
expected = 1.0 / T_eff
time_sub = np.linspace(0, 1, T_eff)
ds_vis = max(1, T_eff // 1000)
row_ds = row_mass[:T_eff//ds_vis*ds_vis].reshape(-1, ds_vis).mean(axis=1)
time_ds = time_sub[:T_eff//ds_vis*ds_vis].reshape(-1, ds_vis).mean(axis=1)
axes[1].plot(time_ds, row_ds, color='coral', linewidth=1.0, label=f'Row marginal (mean={row_mass.mean():.6f})')
axes[1].axhline(y=expected, color='steelblue', linestyle='--', linewidth=0.8, label=f'Expected (1/T={expected:.6f})')
axes[1].set_xlabel('Normalized audio time')
axes[1].set_ylabel('Row sum')
axes[1].set_title('Row Marginal (should be ~uniform for constraint satisfaction)')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle(f"Sinkhorn Transport Plan Pi — Marginal Validation\n{SAMPLE_ID}", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "attn_sinkhorn_pi_full.png", dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved: attn_sinkhorn_pi_full.png")

# ── Quantitative comparison ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"QUANTITATIVE COMPARISON")
print(f"{'='*60}")

if ca_centroid is not None:
    ca_spearman = spearmanr(ca_centroid, np.linspace(0, 1, T_eff))[0]
    print(f"Centroid vs time (baseline cross-attn):  rho={ca_spearman:.4f}")
pi_spearman = spearmanr(pi_centroid, np.linspace(0, 1, T_eff))[0]
print(f"Centroid vs time (Sinkhorn Pi):           rho={pi_spearman:.4f}")
print(f"Pi row_error:  {diag.get('row_error', -1):.6f}")
print(f"Pi col_error:  {diag.get('col_error', -1):.6f}")
print(f"Pi entropy:    {diag.get('entropy', -1):.6f}")
print(f"Row mass check: {Pi_np.sum(axis=1).mean():.4f} (should be ~{1/T_eff:.4f})")
print(f"Col mass check: {Pi_np.sum(axis=0).mean():.4f} (should be ~{mu.cpu().numpy().mean():.4f})")

# Per-unit coverage
unit_coverage = Pi_np.sum(axis=0)
unit_coverage_norm = unit_coverage / unit_coverage.sum()
print("\nPer-unit Sinkhorn coverage (top 10):")
top_k = np.argsort(unit_coverage_norm)[-10:][::-1]
for idx in top_k:
    sid = int(u_section_ids[idx].item())
    sn = SECTION_NAMES.get(sid, "?")
    print(f"  Unit {idx:2d} [{sn:>8}] pos={c_unit_np[idx]:.3f} mass_target={mu.cpu().numpy()[idx]:.4f} "
          f"coverage={unit_coverage_norm[idx]:.4f}")

print(f"\nAll figures saved to {OUTPUT_DIR}")
