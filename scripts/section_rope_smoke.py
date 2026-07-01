#!/usr/bin/env python3
"""
Section-RoPE Offset Smoke Test.

1. Checks section_ids distribution across dataset
2. Runs 1 batch forward/backward to verify gradient flow
3. Verifies config flags
"""

import sys, os, warnings
from pathlib import Path
from collections import Counter

import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

TENSOR_DIR = "/root/autodl-tmp/musicdata/train_tensors"

# ====================================================================
# STEP 1: section_ids distribution
# ====================================================================
print("=" * 70)
print("  STEP 1: Section IDs Distribution")
print("=" * 70)

from acestep.tgca.lyrics_parser import LyricsStructureParser, SECTION_VOCAB

ID_TO_NAME = {v: k for k, v in SECTION_VOCAB.items()}
parser = LyricsStructureParser()
pt_files = sorted(Path(TENSOR_DIR).glob("*.pt"))
print(f"  Files: {len(pt_files)}")

sample_files = pt_files[:200]
total_counts = Counter()
total_tokens = 0

for fpath in sample_files:
    data = torch.load(fpath, map_location="cpu", weights_only=True)
    metadata = data.get("metadata", {})
    lyrics_text = metadata.get("lyrics", "") if isinstance(metadata, dict) else ""
    L = data["encoder_hidden_states"].shape[0]
    total_tokens += L
    if lyrics_text:
        output = parser.parse(lyrics_text, num_chunks=L)
        for sid in output.section_type_ids.tolist():
            total_counts[sid] += 1
    else:
        total_counts[0] += L

print(f"\n  Distribution (200 files, {total_tokens} tokens):")
print(f"  {'ID':>4}  {'Name':<18}  {'Tokens':>10}  {'Ratio':>8}")
print("  " + "-" * 44)
for sid in sorted(total_counts.keys()):
    print(f"  {sid:>4}  {ID_TO_NAME.get(sid, '?'):<18}  {total_counts[sid]:>10}  {total_counts[sid]/total_tokens:>7.1%}")

unk_ratio = total_counts.get(0, 0) / total_tokens
if unk_ratio > 0.9:
    print(f"\n  WARNING: UNKNOWN {unk_ratio:.0%} > 90%")
else:
    print(f"\n  Coverage OK (UNKNOWN {unk_ratio:.1%})")

# ====================================================================
# STEP 2: Gradient flow test (loaded pretrained model)
# ====================================================================
print("\n" + "=" * 70)
print("  STEP 2: Gradient Flow Test (loaded model)")
print("=" * 70)

from acestep.handler import AceStepHandler
from acestep.tgca.section_rope import SectionRoPEOffset

dit_handler = AceStepHandler()
dit_status, dit_success = dit_handler.initialize_service(
    project_root="/root/autodl-tmp/Ace-Step1.5",
    config_path="acestep-v15-sft", device="cuda",
    use_flash_attention=False, compile_model=False, offload_to_cpu=False,
)
if not dit_success:
    print(f"  Model load failed: {dit_status}")
    sys.exit(1)

model = dit_handler.model
model.train()

# Disable PhaseMemory on all layers
for lm in model.decoder.layers:
    if getattr(lm, "use_phase_memory", False):
        lm.use_phase_memory = False

# Enable Section-RoPE on layer 12
model.config.use_section_rope_offset = True
model.config.section_rope_layers = [12]
model.config.use_pm = False
model.config.use_pm_kv = False
model.config.use_traj = False

layer12 = model.decoder.layers[12]
layer12.use_section_rope = True
layer12.section_rope_offset_module = SectionRoPEOffset(
    num_section_types=8, rope_pair_dim=16,
    max_offset=0.2, init_scale=0.01,
).cuda().to(model.dtype)
layer12.section_rope_time_dim = 32

# Freeze all except section_rope
for param in model.parameters():
    param.requires_grad = False
for name, mod in model.named_modules():
    if "section_rope_offset_module" in name.split("."):
        for p in mod.parameters():
            p.requires_grad = True

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total:,}")
print(f"  Trainable params: {trainable:,}")

# Get batch shapes
sample = torch.load(pt_files[0], map_location="cpu", weights_only=True)
B, T = 2, sample["target_latents"].shape[0]
L = 128
dtype = torch.bfloat16

sample2 = torch.load(pt_files[1], map_location="cpu", weights_only=True)
lyrics_text = sample2.get("metadata", {}).get("lyrics", "")
output = parser.parse(lyrics_text, num_chunks=L)
section_ids = output.section_type_ids.to(torch.long)
print(f"  Section IDs: {section_ids.unique().tolist()}")

# Forward/backward
hidden = torch.randn(B, T, 64, device="cuda", dtype=dtype, requires_grad=True)
context = torch.randn(B, T, 128, device="cuda", dtype=dtype)
enc_h = torch.randn(B, L, 2048, device="cuda", dtype=dtype)
enc_mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
attn_mask = torch.ones(B, T, dtype=dtype, device="cuda")

out = model.decoder(
    hidden_states=hidden,
    timestep=torch.full((B,), 0.5, device="cuda", dtype=dtype),
    timestep_r=torch.full((B,), 0.5, device="cuda", dtype=dtype),
    attention_mask=attn_mask,
    context_latents=context,
    encoder_hidden_states=enc_h,
    encoder_attention_mask=enc_mask,
    section_ids=section_ids.unsqueeze(0).expand(B, -1).cuda(),
)

loss = out[0].mean()
loss.backward()

sec_mod = layer12.section_rope_offset_module
w_gn = sec_mod.section_phase.weight.grad.norm().item() if sec_mod.section_phase.weight.grad is not None else 0.0
ls_g = sec_mod.log_scale.grad.item() if sec_mod.log_scale.grad is not None else 0.0

print(f"\n  Loss: {loss.item():.4f}")
print(f"  section_phase.weight.grad_norm: {w_gn:.8f}")
print(f"  log_scale.grad: {ls_g:.8f}")
print(f"  {'GRADIENT FLOWS!' if w_gn > 0 else 'grad_norm = 0'}")

# ====================================================================
# STEP 3: Verify flags
# ====================================================================
print("\n" + "=" * 70)
print("  STEP 3: Config Verification")
print("=" * 70)
for flag in ["use_pm", "use_pm_kv", "use_traj", "use_anchor",
             "use_entropy_controller", "use_kl_loss"]:
    print(f"  {flag}: {getattr(model.config, flag, False)}")
print(f"  Layer 12 use_phase_memory: {layer12.use_phase_memory}")
print(f"  Layer 12 use_section_rope: {layer12.use_section_rope}")
print(f"  Layer 12 has phase_memory: {hasattr(layer12, 'phase_memory')}")
print(f"  Layer 12 has section_rope: {hasattr(layer12, 'section_rope_offset_module')}")

print("\n" + "=" * 70)
print("  SMOKE TEST COMPLETE")
print("=" * 70)
