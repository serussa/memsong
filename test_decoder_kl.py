"""Test gradient flow through decoder with gradient checkpointing enabled."""
import os
os.environ["ACESTEP_LOCAL_MODEL_CODE"] = "1"
os.environ["SIDESTEP_SAFE_ROOT"] = "/"

import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from acestep.models.sft.modeling_acestep_v15_base import AceStepConditionGenerationModel
from acestep.models.sft.configuration_acestep_v15 import AceStepConfig
from acestep.training_v2.trainer_helpers import configure_memory_features

import warnings
warnings.filterwarnings("ignore")

print("Loading model...")
config = AceStepConfig.from_pretrained("/root/autodl-tmp/Ace-Step1.5")
model = AceStepConditionGenerationModel(config)
model = model.to("cuda", dtype=torch.bfloat16)
model.decoder.gradient_checkpointing = False  # start without ckpt

# Find PhaseMemory
pm = None
for m in model.modules():
    if hasattr(m, "kl_loss"):
        pm = m
        break
print(f"PhaseMemory found: {pm is not None}")
if pm:
    print(f"bias_net[-1].weight[:3]: {pm.bias_net[-1].weight[0, :3].detach().cpu().tolist()}")

# Create dummy batch
B, T = 2, 7500  # 30s at 25Hz
T_patches = T // config.patch_size  # 3750
device = "cuda"
dtype = torch.bfloat16

hidden_states = torch.randn(B, T, 64, device=device, dtype=dtype)
attention_mask = torch.ones(B, T, device=device, dtype=dtype)
src_latents = torch.randn(B, T, 64, device=device, dtype=dtype)
chunk_masks = torch.ones(B, T, 64, device=device, dtype=dtype)
is_covers = torch.zeros(B, device=device, dtype=torch.long)
silence_latent = torch.randn(B, T, 64, device=device, dtype=dtype)
text_hidden_states = torch.randn(B, 77, 1024, device=device, dtype=dtype)
text_attention_mask = torch.ones(B, 77, device=device, dtype=dtype)
lyric_hidden_states = torch.randn(B, 123, 1024, device=device, dtype=dtype)
lyric_attention_mask = torch.ones(B, 123, device=device, dtype=dtype)
refer_audio = torch.randn(3, 750, 64, device=device, dtype=dtype)
refer_mask = torch.tensor([0, 0, 1], device=device)
beat_phase = torch.rand(B, T, device=device, dtype=dtype)

# Test 1: WITHOUT gradient checkpointing
print("\n--- Test 1: No gradient checkpointing ---")
with torch.no_grad():
    out = model.decoder(
        hidden_states=src_latents,
        timestep=torch.full((B,), 0.5, device=device),
        timestep_r=torch.full((B,), 0.5, device=device),
        attention_mask=attention_mask,
        encoder_hidden_states=text_hidden_states,
        encoder_attention_mask=text_attention_mask,
        context_latents=torch.cat([src_latents, chunk_masks], dim=-1),
        beat_phase=beat_phase,
    )
    print(f"decoder output tuple length: {len(out)}")
    print(f"out[2] exists: {len(out) >= 3}, type={type(out[2]).__name__ if len(out) >= 3 else 'N/A'}")
    if len(out) >= 3 and isinstance(out[2], torch.Tensor):
        print(f"out[2] (kl_loss): {out[2].item():.6f}")

print("\nDone. Test 1 passed (forward only, no backward needed).")
