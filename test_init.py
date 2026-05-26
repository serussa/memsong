import torch
import sys
from acestep.training_v2.fixed_lora_module import FixedLoRAModule

class DummyArgs:
    model_variant = 'sft'
    checkpoint_dir = '/root/autodl-tmp/Ace-Step1.5/checkpoints'
    adapter_type = 'phase_memory'
    precision = 'bf16'
    target_modules = []
    # Any other required fields
    lora_rank = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lokr_factor_a = 4
    init_scale = 0.01

args = DummyArgs()
mod = FixedLoRAModule(args)

phase_mem = mod.model.decoder.layers[12].phase_memory
print("\n=== DEBUG ===")
print("proj_out.weight.abs().sum():", phase_mem.proj_out.weight.abs().sum().item())
print("proj_out.bias.abs().sum():", phase_mem.proj_out.bias.abs().sum().item() if phase_mem.proj_out.bias is not None else "No bias")
print("z_init_real.abs().sum():", phase_mem.z_init_real.abs().sum().item())
print("================\n")
