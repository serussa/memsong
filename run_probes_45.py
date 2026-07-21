#!/usr/bin/env python3
"""Run probes 4 and 5 with fixed attention perturbation hooks."""
import sys, os
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
import logging, warnings
logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

handler = AceStepHandler()
handler.initialize_service(
    project_root='/root/autodl-tmp/Ace-Step1.5',
    config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False,
    compile_model=False, offload_to_cpu=False,
)
model = handler.model.eval()
model.config.use_section_rope_offset = False
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False

llm = LLMHandler()
llm.initialize(
    checkpoint_dir='/root/autodl-tmp/Ace-Step1.5/checkpoints',
    lm_model_path='acestep-5Hz-lm-1.7B',
    backend='pt', device='cuda',
)

from run_baseline_progression_probe import (
    AttentionPerturbationEngine, run_baseline,
    probe4_layer_sensitivity, probe5_step_sensitivity,
    OUTPUT_DIR, SHORT_LYRICS, BASE_CAPTION, SEEDS, BASE_DURATION,
    save_table, collect_attention_statistics
)

# Quick test
print('Testing perturbation hook...')
pert = AttentionPerturbationEngine(model)
pert.apply_caption_attenuation(12, 0.2, n_caption_tokens=10)

from acestep.inference import GenerationParams, GenerationConfig, generate_music
params = GenerationParams(
    task_type='text2music',
    caption=BASE_CAPTION, lyrics=SHORT_LYRICS,
    instrumental=False, bpm=120, keyscale='C major',
    timesignature='4', vocal_language='zh',
    duration=15, inference_steps=20,
    guidance_scale=7.0, seed=42,
    thinking=True, use_cot_metas=True, use_cot_caption=True,
    lm_temperature=0.75,
)
config = GenerationConfig(
    batch_size=1, audio_format='flac',
    use_random_seed=False, seeds=[42],
)

out = None
try:
    result = generate_music(dit_handler=handler, llm_handler=llm,
                          params=params, config=config,
                          save_dir='/tmp/test_pert_v3')
    print(f'Test succeeded: {result.success}')
    out = result
except Exception as e:
    print(f'Test error: {e}')
    import traceback; traceback.print_exc()
finally:
    pert.clear_all()

if out is None or not out.success:
    print('Test failed, aborting probes 4/5')
    sys.exit(1)

print()
print('Running Probe 4...')
r4 = probe4_layer_sensitivity(model, llm, handler, nprompts=2)
s4 = sum(1 for r in r4 if r.get('success'))
print(f'  Done: {s4}/{len(r4)} successful')

print()
print('Running Probe 5...')
r5 = probe5_step_sensitivity(model, llm, handler, nprompts=2)
s5 = sum(1 for r in r5 if r.get('success'))
print(f'  Done: {s5}/{len(r5)} successful')

print()
print('Running attention statistics...')
collect_attention_statistics(model, llm, handler, nprompts=2)

print()
print('All complete!')
