#!/usr/bin/env python3
"""Continue song_progress experiment: generate missing trajectories."""
import sys, os, warnings
from pathlib import Path
import numpy as np
import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
warnings.filterwarnings("ignore")

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

TRAJ_DIR = Path("output/song_progress/trajectories")
LYRICS_DIR = Path("/root/autodl-tmp/musicdata/audios")
SEEDS = [42, 123, 999]

# Missing runs to generate
PENDING = [
    ("complex", "rap, hip hop, male vocal, drums, bass, synthesizer, energetic, 141 bpm, D major",
     "02146eebf9c7af14bfdf7fd4235bc060d648b8ed_984.lyrics.txt", [123, 999]),
    ("repetitive", "pop, female vocal, piano, drums, bass, emotional, 120 bpm, D major",
     "3fcb749666691c3cd973f27f4dd3332ffd322d02_999.lyrics.txt", [42, 123, 999]),
]

class Collector:
    def __init__(self):
        self.states = []
    def __call__(self, mod, inp, out):
        hs = out[0]
        if hs.shape[0] > 1: hs = hs[:1]
        self.states.append(hs.detach().cpu())
    def get_traj(self):
        if not self.states: return None
        return torch.cat(self.states, dim=0).float().numpy()
    def reset(self):
        self.states = []

print("Loading model...")
dit_handler = AceStepHandler()
dit_status, dit_success = dit_handler.initialize_service(
    project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
    device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
if not dit_success: print("FAILED"); sys.exit(1)
for lm in dit_handler.model.decoder.layers:
    if getattr(lm, "use_phase_memory", False): lm.use_phase_memory = False
collector = Collector()
handle = dit_handler.model.decoder.layers[12].register_forward_hook(collector)

total = sum(len(s) for _,_,_, s in PENDING) * 2  # instrumental + lyrics
done = 0
for name, caption, lpath, seeds in PENDING:
    lyrics_text = open(LYRICS_DIR / lpath).read().strip()
    for seed in seeds:
        for mode, lyrics_val in [("instrumental", "[Instrumental]"), ("lyrics", lyrics_text)]:
            key = f"traj_{name}_{mode}_s{seed}.npy"
            if (TRAJ_DIR / key).exists():
                print(f"  SKIP {key}")
                continue
            collector.reset()
            try:
                params = GenerationParams(caption=caption, lyrics=lyrics_val,
                    instrumental=(mode=="instrumental"), duration=30,
                    inference_steps=50, guidance_scale=5.0, seed=seed,
                    thinking=False, use_cot_caption=False, use_cot_metas=False)
                config = GenerationConfig(batch_size=1, audio_format="mp3", use_random_seed=False)
                generate_music(dit_handler, None, params, config)
                X = collector.get_traj()
                if X is not None:
                    np.save(TRAJ_DIR / key, X)
                    done += 1
                    print(f"  OK {key} ({X.shape}) [{done}/{total}]")
                else:
                    print(f"  FAIL {key}: no states")
            except Exception as e:
                print(f"  ERROR {key}: {e}")

handle.remove()
print(f"\nDone. Generated {done} trajectories. Total in dir: {len(list(TRAJ_DIR.glob('*.npy')))}")
