#!/usr/bin/env python3
"""Generate a song using dataset lyrics and caption."""
import os, sys
from pathlib import Path

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
CHECKPOINT_DIR = Path("/root/autodl-tmp/lyrics_checkpoints/checkpoints")

# Pick latest PhaseMemory checkpoint
ckpts = sorted(CHECKPOINT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
LATEST_CKPT = ckpts[-1]

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.training.phase_memory_checkpoint import load_phase_memory_weights

# ========== Choose a song from the dataset ==========
data_dir = Path("/root/autodl-tmp/musicdata/audios")

songs = sorted(data_dir.glob("*.caption.txt"))
print("Available songs:")
for i, p in enumerate(songs[:20]):
    sid = p.stem.replace(".caption", "")
    cap = p.read_text().strip()[:60]
    print(f"  [{i+1}] {sid}: {cap}...")
print()

# Pick song 4 (ballad, female vocal, piano, 131bpm) which has nice structure
song_id = songs[3].stem.replace(".caption", "")

with open(data_dir / f"{song_id}.caption.txt") as f:
    caption = f.read().strip()
with open(data_dir / f"{song_id}.lyrics.txt") as f:
    lyrics = f.read().strip()

# Parse caption for BPM/key
# "ballad, female vocal, piano, strings, romantic, intense, 131 bpm, G major"
bpm = None
key = ""
for part in caption.split(","):
    part = part.strip()
    if "bpm" in part.lower():
        try:
            bpm = int(part.lower().replace("bpm", "").strip())
        except:
            pass
    elif len(part.split()) == 2 and part.split()[1].lower() in ["major", "minor"]:
        key = part

duration = 180  # 3 minutes
print(f"Song: {song_id}")
print(f"Caption: {caption}")
print(f"BPM: {bpm}, Key: {key}")
print(f"Lyrics ({len(lyrics)} chars)")
print(f"Checkpoint: {LATEST_CKPT.name}")

# ========== Init models ==========
print("\n[1/4] Initializing DiT...")
dit = AceStepHandler()
status, ok = dit.initialize_service(
    project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
    device="cuda", use_flash_attention=False, compile_model=False,
)
if not ok:
    print(f"DiT init failed: {status}")
    sys.exit(1)

print("[2/4] Loading PhaseMemory checkpoint...")
load_phase_memory_weights(dit.model, str(LATEST_CKPT))
dit.model.eval()

print("[3/4] Initializing LLM...")
llm = LLMHandler()
ok = llm.initialize(
    checkpoint_dir=str(MODEL_ROOT), lm_model_path="acestep-5Hz-lm-1.7B",
    backend="pt", device="cuda",
)
if not ok:
    print("LLM init failed")
    sys.exit(1)

# ========== Generate ==========
print("[4/4] Generating...\n")
params = GenerationParams(
    task_type="text2music",
    caption=caption,
    lyrics=lyrics,
    instrumental=False,
    bpm=bpm,
    keyscale=key,
    vocal_language="zh",
    duration=duration,
    inference_steps=30,
    guidance_scale=7.0,
    seed=42,
    thinking=True,
    use_cot_metas=True,
    use_cot_caption=True,
    lm_temperature=0.75,
    use_beat_alignment=True,
)
config = GenerationConfig(batch_size=1, use_random_seed=False)

OUTPUT_DIR = ACE_STEP_ROOT / "output" / f"generate_{song_id}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

result = generate_music(dit, llm, params, config, save_dir=str(OUTPUT_DIR))
if result.success:
    print(f"\n✓ Song generated!")
    for a in result.audios:
        print(f"  → {a['path']}")
else:
    print(f"\n✗ Failed: {result.error}")
