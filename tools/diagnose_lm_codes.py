#!/root/miniconda3/envs/musicgen/bin/python
"""
Trace where code count is decided: LM generation, constrained decoding, or postproc.
"""
import sys, os, json, re, pickle, subprocess, difflib
from pathlib import Path

PROJECT_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
os.environ["SIDESTEP_SAFE_ROOT"] = "/root/autodl-tmp"

import torch
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.handler import AceStepHandler

TEST_JSONL = PROJECT_ROOT / "Muse/infer/test.jsonl"
QWEN_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-ASR-1.7B")
PIPELINE_DIR = PROJECT_ROOT / "Muse/eval_pipeline"

# Track 3 songs across the duration range
for idx in [0, 3, 7]:
    print(f"\n{'='*60}")
    print(f"Song {idx}: Full diagnostic")
    print(f"{'='*60}")

    # Parse
    with open(TEST_JSONL) as f:
        for i, l in enumerate(f):
            if i == idx: d = json.loads(l); break
    msgs = d["messages"]
    style = msgs[0]["content"].split(chr(10))[0].replace(
        "Please generate a song in the following style:", "").strip()
    seen, parts = set(), []
    for msg in msgs:
        c = msg.get("content", "")
        m = re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m: continue
        n = m.group(1)
        if n in seen: continue
        seen.add(n)
        rest = c[m.end():]; l = ""
        if "[lyrics:" in rest: l = rest.split("[lyrics:")[1].split("]")[0].strip()
        parts.append(m.group(0))
        if l: parts.append(l)
        parts.append("")
    lyrics = chr(10).join(parts).strip()
    clean_lyrics = re.sub(r'\[[^\]]+\]\n*', '', lyrics).strip()
    char_count = len(clean_lyrics)

    # Load LLM
    llm = LLMHandler()
    llm.initialize(checkpoint_dir="/root/autodl-tmp/Ace-Step1.5/checkpoints",
                   lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")

    # ========== Phase 1: CoT ==========
    cot_res = llm.generate_with_stop_condition(
        caption=style, lyrics=lyrics, infer_type="dit",
        temperature=0.75, cfg_scale=2.0, target_duration=180,
        use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
        use_constrained_decoding=True, batch_size=1, seeds=[42])
    meta = cot_res["metadata"]
    lm_dur = meta.get("duration", "N/A")
    print(f"\n  Phase 1 (CoT):")
    print(f"    LM says duration: {lm_dur}")
    print(f"    Lyrics chars: {char_count}")

    # ========== Phase 2: Codes with high debug ==========
    raw_codes = ""
    print(f"\n  Phase 2 (codes):")
    full_res = llm.generate_with_stop_condition(
        caption=style, lyrics=lyrics, infer_type="llm_dit",
        temperature=0.75, cfg_scale=2.0, target_duration=None,  # ← Key: no constraint
        use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
        use_constrained_decoding=True, batch_size=1, seeds=[42])

    r_codes = full_res["audio_codes"]
    if isinstance(r_codes, list) and len(r_codes) > 0:
        raw_codes = r_codes[0]
    elif isinstance(r_codes, str):
        raw_codes = r_codes

    raw_count = len(raw_codes.split("<|audio_code_")) - 1 if raw_codes else 0

    # Find EOS in raw output
    eos_pos = -1
    eos_id = llm.llm_tokenizer.eos_token_id
    # We can also look for <|im_end|> in the raw text
    extra = full_res.get("extra_outputs", {})
    # The raw text from the output is not directly exposed, but we can check
    # via the constrained processor logs

    print(f"    Raw codes string length: {len(raw_codes)}")
    print(f"    Raw code count: {raw_count}")
    print(f"    Expected at 5Hz: {raw_count/5:.1f}s")

    # ========== Step 3: Decode codes back to latent ==========
    print(f"\n  DiT decode:")
    dt = AceStepHandler()
    dt.initialize_service(project_root="/root/autodl-tmp/Ace-Step1.5",
        config_path="acestep-v15-sft", device="cuda",
        use_flash_attention=False, compile_model=False, offload_to_cpu=False)
    model = dt.model.eval()
    model.config.use_section_rope_offset = False

    has_cjk = any('一' <= c <= '鿿' for c in lyrics)
    params = GenerationParams(
        task_type='text2music', caption=style, lyrics=lyrics,
        instrumental=False, bpm=120, keyscale='C major', timesignature='4',
        vocal_language='zh' if has_cjk else 'en',
        duration=180, inference_steps=50, guidance_scale=7.0,
        seed=42, thinking=False, use_cot_metas=False, use_cot_caption=False,
        audio_codes=raw_codes)
    config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[42])

    out_dir = Path(f"/tmp/lm_diag_codes_{idx}")
    out_dir.mkdir(exist_ok=True)
    result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir=str(out_dir))

    if result.audios:
        import soundfile as sf
        data, sr = sf.read(result.audios[0]['path'])
        dur_real = len(data) / sr
        n_frames = result.extra_outputs.get("latent_shape", [0, 0, 0])[2] if result.extra_outputs else 0
        print(f"    Codes→DiT duration: {dur_real:.1f}s")
        print(f"    Latent frames: {dur_real*25:.0f} (target: {raw_count/5*25:.0f})")

    # ========== ASR + PER on this audio ==========
    print(f"\n  ASR + PER:")
    audio_dir = Path(f"/tmp/lm_diag_codes_{idx}")
    subprocess.run([sys.executable, str(PIPELINE_DIR / "transcribe_local.py"),
        "--input_dir", str(audio_dir), "--output", str(audio_dir / "trans.jsonl"),
        "--model_path", str(QWEN_MODEL_PATH)], capture_output=True, timeout=300)
    subprocess.run([sys.executable, str(PIPELINE_DIR / "calc_per_long.py"),
        "--hyp_file", str(audio_dir / "trans.jsonl"),
        "--gt_file", str(PIPELINE_DIR / "gt_lyrics" / "zh.jsonl"),
        "--model_name", f"diag_{idx}", "--output_dir", str(audio_dir / "per")],
        capture_output=True, timeout=60)

    per_path = audio_dir / "per" / "songs.jsonl"
    if per_path.exists():
        for line in open(per_path):
            d = json.loads(line)
            print(f"    PER: {d['overall_per']:.4f} early={d['early_per']:.4f} late={d['late_per']:.4f} ldg={d['ldg']:.4f}")

    print(f"\n  {'='*40}")
    print(f"  SUMMARY song {idx}: {char_count}chars → {raw_count}codes ≈ {raw_count/5:.1f}s")
    print(f"  DiT actual: {dur_real:.1f}s, PER={d.get('overall_per', 0):.4f}")
