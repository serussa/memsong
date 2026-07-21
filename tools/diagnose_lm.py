#!/root/miniconda3/envs/musicgen/bin/python
"""
LM Diagnostic: probe whether 5Hz audio_codes contain recoverable lyrics.

Tests:
  1. Codes→text: ask LM to transcribe its own audio_codes back to lyrics
  2. Audio→text: normal ASR on generated audio

Usage:
  python tools/diagnose_lm.py <song_idx>
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

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"[lm_diag] Song {idx}")

# Parse entry
with open(TEST_JSONL) as f:
    for i, line in enumerate(f):
        if i == idx: data = json.loads(line); break
msgs = data["messages"]
style = msgs[0]["content"].split("\n")[0].replace(
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
lyrics = "\n".join(parts).strip()
clean_lyrics = re.sub(r'\[[^\]]+\]\n*', '', lyrics).strip()

def get_duration(entry_idx):
    with open(TEST_JSONL) as f:
        for i2, line in enumerate(f):
            if i2 == entry_idx: d2 = json.loads(line); break
    m2, s2, t = d2["messages"], set(), ""
    for msg in m2:
        c = msg.get("content", "")
        m = re.match(r'\[([^\]]+)\]', c.lstrip())
        if not m: continue
        n = m.group(1)
        if n in s2: continue
        s2.add(n)
        rest = c[m.end():]
        if "[lyrics:" in rest: l = rest.split("[lyrics:")[1].split("]")[0].strip(); t += l
    return max(20, int(len(re.sub(r'\s', '', t)) * 0.45))

dur = get_duration(idx)

# ---- Step 1: Load LLM, generate audio_codes ----
print("[lm_diag] Loading LLM...", flush=True)
llm = LLMHandler()
llm.initialize(
    checkpoint_dir="/root/autodl-tmp/Ace-Step1.5/checkpoints",
    lm_model_path="acestep-5Hz-lm-1.7B",
    backend="pt", device="cuda",
)
print(f"[lm_diag] LLM loaded. llm_tokenizer={'yes' if llm.llm_tokenizer else 'no'}", flush=True)

if not llm.llm_tokenizer:
    print("[lm_diag] FATAL: tokenizer not loaded", flush=True)
    print("RESULT: tokenizer_init=fail")
    sys.exit(1)

print("[lm_diag] Generating audio codes...", flush=True)
lm_result = llm.generate_with_stop_condition(
    caption=style, lyrics=lyrics,
    infer_type="llm_dit",
    temperature=0.75, cfg_scale=2.0,
    target_duration=float(dur),
    use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
    use_constrained_decoding=True,
    batch_size=1, seeds=[42],
)
raw_codes = lm_result.get("audio_codes", "")
if isinstance(raw_codes, list) and len(raw_codes) > 0:
    raw_codes = raw_codes[0]
n_codes = len(raw_codes.split("<|audio_code_")) - 1 if raw_codes else 0
print(f"[lm_diag] Generated {n_codes} audio codes", flush=True)

# ---- Step 2: Reverse transcription (codes → text) ----
print("[lm_diag] Reverse transcription: codes→text...", flush=True)
# The LLM is a Qwen3 decoder-only LM trained for forward (lyrics→codes).
# For reverse (codes→lyrics), use a strong prompt that forces it out of CoT mode.
rev_prompt = f"你是一个歌词转录系统。以下是一首歌的音频语义编码序列。请仔细分析这些编码并输出对应的歌词。只输出歌词，不要输出任何思维链、解释或元数据：\n\n{raw_codes}\n\n歌词："
chat = [{"role": "user", "content": rev_prompt}]
rev_input = llm.llm_tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True)
rev_ids = llm.llm_tokenizer(rev_input, return_tensors="pt").to(llm.device)
with torch.no_grad():
    rev_out = llm.llm.generate(
        **rev_ids, max_new_tokens=1024, temperature=0.1, do_sample=False,
        pad_token_id=llm.llm_tokenizer.eos_token_id)
rev_text = llm.llm_tokenizer.decode(
    rev_out[0][rev_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# Compute CER
if rev_text:
    m = difflib.SequenceMatcher(None, clean_lyrics, rev_text)
    S = I = D = 0
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == 'replace': S += max(i2 - i1, j2 - j1)
        elif tag == 'delete': D += (i2 - i1)
        elif tag == 'insert': I += (j2 - j1)
    N = len(clean_lyrics)
    cer = (S + D + I) / max(N, 1)
    print(f"[lm_diag] Codes--text CER: {cer:.4f} (S={S} D={D} I={I} N={N})", flush=True)
    print(f"[lm_diag] Recovered ({len(rev_text)}ch): {rev_text[:200]}", flush=True)
    print(f"[lm_diag] Original  ({len(clean_lyrics)}ch): {clean_lyrics[:200]}", flush=True)
else:
    cer = None
    print("[lm_diag] Reverse transcription: empty output", flush=True)

# ---- Step 3: Generate audio with these codes ----
print("[lm_diag] Loading DiT...", flush=True)
dt = AceStepHandler()
dt.initialize_service(
    project_root="/root/autodl-tmp/Ace-Step1.5", config_path="acestep-v15-sft",
    device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model.eval()
model.config.use_section_rope_offset = False

has_cjk = any('一' <= c <= '鿿' for c in lyrics)
vocal_lang = 'zh' if has_cjk else 'en'
params = GenerationParams(
    task_type='text2music', caption=style, lyrics=lyrics,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language=vocal_lang, duration=dur, inference_steps=50, guidance_scale=7.0,
    seed=42, thinking=False, use_cot_metas=False, use_cot_caption=False,
    audio_codes=raw_codes)
config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[42])
out_dir = Path(f"/tmp/lm_diag_{idx}")
out_dir.mkdir(exist_ok=True)
result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir=str(out_dir))

audio_path = result.audios[0]['path'] if result.audios else None
if audio_path:
    print(f"[lm_diag] Audio: {audio_path}", flush=True)

    # ---- Step 4: ASR on generated audio ----
    print("[lm_diag] Running ASR...", flush=True)
    trans_file = out_dir / "trans.jsonl"
    subprocess.run([
        sys.executable, str(PIPELINE_DIR / "transcribe_local.py"),
        "--input_dir", str(Path(audio_path).parent), "--output", str(trans_file),
        "--model_path", str(QWEN_MODEL_PATH),
    ], capture_output=True, timeout=300)

    hyp_text = ""
    if trans_file.exists():
        for line in open(trans_file):
            d = json.loads(line)
            hyp_text = d.get("hyp_text", "")

    if hyp_text:
        m2 = difflib.SequenceMatcher(None, clean_lyrics, hyp_text)
        Sa = Ia = Da = 0
        for tag, i1, i2, j1, j2 in m2.get_opcodes():
            if tag == 'replace': Sa += max(i2 - i1, j2 - j1)
            elif tag == 'delete': Da += (i2 - i1)
            elif tag == 'insert': Ia += (j2 - j1)
        Na = len(clean_lyrics)
        audio_cer = (Sa + Da + Ia) / max(Na, 1)
        print(f"[lm_diag] Audio--text CER: {audio_cer:.4f} (S={Sa} D={Da} I={Ia} N={Na})", flush=True)
        print(f"[lm_diag] ASR ({len(hyp_text)}ch): {hyp_text[:200]}", flush=True)
    else:
        audio_cer = None
        print("[lm_diag] ASR: no output", flush=True)
else:
    print("[lm_diag] No audio output", flush=True)
    audio_cer = None

# ---- Summary ----
print("\n[lm_diag] " + "=" * 50, flush=True)
print(f"[lm_diag] Song {idx}: {n_codes} codes", flush=True)
print(f"[lm_diag] Codes--text CER: {cer:.4f}" if cer else "[lm_diag] Codes--text: FAIL", flush=True)
print(f"[lm_diag] Audio--text CER: {audio_cer:.4f}" if audio_cer else "[lm_diag] Audio--text: FAIL", flush=True)

if cer is not None and audio_cer is not None:
    if cer < 0.3 and audio_cer > 0.3:
        print("[lm_diag] CONCLUSION: Codes contain lyrics (CER<0.3), but audio PER is high.", flush=True)
        print("[lm_diag]   → Problem is in DiT execution, NOT LM planning.", flush=True)
    elif cer >= 0.3 and audio_cer < 0.3:
        print("[lm_diag] CONCLUSION: Codes miss lyrics but audio is OK.", flush=True)
        print("[lm_diag]   → ASR recovers from audio directly, LM codes are redundant.", flush=True)
    elif cer < 0.3 and audio_cer < 0.3:
        print("[lm_diag] CONCLUSION: Both codes and audio work well.", flush=True)
        print("[lm_diag]   → LM and DiT both functional for PER.", flush=True)
    else:
        print("[lm_diag] CONCLUSION: Both codes and audio degrade.", flush=True)
        print("[lm_diag]   → Either LM codes miss info, or ASR is the bottleneck.", flush=True)
