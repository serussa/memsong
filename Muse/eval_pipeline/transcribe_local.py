#!/usr/bin/env python3
"""
Local ASR Transcription using Qwen3-ASR-1.7B (local model)

Usage:
    python transcribe_local.py --input_dir <audio_dir> --output <output.jsonl> [--language zh]

Output: JSONL with fields file_path, file_name, file_idx, hyp_text
"""
import argparse, json, os, re, glob
from tqdm import tqdm
import torch
from qwen_asr import Qwen3ASRModel

os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None


def load_existing(output_path):
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                existing[os.path.abspath(rec['file_path'])] = rec
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Audio directory")
    parser.add_argument("--output", required=True, help="Output transcription file (jsonl)")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-ASR-1.7B",
                        help="Local Qwen3-ASR model path")
    parser.add_argument("--force", action="store_true", help="Force re-transcribe all files")
    parser.add_argument("--language", type=str, default=None,
                        help="Language hint for ASR (zh, en, or None for auto)")

    args = parser.parse_args()

    files = sorted(
        glob.glob(f"{args.input_dir}/*.wav") +
        glob.glob(f"{args.input_dir}/*.mp3") +
        glob.glob(f"{args.input_dir}/*.flac")
    )
    print(f"Found {len(files)} audio files")

    if not files:
        print("No audio files found, exiting.")
        return

    existing = {} if args.force else load_existing(args.output)
    pending = [f for f in files if os.path.abspath(f) not in existing]
    done_count = len(files) - len(pending)

    if not pending:
        print(f"All {done_count} files already transcribed. Model load skipped.")
        return

    if done_count > 0:
        print(f"{done_count} files already done, {len(pending)} remaining.")

    print("Loading local Qwen3-ASR model ...")
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=2048,
    )
    print("Model loaded.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for rec in existing.values():
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        for audio_path in tqdm(pending, desc="Transcribing"):
            audio_path_abs = os.path.abspath(audio_path)
            filename = os.path.basename(audio_path_abs)
            idx = extract_idx(filename)
            try:
                results = model.transcribe(audio=audio_path_abs, language=args.language)
                hyp_text = results[0].text
            except Exception as e:
                print(f"ASR Error {audio_path}: {e}")
                hyp_text = ""

            rec = {
                "file_path": audio_path_abs,
                "file_name": filename,
                "file_idx": idx,
                "hyp_text": hyp_text
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
