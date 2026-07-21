#!/usr/bin/env python3
"""
Audiobox Evaluation: Evaluate audio aesthetic scores
Usage: python eval_audiobox.py --input_dir <audio_directory> --model_name <model_name> --output <output_file>
Output: Summary results + _details.jsonl detailed results
"""
import argparse, json, os, glob, re
import librosa
import torch

# Monkey-patch audiobox_aesthetics to use librosa instead of torchaudio
import audiobox_aesthetics.infer as ab_infer
_orig_read_wav = ab_infer.read_wav

def _patched_read_wav(meta):
    path = meta if isinstance(meta, str) else meta.get("path", "")
    wav, sr = librosa.load(path, sr=16000, mono=True)
    wav_t = torch.from_numpy(wav).unsqueeze(0).float()
    return wav_t, sr

ab_infer.read_wav = _patched_read_wav

from audiobox_aesthetics.infer import AesPredictor


def extract_idx(filename):
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", default="xxx/audiobox-aesthetics_ckpt/checkpoint.pt")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.input_dir}/*.wav") + glob.glob(f"{args.input_dir}/*.mp3") + glob.glob(f"{args.input_dir}/*.flac"))
    if not files:
        print("No audio files found")
        return

    ckpt = args.ckpt if os.path.exists(args.ckpt) else None
    predictor = AesPredictor(checkpoint_pth=ckpt)

    batch = [{"path": os.path.abspath(f)} for f in files]
    all_rows = predictor.forward(batch)

    metrics = {"CE": [], "CU": [], "PC": [], "PQ": []}
    details = []

    for i, (f, row) in enumerate(zip(files, all_rows)):
        scores = {k: row[k] for k in ["CE", "CU", "PC", "PQ"]}
        scores["Score"] = sum(scores.values())
        filename = os.path.basename(f)
        for k in metrics:
            metrics[k].append(scores[k])
        details.append({
            "file": filename,
            "idx": extract_idx(filename),
            "scores": scores,
        })

    avg = {k: sum(v)/len(v) if v else 0 for k, v in metrics.items()}
    avg["Score"] = sum(avg.values())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({"model": args.model_name, "metrics": avg, "count": len(files)}, f, indent=2)

    details_file = args.output.replace('.json', '_details.jsonl')
    with open(details_file, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"Saved: {args.output}")
    print(f"Details: {details_file}")


if __name__ == "__main__":
    main()
