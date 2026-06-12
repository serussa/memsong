import os
from pathlib import Path
import librosa
import numpy as np

def extract_beat_phase(
    audio_path: str,
    sr: int = 48000,
    hop_length: int = 512,
    target_fps: int = 25,
) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units='frames'
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    total_seconds = len(y) / sr
    num_frames = int(np.ceil(total_seconds * target_fps))
    beat_phase = np.zeros(num_frames, dtype=np.float32)

    if len(beat_times) < 2:
        return beat_phase

    beat_indices = (beat_times * target_fps).astype(int)
    beat_indices = np.clip(beat_indices, 0, num_frames - 1)

    for i in range(len(beat_indices) - 1):
        start, end = beat_indices[i], beat_indices[i+1]
        if end <= start:
            continue
        length = end - start
        beat_phase[start:end] = np.linspace(0, 1, length, endpoint=False)

    if len(beat_indices) > 0:
        last_beat = beat_indices[-1]
        if last_beat < num_frames:
            remaining = num_frames - last_beat
            beat_phase[last_beat:] = np.linspace(0, 1, remaining, endpoint=False)

    return beat_phase


# ============ 在这里直接修改路径 ============
INPUT_DIR = "/root/autodl-tmp/musicdata/audios"  # 音频文件夹路径
OUTPUT_DIR = "/root/autodl-tmp/musicdata/beat_phases"  # npy 输出文件夹路径（若与输入相同则设为 None）
EXTENSION = ".mp3"                        # 音频文件扩展名
# ===========================================

def main():
    input_path = Path(INPUT_DIR)
    if OUTPUT_DIR:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    for audio_file in input_path.glob(f"*{EXTENSION}"):
        stem = audio_file.stem
        npy_file = output_path / f"{stem}.npy"
        if npy_file.exists():
            print(f"Skipping {stem} (already exists)")
            continue
        try:
            bp = extract_beat_phase(str(audio_file))
            np.save(str(npy_file), bp)
            print(f"Saved {npy_file.name} ({bp.shape[0]} frames)")
        except Exception as e:
            print(f"Failed {audio_file.name}: {e}")

if __name__ == "__main__":
    main()