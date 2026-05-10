import librosa
import torch
import numpy as np

def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    
    # 1. 使用 librosa 读取并自动重采样到 target_sample_rate，保持原声道数
    audio_np, sr = librosa.load(str(audio_path), sr=target_sample_rate, mono=False)

    # 2. 形状对齐：librosa 读单声道时返回 1D 数组 (samples,)，需要补齐为 (1, samples)
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    # 3. 转换为 PyTorch Tensor
    audio = torch.from_numpy(audio_np).float()

    # 4. 强制转换为双声道 (Stereo)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    # 5. 按照最大时长截断
    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    # 返回处理好的 tensor 和目标采样率
    return audio, target_sample_rate