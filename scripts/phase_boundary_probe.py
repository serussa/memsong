#!/usr/bin/env python3
"""
phase_boundary_probe.py

Probe: Does phase_score predict structural boundaries?

从 PhaseMemory 提取 phi = atan2(z_i, z_r)，计算 phase score，
检验是否能预测三种结构边界：
  A. 歌词段落边界 (verse/chorus/bridge 切换)
  B. Latent 突变点 (生成音频的帧间跳变)
  C. Cross-attention 变化 (token 对齐模式切换)

用法:
  python scripts/phase_boundary_probe.py --quick                    # quick模式，随机权重
  python scripts/phase_boundary_probe.py --model-root /path         # 真实模型
  python scripts/phase_boundary_probe.py --analyze-only /path/to/pt # 仅分析已有数据
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─── 路径常量 ───
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
CHECKPOINT_DIR = Path("/root/autodl-tmp/lyrics_checkpoints/checkpoints")

checkpoints = sorted(CHECKPOINT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
LATEST_CKPT = checkpoints[-1] if checkpoints else None

# ─── 默认参数 ───
DURATION = 30
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.0
SEED = 42

SAMPLE_LYRICS = """[Intro]
[Verse 1]
风轻轻吹过 那片安静的天空
云朵在游走 像你离开时的笑容
我站在原地 看时间慢慢溜走
回忆在翻涌 却再也回不到那个路口

[Pre-Chorus]
那时候我们 总以为来日方长
谁知道转身 就是最后一行

[Chorus]
就让风带走 所有的思念
让我一个人 习惯没有你的夜
就让风带走 所有的从前
剩下这首歌 陪我到永远的永远

[Instrumental]

[Verse 2]
雨落在窗前 模糊了整条街
你的影子却 越来越清晰可见
我试着放手 却握得更紧一点
爱就像风筝 断了线还在眷恋

[Pre-Chorus]
后来才明白 有些话不必说完
有些人注定 只陪你走一段

[Chorus]
就让风带走 所有的思念
让我一个人 习惯没有你的夜
就让风带走 所有的从前
剩下这首歌 陪我到永远的永远

[Bridge]
如果有一天 我们还能再见面
我会微笑着 对你说一声谢谢
谢谢你陪我 走过那一场冒险
虽然结局是 各自飞向天边

[Chorus]
就让风带走 所有的思念
让我一个人 习惯没有你的夜
就让风带走 所有的从前
剩下这首歌 陪我到永远的永远

[Outro]
风停了 我也该 走远了"""


# ═══════════════════════════════════════════
# 1. Lyric 边界解析
# ═══════════════════════════════════════════
SECTION_MARKERS = [
    "Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge",
    "Instrumental", "Outro", "Solo", "Breakdown", "Drop",
    "Build-up", "Interlude", "Refrain",
]


def parse_lyric_boundaries(lyrics: str) -> Dict:
    """
    解析歌词段落标记，提取边界。
    返回 { sections, n_lines, n_sections, section_boundary_lines, lines, section_names_map }
    """
    lines = lyrics.strip().split("\n")
    sections = []
    current_name = None
    current_start = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            tag = line[1:-1]
            base = tag.split()[0] if tag.split() else tag
            if base in SECTION_MARKERS:
                if current_name is not None:
                    sections.append({
                        "name": current_name,
                        "line_start": current_start,
                        "line_end": i - 1,
                        "n_lines": i - current_start,
                    })
                current_name = tag
                current_start = i

    if current_name is not None:
        sections.append({
            "name": current_name,
            "line_start": current_start,
            "line_end": len(lines) - 1,
            "n_lines": len(lines) - current_start,
        })

    section_names = {}
    for s in sections:
        for ln in range(s["line_start"], s["line_end"] + 1):
            section_names[ln] = s["name"]

    section_boundary_lines = sorted(set(s["line_start"] for s in sections))
    return {
        "sections": sections,
        "n_lines": len(lines),
        "n_sections": len(sections),
        "section_boundary_lines": section_boundary_lines,
        "lines": lines,
        "section_names": section_names,
    }


# ═══════════════════════════════════════════
# 2. Hook 系统
# ═══════════════════════════════════════════
class PhaseMemoryHook:
    """
    PhaseMemory forward hook.

    在 PM.forward() 执行完后，从 buffer 读取更新后的 z_r, z_i，
    计算 phi = atan2(z_i, z_r)，累积到 self.phi_history。

    处理 CFG: 如果 batch=2（cond + uncond），只取 cond 分支 ([:1])。
    """

    def __init__(self):
        self.phi_history = []
        self.handle = None

    def _hook_fn(self, module, inputs, output):
        z_r = module.z_r.detach()  # [B, T, D]
        z_i = module.z_i.detach()
        # CFG handling: take cond branch (first half)
        if z_r.shape[0] > 1:
            z_r = z_r[:1]
            z_i = z_i[:1]
        phi = torch.atan2(z_i, z_r)
        self.phi_history.append(phi.cpu().float())

    def register(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == "PhaseMemory":
                self.handle = module.register_forward_hook(self._hook_fn)
                print(f"  [Hook] PhaseMemory → {name}")
                return
        raise RuntimeError("No PhaseMemory module found in model")

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_phi(self, unwrap: bool = True) -> torch.Tensor:
        """返回 [T_steps, T_frames, D_mem]"""
        if not self.phi_history:
            return torch.empty(0)
        phi = torch.stack(self.phi_history, dim=0).squeeze(1)
        if unwrap and phi.shape[0] > 1:
            phi = torch.diff(phi, dim=0, prepend=phi[:1])
            phi = torch.cumsum(phi, dim=0)
        return phi


class CrossAttentionHook:
    """
    Cross-attention hook: 在 PhaseMemory 所在层捕获 attention weights.

    注意: 由于 Flash Attention 可能不返回权重，这个 hook 可能收集不到数据。
    这是一个可选的分析，收集不到不影响主分析。
    """

    def __init__(self):
        self.attn_maps = []
        self.handle = None

    def _hook_fn(self, module, inputs, output):
        # 尝试从 output tuple 中提取 attention weights
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[-1]
            if isinstance(attn, torch.Tensor):
                if attn.shape[0] == 2:  # CFG
                    attn = attn[:1]
                self.attn_maps.append(attn.detach().cpu())

    def register(self, model):
        # PhaseMemory 在 decoder.layers.12 (middle layer)
        for name, module in model.named_modules():
            if "decoder.layers.12" in name and "cross_attn" in name:
                if hasattr(module, "out_proj"):
                    self.handle = module.register_forward_hook(self._hook_fn)
                    print(f"  [Hook] CrossAttn → {name}")
                    return
        print("  [Hook] WARNING: cross_attn not found at layer 12 (not fatal for probe)")

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_attn_maps(self) -> torch.Tensor:
        if not self.attn_maps:
            return torch.empty(0)
        return torch.stack(self.attn_maps, dim=0)


class LatentCaptureHook:
    """捕获生成的 audio latents（最后一个 decoder step 的输出）"""

    def __init__(self):
        self.latents = None
        self.handle = None

    def _hook_fn(self, module, input, output):
        # output[0] = hidden_states [B, T, D]
        hs = output[0].detach()
        if hs.shape[0] > 1:  # CFG
            hs = hs[:1]
        self.latents = hs.cpu()

    def register(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == "AceStepDiTModel":
                # hook the final return, but that's not straightforward
                # instead, hook the last layer's output
                pass
            if module.__class__.__name__ == "AceStepDiTModel":
                # Register on the forward method of the decoder
                # Actually, we want the decoder.layers[-1] output
                pass
        # Simpler: try to find the last decoder layer's output
        for name, module in model.named_modules():
            if "decoder.layers.23" in name and isinstance(module, nn.Module):
                # Last layer before norm_out
                pass

    def remove(self):
        pass


# ═══════════════════════════════════════════
# 3. Phase Score 计算
# ═══════════════════════════════════════════
def compute_phase_scores(phi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    从 phi [T_steps, T_frames, D_mem] 计算 phase scores.

    返回 dict:
      velocity:            [T-1, S]   每 step 的 |Δφ| (dim mean)
      accel:               [T-2, S]   每 step 的 |Δ²φ|
      entropy:             [T, S]     circular variance (0=aligned, 1=uniform)
      final_vel:           [S]        最后 30% steps 的 mean velocity
      final_accel:         [S]        最后 30% steps 的 mean accel
      total_drift:         [S]        |phi[T] - phi[0]| (dim mean)
      coherence:           [T, S-1]   相邻帧相位一致性: mean(cos(Δφ across frames))
      coherence_min:       [S]        每帧与左右邻帧的最小 coherence
      composite_boundary:  [S]        z-scored 组合 (vel + drift - coherence)
    """
    T, S, D = phi.shape
    eps = 1e-8
    scores = {}

    # Velocity: mean(|Δφ|) over dims
    vel = (phi[1:] - phi[:-1]).abs().mean(dim=-1)             # [T-1, S]
    scores["velocity"] = vel

    # Acceleration: |Δ²φ| mean over dims
    acc = (vel[1:] - vel[:-1]).abs() if T > 2 else vel.clone()  # [T-2, S]
    scores["accel"] = acc

    # Entropy: circular variance = 1 - |mean(exp(iφ))|
    cos_mean = phi.cos().mean(dim=-1)                          # [T, S]
    sin_mean = phi.sin().mean(dim=-1)
    circ_var = 1 - torch.sqrt(cos_mean ** 2 + sin_mean ** 2 + eps)  # [T, S]
    scores["entropy"] = circ_var

    # Final-stage aggregates
    cutoff = int(T * 0.7)
    scores["final_vel"] = vel[cutoff:].mean(dim=0) if cutoff < T - 1 else vel.mean(dim=0)
    if cutoff < T - 2:
        scores["final_accel"] = acc[cutoff:].mean(dim=0)
    else:
        scores["final_accel"] = acc.mean(dim=0)

    # Total drift
    scores["total_drift"] = (phi[-1] - phi[0]).abs().mean(dim=-1)

    # Frame coherence per step: mean(cos(phi[f+1] - phi[f])) across dims
    frame_diff = phi[:, 1:] - phi[:, :-1]
    coh = frame_diff.cos().mean(dim=-1)                        # [T, S-1], 1=aligned
    scores["coherence"] = coh

    # Position-wise coherence (min of left/right neighbor coherence)
    mean_coh = coh.mean(dim=0)                                  # [S-1]
    left = F.pad(mean_coh.unsqueeze(0), (1, 0), value=1.0).squeeze(0)  # [S]
    right = F.pad(mean_coh.unsqueeze(0), (0, 1), value=1.0).squeeze(0)  # [S]
    scores["coherence_min"] = torch.minimum(left, right)

    # Composite boundary score
    fv = scores["final_vel"]
    td = scores["total_drift"]
    cm = scores["coherence_min"]
    scores["composite_boundary"] = (
        (fv - fv.mean()) / (fv.std() + eps)
        + (td - td.mean()) / (td.std() + eps)
        - (cm - cm.mean()) / (cm.std() + eps)
    )

    return scores


# ═══════════════════════════════════════════
# 4. 边界预测评价
# ═══════════════════════════════════════════
def evaluate_boundary_prediction(
    score: np.ndarray,
    true_boundaries: np.ndarray,
) -> Dict:
    """
    评估连续 score 预测二元边界的能力。

    Args:
        score: [S] float 连续值
        true_boundaries: [S] bool

    Returns:
        dict: auc, precision@k, recall@k, f1@k, pearson_corr, d_prime
    """
    min_len = min(len(score), len(true_boundaries))
    score = score[:min_len]
    true = true_boundaries[:min_len].astype(np.float32)

    n_pos = int(true.sum())
    n_neg = min_len - n_pos

    if n_pos == 0 or n_neg == 0:
        return {
            "auc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hit_rate_top10": 0.0,
            "pearson_corr": 0.0,
            "d_prime": 0.0,
            "n_true_boundaries": n_pos,
            "k": 0,
            "score_mean": float(score.mean()),
            "score_std": float(score.std()),
        }

    # AUC (prob that boundary > non-boundary)
    pos_s = score[true == 1]
    neg_s = score[true == 0]
    auc_val = (pos_s[:, None] > neg_s[None, :]).mean()
    auc_val += 0.5 * (pos_s[:, None] == neg_s[None, :]).mean()

    # Top-K precision/recall/F1 (K = n_boundaries)
    k = n_pos
    topk_idx = np.argsort(score)[-k:]
    hits = true[topk_idx].sum()
    prec = hits / k
    rec = hits / n_pos
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    # Hit rate @ top 10%
    top10 = max(1, min_len // 10)
    top10_idx = np.argsort(score)[-top10:]
    hit10 = true[top10_idx].sum()
    hit_rate = hit10 / n_pos

    # d' = (μ_boundary - μ_non) / σ_pooled
    d_prime = (pos_s.mean() - neg_s.mean()) / np.sqrt(
        (pos_s.var() + neg_s.var()) / 2 + 1e-8
    )

    # Pearson correlation with Gaussian-smoothed soft boundary
    from scipy.ndimage import gaussian_filter
    soft_true = gaussian_filter(true, sigma=2.0)
    corr = np.corrcoef(score, soft_true)[0, 1]

    return {
        "auc": float(auc_val),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "hit_rate_top10": float(hit_rate),
        "pearson_corr": float(corr),
        "d_prime": float(d_prime),
        "n_true_boundaries": n_pos,
        "k": k,
    }


# ═══════════════════════════════════════════
# 5. 生成 Pipeline
# ═══════════════════════════════════════════
def run_generation(
    model_root: str,
    checkpoint_dir: str,
    lyrics: str,
    duration: int = 30,
    inference_steps: int = 50,
    guidance_scale: float = 7.0,
    seed: int = 42,
    quick: bool = False,
) -> Dict:
    """
    运行生成，hook 收集 phi / attn / latents.

    Returns dict with keys:
      phi, phi_raw, attn_maps, latents, audio, phase_scores,
      lyric_info, config
    """
    sys.path.insert(0, str(ACE_STEP_ROOT))

    pm_hook = PhaseMemoryHook()
    attn_hook = CrossAttentionHook()

    if quick:
        print("=" * 60)
        print(" [QUICK MODE] 随机权重")
        print("=" * 60)
        from acestep.phase_memory import PhaseMemory

        T_frames, D_model, D_mem, N_steps = 120, 64, 32, inference_steps

        class MiniModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pm = PhaseMemory(dim=D_model, mem_dim=D_mem)

            def forward(self):
                h = torch.randn(1, T_frames, D_model)
                for step in range(N_steps):
                    step_ratio = torch.tensor([step / N_steps * 1000.0])
                    h, _ = self.pm(h, step_ratio)
                return h

        model = MiniModel()
        pm_hook.register(model)
        attn_hook.register(model)
        print(f"\n[Quick] Forward {N_steps} steps...")
        _ = model()
        latents = torch.randn(1, T_frames, 128)
        audio = torch.randn(1, T_frames * 320)
    else:
        print("=" * 60)
        print(" 加载真实模型 ...")
        print("=" * 60)
        os.environ["ACESTEP_OFFLINE"] = "1"
        os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        dit_handler = AceStepHandler()
        print("\n[1/3] Init DiT ...")
        status, ok = dit_handler.initialize_service(
            project_root=str(model_root),
            config_path="acestep-v15-sft",
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"Init failed: {status}")

        model = dit_handler.model
        if checkpoint_dir:
            from acestep.training.phase_memory_checkpoint import load_phase_memory_weights
            ckpts = sorted(Path(checkpoint_dir).glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
            if ckpts:
                print(f"  Load PM weights: {ckpts[-1].name}")
                load_phase_memory_weights(model, str(ckpts[-1]))

        pm_hook.register(model)
        attn_hook.register(model)

        params = GenerationParams(
            task_type="text2music",
            caption="pop ballad, piano, emotional female vocal, strings, warm",
            lyrics=lyrics,
            instrumental=False,
            bpm=80,
            vocal_language="zh",
            duration=duration,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
        )
        config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)

        print(f"\n[2/3] Generate ({duration}s, {inference_steps} steps)...")
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=None,
            params=params,
            config=config,
            save_dir="/tmp/pm_probe_output",
        )
        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")
        print("  ✓ Done")

        # Extract decoded audio waveform for transition detection
        audio = torch.empty(0)
        if hasattr(result, "audios") and result.audios and len(result.audios) > 0:
            first = result.audios[0]
            if isinstance(first, dict) and "tensor" in first:
                audio = first["tensor"].float()  # [1, T] tensor
            elif isinstance(first, dict) and "path" in first:
                import soundfile as sf
                wav, sr = sf.read(first["path"])
                audio = torch.from_numpy(wav).float().unsqueeze(0)
            elif isinstance(first, torch.Tensor):
                audio = first.float()
        latents = torch.empty(0)  # not captured yet

    pm_hook.remove()
    attn_hook.remove()

    # ── Collect ──
    phi = pm_hook.get_phi(unwrap=True)
    phi_raw = pm_hook.get_phi(unwrap=False)
    attn_maps = attn_hook.get_attn_maps()
    lyric_info = parse_lyric_boundaries(lyrics)
    phase_scores = compute_phase_scores(phi)

    print(f"\n  phi={phi.shape}  attn={attn_maps.shape}  latents={latents.shape if hasattr(latents,'shape') else 'N/A'}")

    return {
        "phi": phi,
        "phi_raw": phi_raw,
        "attn_maps": attn_maps,
        "latents": latents,
        "audio": audio,
        "phase_scores": phase_scores,
        "lyric_info": lyric_info,
        "config": {
            "quick": quick,
            "duration": duration,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        },
    }


# ═══════════════════════════════════════════
# 6. 核心分析
# ═══════════════════════════════════════════
SCORE_NAMES = [
    "final_vel", "final_accel", "total_drift",
    "composite_boundary", "entropy", "coherence_min",
]


def analyze(data: Dict) -> Dict:
    """
    对 A/B/C 三种边界，计算每个 phase score 的预测能力。
    """
    phi = data["phi"]
    pscores = data["phase_scores"]
    lyric_info = data["lyric_info"]
    T, S, D = phi.shape
    results = {}

    ############ A: Lyric Section Boundaries ############
    # 将歌词行边界映射到音频帧
    n_lines = lyric_info["n_lines"]
    lpf = max(1, n_lines / S)
    bound_frames = set()
    for li in lyric_info["section_boundary_lines"]:
        f = min(int(li / lpf), S - 1)
        for df in range(-1, 2):
            if 0 <= f + df < S:
                bound_frames.add(f + df)
    true_lyric = np.zeros(S, dtype=bool)
    for f in bound_frames:
        true_lyric[f] = True

    met_A = {}
    for sk in SCORE_NAMES:
        if sk not in pscores:
            continue
        sc = pscores[sk]
        if sc.ndim > 1:
            sc = sc.mean(dim=0)
        met_A[sk] = evaluate_boundary_prediction(sc.float().cpu().numpy()[:S], true_lyric)
    results["A_lyric_boundary"] = {
        "n_sections": lyric_info["n_sections"],
        "n_boundary_frames": int(true_lyric.sum()),
        "metrics": met_A,
    }

    ############ B: Latent / Audio Frame Transitions ############
    # 从 audio 波形计算光谱变化
    audio = data.get("audio", torch.empty(0))
    true_latent = np.zeros(S, dtype=bool)
    cos_curve = np.ones(S)
    cos_threshold = 0.0
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        # 使用谱图帧间余弦相似度检测边界
        x = audio
        if x.dim() > 1:
            x = x[0]  # [T_samples]
        # 简单 mel-like 谱图: 用短时 FFT
        n_fft = 512
        hop_length = max(1, x.shape[-1] // (S * 2))  # oversample for smoothness
        spec = torch.stft(x.float(), n_fft=n_fft, hop_length=hop_length,
                          window=torch.hann_window(n_fft).to(x.device),
                          return_complex=True)  # [freq, time]
        mag = spec.abs()  # [freq, time_steps]
        # 插值时间维度到 S 帧: [1, freq, time] → [1, freq, S]
        mag = F.interpolate(mag.unsqueeze(0), size=S, mode="linear").squeeze(0)  # [freq, S]
        mag = mag.T  # [S, freq]
        cos_sim = F.cosine_similarity(mag[1:], mag[:-1], dim=-1)  # [S-1]
        cos_curve = F.pad(cos_sim, (1, 0), value=1.0).cpu().numpy()
        threshold = cos_curve.mean() - 1.5 * cos_curve.std()
        jumps = np.where(cos_curve < threshold)[0]
        for f in jumps:
            if 0 <= f < S:
                true_latent[f] = True

    met_B = {}
    for sk in SCORE_NAMES:
        if sk not in pscores:
            continue
        sc = pscores[sk]
        if sc.ndim > 1:
            sc = sc.mean(dim=0)
        met_B[sk] = evaluate_boundary_prediction(sc.float().cpu().numpy()[:S], true_latent)
    results["B_latent_transition"] = {
        "n_transitions": int(true_latent.sum()),
        "cos_threshold": float(cos_threshold),
        "metrics": met_B,
    }

    ############ C: Cross-Attention Changes ############
    attn = data["attn_maps"]
    true_attn = np.zeros(S, dtype=bool)
    kl_curve = np.array([])
    if attn.numel() > 0 and attn.shape[0] > 1:
        n_step = attn.shape[0]
        kl_list = []
        for t in range(1, n_step):
            p = attn[t - 1].reshape(attn.shape[1], -1).clamp_min(1e-8)
            q = attn[t].reshape(attn.shape[1], -1).clamp_min(1e-8)
            kl = F.kl_div(p.log(), q, reduction="none").sum(dim=-1)
            kl_list.append(kl.mean().item())
        kl_curve = np.array(kl_list)
        kl_thresh = kl_curve.mean() + 2.0 * kl_curve.std()
        jumps = np.where(kl_curve > kl_thresh)[0]
        for j in jumps:
            f = min(int(j * S / max(1, len(kl_curve))), S - 1)
            true_attn[f] = True

    met_C = {}
    for sk in SCORE_NAMES:
        if sk not in pscores:
            continue
        sc = pscores[sk]
        if sc.ndim > 1:
            sc = sc.mean(dim=0)
        met_C[sk] = evaluate_boundary_prediction(sc.float().cpu().numpy()[:S], true_attn)
    results["C_attn_change"] = {
        "n_transitions": int(true_attn.sum()),
        "metrics": met_C,
    }

    ############ Summary ############
    summary = {}
    for key, rd in results.items():
        best = max(
            ((sk, m.get("f1", 0), m.get("auc", 0), m.get("d_prime", 0))
             for sk, m in rd.get("metrics", {}).items()),
            key=lambda x: x[1], default=("", 0, 0.5, 0)
        )
        summary[key] = {
            "best_score": best[0],
            "best_f1": best[1],
            "best_auc": best[2],
            "best_d_prime": best[3],
        }
    results["summary"] = summary
    return results


# ═══════════════════════════════════════════
# 7. 可视化
# ═══════════════════════════════════════════
def plot_results(data: Dict, results: Dict, save_dir: str = "/tmp/pm_probe"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available")
        return

    os.makedirs(save_dir, exist_ok=True)
    phi = data["phi"]
    pscores = data["phase_scores"]
    lyric_info = data["lyric_info"]
    S = phi.shape[1]
    frame_idx = np.arange(S)

    # ─── Panel: Main probe panel ───
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1a: φ heatmap (first 16 dims, averaged into 4 groups)
    ax = axes[0, 0]
    phi_disp = phi[:, :, :16].cpu()
    # group dims into 4 for display
    group_size = max(1, 16 // 4)
    groups = []
    for g in range(4):
        gs = g * group_size
        ge = min((g + 1) * group_size, 16)
        groups.append(phi_disp[:, :, gs:ge].mean(dim=-1).numpy())
    combined = np.stack(groups, axis=0).reshape(-1, S)
    im = ax.imshow(combined, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("φ Trajectory (4 groups of dims)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Step × Group")
    plt.colorbar(im, ax=ax)

    # 1b: Phase scores overview
    ax = axes[0, 1]
    for sk in ["final_vel", "total_drift", "composite_boundary"]:
        if sk in pscores:
            sc = pscores[sk].float().cpu().numpy()[:S]
            ax.plot(frame_idx, sc, label=sk, lw=1)
    ax.set_title("Phase Scores")
    ax.set_xlabel("Frame")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2a: Lyric boundaries
    ax = axes[1, 0]
    cb = pscores.get("composite_boundary", torch.zeros(S)).float().cpu().numpy()[:S]
    ax.plot(frame_idx, cb, "b-", lw=1, label="composite")
    sections = lyric_info.get("sections", [])
    n_lines = lyric_info.get("n_lines", 1)
    lpf = max(1, n_lines / S)
    for s in sections:
        f = min(int(s["line_start"] / lpf), S - 1)
        ax.axvline(f, color="r", alpha=0.5, ls="--", lw=1.5)
        ax.text(f, ax.get_ylim()[1] * 0.85, s["name"][:6], rotation=45,
                fontsize=6, color="darkred")
    ax.set_title("A: Lyric Section Boundaries (red)")
    ax.set_xlabel("Frame")
    ax.grid(True, alpha=0.3)

    # 2b: Latent transitions
    ax = axes[1, 1]
    ax.plot(frame_idx, cb, "b-", lw=1, label="composite")
    lat_trans = results.get("B_latent_transition", {})
    # Mark transitions from the raw audio STFT
    if lat_trans.get("n_transitions", 0) > 0:
        for f in np.where(np.array([True]))[0]:  # placeholder
            pass
    ax.set_title("B: Latent / Audio Transitions")
    ax.set_xlabel("Frame")
    ax.grid(True, alpha=0.3)

    # 3a: Cross-attention changes
    ax = axes[2, 0]
    ax.plot(frame_idx, cb, "b-", lw=1, label="composite")
    ax.set_title("C: Cross-Attention Changes")
    ax.set_xlabel("Frame")
    ax.grid(True, alpha=0.3)

    # 3b: Summary bar chart
    ax = axes[2, 1]
    summary = results.get("summary", {})
    labels, f1s, aucs, dprimes = [], [], [], []
    for k, v in summary.items():
        labels.append(k.split("_")[0])
        f1s.append(v.get("best_f1", 0))
        aucs.append(v.get("best_auc", 0))
        dprimes.append(v.get("best_d_prime", 0))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, f1s, w, label="F1@k", alpha=0.8)
    ax.bar(x, aucs, w, label="AUC", alpha=0.8)
    ax.bar(x + w, dprimes, w, label="d'", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Boundary Prediction Summary")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Score")

    plt.tight_layout()
    path = os.path.join(save_dir, "probe_panel.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {path}")

    # ─── Score-specific panels ───
    for sk in ["composite_boundary", "final_vel"]:
        if sk not in pscores:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        sc = pscores[sk]
        if sc.ndim > 1:
            sc = sc.mean(dim=0)
        sc_np = sc.float().cpu().numpy()[:S]
        for ax, (label, key) in zip(axes, [
            (f"A: Lyric × {sk}", "A_lyric_boundary"),
            (f"B: Latent × {sk}", "B_latent_transition"),
            (f"C: Attn × {sk}", "C_attn_change"),
        ]):
            ax.plot(frame_idx, sc_np, "b-", lw=1)
            ax.set_title(label)
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f"probe_{sk}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] {path}")

    # ─── Text report ───
    rpath = os.path.join(save_dir, "probe_report.txt")
    with open(rpath, "w") as f:
        f.write("Phase Boundary Probe Report\n" + "=" * 50 + "\n\n")
        for pk, pd in results.items():
            if pk == "summary":
                continue
            f.write(f"\n--- {pk} ---\n")
            for sk, sm in pd.get("metrics", {}).items():
                f.write(f"  [{sk:20s}] AUC={sm['auc']:.3f}  F1={sm['f1']:.3f}  "
                        f"d'={sm['d_prime']:.3f}  prec={sm['precision']:.3f}  "
                        f"rec={sm['recall']:.3f}\n")
        f.write("\n--- Summary ---\n")
        for sk, sv in summary.items():
            f.write(f"  {sk}: best={sv['best_score']:20s}  F1={sv['best_f1']:.3f}  "
                    f"AUC={sv['best_auc']:.3f}  d'={sv['best_d_prime']:.3f}\n")
    print(f"  [Report] {rpath}")


# ═══════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase Boundary Probe")
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--duration", type=int, default=DURATION)
    parser.add_argument("--inference-steps", type=int, default=INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save-dir", default="/tmp/pm_probe")
    parser.add_argument("--analyze-only", default=None)
    args = parser.parse_args()

    if args.analyze_only:
        print(f"Loading {args.analyze_only} ...")
        data = torch.load(args.analyze_only, map_location="cpu")
        results = analyze(data)
    else:
        data = run_generation(
            model_root=args.model_root,
            checkpoint_dir=args.checkpoint_dir,
            lyrics=SAMPLE_LYRICS,
            duration=args.duration,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            quick=args.quick,
        )
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "probe_data.pt")
        torch.save(data, save_path)
        print(f"\n[Data] Saved: {save_path}")

        results = analyze(data)

    # Save results JSON
    def json_convert(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    res_path = os.path.join(args.save_dir, "probe_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=json_convert, ensure_ascii=False)
    print(f"[Results] Saved: {res_path}")

    plot_results(data, results, save_dir=args.save_dir)

    # ─── Print summary ───
    print("\n" + "=" * 60)
    print("  PROBE SUMMARY")
    print("=" * 60)
    for pk, sv in results.get("summary", {}).items():
        print(f"  {pk:20s}: best={sv['best_score']:20s}  "
              f"F1={sv['best_f1']:.3f}  AUC={sv['best_auc']:.3f}  d'={sv['best_d_prime']:.3f}")
    print("=" * 60)

    aucs = [sv["best_auc"] for sv in results.get("summary", {}).values()]
    f1s = [sv["best_f1"] for sv in results.get("summary", {}).values()]
    avg_auc = np.mean(aucs) if aucs else 0
    avg_f1 = np.mean(f1s) if f1s else 0

    print(f"\n  avg AUC={avg_auc:.3f}  avg F1={avg_f1:.3f}")
    if avg_auc > 0.65:
        print("  ✓ Phase score 能预测结构边界")
        print("  ✓ phase 确实知道结构")
    elif avg_auc > 0.55:
        print("  ~ Phase score 有微弱预测能力")
        print("  ~ 需要更大规模验证")
    else:
        print("  ✗ Phase score 未能有效预测边界")
        print("  ✗ 可能需要不同的 phase score 设计")
    print("=" * 60)


if __name__ == "__main__":
    main()
