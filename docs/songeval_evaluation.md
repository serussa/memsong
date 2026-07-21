# SongEval 评估

## 概述

SongEval 是一个歌曲美学评估工具，从 5 个感知维度自动评价生成的歌曲。基于 [SongEval 数据集](https://huggingface.co/datasets/ASLP-lab/SongEval) 训练。

## 评估维度

| 维度 | 范围 | 说明 |
|------|------|------|
| Coherence | 1~5 | 整体连贯性 |
| Musicality | 1~5 | 整体音乐性 |
| Memorability | 1~5 | 记忆点/抓耳程度 |
| Clarity | 1~5 | 歌曲结构清晰度 |
| Naturalness | 1~5 | 人声呼吸和乐句的自然度 |

## 模型架构

SongEval 使用两阶段架构：

1. **MuQ 编码器** (`OpenMuQ/MuQ-large-msd-iter`): 将音频（24kHz）编码为 SSL 特征。取第 6 层 hidden state（1024维），作为下游输入。
2. **Generator 评分头**: 4 层 MultiheadAttention + FFN 的 Transformer，输出 5 维分数（每维映射到 1~5 范围）。

### 数据流

```
音频 (24kHz) → MuQ 编码器 → hidden_states[6] (1024-dim) → Generator → [Coherence, Musicality, Memorability, Clarity, Naturalness]
```

## 检查点

| 检查点 | 路径 | 来源 |
|--------|------|------|
| SongEval Generator | `SongEval/ckpt/model.safetensors` | 本仓库自带的评分头权重 |
| MuQ 编码器 | HF 缓存（自动下载） | `OpenMuQ/MuQ-large-msd-iter` |

MuQ 编码器会通过 `HF_HOME` 环境变量自动下载到本地缓存。在 ACE-Step 项目中，
已在 `eval.py` 中设置 `HF_HOME=/root/autodl-tmp/hf_cache`。

## 使用方法

### 评估单个文件

```bash
cd /root/ACE-Step-1.5/SongEval
python eval.py -i /path/to/audio.flac -o /path/to/output_dir
```

### 评估一个目录下的所有文件

```bash
cd /root/ACE-Step-1.5/SongEval
python eval.py -i /path/to/audio_folder -o /path/to/output_dir
```

### 使用一键脚本（推荐）

```bash
bash scripts/run_songeval.sh /path/to/input_dir /path/to/output_dir
```

## 输出格式

结果保存在 `{output_dir}/result.json`，格式为：

```json
{
    "filename_without_ext": {
        "Coherence": 4.28,
        "Musicality": 4.12,
        "Memorability": 4.21,
        "Clarity": 4.18,
        "Naturalness": 4.22
    }
}
```

## 代码位置

- `SongEval/eval.py` — 主评估脚本
- `SongEval/model.py` — Generator 模型定义
- `SongEval/config.yaml` — Generator 配置
- `SongEval/ckpt/model.safetensors` — 预训练权重

## 注意事项

- 脚本需要从 `SongEval/` 目录下运行，因为 checkpoint 路径是相对路径 `ckpt/model.safetensors`
- 支持的音频格式：`.wav`, `.mp3`, `.flac`
- 内部会重采样到 24kHz
- MuQ 编码器需要从 HuggingFace 下载（约 2GB），首次运行需联网
