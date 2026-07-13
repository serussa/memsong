# Aesthetic 评估

## 评估工具

本项目使用两个独立的评估模型对生成音频进行客观质量评分：

1. **Audiobox-Aesthetics**（Meta） — 通用音频美学评估（4 维度）
2. **SongEval**（ASLP-lab） — 歌曲美学评估（5 维度）

---

## 一、Audiobox-Aesthetics

### 评估维度

| 维度 | 全称 | 范围 | 说明 |
|------|------|------|------|
| CE | Content Enjoyment | 0~10 | 内容愉悦度 |
| CU | Content Usefulness | 0~10 | 内容有用性 |
| PC | Production Complexity | 0~10 | 制作复杂度 |
| PQ | Production Quality | 0~10 | 制作质量 |

### 模型架构

WavLM 编码器 + 4 个独立 MLP 预测头。

### 数据流

```
音频 → 重采样 16kHz 单声道 → 分窗（10秒窗口，重叠10秒）→ 加权平均 → [CE, CU, PC, PQ]
```

### 检查点

| 检查点 | 路径 | 大小 |
|--------|------|------|
| 主模型权重 | `/root/autodl-tmp/models/audiobox-aesthetics/checkpoint.pt` | ~400MB |

### 使用方法

```bash
bash scripts/run_audiobox_aes.sh <input_dir> <output_dir>
```

### 代码位置

- `audiobox-aesthetics/src/audiobox_aesthetics/infer.py` — 推理逻辑
- `audiobox-aesthetics/src/audiobox_aesthetics/model/aes.py` — 模型定义

---

## 二、SongEval

### 评估维度

| 维度 | 范围 | 说明 |
|------|------|------|
| Coherence | 1~5 | 整体连贯性 |
| Musicality | 1~5 | 整体音乐性 |
| Memorability | 1~5 | 记忆点/抓耳程度 |
| Clarity | 1~5 | 歌曲结构清晰度 |
| Naturalness | 1~5 | 人声呼吸和乐句的自然度 |

### 模型架构

MuQ 编码器（24kHz）→ hidden_states[6]（1024维）→ Generator（4层 Attention + FFN）→ 5维输出

### 检查点

| 检查点 | 路径 | 大小 |
|--------|------|------|
| SongEval Generator | `SongEval/ckpt/model.safetensors` | ~96MB |
| MuQ 编码器 | HuggingFace `OpenMuQ/MuQ-large-msd-iter`（自动缓存） | ~2GB |

### 使用方法

```bash
bash scripts/run_songeval.sh <input_dir> <output_dir>
```

### 代码位置

- `SongEval/eval.py` — 主评估脚本
- `SongEval/model.py` — Generator 模型定义
- `SongEval/config.yaml` — 模型配置
- `SongEval/ckpt/model.safetensors` — 预训练权重

---

## 三、评估结果归档

每次评估的结果保存在 `{output_dir}/result.json` 中。

### 输出格式

Audiobox-Aesthetics:
```json
{"CE": 7.63, "CU": 7.77, "PC": 6.63, "PQ": 8.15}
```

SongEval:
```json
{
    "filename": {
        "Coherence": 4.28,
        "Musicality": 4.12,
        "Memorability": 4.21,
        "Clarity": 4.18,
        "Naturalness": 4.22
    }
}
```
