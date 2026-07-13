# PER 评估

## 概述

PER（Phoneme Error Rate，音素错误率）评估通过 ASR 转录生成音频的歌词，并与原始歌词计算音素级别的错误率，用于衡量生成歌曲的歌词可懂度。

## 评估维度

| 指标 | 范围 | 说明 |
|------|------|------|
| PER | 0~1+ | 音素错误率，越低越好（0=完全匹配） |

PER = (S + D + I) / N，其中 S=替换错误数，D=删除错误数，I=插入错误数，N=参考音素数。

## 模型架构

两阶段流水线：

1. **ASR 转录**（Qwen3-ASR-1.7B）：将生成音频转写为文本
2. **PER 计算**：将转录文本与 GT 歌词分别转为音素序列，计算 Levenshtein 距离

### 数据流

```
生成音频 → Qwen3-ASR-1.7B → 转录文本 → 音素转换 → PER vs GT 歌词
                                                            ↑
                                              GT 歌词 → 音素转换
```

### 音素转换

- **中文**：jieba 分词 → pypinyin 转拼音 → 拆分为声母/韵母（含多音字校正）
- **英文**：g2p_en（基于 CMUDict）直接转音素

## 检查点

| 模型 | 路径 | 大小 |
|------|------|------|
| Qwen3-ASR-1.7B | `/root/autodl-tmp/models/Qwen3-ASR-1.7B` | ~3.4GB |
| GT 歌词 (中文) | `Muse/eval_pipeline/gt_lyrics/zh.jsonl` | 50 条 |
| GT 歌词 (英文) | `Muse/eval_pipeline/gt_lyrics/en.jsonl` | 50 条 |

## 使用方法

### 准备音频

音频文件需按索引命名（与 GT 的 `file_index` 对齐）：

- 中文音频：`000000.wav` ~ `000049.wav`（对应 GT file_index 0~49）
- 英文音频：`000000.wav` ~ `000049.wav`（对应 GT file_index 0~49）

### 一键评估

```bash
bash scripts/run_per.sh <input_dir> <language:cn/en> [output_dir]
```

示例：
```bash
bash scripts/run_per.sh /path/to/audio_cn cn /path/to/results
```
### 脚本路径说明

PER 评估复用 `Muse/eval_pipeline/` 下的流水线：

| 脚本 | 路径 | 说明 |
|------|------|------|
| ASR 转录 | `Muse/eval_pipeline/transcribe_local.py` | 调用本地 Qwen3-ASR 模型 |
| PER 计算 | `Muse/eval_pipeline/calc_per.py` | 计算音素错误率 |
| 音素工具 | `Muse/eval_pipeline/phoneme_utils.py` | 中英文音素转换 |
| GT 歌词 | `Muse/eval_pipeline/gt_lyrics/` | 中文/英文 GT 歌词 |
| 入口脚本 | `scripts/run_per.sh` | 一键运行 |

## 输出格式

结果保存在 `{output_dir}/per_result.json`：

```json
{
    "model": "model_name",
    "metrics": {"PER": 0.5132},
    "count": 50
}
```

详细结果在 `{output_dir}/per_result_details.jsonl`，每条包含：
- `file`：文件名
- `per`：单样本 PER
- `ref_text`：GT 歌词
- `hyp_text`：ASR 转录文本
- `ref_phonemes`：GT 音素序列
- `hyp_phonemes`：转录音素序列

## 注意事项

- 需要 GPU（显存 ≥ 6GB），Qwen3-ASR 模型约占用 4-5GB
- 模型首次运行会加载到 GPU，后续转录每个文件约 3-6 秒
- 音素转换依赖 nltk cmudict 和 jieba 分词
- 中文 GT 有 50 条（file_index 0-49），英文 GT 有 50 条（file_index 0-49）
