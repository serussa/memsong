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

- 中文音频：`000000.flac` ~ `000049.flac`（对应 GT file_index 0~49）
- 英文音频：`000050.flac` ~ `000099.flac`（对应 GT file_index 0~49，使用 `--offset 50`）

注意：英文音频使用 test.jsonl 中的 entry index（50~99）作为文件名，而非 0~49。
`calc_per_long.py` 通过 `--offset 50` 将 GT 的 `file_index 0` 匹配到音频 `000050.flac`。

### 一键评估

```bash
# 中文
bash scripts/run_per.sh /path/to/audio_cn cn /path/to/results

# 英文（带 offset）
bash scripts/run_per_long.sh \
    /path/to/transcription_en.jsonl \
    Muse/eval_pipeline/gt_lyrics/en.jsonl \
    /path/to/output \
    my_model \
    --offset 50
```

### 批量评估（推荐）

使用 `tools/run_tsm_eval.py` 一键完成生成 + 评估：

```bash
# 批量生成（3 方法 × 2 语言 × 5 首）
python tools/run_tsm_eval.py generate

# 评估（SongEval + AudioBox + ASR + PER）
python tools/run_tsm_eval.py evaluate

# 全部
python tools/run_tsm_eval.py all
```

脚本自动处理：
- 英文文件命名（entry index 50~99）
- PER offset 参数
- 检查点选择（baseline / transport_only / sinkhorn_tsm）

### 脚本路径说明

PER 评估复用 `Muse/eval_pipeline/` 下的流水线：

| 脚本 | 路径 | 说明 |
|------|------|------|
| ASR 转录 | `Muse/eval_pipeline/transcribe_local.py` | 调用本地 Qwen3-ASR 模型 |
| PER 计算 | `Muse/eval_pipeline/calc_per.py` | 计算音素错误率 |
| 音素工具 | `Muse/eval_pipeline/phoneme_utils.py` | 中英文音素转换 |
| GT 歌词 | `Muse/eval_pipeline/gt_lyrics/` | 中文/英文 GT 歌词 |
| 入口脚本 | `scripts/run_per.sh` | 一键运行 |
| 批量入口 | `tools/run_tsm_eval.py` | 批量生成+评估 |

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

## 长程 PER 评估（分段指标）

### 概述

长程 PER (`calc_per_long.py`) 在原有整体 PER 基础上，增加了按歌词位置分段的评估能力。

核心改进：
- **不截断**：使用完整的参考歌词和 ASR 转写，不允许按最短长度截断
- **全局对齐**：对完整音素序列做一次 Levenshtein 对齐，而不是分段独立计算
- **分段统计**：按参考音素位置均分 Early / Middle / Late 三段，在同一次全局对齐中分别统计每段的 S/D/I
- **Insertion 归属**：根据全局对齐中的相邻参考位置归入对应分段，而非分段独立对齐

### 分段指标

| 指标 | 说明 |
|------|------|
| Overall PER | 整句音素错误率 (S+D+I)/N |
| Early PER | 前 1/3 歌词位置的音素错误率 |
| Middle PER | 中间 1/3 歌词位置的音素错误率 |
| Late PER | 后 1/3 歌词位置的音素错误率 |
| LDG | Late PER - Early PER，正数表示结尾比开头差 |

### 数据流

```
完整参考歌词 ──→ 音素转换 ──→ ┐
                                ├──→ 一次全局 Levenshtein 对齐 ──→ 分段 S/D/I 统计
完整 ASR 转写 ──→ 音素转换 ──→ ┘
```

### 使用方法

```bash
python Muse/eval_pipeline/calc_per_long.py \
    --hyp_file <ASR转写.jsonl> \
    --gt_file <GT歌词.jsonl> \
    --model_name <模型名> \
    --output_dir <输出目录>
```

### 一键评估

```bash
bash scripts/run_per_long.sh \
    <ASR转写.jsonl> <GT歌词.jsonl> <output_dir> [model_name]
```

### 输出格式

`songs.jsonl`（每首歌的明细）：
```
{"file_name": "000000.wav", "file_index": 0, "overall_per": 0.5132,
 "early_per": 0.4800, "middle_per": 0.5200, "late_per": 0.5400,
 "ldg": 0.0600, "original_per": 0.5132,
 "ref_phonemes": "...", "hyp_phonemes": "...", ...}
```

`summary.csv`（模型级汇总）：

| model | metric | mean | std | count |
|-------|--------|------|-----|-------|
| my_model | overall_per | 0.5132 | 0.1200 | 50 |
| my_model | early_per | 0.4900 | 0.1100 | 50 |
| my_model | middle_per | 0.5200 | 0.1300 | 50 |
| my_model | late_per | 0.5300 | 0.1400 | 50 |
| my_model | ldg | 0.0400 | 0.0800 | 50 |
| my_model | original_per | 0.5132 | 0.1200 | 50 |

### 相关脚本

| 脚本 | 路径 | 说明 |
|------|------|------|
| Long-form PER 计算 | `Muse/eval_pipeline/calc_per_long.py` | 全局对齐 + 分段 PER |
| Original PER 计算 | `Muse/eval_pipeline/calc_per.py` | 原版整体 PER |
| 音素工具 | `Muse/eval_pipeline/phoneme_utils.py` | 中英文音素转换 |
| 入口脚本 | `scripts/run_per_long.sh` | 一键运行分段 PER |

### 最小测试

```bash
python Muse/eval_pipeline/calc_per_long.py --test
```

## 注意事项

- 需要 GPU（显存 ≥ 6GB），Qwen3-ASR 模型约占用 4-5GB
- 模型首次运行会加载到 GPU，后续转录每个文件约 3-6 秒
- 音素转换依赖 nltk cmudict 和 jieba 分词
- 中文 GT 有 50 条（file_index 0-49），英文 GT 有 50 条（file_index 0-49）
