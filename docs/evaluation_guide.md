# ACE-Step 评估体系使用指南

给合作者的完整操作文档。所有评估脚本都支持一键运行，按此文档操作即可复现全部指标。

---

## 1. 环境准备

```bash
# 激活环境（ACE-Step 常用环境）
conda activate musicgen  # 或其他包含 torch + CUDA 的环境

# 安装评估依赖
pip install audiobox-aesthetics   # Audiobox-Aesthetics
cd SongEval && pip install -r requirements.txt && cd ..   # SongEval

# PER 依赖（ASR + 音素转换）
pip install qwen-asr jieba pypinyin g2p_en pypinyin_dict
python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## 2. 检查点准备

所有检查点路径硬编码在脚本中，下表为参考：

| 评估工具 | 检查点位置 | 大小 |
|----------|-----------|------|
| **Audiobox-Aesthetics** | `/root/autodl-tmp/models/audiobox-aesthetics/checkpoint.pt` | 793 MB |
| **SongEval Generator** | `SongEval/ckpt/model.safetensors` | 97 MB |
| **MuQ 编码器** | HuggingFace 自动缓存 (`OpenMuQ/MuQ-large-msd-iter`) | ~2 GB |
| **Qwen3-ASR** | `/root/autodl-tmp/models/Qwen3-ASR-1.7B` | 4.4 GB |
| **GT 歌词 (中文)** | `Muse/eval_pipeline/gt_lyrics/zh.jsonl` | — |
| **GT 歌词 (英文)** | `Muse/eval_pipeline/gt_lyrics/en.jsonl` | — |

如果路径不同，编辑脚本顶部的 `CKPT` / `MODEL_PATH` 变量即可。

---

## 3. 音频文件命名规范

### PER 评估（必须遵守）

ASR 转写 + PER 计算依赖 **文件名末尾的数字序号** 匹配 GT 歌词的 `file_index`。

```
音频文件命名规则:  XXXXXX.wav 或 XXXXXX.flac，末尾数字即 index

中文音频 50 首:  000000.flac ~ 000049.flac  (对应 zh.jsonl: file_index 0~49)
英文音频 50 首:  000000.flac ~ 000049.flac  (对应 en.jsonl: file_index 0~49)
```

若音频按其他规则命名（如 1~50），可传入 `--offset` 调整索引偏移。

### SongEval / Audiobox-Aesthetics

无命名限制，直接扫描目录下所有 `.wav/.mp3/.flac` 文件。

---

## 4. 评估流程

建议按顺序执行：**Audiobox → SongEval → ASR 转录 → PER → 长程 PER**

### 4.1 Audiobox-Aesthetics

```bash
bash scripts/run_audiobox_aes.sh /path/to/audio_dir /path/to/output_dir
```

输出：
```
output_dir/result.json
[
  {"CE": 7.63, "CU": 7.77, "PC": 6.63, "PQ": 8.15},
  ...
]
```

### 4.2 SongEval

```bash
bash scripts/run_songeval.sh /path/to/audio_dir /path/to/output_dir
```

输出：
```
output_dir/result.json
{
  "filename1": {"Coherence": 4.28, "Musicality": 4.12, ...},
  "filename2": {...}
}
```

### 4.3 ASR 转录（PER 的前置步骤）

```bash
python Muse/eval_pipeline/transcribe_local.py \
    --input_dir /path/to/audio_dir \
    --output /path/to/transcription.jsonl \
    --model_path /root/autodl-tmp/models/Qwen3-ASR-1.7B
```

- 支持 `--force` 强制重新转录
- 已存在的结果会自动跳过（按文件路径去重）
- 按文件名末尾数字提取 `file_idx`，与 GT 匹配

### 4.4 PER（基本版）

```bash
bash scripts/run_per.sh /path/to/audio_dir cn /path/to/output_dir
```

或分步执行：
```bash
# Step 1: ASR 转录（如果还没做）
python Muse/eval_pipeline/transcribe_local.py \
    --input_dir /path/to/audio_cn \
    --output /path/to/output/transcription.jsonl

# Step 2: 计算 PER
python Muse/eval_pipeline/calc_per.py \
    --hyp_file /path/to/output/transcription.jsonl \
    --gt_file Muse/eval_pipeline/gt_lyrics/zh.jsonl \
    --model_name my_model \
    --output /path/to/output/per_result.json
```

### 4.5 长程 PER（分段指标）

```bash
bash scripts/run_per_long.sh \
    /path/to/transcription.jsonl \
    Muse/eval_pipeline/gt_lyrics/zh.jsonl \
    /path/to/output \
    my_model
```

输出明细（songs.jsonl）：
```json
{"file_index": 0, "overall_per": 0.5132, "early_per": 0.48,
 "middle_per": 0.52, "late_per": 0.54, "ldg": 0.06, ...}
```

输出汇总（summary.csv）：

| model | metric | mean | std | count |
|-------|--------|------|-----|-------|
| my_model | overall_per | 0.5132 | 0.1200 | 50 |
| my_model | early_per | 0.4900 | 0.1100 | 50 |
| my_model | middle_per | 0.5200 | 0.1300 | 50 |
| my_model | late_per | 0.5300 | 0.1400 | 50 |
| my_model | ldg | 0.0400 | 0.0800 | 50 |

> `original_per` 使用原版 calc_per.py 的截断策略，可供对照验证。

---

## 5. 完整端到端示例

```bash
cd /root/ACE-Step-1.5

# 假设音频在 output/baseline_270_50step/ 下

# 1. Audiobox
bash scripts/run_audiobox_aes.sh output/baseline_270_50step output/baseline_270_50step/audiobox_aes_results

# 2. SongEval
bash scripts/run_songeval.sh output/baseline_270_50step output/baseline_270_50step/songeval_results

# 3. ASR 转录（中文）
python Muse/eval_pipeline/transcribe_local.py \
    --input_dir output/baseline_270_50step \
    --output output/baseline_270_50step/per_results/transcription.jsonl

# 4. PER（基本版）
bash scripts/run_per.sh output/baseline_270_50step cn output/baseline_270_50step/per_results

# 5. 长程 PER
bash scripts/run_per_long.sh \
    output/baseline_270_50step/per_results/transcription.jsonl \
    Muse/eval_pipeline/gt_lyrics/zh.jsonl \
    output/baseline_270_50step/per_results_long \
    baseline_270_50step
```

---

## 6. 文件索引

### 评估脚本

| 脚本 | 用途 |
|------|------|
| `scripts/run_audiobox_aes.sh` | Audiobox-Aesthetics 一键评估 |
| `scripts/run_songeval.sh` | SongEval 一键评估 |
| `scripts/run_per.sh` | PER 一键评估（ASR 转录 + 计算） |
| `scripts/run_per_long.sh` | 长程 PER（输入已转录的 JSONL） |

### 核心 Python

| 文件 | 用途 |
|------|------|
| `Muse/eval_pipeline/transcribe_local.py` | 本地 Qwen3-ASR 转录 |
| `Muse/eval_pipeline/calc_per.py` | 基本 PER 计算（Muse 原始版） |
| `Muse/eval_pipeline/calc_per_long.py` | 长程 PER（全局对齐 + 分段统计） |
| `Muse/eval_pipeline/phoneme_utils.py` | 中英文音素转换 |
| `audiobox-aesthetics/src/audiobox_aesthetics/infer.py` | Audiobox 推理 |
| `SongEval/eval.py` | SongEval 推理 |

### 文档

| 文件 | 内容 |
|------|------|
| `docs/aesthetic_evaluation.md` | Audiobox + SongEval 评估说明 |
| `docs/songeval_evaluation.md` | SongEval 单独说明 |
| `docs/per_evaluation.md` | PER 评估说明（含长程 PER） |

---

## 7. 常见问题

**Q: 显存不够？**
- Audiobox-Aesthetics: ~4 GB
- SongEval: ~4 GB
- Qwen3-ASR 转录: ~5 GB
- 可设 `CUDA_VISIBLE_DEVICES=0` 指定 GPU

**Q: 检查点路径变了？**
- 编辑对应 `.sh` 脚本顶部的 `CKPT` / `MODEL_PATH` / `GT_FILE` 变量

**Q: 音频不是按 000000 命名的？**
- PER 用 `--offset` 参数，或直接修改 `transcribe_local.py` 的 `extract_idx()` 正则
- SongEval / Audiobox 无命名限制

**Q: 想只跑英文？**
- PER: `bash scripts/run_per.sh /path/to/audio_dir en /path/to/output`
- PER long: 换成 `gt_lyrics/en.jsonl`
