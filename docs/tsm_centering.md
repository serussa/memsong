# TSM Mu-Weighted Centering

## 问题

Sinkhorn coupling P 的列 profile 有显著差异（`col_cosine ≈ 0.36`），但 pooling 得到的 structural slots 几乎相同（`raw_cosine ≈ 0.99`）。

原因：隐藏状态 H 被**强公共方向**主导。不同歌词单元的 P 列从不同时间区域加权 H，但各区域的公共分量远大于差异分量，导致所有 slot 收敛到同一个全局均值。

## 修改

### 文件

`acestep/modules/transported_structural_memory.py` — `TSM.forward()`

### 公式

**Mu 权重**（来自 coupling column mass）：

```
mu_u = col_mass_u / sum_v col_mass_v
```

**中心化**（pooling 后，mixer 前）：

```
slot_mean = sum_u mu_u * M_u
M_centered = M - slot_mean
```

**Mixer 后再次中心化**（防止 mixer 重新产生公共方向）：

```
M_mixed_centered = M_mixed - sum_u mu_u * M_mixed_u
```

### 改动

1. **第 3a 步** — 计算 `mu` 权重，做 `mu` 加权中心化得到 `M_centered`
2. **第 4 步** — slot mixer 输入从 `M` 改为 `M_centered`
3. **第 4a 步** — mixer 后再次中心化得到 `M_mixed_centered`
4. **第 6 步** — broadcast 从 `M_mixed_centered` 读取

不新增可训练参数，不改动外围模块。

### 诊断行

```
[TSM-COS] col=0.359 raw=0.977 cntr=-0.016 mx=-0.008 r∅r=0.140 eff_rk=4.3
```

| 字段 | 含义 | 修改前 | 修改后 |
|------|------|--------|--------|
| `col` | coupling 列 profile 余弦 | 0.36 | 0.36 |
| `raw` | 原始 M 余弦 | 0.91-1.00 | 0.91-1.00 |
| `cntr` | 中心化后 M 余弦 | ~0.0 | ~0.0 |
| `mx` | 中心化后 mixed 余弦 | 0.99 | ~0.0 |
| `r∅r` | 中心化/原始 RMS 比 | N/A | 0.06-0.31 |
| `eff_rk` | 中心化 slot 有效秩 | N/A | 3.0-5.6 |

## 启动命令

从头训练（不加载预训练 checkpoint）：

```bash
SIDESTEP_SAFE_ROOT=/root/autodl-tmp python train.py --yes fixed \
  --checkpoint-dir /root/autodl-tmp/Ace-Step1.5/checkpoints \
  --model-variant sft \
  --dataset-dir /root/autodl-tmp/musicdata/train_tensors \
  --output-dir /root/autodl-tmp/tsm_sinkhorn \
  --adapter-type pm_retrieval \
  --use-transport-retrieval \
  --transport-qk-scale 0 \
  --lr 5e-5 --batch-size 1 --gradient-accumulation 4 \
  --epochs 1 --warmup-steps 100 --weight-decay 0.01 \
  --max-grad-norm 1.0 --seed 42 --precision bf16 \
  --save-every 1 --log-every 10 --log-heavy-every 50 \
  --use-tsm --tsm-mode sinkhorn_tsm \
  --tsm-num-heads 4 --tsm-ffn-dim 512 --tsm-slot-layers 1 \
  --tsm-memory-dim 256 --tsm-dropout 0
```

## 批量评估

使用 `tools/run_tsm_eval.py` 一键生成 + 评估三个方法（baseline / transport_only / sinkhorn_tsm）：

```bash
# 生成（30 文件：3 方法 × 2 语言 × 5 首）
python tools/run_tsm_eval.py generate

# 评估（SongEval + AudioBox + PER）
python tools/run_tsm_eval.py evaluate

# 全部
python tools/run_tsm_eval.py all
```

输出目录：`/root/autodl-tmp/tsm_eval_results/{method}/{audio_zh,audio_en,results,per_results_zh,per_results_en}/`

### 修复的 Bug

1. **生成时长** — 原使用音素计数 /25，导致 180s 歌只生成了 35s。改为歌词字符数 ×0.45
2. **歌词解析** — `parse_entry()` 改用 `re.match` 只提取最外层 `[Section]` 标签，过滤 `[desc:][lyrics:][phoneme:]` 内部标签
3. **英文 offset** — 英文音频命名用 test.jsonl entry index（50~99），PER 计算用 `--offset 50`
4. **FLAC 支持** — SongEval / AudioBox glob 增加 `*.flac`

### 实验结果（2000 steps 训练后）

| 指标 | baseline | transport_only | sinkhorn_tsm |
|------|:--------:|:--------------:|:------------:|
| **PER zh** ↓ | 0.1781 | 0.1777 | **0.1669** |
| **PER en** ↓ | 0.3142 | 0.3159 | **0.3101** |
| **SongEval zh** ↑ | 4.15~4.34 | 4.15~4.34 | **4.17~4.35** |
| **SongEval en** ↑ | 3.91~4.15 | 3.84~4.09 | 3.91~4.15 |
| **AudioBox zh** ↑ | 30.32 | 30.31 | **30.32** |
| **AudioBox en** ↑ | 30.09 | 29.99 | 30.04 |

Sinkhorn TSM 在中文 PER 上领先 baseline ~0.011（6.3% 相对提升），在英文 SongEval 上与 baseline 持平。差距较小，TSM output_proj 初始化 std（当前 0.01）可能需要增大以产生更显著的注入效果。
