# Section-RoPE Offset

## 动机

原始 ACE-Step 把歌词中的 `[Verse]` / `[Chorus]` / `[Bridge]` 等结构化标注当作普通文本 token 输入 lyric encoder，没有任何特殊处理。模型需要**自己从数据中隐式学习**这些标注与音频结构之间的关联，没有先验的归纳偏置。

Section-RoPE Offset 的实验假设是：

> 把 section type 信息作为 cross-attention K 侧的**相位偏置**注入 RoPE，能否改善长程歌词/结构稳定性？

## 核心修改点

### 1. 新增模块

| 文件 | 说明 |
|------|------|
| `acestep/tgca/section_rope.py` | `SectionRoPEOffset`（产生 section-type 相位偏置）+ `rope_with_phase_offset`（旋转 K 的前 32 维） |
| `acestep/tgca/lyrics_parser.py` | 歌词结构解析器，将 `[Verse]` / `[Chorus]` 映射到 8 类 section type ID |

### 2. 修改的文件

| 文件 | 修改 |
|------|------|
| `acestep/models/sft/configuration_acestep_v15.py` | 新增 `use_section_rope_offset`、`section_rope_layers`、`section_time_dim` 等配置项，全部 PM 功能默认关闭 |
| `acestep/models/sft/modeling_acestep_v15_base.py` | `AceStepAttention` forward 支持 `section_rope_offset` + `section_rope_time_dim` 参数；`AceStepDiTLayer.__init__` 可选注入 `SectionRoPEOffset` 模块；forward 从 kwargs 读取 `section_ids` 并传给 cross-attention；`AceStepDiTModel.forward` 提取 `section_ids` 传给各层 |
| `acestep/training/data_module.py` | `PreprocessedTensorDataset.__getitem__` 实时从 `metadata["lyrics"]` 解析 `section_ids [B, L]`；collator 补齐 |
| `acestep/training_v2/fixed_lora_module.py` | 新增 `adapter_type="section_rope"` + `_inject_section_rope()`，冻结 backbone 只训 129 个参数 |

### 3. 删除的

- `acestep/tgca/time_geometric_cross_attention_bias.py`（整个 TGCA 模块）
- TGCA 所有配置项（`use_tgca`、`tgca_*`）
- `TGCAConfigV2` 数据类
- `_inject_tgca` 及相关代码

## 架构

```
Python输入: section_ids [B, L]  (0=UNKNOWN, 1=INTRO, 2=VERSE, 3=PRECHORUS, 4=CHORUS, 5=BRIDGE, 6=OUTRO, 7=INSTRUMENTAL)
SectionRoPEOffset:
  Embedding([8, 16]) → delta_theta [B, L, 16]   scale=exp(log_scale)
  clamp(-0.2, 0.2) → delta_theta [B, L, 16]

rope_with_phase_offset(K, time_dim=32, phase_offset=delta_theta):
  K_time = K[..., :32]    # 仅前 32 维
  按频率对拆分 → 每对旋转 delta_theta
  K[..., :32] ← 旋转后结果
  K[..., 32:] 不变

结果: Q·K^T 中，同一个 lyric token 的 K 向量会根据其 section type
     在前 32 维上旋转不同角度，从而影响 attention score
```

设备确保开关关闭时与 baseline 等价（zero-init）。

## 训练

### 烟测验证（已通过）

```bash
python scripts/section_rope_smoke.py
```

检查通过：
- Section IDs 覆盖率: INTRO 13.9% / VERSE 31.1% / CHORUS 33.3% / UNKNOWN **0.6%**
- 可训练参数: **129**（128 + 1）
- 梯度流: ✅ `section_phase.weight.grad_norm = 3.76e-6` > 0

### 正式训练

```bash
cd /root/ACE-Step-1.5

ACESTEP_LOCAL_MODEL_CODE=1 SIDESTEP_SAFE_ROOT="/" python train.py fixed \
  --adapter-type section_rope \
  --learning-rate 5e-4 \
  --epochs 5 \
  --batch-size 8 \
  --dataset-dir /root/autodl-tmp/musicdata/train_tensors \
  --checkpoint-dir /root/autodl-tmp/Ace-Step1.5/checkpoints \
  --model-variant sft \
  --output-dir /root/autodl-tmp/section_ckpt
```

### 关键训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `adapter_type` | `section_rope` | 冻结 backbone，只解冻 `section_rope_offset_module` |
| `learning_rate` | 5e-4 | 因参数仅 129 个且 scale 压小了梯度，比默认 1e-4 更高 |
| `epochs` | 5 | 第一版不必过长 |
| `batch_size` | 2 | 视显存调整 |
| `gradient_accumulation` | 8 | 有效 batch size ≈ 16 |

### 日志监控

训练中每 50 步打印并记录到 TensorBoard：

| 指标 | 说明 |
|------|------|
| `section_rope/phase_norm` | section_phase 权重范数，从 0 → 非零说明学到东西 |
| `section_rope/delta_mean_abs` | 相位偏置平均绝对值，从 0 → 非零 |
| `section_rope/delta_max_abs` | 相位偏置最大绝对值，不应长期顶到 max_offset=0.2 |
| `section_rope/log_scale` | log_scale 值，随训练调整 |

启动时打印配置校验：

```
[Section-RoPE] use_pm=false  use_pm_kv=false  use_traj=false  use_anchor=false ...
[Section-RoPE] section_type_vocab: UNKNOWN=0 INTRO=1 VERSE=2 PRECHORUS=3 CHORUS=4 ...
```

### 消融实验

| 配置 | 含义 |
|------|------|
| A. `adapter_type lora`（无 section_rope） | baseline |
| B. `adapter_type section_rope`（真实 section_ids） | 实验组 |
| C. `adapter_type section_rope`（打乱的 section_ids） | 对照：如果 B > C 说明真用了 section 信息 |

## 数据集覆盖率

对 200 条训练数据统计：

| ID | Section | Token 占比 |
|----|---------|-----------|
| 0 | UNKNOWN | 0.6% |
| 1 | INTRO | 13.9% |
| 2 | VERSE | 31.1% |
| 3 | PRECHORUS | 6.5% |
| 4 | CHORUS | 33.3% |
| 5 | BRIDGE | 2.7% |
| 6 | OUTRO | 3.2% |
| 7 | INSTRUMENTAL | 8.8% |

VERSE + CHORUS 占 64%，覆盖率充足。

## 不得做的事

- ❌ PM / trajectory / anchor / entropy / KL 全部关闭
- ❌ 不改 Q、V、hidden residual
- ❌ 不改 self-attention
- ❌ 不改 lyric encoder
- ❌ 不解冻 DiT backbone
- ❌ 不加 LoRA
- ❌ 不加 Gumbel / hard alignment / confidence gate
