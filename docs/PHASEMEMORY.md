# PhaseMemory — PMDC: PhaseMemory-Conditioned Duration Clock

## 概述

将旧 PhaseMemory（hidden/KV 内容注入器）替换为 **progress-only duration clock**，作为 ACE-Step 歌词进度控制和 cross-attention 时间对齐的基础组件。

### 设计原则

```
p_base → PMDC clock → p_final → duration bias → mass-preserving attention
```

- PMDC **不修改 hidden state、不修改 K/V、不注入 residual**
- PMDC 只输出一个 bounded log-speed residual `s_pm`，用来 warp 线性时间进度
- 通过 duration-interval bias 引导 cross-attention 的时间对齐
- 支持免训练 static 模式和可训练的 residual clock

---

## 核心模块 (acestep/phase_memory.py)

### PhaseMemoryDurationClock（复杂值循环 clock）

适合推理时使用，含逐 token 复数相位状态更新。

```python
class PhaseMemoryDurationClock(nn.Module):
    def forward(self, h, p_base=None) -> (s_pm, p_final)
```

- `mem_dim`: 复数记忆维度（默认 128）
- `beta`: log-speed residual 强度（默认 0.1）
- `use_delta_h`: 是否使用 `h_i - h_{i-1}` 作为输入
- per-step state normalization 防止数值漂移
- zero-init final layer → 初始 `p_final ≈ p_base`

### PMDCResidualClock（轻量 MLP clock - 训练用）

非循环结构，适用于可训练场景：

```python
class PMDCResidualClock(nn.Module):
    def forward(self, h, p_base=None) -> (speed_residual, p_final, log_v)
```

- `hidden_dim=128`, `beta_init=0.05`, `beta_max=0.15`
- MLP: `LayerNorm → Linear(dim→128) → SiLU → Linear(128→1)`
- 最后一层 **zero-init** → 初始 `speed_residual≈0` → `p_final≈p_base`
- `beta = beta_max * sigmoid(beta_logit)`，可训练

### PMDCResidualClock 训练现状

2026-06-28 确认训练困境：grad(flow_loss, p_final) ≈ 0。因为 bias 改变 attention logits → 经过 softmax 饱和区 → 再经过 12 层 residual，梯度衰减到 ~1e-17，PMDCResidualClock 训不动。

所以后续重心转向了 **PMRetrievalPhaseMemory + LyricRetrievalAdapter**（通过 hidden residual 路径训练）。

### PMRetrievalPhaseMemory（新 PM，hidden residual 路径）

```python
class PMRetrievalPhaseMemory(nn.Module):
    def forward(self, h, diffusion_step=None) -> (hidden_out, kl_loss)
```

- pool hidden → 单步复数循环 → pm_state [B, T, 256]
- zero-init out_proj → 初始 residual = 0
- 训练时通过 hook 注入 hidden residual（不用改 layer forward）
- 训练时 residual 路径的通：grad 走 12 层 → residual_head，已经验证

### LyricRetrievalAdapter（旁路歌词检索分支）

```python
class LyricRetrievalAdapter(nn.Module):
    def forward(self, hidden_states, text_hidden, pm_state,
                p_audio, c_text, section_id, token_type_id, ...) -> (ret_res, attn_r)
```

- 输入：PM state + 歌词结构坐标 + text encoder hidden + scaffold prior
- 内部：q = f(pm_state, audio_coord, timestep)；k = g(text_hidden, text_coord, section)
- 输出：gated hidden residual [B, T, D] + attention [B, T, L]
- 不修改原 cross-attention Q/K/V
- 内部有 scaffold prior：`score += -lambda * dist(p_audio, c_text)² / sigma²`

### LyricUnit

```python
@dataclass
class LyricUnit:
    unit_id: int
    section: str
    text: str
    char_count: int = 0
    token_indices: list[int] = field(default_factory=list)
    occurrence_id: int = 0
    is_silence: bool = False
    is_control: bool = False
    is_lyric: bool = True
    duration_weight: float = 0.0
```

### 辅助函数

| 函数 | 用途 |
|------|------|
| `parse_lyrics_to_units()` | tag-aware 歌词解析，生成 LyricUnit + debug_info |
| `build_duration_scaffold()` | 从 LyricUnit 构造时间边界 + mask |
| `build_duration_interval_bias()` | 从 p_final 构造 cross-attention bias |
| `mass_preserving_attention()` | text-mass-preserving split-softmax |
| `insert_control_lines()` | 在 tokenizer 前插入 control line |
| `is_natural_control_line()` | 检测 bracketed line 是否为 control line |
| `get_scheduled_gate()` | denoising gate fadeout 调度 |
| `scaffold_progress()` | 从 scaffold 提取 p_audio / c_text / token_type_id |
| `freeze_except_pmdc_clock()` | 冻结 backbone，只保留 PMDC + gate |

---

## Tag-Aware Lyrics Parser

### 解析规则

| 文本行 | 处理方式 |
|--------|---------|
| `[Verse]` / `[Chorus]` / `[Bridge]` / `[Pre-chorus]` | 仅切换 current_section，不作为 unit，token 进入 `tag_control_mask` |
| `[Intro]` / `[Outro]` | 作为 silence unit（无歌词），`is_silence=True, is_lyric=False` |
| 真实歌词行 | 作为 lyric unit，`is_lyric=True, is_silence=False` |
| `[Intro - Instrumental]`（含描述） | natural control line，`is_control=True, is_silence=True` |

### Mask 族

| Mask | 含义 | 769 token 示例 |
|------|------|----------------|
| `lyric_mask` | 真实歌词 token | ~701 true |
| `control_mask` | tag / control token | ~68 true |
| `attendable_mask` | `lyric_mask \| control_mask` | ~769 true |
| `tag_control_mask` | 裸标签（debug info 内） | — |

---

## Duration Scaffold

### 权重公式

```
lyric unit:   w = max(char_count, 1)^gamma * section_multiplier[section]
silence unit: w = avg_lyric_weight * silence_multiplier
```

### Section Multiplier

```
CHORUS: 1.15   PRECHORUS: 1.05   BRIDGE: 1.05
VERSE:  1.00   INTRO:    1.00   OUTRO:  1.00
INSTR:  1.00   UNKNOWN:  1.00
```

### 输出

```python
{
    "unit_boundaries": [U+1],      # 归一化 [0,1]
    "unit_duration": [U],
    "token_to_unit": [L],
    "lyric_mask": [L],              # bool
    "control_mask": [L],            # bool
    "attendable_mask": [L],         # bool = lyric | control
}
```

---

## Duration Interval Bias

```python
bias = build_duration_interval_bias(
    p_final=p_final,
    unit_boundaries=...,
    token_to_unit=...,
    attendable_mask=...,
    sigma=0.03, lambda_=0.5, max_bias=1.0,
)
```

公式：
```
u = token_to_unit[j]
interval = [boundaries[u], boundaries[u+1]]
distance = |p_final[i] - centre(interval)|
bias[i,j] = -lambda * (distance / sigma)²
```

- 只对 `attendable_mask=True` 的 token 施加 bias（lyric + control 都生效）
- non-attendable token 的 bias = 0

---

## Mass-Preserving Attention

```python
text_mask = lyric_mask | control_mask
attn_base = softmax(logits)
text_mass_base = attn_base[text_mask].sum()

text_logits = logits + gate * bias
attn_text = softmax(text_logits, text_mask)

non_text_logits = logits
attn_non_text = softmax(non_text_logits, ~text_mask)

attn_new = attn_text * text_mass_base + attn_non_text * (1 - text_mass_base)
```

- `text_mass_new ≈ text_mass_base`
- duration bias 对 lyric 和 control token **都生效**

---

## 训练设施

### 适配器类型

| adapter-type | 模块 | 训练路径 | 状态 |
|---|---|---|---|
| `pmdc_clock` | PMDCResidualClock | attention bias → softmax → decoder → flow_loss | ⚠️ 梯度被 softmax 饱和吃掉，训不动 |
| `pm_retrieval` | PMRetrievalPhaseMemory + LyricRetrievalAdapter | hidden residual → decoder → flow_loss | ✅ 梯度连通，1 epoch real-data dry run 通过 |
| `phase_memory` | 旧 PhaseMemory | hidden residual（注入 layer forward） | ✅ 原样保留 |

### pm_retrieval 训练流程

```
train.py
  └── fixed 子命令
        └── FixedLoRATrainer
              └── FixedLoRAModule (adapter_type == "pm_retrieval")
                    ├── _inject_pm_retrieval()
                    │     ├── 创建 PMRetrievalPhaseMemory + LyricRetrievalAdapter
                    │     ├── 冻结 backbone
                    │     └── 注册模块到 model
                    └── _pm_retrieval_training_step()
                          ├── Warmup forward → hook layer 12 → H
                          ├── PMRetrievalPhaseMemory(H) → pm_state（不注入 residual）
                          ├── scaffold_progress(scaffold, T) → p_audio, c_text
                          ├── LyricRetrievalAdapter(pm_state, text, scaffold) → ret_res
                          ├── Hook 替换 layer 12 hidden = H + gamma * ret_res
                          ├── 从 layer 13 继续走完 decoder → flow_loss
                          └── backward → 梯度走 12 层 → PM + Adapter
```

### 冻结规则

冻结：DiT backbone, Qwen text encoder, VAE, Q/K/V/O, pm_kv_proj, 旧 PM hidden residual
解冻：PMRetrievalPhaseMemory + LyricRetrievalAdapter 全部参数

### Loss

```python
loss = flow_loss  # 只用主 loss，不加额外正则
```

### 训练命令

```bash
# pm_retrieval（新 PM + LyricRetrievalAdapter）
# ⚠️ 必须带 --no-gradient-checkpointing，否则梯度被 checkpoint 切断
ACESTEP_LOCAL_MODEL_CODE=1 SIDESTEP_SAFE_ROOT="/" python train.py --yes fixed \
  --adapter-type pm_retrieval \
  --checkpoint-dir /root/autodl-tmp/Ace-Step1.5/checkpoints \
  --model-variant sft \
  --dataset-dir /root/autodl-tmp/musicdata/train_tensors \
  --output-dir /root/autodl-tmp/pm_retrieval_smoke \
  --learning-rate 5e-5 \
  --no-gradient-checkpointing \
  --batch-size 16 \
  --epochs 1 \
  --device auto
```

### Checkpoint 保存

```
{output_dir}/final/pm_retrieval.pt
  ├── "phase_memory":        PMRetrievalPhaseMemory state_dict
  └── "retrieval_adapter":   LyricRetrievalAdapter state_dict
```

---

## 2026-06-30: Gradient Checkpointing 阻断梯度（关键发现）

### 问题

pm_retrieval 的 hook 注入方案（Warmup forward → H → PM → adapter → 替换 layer 12 hidden → 从 layer 13 继续 forward）存在一个致命隐患：**gradient checkpointing 切断了梯度图**。

梯度 checkpointing 的工作原理是在 forward 时丢弃中间激活，backward 时重新计算。当 hook 替换 `o[0]` 为 `final_h`，这个替换后的 tensor 是在 checkpoint 区域 **之外** 创建的。如果 decoder 内部启用了 checkpointing，从 `final_h` 到 PM/adapter 参数的 backward 路径会被 checkpoint 的 `no_grad` 上下文截断。

### 验证（2026-06-30 20-step parameter delta）

```text
配置: γ=0.1, lr=5e-5, 50 samples, batch=1, grad_acc=1

GC 开启（默认）:
  pm_proj_r_delta  = 0.00e+00
  q_mlp_delta      = 0.00e+00
  k_mlp_delta      = 0.00e+00
  out_proj_delta   = 0.00e+00
  gamma_r_delta    = 0.00e+00
  → 参数完全没更新 ←

GC 关闭（--no-gradient-checkpointing）:
  pm_proj_r_delta  = 1.59e-05  ✓
  q_mlp_delta      = 1.45e-05  ✓
  k_mlp_delta      = 1.44e-05  ✓
  v_mlp_delta      = 5.10e-05  ✓
  out_proj_delta   = 1.50e-04  ✓
  gamma_r_delta    = 3.58e-04  ✓
  → 全部非零 ←
```

### 影响

| 训练 | GC 状态 | 参数是否更新 | 结论 |
|------|---------|:-----------:|:----:|
| 1 epoch（2026-06-30, γ=0.01, lr=2e-5） | 默认开启 | ❌ 全部为 0 | **无效** |
| 50-step（γ=0.1, lr=5e-5） | 默认开启 | ❌ 全部为 0 | **无效** |
| 20-step（γ=0.1, lr=5e-5） | 关闭 | ✅ 全部非零 | 有效 |

**之前所有的训练结果全部无效，checkpoint 权重 = 初始化。**

### 修正

pm_retrieval 训练时必须禁用 gradient checkpointing:

```bash
python train.py --yes fixed \
  --adapter-type pm_retrieval \
  ... \
  --no-gradient-checkpointing
```

### 修复思路（TODO）

长期方案是在 hook 方案中显式处理 checkpoint 边界，让 `final_h` 参与 checkpoint 区域的重计算。当前先用 `--no-gradient-checkpointing`。

### 有效配置（2026-06-30 20-step micro-train）

| 参数 | 值 |
|------|-----|
| out_proj init | std=1e-3（原 1e-5） |
| gamma_r_init | 0.1（原 0.01） |
| lr | 5e-5（原 2e-5） |
| residual_scale | 0.1（tanh 限幅） |
| gradient_checkpointing | ❌ 关闭 |
| 20 步后 hidden_delta_norm | ~5e-4（原 3e-8） |
| 20 步后 gamma_r | 0.0996（从 0.1 下降≈0.0004，正常波动） |

### 待修复问题

| 问题 | 原因 |
|------|------|
| 之前所有训练无效 | gradient checkpointing 阻断梯度 |
| grad_norm 始终为 0（即使 GC 关闭） | loss.register_hook 在 grad 计算完前触发 |
| q/k 梯度弱于 out_proj | scaffold prior 占据注意力分配主导 |
| PM omega 梯度极弱（1.78e-07） | 单步 state 更新，omega 梯度需多步累积 |

---

## 免训练模式

### 推荐参数 (clean_parser_no_control)

```yaml
sigma: 0.03
lambda: 0.5
gate: 0.35
max_bias: 1.0
```

### 支持的 mode

| Mode | Parser | Control Lines | Gate | 说明 |
|------|--------|---------------|------|------|
| `baseline` | — | — | — | 原始模型 |
| `fixed_linear` | — | — | 0.3 | token-level 线性 bias |
| `duration_weak` | clean | — | 0.5 | 最弱干预 |
| `clean_parser_no_control` | clean | — | 0.35 | **当前最优免训练** |
| `clean_parser_intro_outro_control` | clean | `[Intro - Instrumental]` + ... | 0.35 | control line 实验 |
| `clean_parser_intro_outro_control_fadeout` | clean | 同上 | 0.35→0.05 | gate fadeout |

---

## 生成推理脚本

### acestep/run_inference_pmdc.py

| 参数 | 效果 |
|------|------|
| 不加参数 | static clean_parser_no_control |
| `--pmdc-ckpt path` | 已训练的 PMDCResidualClock（注：训不动，等同 static） |
| `--force-p-final-base` | ablation: 禁 PMDC residual |
| `--duration-bias-off` | ablation: 关所有 bias |

**当前不支持 `pm_retrieval` 模式推理。** 需要等 3 epoch smoke 训练完成后，再写推理加载逻辑。

---

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `acestep/phase_memory.py` | 重写 | 新增 PhaseMemoryDurationClock, PMDCResidualClock, PMRetrievalPhaseMemory, LyricRetrievalAdapter, parse_lyrics_to_units, build_duration_scaffold, build_duration_interval_bias, mass_preserving_attention, scaffold_progress, insert_control_lines, freeze_except_pmdc_clock |
| `acestep/training/configs.py` | 修改 | 新增 `PMDCConfig` |
| `acestep/training_v2/configs.py` | 修改 | 新增 `PMDCConfigV2`；`TrainingConfigV2` 新增 `pmdc_*` 字段 |
| `acestep/training_v2/fixed_lora_module.py` | 修改 | 新增 `_inject_pm_retrieval()`，`_pm_retrieval_training_step()`，`training_step()` 分派，`_collect_pm_diag()` 兼容新旧 PM |
| `acestep/training_v2/cli/args.py` | 修改 | `--adapter-type` 新增 `pmdc_clock`、`pm_retrieval`；新增 `--pmdc-*` 参数 |
| `acestep/training_v2/cli/config_builder.py` | 修改 | `build_configs` 新增 `pmdc_clock` / `pm_retrieval` 分支 |
| `acestep/training_v2/trainer_helpers.py` | 修改 | `save_adapter_flat` 支持 `pm_retrieval` checkpoint |
| `acestep/training_v2/trainer_fixed.py` | 修改 | adapter_label 支持全部类型 |
| `acestep/run_inference_pmdc.py` | 新增 | PMDC 推理脚本（static + 已训练 PMDCResidualClock） |
| `scripts/pmdc_dataset_generation_ablation.py` | 修改 | 修复 SDPA bypass bug（强制 `_attn_implementation = "eager"`） |

---

## 实验结论

### Temporal Localization（Stage 2.6）

| Mode | center_error ↓ | out_of_interval ↓ | center_spearman ↑ | KL |
|------|---------------|-------------------|-------------------|---|
| baseline | 0.251 | 0.976 | 0.06 | — |
| fixed_linear | **0.183** (-27%) | 0.959 | **0.975** | 0.033 |
| duration_weak | 0.243 (-3%) | 0.963 | 0.733 | **0.005** |
| duration_mid | 0.237 (-6%) | **0.944** (-3.3%) | 0.874 | 0.019 |
| duration_strong | **0.177** (-30%) | 0.956 | **0.974** | 0.041 |

### Attention Trace（Stage 3.5）

- baseline: lyric_mass ≈ 0.929, control_mass ≈ 0.071
- clean_parser_no_control: lyric_mass ≈ 0.929, KL≈0.0024

### PMDCResidualClock 训练失败原因

根因：SDPA bypass。`_attn_implementation = "sdpa"` 跳过了 `eager_attention_forward` 的 patch。2026-06-28 修复（强制 `_attn_implementation = "eager"`）。

修复后验证：grad(flow_loss, p_final) ≈ 6e-17。即使 patch 生效，梯度经过 softmax 饱和区衰减到零。**这不是 bug，是架构信号衰减。**

### pm_retrieval 梯度验证（2026-06-28）

| 组件 | 梯度状态 |
|------|---------|
| PMRetrievalPhaseMemory out_proj | ✅ 非零（~1e-3） |
| LyricRetrievalAdapter q_mlp | ❌ ≈0（scaffold prior 主导） |
| LyricRetrievalAdapter k_mlp | ❌ ≈0 |
| LyricRetrievalAdapter v_mlp | ✅ 非零（~1e-7） |
| LyricRetrievalAdapter out_proj | ✅ 非零（~1e-7） |
| hidden residual norm | ✅ 0.96（正常） |
| attn_r entropy | 4.55（均匀=5.55） |
| attn_r center-p_audio 相关系数 | 0.987 |

---

## 技术债务

1. **`_attn_implementation = "eager"`** 是全局强制的，训练时会影响 self-attention 的性能。需要确认 SDPA 和 eager 在训练加速上的差异。
2. **`pm_retrieval` 推理脚本未实现。** 训练出的 checkpoint 不能直接用于生成。
3. **hook 方案是临时方案。** 不会污染 modeling 文件，但长期需要改成 layer processor。
4. **PM state 膨胀（norm 无约束）。** 需要加 normalize 或 weight decay。
5. **q/k 梯度 ≈ 0。** scaffold prior 太强，learnable 检索信号不够。

---

## 诊断信号

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| `mean_abs_p_delta < 0.003` | PMDC 没动 | 降低 `w_pbase` 或提高 `beta_max` |
| `pm_state_norm` 持续上涨 | 复数循环无约束 | 加 normalize 或 weight decay |
| `q_mlp grad ≈ 0` | scaffold prior 主导检索 | 降低 prior lambda 或提高 lr |
| `ret_res_norm` 爆炸 | residual 不稳定 | 降低 `lr`、加 `grad_clip` |
| `gate ≈ 0` | duration bias 被关闭 | 检查 gate_logit 梯度 |
