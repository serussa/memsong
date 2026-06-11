# PhaseMemory 功能总结

## 概述

为 ACE-Step-1.5 DiT 模型新增 **最小循环复数相位状态 (PhaseMemory)** 模块，作为除 LoRA/LoKR 之外的第三种轻量级微调适配器。

---

## 核心设计

### 数学公式

```text
h_pool  = mean(h, dim=1)                                     # [B, H]
z_input = normalize(proj_in(h_pool))                         # 归一化输入驱动力
ω       = π · tanh(proj_omega([h_pool, Re(z_t)]))             # 有界旋转频率 ∈ [-π, π]
ω_pos   = π · tanh(proj_pos(rope(t)))                         # RoPE 时间锚点 ∈ [-π, π]
g_t     = sigmoid(proj_gate([h_pool, Re(z_t)]))               # scalar input gate (初始≈0.05)
r_t     = sigmoid(proj_read([h_pool, Re(z_t)]))               # scalar read gate
z_{t+1} = (1 - g_t) · (z_t · exp(i · (ω + α·ω_pos))) + g_t · z_input
h       = h + r_t · proj_out(Re(z_{t+1}))                     # 极弱残差注入
```

其中：
- `z` ∈ ℂ^mem_dim：复数相位状态
- `h_pool` = mean(h)：单层 token 均值汇聚
- `z_input` = normalize(Linear(h_pool)) → ℂ^mem_dim：归一化输入驱动力
- `g_t` ∈ (0,1)：scalar input gate，控制 memory 写入
- `r_t` ∈ (0,1)：scalar read gate，控制回注入强度
- `ω` = π · tanh(…) → [-π, π]：有界旋转频率，支持正反向相位流
- `ω_pos` = π · tanh(…) → [-π, π]：RoPE 时间锚点相位偏移
- `α`：可学习标量（默认 0.1），控制时间锚点强度
- `proj_out`：复数实部 → 隐藏维度的线性投影（**强制零初始化**）
- `z_init`：可学习的初始复数相位

### 训练/推理双模式

| 模式 | 状态来源 | 梯度行为 |
|------|---------|---------|
| **训练** (`self.training=True`) | `z_init` 每步独立展开，无跨 batch 缓冲区 | 全微分，无 detach |
| **推理** (`self.training=False`) | persistent buffer 跨 denoising step 递推 | `.detach()` 截断梯度 |

### 注入位置

**仅 Layer 12**（`num_layers // 2`），24 层 DiT 中只注入了 1 层。

### 参数规模

- `mem_dim` 默认 **128**，可训练参数 **~1.1M**（`hidden_size=4096` 时）
- 总模型参数：~2.4B
- 占比：**0.05%**
- 可通过 `--phase-mem-dim` 调整（64→0.6M, 256→2.1M）

---

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `acestep/phase_memory.py` | **新增** | PhaseMemory 核心模块 + `freeze_except_phase_memory` / `unfreeze_all` 工具函数 |
| `acestep/models/sft/modeling_acestep_v15_base.py` | 修改 | `AceStepDiTLayer` 支持 `use_phase_memory`；`AceStepDiTModel` 仅 Layer 12 注入；`_init_weights` 保护 `proj_out` zero-init |
| `acestep/training/configs.py` | 修改 | 新增 `PhaseMemoryConfig` |
| `acestep/training_v2/configs.py` | 修改 | 新增 `PhaseMemoryConfigV2`；`TrainingConfigV2.adapter_type` 支持 `"phase_memory"` |
| `acestep/training_v2/fixed_lora_module.py` | 修改 | 新增 `_inject_phase_memory()` 方法 |
| `acestep/training_v2/trainer_helpers.py` | 修改 | `save_adapter_flat` / `resume_checkpoint` 支持 PhaseMemory |
| `acestep/training/phase_memory_checkpoint.py` | **新增** | PhaseMemory 权重保存/加载（safetensors 格式） |
| `acestep/training_v2/trainer_basic_loop.py` | 修改 | PhaseMemory gate 观测日志 |
| `acestep/training_v2/trainer_fixed.py` | 修改 | Fabric 训练路径写入 PhaseMemory 观测日志（TensorBoard） |
| `acestep/training/path_safety.py` | 修改 | 支持 `SIDESTEP_SAFE_ROOT` 环境变量 |
| `acestep/phase_memory.py` | 修改 | 新增诊断统计 `last_*`；`diffusion_step` 形状处理更鲁棒 |
| `acestep/models/sft/modeling_acestep_v15_base.py` | 修改 | PhaseMemory 初始化支持 `phase_mem_dim/phase_mem_init_scale` 配置覆盖 |

---

## 两个关键 Bug 修复

### Bug 1：`z_init_real` 在训练启动时 NaN

**根因**：HuggingFace `from_pretrained` → `model.to(bf16)` 流程中 `z_init_real` 被设为 NaN

**修复**：`_init_weights` 的 `isinstance(module, PhaseMemory)` 分支中显式重新初始化：
```python
with torch.no_grad():
    module.z_init_real.normal_(0.0, 0.01)
    module.z_init_imag.normal_(0.0, 0.01)
```

### Bug 2：`proj_out` 的 zero-init 被覆盖

**根因**：`nn.Module.apply()` post-order 遍历时，`proj_out`（`nn.Linear`）先被 `_init_weights` 的 `isinstance(Linear)` 分支 `normal_` 初始化

**修复**（双重保护）：
1. `proj_out` 设置 `_pm_safe_output = True` 标记
2. `_init_weights` 的 `nn.Linear` 分支检查此标记 → 跳过 normal init
3. `isinstance(PhaseMemory)` 分支强制 `nn.init.zeros_`（belt-and-suspenders）

---

## 使用方法

### 1. 准备数据集

预处理后的 `.pt` 文件放入同一目录：
```
train_tensors/
  sample_001.pt
  sample_002.pt
  ...
```

### 2. 运行训练

```bash
cd /root/ACE-Step-1.5

# 使用 musicgen conda 环境
conda activate musicgen

# PhaseMemory 训练
ACESTEP_LOCAL_MODEL_CODE=1 SIDESTEP_SAFE_ROOT="/" python train.py --yes fixed \
    --adapter-type phase_memory \
    --checkpoint-dir /root/autodl-tmp/Ace-Step1.5/checkpoints \
    --model-variant sft \
    --dataset-dir /root/autodl-tmp/musicdata/train_tensors \
    --output-dir /root/autodl-tmp/lyrics_checkpoints \
    --learning-rate 1e-4 \
    --batch-size 16 \
    --gradient-accumulation 1 \
    --epochs 50 \
    --save-every 5 \
    --device auto
```

### 3. 关键参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--adapter-type` | 必须设为 `phase_memory` | `phase_memory` |
| `--model-variant` | 模型版本 | `sft` |
| `--learning-rate` | 学习率 | `1e-4` ~ `2e-4` |
| `--batch-size` | 每卡 batch size | RTX 5090: `4-8` |
| `--gradient-accumulation` | 梯度累积步数 | `2-4` |
| `--epochs` | 训练轮数 | 建议 `50+`（参数少，需多轮） |
| `--save-every` | 每 N epoch 保存 | `10` |
| `--device` | 设备 | `auto` |
| `SIDESTEP_SAFE_ROOT` | 路径安全检查 | 设为 `"/"` 跳过限制 |

### 4. 断点续训

```bash
SIDESTEP_SAFE_ROOT="/" python train.py --yes fixed \
    --adapter-type phase_memory \
    --checkpoint-dir /root/autodl-tmp/Ace-Step1.5/checkpoints \
    --model-variant sft \
    --dataset-dir /root/autodl-tmp/musicdata/train_tensors \
    --output-dir /root/autodl-tmp/newest_checkpoints \
    --resume-from /root/autodl-tmp/newest_checkpoints/checkpoints/epoch_10_loss_0.8453 \
    ...  # 其他参数同上
```

### 5. 查看训练进度

```bash
# TensorBoard
tensorboard --logdir /root/autodl-tmp/new_checkpoints/runs --bind_all
```

### 6. 训练输出

训练完成后，`output_dir` 下会生成：
- `phase_memory_weights.safetensors` — PhaseMemory 权重
- `phase_memory_config.json` — 配置元数据
- `training_state.pt` — 断点续训状态
- `runs/` — TensorBoard 日志

---

## RTX 5090 超参数建议

| 参数 | 保守 | 推荐 | 激进 |
|------|------|------|------|
| `batch_size` | 4 | 8 | 12 |
| `gradient_accumulation` | 4 | 2 | 1 |
| 有效 batch | 16 | 16 | 12 |
| `learning_rate` | 1e-4 | 2e-4 | 5e-4 |
| `epochs` | 30 | 50 | 100 |

PhaseMemory 仅 2.1M 参数，5090 上训练非常快

---

## 技术要点

1. **Zero-init 是强制要求**：`proj_out` 必须严格零初始化，确保 Step 0 注入 = 0，避免灾难性遗忘
2. **归一化输入驱动力**：`z_input` 归一化可以抑制幅度漂移与相位爆炸
3. **训练模式无跨 batch 状态**：`self.training=True` 时每步独立展开 `z_init`，不依赖 buffer
4. **推理模式保持状态**：`self.training=False` 时 buffer 跨 denoising step 递推，使用 `.detach()` 截断梯度
5. **路径安全**：数据不在 workspace 子目录时需设置 `SIDESTEP_SAFE_ROOT="/"`

---

## 重构记录 (2026-05-14)

### V3 → V4: Tiny Controlled Phase Memory

**动机**：原 2048-dim 强制正旋转 + 无门控写入，导致 memory 无限制累积、lyric contamination、repetition drift。

| 变更项 | V3 (旧) | V4 (新) |
|--------|---------|---------|
| `mem_dim` | 2048 | **128** |
| 参数量 | 33.6M | **2.1M** |
| ω 范围 | `softplus` → (0, ∞) 单向漂移 | **`π·tanh` → [-π, π] 双向** |
| 写入控制 | 无门控 | **scalar input gate `g_t`** |
| 读出控制 | — | **scalar read gate `r_t`** |
| gate 初始化 | — | **bias = -3 → 初始 g≈0.047** |
| gate 结构 | — | `Linear([h_pool, Re(z_t)], 1) + sigmoid` |

### 设计原理

1. **Input gate 比 output gate 更根本**：output gate 只阻止"被污染的 memory 输出"，但污染已经写入 memory；input gate 在写入侧阻止污染进入。
2. **Read gate 提供可观测性**：`r_t` 单独调节回注入强度，便于跟踪与解释。
3. **Bounded ω 避免 phase drift**：`softplus` 强制正向旋转导致长期漂移；`tanh` 约束到 [-π, π] 支持自然相位 recurrence。
4. **Scalar gate 而非 vector gate**：`[B, 1]` 比 `[B, M]` 更强约束，更少参数（8k vs 2.6M），更易解释。
5. **-3 bias 确保冷启动安全**：`sigmoid(-3) ≈ 0.047`，训练初期 gate 几乎关闭，memory 不污染 backbone。

### 分析脚本

```bash
# 训练后的模型分析（PhaseMemory）
python scripts/analyze_phase_dynamics.py \
    --model-root /root/autodl-tmp/Ace-Step1.5 \
    --pm-dir /root/autodl-tmp/new_checkpoints/checkpoints/epoch_N_loss_X \
    --output-dir /root/ACE-Step-1.5/output/phase_memory_analysis

# 基线（PhaseMemory 未训练）
python scripts/analyze_phase_dynamics_baseline.py \
    --model-root /root/autodl-tmp/Ace-Step1.5 \
    --output-dir /root/ACE-Step-1.5/output/phase_memory_analysis_baseline

# 原模型（无 PhaseMemory，同一层的 hidden state 随机投影作为对照）
python scripts/analyze_hidden_state_phase.py \
    --model-root /root/autodl-tmp/Ace-Step1.5 \
    --output-dir /root/ACE-Step-1.5/output/hidden_state_phase_analysis
```

### 可能的 failure mode

| 问题 | 诊断 | 处理 |
|------|------|------|
| Gate collapse (g→0) | mean gate 持续下降，`proj_gate` 梯度消失 | 降低 gate 权重衰减；check 激活函数梯度 |
| Read collapse (r→0) | read gate 长期趋近 0 | 检查 `proj_read` 初始化与学习率 |
| Gate all-open (g→1) | bias 被优化器推翻，memory 瞬间污染 | 提高 gate 的 weight_decay |
| Phase explosion (|z|→∞) | 归一化失效或数值溢出 | 检查 `normalize` 与 dtype；必要时 clamp |
