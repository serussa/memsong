# 概率偏置软对齐 (Probabilistic Beat Alignment Bias) 改动总结

## 设计思路

在 PhaseMemory 现有流程中（复数旋转 → 熵门控 → 自适应锚点 → update），于 update 完成之后、trajectory memory 和 stabilize 之前，插入一个 **概率偏置网络**。该网络根据当前相位状态、去噪进度和 beat_phase 预测一个偏移量 `delta` 并旋转相位。`delta` 从 `N(mu, sigma^2)` 采样得到，通过 KL 散度正则化到 `N(0,1)`，模型可以自主决定是否采纳节拍信息。

## 修改文件清单

### 1. `acestep/phase_memory.py` — 核心模块改动

**`__init__`**:
- 新增 `self.bias_net`：3 层 MLP（4→64→64→2），输入 `[zr_mean, zi_mean, step_ratio, beat_phase]`，输出 `[mu, log_var]`
- 输出层 normal_(0,0.01) 初始化，log_var bias 设为 -5.0，保证梯度能反向传播到前层
- 新增 `self.kl_loss = None` 和 `self.max_delta = 0.3`

**`forward(self, h, diffusion_step, beat_phase=None)`**:
- 新增 `beat_phase` 参数，`None` 时跳过偏置逻辑（退化为原行为）
- 在 update 完成后（zr_new/zi_new 之后）、trajectory memory 之前：
  1. 计算 `zr_mean, zi_mean`（按 latent dim 平均）
  2. 拼接 `[zr_mean, zi_mean, step_ratio, beat_phase]` 作为 bias_net 输入
  3. 输出 `mu, log_var`，训练时重参数化采样 `delta`，推理时直接用 `mu`
  4. `delta = 0.3 * tanh(raw_delta)` 限幅
  5. 旋转：`zr_new *= cos(delta); zi_new = zr_new*sin + zi_new*cos`（注意此处旋转实现绕自身，始终保持幅度稳定）
  6. 计算 KL 散度：`-0.5 * (1 + log_var - mu^2 - exp(log_var))` → 取 mean，存到 `self.kl_loss`

**`reset()`**: 新增 `self.kl_loss = None`

### 2. `acestep/models/sft/modeling_acestep_v15_base.py` — 模型前向传播改动

**`AceStepDiTLayer.forward`**:
- 参数列表新增 `beat_phase: Optional[torch.Tensor] = None`
- 调用 `self.phase_memory()` 时传入 `beat_phase=beat_phase`

**`AceStepDiTModel.forward`**:
- 参数列表新增 `beat_phase: Optional[torch.Tensor] = None`
- 调用 layer_module 时传入 `beat_phase=beat_phase`

**`AceStepConditionGenerationModel.forward`** (训练):
- 参数列表新增 `beat_phase: Optional[torch.Tensor] = None`
- 调用 decoder 时传入 `beat_phase=beat_phase`
- forward 返回值新增 `"kl_loss"`：遍历所有子模块收集 PhaseMemory 的 `kl_loss`

**`AceStepConditionGenerationModel.generate_audio`** (推理):
- 参数列表新增 `beat_phase: Optional[torch.Tensor] = None`
- decoder 推理调用时传入 `beat_phase=beat_phase`

### 3. `acestep/training_v2/fixed_lora_module.py` — 训练接入

- 在 decoder forward 前，检查 batch 中 `has_beat_phase`，有数据时传 `beat_phase` 给 decoder
- Decoder forward 后，计算 KL loss：遍历所有子模块收集 `module.kl_loss`，乘以 `kl_lambda` 加到 `diffusion_loss`
- **删除**了原有的 `_compute_beat_align_loss()` 方法（Huber loss 硬对齐）及其调用、section header

### 4. `acestep/training_v2/configs.py` — 配置

- 新增 `kl_lambda: float = 0.001` 字段
- 删除 `beat_align_lambda` 字段
- 删除 `to_dict()` 中的 `beat_align_lambda` 序列化

### 5. `acestep/training_v2/cli/args.py` — CLI 参数

- 新增 `--kl-lambda`（默认 0.001）
- 删除 `--beat-align-lambda`

### 6. `acestep/training_v2/cli/config_builder.py` — 配置组装

- 删除 `beat_align_lambda=getattr(...)` 行
- 新增 `kl_lambda=getattr(...)` 行

### 7. `docs/PHASEMEMORY.md` — 训练命令

- 新增 `--beat-phase-dir /root/autodl-tmp/musicdata/beat_phases` 和 `--kl-lambda 0.05`
- 删除 `--beat-align-lambda`

## 数据流

```
数据集 .npy 文件
  → PreprocessedTensorDataset (data_module.py)
  → batch["beat_phase"] [B, T], batch["has_beat_phase"] [B]
  → FixedLoRAModule.training_step()
  → decoder(**kwargs, beat_phase=beat_phase)
  → AceStepDiTModel → AceStepDiTLayer
  → PhaseMemory.forward(h, diffusion_step, beat_phase=beat_phase)
    → bias_net 输出 mu, log_var
    → 采样 delta → 旋转 zr_new/zi_new
    → 计算 kl_loss 存到 self.kl_loss
  → training_step() 收集 kl_loss → diffusion_loss += kl_lambda * kl_loss
```

## 训练命令

```bash
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
    --device auto \
    --beat-phase-dir /root/autodl-tmp/musicdata/beat_phases \
    --kl-lambda 0.05
```

## 调参建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `--kl-lambda` | 0.001 | KL 权重，观察 TensorBoard 调到 diffusion loss 的 1%~5% |
| `--beat-phase-dir` | /root/autodl-tmp/musicdata/beat_phases | beat_phase .npy 文件目录 |
| `max_delta`（代码硬编码） | 0.3 rad | 单步最大相位偏移，可在 PhaseMemory.__init__ 调整 |
