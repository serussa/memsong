#!/usr/bin/env bash
# PD-SBAR Training — 3 epochs, LoRA rank 16 on rewired layers 8/12/16/20
set -e

export ACESTEP_LOCAL_MODEL_CODE=1
export SIDESTEP_SAFE_ROOT="/"

python train.py --yes fixed \
  --adapter-type pd_sbar \
  --checkpoint-dir /root/autodl-tmp/Ace-Step1.5/checkpoints \
  --model-variant sft \
  --dataset-dir /root/autodl-tmp/musicdata/train_tensors \
  --output-dir /root/autodl-tmp/pd_sbar_exp_final1 \
  --learning-rate 1e-4 \
  --batch-size 1 \
  --epochs 3 \
  --save-every 1 \
  --no-gradient-checkpointing \
  --seed 42

echo "Training complete."
