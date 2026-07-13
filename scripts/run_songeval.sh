#!/bin/bash
#
# SongEval 一键评估脚本
# 评估一个文件夹下所有音频文件（.wav/.mp3/.flac），输出结果到指定目录
#
# 用法:
#   bash scripts/run_songeval.sh <input_dir> <output_dir>
#
# 示例:
#   bash scripts/run_songeval.sh /path/to/generated_songs /path/to/eval_results
#
# 依赖:
#   - SongEval 代码位于项目根目录下的 SongEval/
#   - 预训练权重位于 SongEval/ckpt/model.safetensors
#   - MuQ 编码器通过 HF_HOME 自动加载缓存

set -euo pipefail

# ── 参数 ──────────────────────────────────────────────
INPUT_DIR="${1:?错误: 请指定输入目录作为第一个参数}"
OUTPUT_DIR="${2:?错误: 请指定输出目录作为第二个参数}"
GPU="${3:-true}"  # 第三个参数可选: true=GPU, false=CPU

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"       # 项目根目录
SONGEVAL_DIR="$SCRIPT_DIR/SongEval"

# ── 检查 ──────────────────────────────────────────────
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$SONGEVAL_DIR/ckpt/model.safetensors" ]; then
    echo "错误: SongEval 检查点不存在: $SONGEVAL_DIR/ckpt/model.safetensors"
    exit 1
fi

# ── 统计音频文件数量 ────────────────────────────────
AUDIO_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" \) | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "错误: 输入目录中没有音频文件 (.wav/.mp3/.flac): $INPUT_DIR"
    exit 1
fi

echo "==================================="
echo " SongEval 评估"
echo "==================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "音频文件: ${AUDIO_COUNT} 个"
echo "GPU: $GPU"
echo "==================================="

# ── 运行评估 ──────────────────────────────────────────
cd "$SONGEVAL_DIR"

if [ "$GPU" = "true" ]; then
    python eval.py -i "$INPUT_DIR" -o "$OUTPUT_DIR"
else
    python eval.py -i "$INPUT_DIR" -o "$OUTPUT_DIR" --use_cpu True
fi

echo ""
echo "==================================="
echo " 评估完成！"
echo " 结果: $OUTPUT_DIR/result.json"
echo "==================================="
