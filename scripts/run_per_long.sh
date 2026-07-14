#!/bin/bash
#
# 长程 PER 一键评估脚本
# 基于全局音素对齐，计算 Overall / Early / Middle / Late PER 和 LDG。
#
# 用法:
#   bash scripts/run_per_long.sh <hyp_file> <gt_file> <output_dir> [model_name]
#
# 示例:
#   bash scripts/run_per_long.sh \
#       output/my_model/per_results/transcription.jsonl \
#       Muse/eval_pipeline/gt_lyrics/zh.jsonl \
#       output/my_model/per_results_long \
#       my_model
#
# 依赖:
#   - Muse PER 管线: calc_per_long.py, phoneme_utils.py
#   - Python 包: jieba, pypinyin, g2p_en, pypinyin_dict

set -euo pipefail

# ── 参数 ──────────────────────────────────────────────
HYP_FILE="${1:?错误: 请指定 ASR 转写 JSONL 路径作为第一个参数}"
GT_FILE="${2:?错误: 请指定 GT 歌词 JSONL 路径作为第二个参数}"
OUTPUT_DIR="${3:?错误: 请指定输出目录作为第三个参数}"
MODEL_NAME="${4:-"$(basename "$(dirname "$HYP_FILE")")"}"

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CALC_LONG="$SCRIPT_DIR/Muse/eval_pipeline/calc_per_long.py"

# ── 检查 ──────────────────────────────────────────────
if [ ! -f "$HYP_FILE" ]; then
    echo "错误: ASR 转写文件不存在: $HYP_FILE"
    exit 1
fi

if [ ! -f "$GT_FILE" ]; then
    echo "错误: GT 歌词文件不存在: $GT_FILE"
    exit 1
fi

if [ ! -f "$CALC_LONG" ]; then
    echo "错误: calc_per_long.py 不存在: $CALC_LONG"
    exit 1
fi

echo "============================================"
echo " 长程 PER 评估 (分段指标)"
echo "============================================"
echo "ASR 转写: $HYP_FILE"
echo "GT 歌词:   $GT_FILE"
echo "输出目录:  $OUTPUT_DIR"
echo "模型名称:  $MODEL_NAME"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"

python3 "$CALC_LONG" \
    --hyp_file "$HYP_FILE" \
    --gt_file "$GT_FILE" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo " 评估完成！"
echo " Per-song:  $OUTPUT_DIR/songs.jsonl"
echo " Summary:   $OUTPUT_DIR/summary.csv"
echo "============================================"
