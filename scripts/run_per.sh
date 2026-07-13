#!/bin/bash
#
# PER 一键评估脚本
# 对输入目录中的音频文件进行 ASR 转录（本地 Qwen3-ASR），并与 GT 歌词计算 PER
#
# 用法:
#   bash scripts/run_per.sh <input_dir> <language:cn/en> [output_dir]
#
# 示例:
#   bash scripts/run_per.sh /path/to/generated_audio_cn cn /path/to/eval_results
#
# 依赖:
#   - Qwen3-ASR-1.7B 本地模型（/root/autodl-tmp/models/Qwen3-ASR-1.7B）
#   - Muse PER 管线（Muse/eval_pipeline/{transcribe_local.py,calc_per.py,phoneme_utils.py,gt_lyrics/})
#   - qwen-asr, jieba, pypinyin, g2p_en, pypinyin_dict, nltk Python 包

set -euo pipefail

# ── 参数 ──────────────────────────────────────────────
INPUT_DIR="${1:?错误: 请指定输入目录作为第一个参数}"
LANG="${2:?错误: 请指定语言 cn 或 en 作为第二个参数}"
OUTPUT_DIR="${3:-"${INPUT_DIR}/per_results"}"

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_PIPELINE="$SCRIPT_DIR/Muse/eval_pipeline"
MODEL_PATH="/root/autodl-tmp/models/Qwen3-ASR-1.7B"

TRANS_SCRIPT="$EVAL_PIPELINE/transcribe_local.py"
CALC_SCRIPT="$EVAL_PIPELINE/calc_per.py"

if [ "$LANG" == "cn" ] || [ "$LANG" == "zh" ]; then
    GT_FILE="$EVAL_PIPELINE/gt_lyrics/zh.jsonl"
    LANG_DISPLAY="中文"
else
    GT_FILE="$EVAL_PIPELINE/gt_lyrics/en.jsonl"
    LANG_DISPLAY="英文"
fi

# ── 检查 ──────────────────────────────────────────────
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$TRANS_SCRIPT" ]; then
    echo "错误: 找不到转录脚本: $TRANS_SCRIPT"
    exit 1
fi

if [ ! -f "$CALC_SCRIPT" ]; then
    echo "错误: 找不到 PER 计算脚本: $CALC_SCRIPT"
    exit 1
fi

if [ ! -f "$GT_FILE" ]; then
    echo "错误: 找不到 GT 歌词文件: $GT_FILE"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 找不到 Qwen3-ASR 模型: $MODEL_PATH"
    exit 1
fi

AUDIO_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" \) | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "错误: 输入目录中没有音频文件 (.wav/.mp3/.flac): $INPUT_DIR"
    exit 1
fi

echo "========================================="
echo " PER 评估 (${LANG_DISPLAY})"
echo "========================================="
echo "输入目录:   $INPUT_DIR"
echo "输出目录:   $OUTPUT_DIR"
echo "音频文件:   ${AUDIO_COUNT} 个"
echo "语言:       ${LANG_DISPLAY}"
echo "ASR 模型:   $MODEL_PATH"
echo "GT 文件:    $GT_FILE"
echo "========================================="

# ── Step 1: ASR 转录 ──────────────────────────────
mkdir -p "$OUTPUT_DIR"
TRANS_FILE="$OUTPUT_DIR/transcription.jsonl"

echo "[1/2] ASR 转录（本地 Qwen3-ASR-1.7B）..."
python3 "$TRANS_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output "$TRANS_FILE" \
    --model_path "$MODEL_PATH"

echo ""

# ── Step 2: 计算 PER ──────────────────────────────
RESULT_FILE="$OUTPUT_DIR/per_result.json"

echo "[2/2] 计算 PER..."
python3 "$CALC_SCRIPT" \
    --hyp_file "$TRANS_FILE" \
    --gt_file "$GT_FILE" \
    --model_name "$(basename "$INPUT_DIR")" \
    --output "$RESULT_FILE"

echo ""
echo "========================================="
echo " 评估完成！"
echo " 转录结果: $TRANS_FILE"
echo " PER 结果: $RESULT_FILE"
echo " 详细结果: ${RESULT_FILE%.json}_details.jsonl"
echo "========================================="
