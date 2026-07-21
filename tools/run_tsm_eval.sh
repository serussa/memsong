#!/bin/bash
# ============================================================
# TSM 评估运行脚本
# 生成 + SongEval + AudioBox + PER (含 Late PER + LDG)
#
# 步骤：
#   1. 从 test.jsonl 选 5 中 + 5 英 样本
#   2. 用 3 种方法生成：baseline / transport_only / sinkhorn_tsm
#   3. 运行 SongEval、AudioBox、PER 全套评估
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_ROOT="/root/autodl-tmp/tsm_eval_results"
TEST_JSONL="$PROJECT_ROOT/Muse/infer/test.jsonl"
GEN_SCRIPT="$PROJECT_ROOT/gen_one.py"

TRANSPORT_CKPT="/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt"
TSM_CKPT="/root/autodl-tmp/tsm_sinkhorn/checkpoints/best_loss/pm_retrieval.pt"

SEED=42
DURATION=30
FILE_INDEXES_ZH=(0 10 20 30 40)
FILE_INDEXES_EN=(0 10 20 30 40)  # relative to en subset (maps to entries 50+)

# ============================================================
# Step 1: Generate audio for each method × sample
# ============================================================
echo "=========================================="
echo "Step 1: Generating audio"
echo "=========================================="

declare -A METHODS
METHODS[baseline]=""                              # no checkpoint
METHODS[transport_only]="$TRANSPORT_CKPT"
METHODS[sinkhorn_tsm]="$TSM_CKPT"

for METHOD in "${!METHODS[@]}"; do
    CKPT="${METHODS[$METHOD]}"

    for LANG in zh en; do
        OUT_DIR="$EVAL_ROOT/$METHOD/audio_${LANG}"
        mkdir -p "$OUT_DIR"

        if [ "$LANG" = "zh" ]; then
            ENTRIES=("${FILE_INDEXES_ZH[@]}")
        else
            ENTRIES=("${FILE_INDEXES_EN[@]}")
        fi

        for FILE_IDX in "${ENTRIES[@]}"; do
            OUTPUT_FILE="$OUT_DIR/$(printf '%06d.wav' $FILE_IDX)"
            if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
                echo "  [SKIP] $METHOD/$LANG index=$FILE_IDX already exists"
                continue
            fi

            echo "  [GEN] $METHOD/$LANG index=$FILE_IDX..."

            # gen_one.py reads pickle from stdin:
            #   (METHOD, CKPT, CAPTION, LYRICS, OUT_DIR, SEED, DURATION)
            python3 -c "
import json, sys, os
sys.path.insert(0, '$PROJECT_ROOT')

with open('$TEST_JSONL') as f:
    lines = f.readlines()

entry_idx = $FILE_IDX if '$LANG' == 'zh' else ($FILE_IDX + 50)
data = json.loads(lines[entry_idx])

# Extract style from first user message
first_msg = data['messages'][0]['content']
style_line = first_msg.split(chr(10))[0]
style = style_line.replace('Please generate a song in the following style:', '').strip()

# Extract all lyrics
full_lyrics = ''
for msg in data['messages']:
    content = msg.get('content', '')
    if '[lyrics:' in content:
        lyrics_part = content.split('[lyrics:')[1].split(']')[0]
        full_lyrics += lyrics_part.strip() + chr(10)

full_lyrics = full_lyrics.strip()

# Pickle the args
import pickle
args = (
    '$METHOD',
    '$CKPT' if '$CKPT' else '',
    style,
    full_lyrics,
    '$OUT_DIR',
    $SEED + $FILE_IDX,  # unique seed per file
    $DURATION,
)
pickle.dump(args, sys.stdout.buffer)
" | python "$GEN_SCRIPT" 2>&1 | tail -1

            echo "    -> $OUTPUT_FILE"
        done
    done
done

echo ""
echo "=========================================="
echo "Generation complete!"
echo "=========================================="
echo ""
ls -la "$EVAL_ROOT"/*/audio_cn/000000.wav 2>/dev/null | head -3
echo "..."

# ============================================================
# Step 2: SongEval + AudioBox evaluation
# ============================================================
echo ""
echo "=========================================="
echo "Step 2: Running SongEval & AudioBox (via Muse pipeline)"
echo "=========================================="

cd "$PROJECT_ROOT/Muse/eval_pipeline"
for METHOD in baseline transport_only sinkhorn_tsm; do
    AUDIO_DIR="$EVAL_ROOT/$METHOD"

    echo ">>> $METHOD"
    echo "  Chinese:  $AUDIO_DIR/audio_cn"
    echo "  English:  $AUDIO_DIR/audio_en"

    # Split audio (create symlinks or copy to expected layout)
    # The pipeline expects audio_cn files named 000000.wav~000049.wav
    # We already have that format, so we can run eval directly if we set up
    # the proper MODEL_NAME structure.

    # SongEval
    for LANG in cn en; do
        MODEL_NAME="${METHOD}_${LANG}"
        SRC_DIR="$AUDIO_DIR/audio_${LANG}"

        if [ ! -d "$SRC_DIR" ] || [ -z "$(ls -A "$SRC_DIR" 2>/dev/null)" ]; then
            echo "  SKIP $MODEL_NAME (no audio)"
            continue
        fi

        COUNT=$(ls "$SRC_DIR"/*.wav 2>/dev/null | wc -l)
        echo "  SongEval [$MODEL_NAME] ($COUNT files)..."

        # SongEval
        conda run --live-stream -n "${ENV_SONGEVAL:-songeval}" \
            env CUDA_VISIBLE_DEVICES=0 \
            python eval_songeval.py \
            --audio_dir "$SRC_DIR" \
            --model_name "$MODEL_NAME" \
            --results_dir "$AUDIO_DIR/results" 2>/dev/null || echo "  WARN: SongEval failed"

        # AudioBox
        conda run --live-stream -n "${ENV_AUDIOBOX:-audiobox}" \
            env CUDA_VISIBLE_DEVICES=0 \
            python eval_audiobox.py \
            --audio_dir "$SRC_DIR" \
            --model_name "$MODEL_NAME" \
            --results_dir "$AUDIO_DIR/results" 2>/dev/null || echo "  WARN: AudioBox failed"

        # PER via Qwen3 ASR
        conda run --live-stream -n "${ENV_PER:-per_eval}" \
            env CUDA_VISIBLE_DEVICES=0 \
            python transcribe.py \
            --audio_dir "$SRC_DIR" \
            --output "$AUDIO_DIR/transcriptions_${LANG}.jsonl" \
            --language "$([ "$LANG" = "cn" ] && echo "zh" || echo "en")" 2>/dev/null || echo "  WARN: Transcribe failed"

        # PER (long-form with late PER + LDG)
        if [ -f "$AUDIO_DIR/transcriptions_${LANG}.jsonl" ]; then
            GT_FILE="$PROJECT_ROOT/Muse/eval_pipeline/gt_lyrics/${LANG}.jsonl"
            if [ -f "$GT_FILE" ]; then
                conda run --live-stream -n "${ENV_PER:-per_eval}" \
                    env CUDA_VISIBLE_DEVICES=0 \
                    python calc_per_long.py \
                    --hyp_file "$AUDIO_DIR/transcriptions_${LANG}.jsonl" \
                    --gt_file "$GT_FILE" \
                    --model_name "$MODEL_NAME" \
                    --output_dir "$AUDIO_DIR/per_results_${LANG}" 2>/dev/null || echo "  WARN: PER calc failed"
            fi
        fi
    done
done

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="

# Summary
echo ""
echo "============= SUMMARY ============="
for METHOD in baseline transport_only sinkhorn_tsm; do
    echo "[$METHOD]"
    for LANG in cn en; do
        PER_FILE="$EVAL_ROOT/$METHOD/per_results_${LANG}/summary.csv"
        if [ -f "$PER_FILE" ]; then
            cat "$PER_FILE"
        fi
    done
    echo ""
done
