#!/bin/bash
#
# Audiobox-Aesthetics 一键评估脚本
# 评估一个文件夹下所有音频文件（.wav/.mp3/.flac），输出结果到指定目录
#
# 用法:
#   bash scripts/run_audiobox_aes.sh <input_dir> <output_dir>
#
# 示例:
#   bash scripts/run_audiobox_aes.sh /path/to/generated_songs /path/to/eval_results
#
# 依赖:
#   - audiobox-aesthetics 包已安装（pip install -e audiobox-aesthetics/）
#   - 预训练权重位于 /root/autodl-tmp/models/audiobox-aesthetics/checkpoint.pt

set -euo pipefail

# ── 参数 ──────────────────────────────────────────────
INPUT_DIR="${1:?错误: 请指定输入目录作为第一个参数}"
OUTPUT_DIR="${2:?错误: 请指定输出目录作为第二个参数}"

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="/root/autodl-tmp/models/audiobox-aesthetics/checkpoint.pt"

# ── 检查 ──────────────────────────────────────────────
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "错误: Audiobox-Aesthetics 检查点不存在: $CKPT"
    exit 1
fi

AUDIO_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" \) | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "错误: 输入目录中没有音频文件 (.wav/.mp3/.flac): $INPUT_DIR"
    exit 1
fi

echo "========================================="
echo " Audiobox-Aesthetics 评估"
echo "========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "音频文件: ${AUDIO_COUNT} 个"
echo "检查点:   $CKPT"
echo "========================================="

# ── 运行评估 ──────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# 生成输入 jsonl
INPUT_JSONL="$OUTPUT_DIR/input.jsonl"
find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" \) | sort | while read -r f; do
    echo "{\"path\":\"$f\"}"
done > "$INPUT_JSONL"

cd "$SCRIPT_DIR"
python3 -c "
import soundfile as sf
import torchaudio
import numpy as np
import torch, json, sys

# Monkey-patch torchaudio.load to use soundfile (FFmpeg compat)
_original_load = torchaudio.load
def patched_load(path, frame_offset=0, num_frames=-1):
    info = sf.info(path)
    sr = info.samplerate
    data, _ = sf.read(path, start=frame_offset, frames=num_frames if num_frames > 0 else -1, dtype='float32')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    return torch.from_numpy(data), sr
torchaudio.load = patched_load

from audiobox_aesthetics.infer import initialize_predictor

with open('$INPUT_JSONL') as f:
    batch = [json.loads(line) for line in f]

predictor = initialize_predictor('$CKPT')
results = predictor.forward(batch)

with open('$OUTPUT_DIR/result.json', 'w') as f:
    json.dump(results, f, indent=2)
for r in results:
    print(json.dumps(r))
"

echo ""
echo "========================================="
echo " 评估完成！"
echo " 结果: $OUTPUT_DIR/result.json"
echo "========================================="
