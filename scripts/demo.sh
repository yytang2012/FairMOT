#!/bin/bash

# FairMOT Demo Script - Simple wrapper for demo.py
# Usage: ./demo.sh [input_video] [output_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_VIDEO="${1:-./samples/MOT16-03.mp4}"
OUTPUT_DIR="$2"
MODEL_PATH="./models/fairmot_dla34.pth"

export CUDA_VISIBLE_DEVICES=0

if [ -n "$OUTPUT_DIR" ]; then
    cd "$SCRIPT_DIR/.." && python3 scripts/demo.py \
        --input_video "$INPUT_VIDEO" \
        --output_dir "$OUTPUT_DIR" \
        --load_model "$MODEL_PATH" \
        --device "auto" \
        --gpus "0" \
        --conf_thres 0.4 \
        --det_thres 0.3 \
        --output_format "video" \
        --quality "high" \
        --input_w 1088 \
        --input_h 608
else
    cd "$SCRIPT_DIR/.." && python3 scripts/demo.py \
        --input_video "$INPUT_VIDEO" \
        --load_model "$MODEL_PATH" \
        --device "auto" \
        --gpus "0" \
        --conf_thres 0.4 \
        --det_thres 0.3 \
        --output_format "video" \
        --quality "high" \
        --input_w 1088 \
        --input_h 608
fi