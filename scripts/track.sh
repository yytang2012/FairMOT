#!/bin/bash

# FairMOT Track Script - Simple wrapper for track.py
# Usage: ./track.sh [dataset] [split] [data_dir]

# Default parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-MOT20}"
SPLIT="${2:-val}"
DATA_DIR="${3:-./datasets}"
MODEL_PATH="./models/mot20_fairmot.pth"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run tracking evaluation with commonly used parameters
cd "$SCRIPT_DIR/.." && python3 scripts/track.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --data_dir "$DATA_DIR" \
    --load_model "$MODEL_PATH" \
    --device "auto" \
    --gpus "0" \
    --conf_thres 0.4 \
    --det_thres 0.3 \
    --nms_thres 0.4 \
    --track_buffer 30 \
    --min_box_area 100 \
    --input_w 1088 \
    --input_h 608 \
    --exp_id "default"