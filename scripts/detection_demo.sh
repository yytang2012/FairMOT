#!/bin/bash

# FairMOT Detection Demo Script - Simple wrapper for detection_demo.py
# Usage: ./detection_demo.sh [weights_path] [data_path]

# Default parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_PATH="${1:-./models/mot20_fairmot.pth}"
DATA_PATH="${2:-./datasets/MOT20/test/MOT20-04/img1}"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run detection demo with commonly used parameters
cd "$SCRIPT_DIR/.." && python3 scripts/detection_demo.py \
    --weights "$WEIGHTS_PATH" \
    --data_path "$DATA_PATH" \
    --conf_thres 0.3 \
    --img_size 640