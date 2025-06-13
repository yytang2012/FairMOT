#!/bin/bash

# FairMOT Train Script - Simple wrapper for train.py
# Usage: ./train.sh [dataset] [epochs] [batch_size] [data_dir]

# Default parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-MOT20}"
EPOCHS="${2:-30}"
BATCH_SIZE="${3:-4}"
DATA_DIR="${4:-./datasets}"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0


cd "$SCRIPT_DIR/.." && python3 scripts/train.py \
    --dataset "$DATASET" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --data_dir "$DATA_DIR" \
    --device "auto" \
    --lr 1e-4 \
    --val_intervals 5 \
    --exp_id "training"