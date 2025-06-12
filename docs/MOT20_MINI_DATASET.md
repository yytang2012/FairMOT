# MOT20 Mini Dataset Guide

This guide explains how to create and use a mini MOT20 dataset for quick testing and development with FairMOT.

## Overview

The MOT20 mini dataset is a subset of the full MOT20 dataset designed for:
- **Quick prototyping**: Test changes without waiting for full dataset processing
- **Development**: Faster iteration during development and debugging  
- **CI/CD**: Lightweight dataset for automated testing
- **Learning**: Smaller dataset for understanding the training pipeline

## Quick Start

### 1. Set Environment Variable

```bash
export FAIRMOT_DATA_ROOT=/path/to/your/dataset/root
# Example: export FAIRMOT_DATA_ROOT=/media/yytang/14T/Dataset/MOT/JDE/
```

### 2. Create Mini Dataset

```bash
cd /path/to/FairMOT/tools/data_preparation
python create_mot20_mini.py
```

### 3. Generate Training Labels

```bash
python gen_labels_20_mini.py
```

### 4. Train Model

```bash
cd /path/to/FairMOT
FAIRMOT_DATA_ROOT=/path/to/your/dataset/root PYTHONPATH=/path/to/FairMOT/src/lib python scripts/train.py mot \
    --dataset jde \
    --data_cfg src/lib/cfg/mot20_mini.json \
    --batch_size 2 \
    --num_epochs 1 \
    --lr 1e-4 \
    --exp_id mot20_mini_test \
    --device cuda:0
```

## Detailed Usage

### Creating Mini Dataset

The `create_mot20_mini.py` script extracts a subset of sequences and frames from the full MOT20 dataset.

#### Basic Usage
```bash
python create_mot20_mini.py
```

#### Advanced Options
```bash
python create_mot20_mini.py \
    --source /path/to/full/MOT20 \
    --output /path/to/MOT20_mini \
    --sequences MOT20-01 MOT20-02 MOT20-03 \
    --frames 50 \
    --skip 3
```

#### Parameters
- `--source`: Path to original MOT20 dataset (default: `/Users/yutao/Dataset/MOT/JDE/MOT20`)
- `--output`: Where to create mini dataset (default: `/Users/yutao/Dataset/MOT/JDE/MOT20_mini`)
- `--sequences`: Which sequences to include (default: `MOT20-01 MOT20-02`)
- `--frames`: Maximum frames per sequence (default: `100`)
- `--skip`: Frame sampling interval - take every Nth frame (default: `5`)

### Generating Labels

The `gen_labels_20_mini.py` script creates YOLO-format training labels for the mini dataset.

#### Basic Usage
```bash
python gen_labels_20_mini.py
```

#### With Custom Dataset Path
```bash
python gen_labels_20_mini.py --dataset_root /path/to/MOT20_mini
```

#### Validation Only
```bash
python gen_labels_20_mini.py --validate
```

## Dataset Structure

The mini dataset maintains the same structure as the full MOT20 dataset:

```
MOT20_mini/
├── images/
│   └── train/
│       ├── MOT20-01/
│       │   ├── img1/
│       │   │   ├── 000001.jpg
│       │   │   ├── 000006.jpg  # Every 5th frame
│       │   │   └── ...
│       │   └── seqinfo.ini
│       └── MOT20-02/
│           └── ...
├── labels_with_ids/
│   └── train/
│       ├── MOT20-01/
│       │   └── img1/
│       │       ├── 000001.txt
│       │       ├── 000006.txt
│       │       └── ...
│       └── MOT20-02/
│           └── ...
├── data/
│   └── mot20_mini.train  # Training data paths
└── mot20_mini.json       # Configuration file
```

## Configuration

### Environment Setup

Set the data root environment variable:
```bash
export FAIRMOT_DATA_ROOT=/path/to/your/dataset/root
```

The mini dataset includes `mot20_mini.json` configuration file with clean relative paths:

```json
{
    "root": "MOT20_mini",
    "train": {
        "mot20_mini": "data/mot20_mini.train"
    },
    "test_emb": {
        "mot20_mini": "data/mot20_mini.train"
    },
    "test": {
        "mot20_mini": "data/mot20_mini.train"
    }
}
```

The training script automatically combines `FAIRMOT_DATA_ROOT` environment variable with these relative paths at runtime.

### Training Configuration

Use the mini dataset in training by specifying the config:

```bash
python train.py --dataset_config lib/cfg/mot20_mini.json
```

## Label Format

The generated labels use YOLO format with tracking IDs:
```
class_id track_id center_x center_y width height
```

Where:
- `class_id`: Always 0 (person class)
- `track_id`: Continuous track identifier
- `center_x, center_y`: Normalized center coordinates [0,1]
- `width, height`: Normalized dimensions [0,1]

Example label file content:
```
0 1 0.512345 0.387654 0.123456 0.234567
0 2 0.678901 0.456789 0.098765 0.187654
```

## Typical Sizes

With default settings:
- **Original MOT20**: ~4 sequences, ~3,000+ frames each
- **Mini dataset**: 2 sequences, ~100 frames each (every 5th frame)
- **Size reduction**: ~95% smaller
- **Processing time**: ~2-5 minutes vs hours

## Troubleshooting

### Common Issues

**Error: "Source dataset not found"**
- Ensure MOT20 dataset is available at the specified path
- Check the `--source` parameter

**Error: "MOT20Labels not found"**
- Ensure MOT20Labels directory exists alongside MOT20 images
- This is needed for ground truth annotations

**Error: "No sequences found"**
- Check that the source dataset has the expected structure
- Verify sequence names match (e.g., MOT20-01, MOT20-02)

**Error: "ModuleNotFoundError: No module named 'datasets'"**
- Ensure PYTHONPATH is set correctly: `PYTHONPATH=/path/to/FairMOT/src/lib`
- The training script requires the src/lib directory to be in Python path

**Error: "invalid literal for int() with base 10: '._000006'"**
- Hidden system files (starting with `._`) are present in the dataset
- Clean them with: `find dataset_path -name "._*" -delete`

**Error: "IsADirectoryError: [Errno 21] Is a directory"**
- The training data file contains directory paths instead of image paths
- Regenerate with: `find images/train -name "*.jpg" | sort > data/mot20_mini.train`

### Validation

Validate dataset structure:
```bash
python gen_labels_20_mini.py --validate
```

Check generated files:
```bash
ls -la /Users/yutao/Dataset/MOT/JDE/MOT20_mini/images/train/
ls -la /Users/yutao/Dataset/MOT/JDE/MOT20_mini/labels_with_ids/train/
```

## Integration with FairMOT

### Training

**Basic Training Test**
```bash
cd /path/to/FairMOT
FAIRMOT_DATA_ROOT=/path/to/your/dataset/root PYTHONPATH=/path/to/FairMOT/src/lib python scripts/train.py mot \
    --dataset jde \
    --data_cfg src/lib/cfg/mot20_mini.json \
    --batch_size 2 \
    --num_epochs 1 \
    --lr 1e-4 \
    --exp_id mot20_mini_test \
    --device cuda:0
```

**Extended Training**
```bash
FAIRMOT_DATA_ROOT=/path/to/your/dataset/root PYTHONPATH=/path/to/FairMOT/src/lib python scripts/train.py mot \
    --dataset jde \
    --data_cfg src/lib/cfg/mot20_mini.json \
    --batch_size 4 \
    --num_epochs 5 \
    --lr 1e-4 \
    --exp_id mot20_mini_extended \
    --device cuda:0 \
    --val_intervals 2
```

**Training Arguments**
- `mot`: Task type (Multi-Object Tracking)
- `--dataset jde`: Use JDE dataset format
- `--data_cfg`: Path to dataset configuration JSON
- `--batch_size`: Number of samples per batch (start with 2 for testing)
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--exp_id`: Experiment identifier for output organization
- `--device`: GPU device (cuda:0, cuda:1, or cpu)
- `--val_intervals`: Run validation every N epochs

### Tracking
```bash
# Test tracking on mini dataset
python track.py \
    --dataset_config lib/cfg/mot20_mini.json \
    --load_model ../models/your_model.pth
```

### Evaluation
```bash
# Quick evaluation
python track.py \
    --dataset_config lib/cfg/mot20_mini.json \
    --test_mot20_mini True
```

## Best Practices

1. **Development Cycle**: Use mini dataset for initial development and testing
2. **Parameter Tuning**: Test hyperparameters on mini dataset first
3. **Code Changes**: Validate modifications with mini dataset before full training
4. **CI/CD**: Use mini dataset in automated testing pipelines
5. **Documentation**: Update mini dataset when making significant data pipeline changes

## Advanced Usage

### Custom Sequence Selection
```bash
# Use specific sequences for targeted testing
python create_mot20_mini.py \
    --sequences MOT20-01 MOT20-05 \
    --frames 200 \
    --skip 2
```

### Multiple Mini Datasets
```bash
# Create different sized mini datasets
python create_mot20_mini.py --output MOT20_tiny --frames 20 --skip 10
python create_mot20_mini.py --output MOT20_small --frames 50 --skip 5
python create_mot20_mini.py --output MOT20_medium --frames 200 --skip 3
```

### Integration with Data Pipeline
```python
# In your training script
import os
if os.environ.get('USE_MINI_DATASET'):
    config_path = 'lib/cfg/mot20_mini.json'
else:
    config_path = 'lib/cfg/mot20.json'
```

This approach allows seamless switching between mini and full datasets during development.