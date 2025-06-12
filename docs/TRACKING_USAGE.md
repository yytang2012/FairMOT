# FairMOT Tracking Usage Guide

## Overview

This document explains how to use FairMOT for multi-object tracking, using MOT20 dataset as an example.

## Environment Setup

### Dataset Path Configuration

FairMOT supports multiple ways to configure dataset paths:

```bash
# Recommended: Use general environment variable
export DATA_ROOT=/path/to/your/datasets

# Legacy FairMOT compatibility
export FAIRMOT_DATA_ROOT=/path/to/your/datasets

# ByteTrack compatibility
export YOLOX_DATADIR=/path/to/your/datasets
```

If no environment variable is set, the system will automatically use the `datasets/` folder in the project root directory.

## Dataset Structure

Make sure your dataset is organized according to the following structure:

```
datasets/
├── MOT20/
│   ├── train/
│   │   ├── MOT20-01/
│   │   │   ├── img1/
│   │   │   ├── gt/
│   │   │   └── det/
│   │   ├── MOT20-02/
│   │   └── ...
│   └── test/
│       ├── MOT20-04/
│       ├── MOT20-06/
│       └── ...
└── MOT17/
    ├── train/
    └── test/
```

## Usage

### 1. MOT20 Validation Set Tracking

```bash
python scripts/track.py mot --val_mot20 1 --load_model ./models/mot20_fairmot.pth
```

### 2. MOT20 Test Set Tracking

```bash
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth
```

### 3. Using CPU Mode

```bash
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --gpus -1
```

### 4. Custom Confidence Threshold

```bash
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --conf_thres 0.4
```

### 5. Specify Data Path

```bash
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --data_dir /custom/path/to/datasets
```

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--load_model` | Pre-trained model path | - |
| `--conf_thres` | Detection confidence threshold | 0.4 |
| `--det_thres` | Detection threshold | 0.3 |
| `--nms_thres` | NMS threshold | 0.4 |
| `--track_buffer` | Tracking buffer size | 30 |
| `--min_box_area` | Minimum detection box area | 200 |
| `--gpus` | GPU device ID, -1 for CPU | 0 |
| `--data_dir` | Dataset root directory | Auto-detect |

## Dataset Options

### MOT20 Dataset

```bash
# Validation set (MOT20-01, MOT20-02, MOT20-03, MOT20-05)
python scripts/track.py mot --val_mot20 1 --load_model ./models/mot20_fairmot.pth

# Test set (MOT20-04, MOT20-06, MOT20-07, MOT20-08) 
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth

# MOT20 Mini dataset
python scripts/track.py mot --val_mot20_mini 1 --load_model ./models/mot20_fairmot.pth
```

### MOT17 Dataset

```bash
# Validation set
python scripts/track.py mot --val_mot17 1 --load_model ./models/fairmot_dla34.pth

# Test set
python scripts/track.py mot --test_mot17 1 --load_model ./models/fairmot_dla34.pth
```

### MOT16 Dataset

```bash
# Validation set
python scripts/track.py mot --val_mot16 1 --load_model ./models/fairmot_dla34.pth

# Test set
python scripts/track.py mot --test_mot16 1 --load_model ./models/fairmot_dla34.pth
```

### MOT15 Dataset

```bash
# Validation set
python scripts/track.py mot --val_mot15 1 --load_model ./models/fairmot_dla34.pth

# Test set
python scripts/track.py mot --test_mot15 1 --load_model ./models/fairmot_dla34.pth
```

## Result Output

Tracking results will be saved in the `results/` directory:

```
results/
└── MOT17_test_public_dla34/
    ├── MOT20-04.txt
    ├── MOT20-06.txt
    ├── MOT20-07.txt
    ├── MOT20-08.txt
    └── summary_MOT17_test_public_dla34.xlsx
```

Each `.txt` file contains tracking results in the format:
```
<frame_id>,<track_id>,<x>,<y>,<width>,<height>,1,-1,-1,-1
```

## Performance Optimization

### Using GPU Acceleration

```bash
# Use single GPU
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --gpus 0

# Multi-GPU (not supported for inference yet)
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --gpus 0
```

### Adjust Tracking Parameters

```bash
# Higher detection confidence to reduce false positives
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --conf_thres 0.6

# Longer tracking buffer for fast-moving targets
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --track_buffer 50

# Larger minimum detection box area to filter small targets
python scripts/track.py mot --test_mot20 1 --load_model ./models/mot20_fairmot.pth --min_box_area 500
```

## Evaluation Metrics

The script automatically calculates the following metrics:

- **MOTA** (Multiple Object Tracking Accuracy)
- **MOTP** (Multiple Object Tracking Precision)  
- **IDF1** (ID F1 Score)
- **IDs** (Identity Switches)
- **Frag** (Fragmentations)
- **FP** (False Positives)
- **FN** (False Negatives)

Evaluation results will be displayed in the terminal and saved as an Excel file.

## Common Issues

### 1. Data Path Error

```
AssertionError: No images found in /path/to/your/datasets/MOT20/test/MOT20-04/img1
```

**Solution**:
- Check if the dataset path is correct
- Set the correct environment variable
- Use the `--data_dir` parameter to specify the path

### 2. Out of Memory

**Solution**:
- Use CPU mode: `--gpus -1`
- Reduce batch size (if supported)
- Increase system memory

### 3. CUDA Version Incompatibility

**Solution**:
- Use CPU mode: `--gpus -1`
- Install PyTorch for the corresponding CUDA version

## Model Download

Make sure to download the corresponding pre-trained models:

- **MOT20**: `mot20_fairmot.pth`
- **MOT17**: `fairmot_dla34.pth` 
- **General**: `fairmot_dla34.pth`

Place the model files in the `models/` directory.

## Complete Example

Here's a complete MOT20 tracking example:

```bash
# 1. Set data path
export DATA_ROOT=/path/to/your/datasets

# 2. Run MOT20 test set tracking
python scripts/track.py mot \
    --test_mot20 1 \
    --load_model ./models/mot20_fairmot.pth \
    --conf_thres 0.4 \
    --gpus 0

# 3. View results
ls results/MOT17_test_public_dla34/
```

This will process all sequences in the MOT20 test set and generate tracking results and evaluation reports in the `results/` directory.