# FairMOT Dataset Setup Guide (ByteTrack Compatible)

## Overview

FairMOT project now fully supports ByteTrack's dataset organization, using the same directory structure and conversion scripts.

## Directory Structure

Following ByteTrack's data organization approach:

```
datasets/
├── mot/                    # MOT17 dataset
│   ├── train/
│   └── test/
├── crowdhuman/             # CrowdHuman dataset
│   ├── Crowdhuman_train/
│   ├── Crowdhuman_val/
│   ├── annotation_train.odgt
│   └── annotation_val.odgt
├── MOT20/                  # MOT20 dataset
│   ├── train/
│   └── test/
├── Cityscapes/             # Cityscapes dataset
│   ├── images/
│   └── labels_with_ids/
└── ETHZ/                   # ETHZ dataset
    ├── eth01/
    ├── eth02/
    ├── ...
    └── eth07/
```

## Quick Start

### 1. Create Directory Structure

```bash
cd /path/to/FairMOT
python src/lib/utils/data_utils.py
```

### 2. Download and Place Datasets

- **MOT17**: Download and place in `datasets/mot/`
- **MOT20**: Download and place in `datasets/MOT20/`  
- **CrowdHuman**: Download and place in `datasets/crowdhuman/`
- **Cityscapes**: Download and place in `datasets/Cityscapes/`
- **ETHZ**: Download and place in `datasets/ETHZ/`

### 3. Convert Datasets to COCO Format

#### Convert all datasets
```bash
python tools/convert_datasets_to_coco.py --dataset all
```

#### Convert specific datasets
```bash
# Convert MOT17
python tools/convert_datasets_to_coco.py --dataset mot17

# Convert MOT20
python tools/convert_datasets_to_coco.py --dataset mot20
```

#### Or run conversion scripts individually
```bash
python tools/convert_mot17_to_coco.py
python tools/convert_mot20_to_coco.py
```

## Environment Variable Configuration

### Optional: Custom Dataset Root Directory

```bash
# Recommended: Use general environment variable
export DATA_ROOT="/path/to/your/datasets"

# Legacy FairMOT compatibility
export FAIRMOT_DATADIR="/path/to/your/datasets"

# ByteTrack compatibility
export YOLOX_DATADIR="/path/to/your/datasets"
```

If no environment variable is set, the system will automatically use the `datasets/` folder in the project root directory.

## ByteTrack Compatibility

### Fully compatible features:

1. **Same directory structure**: Uses exactly the same data organization as ByteTrack
2. **Same conversion scripts**: Conversion logic remains consistent with ByteTrack
3. **Environment variable support**: Supports `YOLOX_DATADIR` environment variable
4. **COCO format output**: Generates annotation files in the same format

### Data Management API

```python
from src.lib.utils.data_utils import (
    get_data_dir,
    get_mot_datadir,
    get_mot20_datadir,
    get_crowdhuman_datadir,
    get_cityscapes_datadir,
    get_ethz_datadir
)

# Get dataset root directory
datadir = get_data_dir()

# Get specific dataset directories
mot17_dir = get_mot_datadir()        # datasets/mot/
mot20_dir = get_mot20_datadir()      # datasets/MOT20/
```

## Show Complete Setup Instructions

```bash
python tools/convert_datasets_to_coco.py --show-setup
```

## File Descriptions

- `src/lib/utils/data_utils.py` - Data directory management tools
- `tools/convert_mot17_to_coco.py` - MOT17 dataset conversion script
- `tools/convert_mot20_to_coco.py` - MOT20 dataset conversion script
- `tools/convert_datasets_to_coco.py` - Unified conversion script

## Dataset Download Links

- **MOT17**: [MOT Challenge](https://motchallenge.net/data/MOT17/)
- **MOT20**: [MOT Challenge](https://motchallenge.net/data/MOT20/)
- **CrowdHuman**: [CrowdHuman Dataset](https://www.crowdhuman.org/)
- **Cityscapes**: [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- **ETHZ**: [ETHZ Dataset](https://data.vision.ee.ethz.ch/cvl/aess/dataset/)

