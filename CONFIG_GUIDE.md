# FairMOT Simple Configuration

Configure your dataset path easily without modifying code.

## Quick Setup

```bash
# Check current configuration
cd src
python -m lib.config check
```

## Set Dataset Path

### Option 1: Environment Variable (Recommended)
```bash
export FAIRMOT_DATA_ROOT="/path/to/your/datasets"
```

### Option 2: Edit Default Path
Edit `src/lib/config.py` and change the default path.

## Expected Dataset Structure

```
${FAIRMOT_DATA_ROOT}/
├── MOT17/
│   ├── images/train/
│   ├── images/test/
│   └── labels_with_ids/train/
├── MOT20/
│   ├── images/train/
│   ├── images/test/
│   └── labels_with_ids/train/
└── crowdhuman/
    ├── images/train/
    ├── images/val/
    └── labels_with_ids/
```

## Usage

All commands work automatically with configured paths:

```bash
cd src
python track.py mot --test_mot20 True --load_model ../models/fairmot_dla34.pth
python gen_labels_20.py
```

