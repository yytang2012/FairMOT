# FairMOT Scripts Usage

This directory contains convenient shell scripts for running FairMOT demos and evaluations.

## Available Scripts

### 1. demo.sh - Video Tracking Demo
```bash
# Basic usage
./scripts/demo.sh [input_video] [output_dir]

# Examples
./scripts/demo.sh ./samples/MOT16-03.mp4
./scripts/demo.sh ./samples/MOT16-03.mp4 ./my_output
```

**Default parameters:**
- Model: `./models/fairmot_dla34.pth`
- Device: Auto (CUDA if available, CPU otherwise)
- Input size: 1088x608
- Confidence threshold: 0.4
- Output format: video (MP4)
- Quality: high

### 2. track.sh - Dataset Evaluation
```bash
# Basic usage
./scripts/track.sh [dataset] [split] [data_dir]

# Examples
./scripts/track.sh MOT16 val ./datasets
./scripts/track.sh MOT20 test ./datasets
```

**Default parameters:**
- Dataset: MOT16
- Split: val
- Data directory: ./datasets
- Model: `./models/fairmot_dla34.pth`
- Device: Auto
- Input size: 1088x608
- Confidence threshold: 0.4

### 3. detection_demo.sh - Detection Visualization
```bash
# Basic usage
./scripts/detection_demo.sh [weights_path] [data_path]

# Examples
./scripts/detection_demo.sh ./models/mot20_fairmot.pth ./datasets/MOT20/test/MOT20-04/img1
./scripts/detection_demo.sh
```

**Default parameters:**
- Weights: `./models/mot20_fairmot.pth`
- Data path: `./datasets/MOT20/test/MOT20-04/img1`
- Confidence threshold: 0.3
- Image size: 640


## Requirements

Make sure you have:
- Python 3.x
- Required dependencies installed
- Model weights in `./models/` directory
- Input data in appropriate directories

## Troubleshooting

1. **Permission denied**: Run `chmod +x scripts/*.sh`
2. **Module not found**: Make sure you're in the FairMOT root directory
3. **CUDA issues**: Scripts will automatically fallback to CPU if CUDA is unavailable