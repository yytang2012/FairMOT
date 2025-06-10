# FairMOT Device Selection Guide

FairMOT now supports more intuitive device selection with automatic CUDA detection and intelligent device selection.

## New Device Selection Options

### 1. Automatic Detection (Recommended)
```bash
# Automatically detect CUDA availability, use CUDA if available, otherwise use CPU
python track.py mot --device auto --load_model ../models/fairmot_dla34.pth --conf_thres 0.6

# Or omit the --device parameter (defaults to auto)
python track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```

### 2. Force CPU Usage
```bash
# Explicitly specify CPU usage
python track.py mot --device cpu --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```

### 3. Specify Specific GPU
```bash
# Use first GPU (cuda:0)
python track.py mot --device cuda:0 --load_model ../models/fairmot_dla34.pth --conf_thres 0.6

# Use second GPU (cuda:1)
python track.py mot --device cuda:1 --load_model ../models/fairmot_dla34.pth --conf_thres 0.6

# Use default CUDA device
python track.py mot --device cuda --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```

## Intelligent Fallback Mechanism

The system automatically detects hardware capabilities:

- **CUDA specified but unavailable**: Automatically fallback to CPU with warning message
- **Auto detection mode**: Prioritize CUDA, fallback to CPU if unavailable
- **Device not found**: Automatically fallback to auto detection mode

## Usage Examples

### macOS (No CUDA)
```bash
# All following commands will automatically use CPU
python track.py mot --device auto --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
python track.py mot --device cuda:0 --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6  # Will warn and fallback to CPU
python track.py mot --device cpu --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```

### Linux with GPU
```bash
# Automatically use GPU
python track.py mot --device auto --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6

# Specify specific GPU
python track.py mot --device cuda:1 --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6

# Force CPU usage (even with GPU available)
python track.py mot --device cpu --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```

## Backward Compatibility

The old `--gpus` parameter is still supported, but the new `--device` parameter is recommended:

```bash
# Old way (still valid, but not recommended)
python track.py mot --gpus -1 --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6

# New way (recommended)
python track.py mot --device cpu --test_mot20 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```

## System Feedback

The system provides clear device selection feedback:

- `AUTO: CUDA detected, using cuda:0` - CUDA automatically detected
- `AUTO: CUDA not available, using CPU` - Automatically fallback to CPU
- `WARNING: CUDA not available, falling back to CPU despite --device cuda:0` - User specified CUDA but unavailable

## Performance Reference

- **CPU (macOS)**: ~0.7 FPS
- **GPU (CUDA)**: ~25+ FPS (depends on GPU model)

It is recommended to use `--device auto` on systems with GPU for optimal performance.