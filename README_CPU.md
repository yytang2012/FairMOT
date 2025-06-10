# FairMOT CPU Setup Guide

Simple guide for running FairMOT on CPU for macOS.

## Quick Setup

### 1. Install Dependencies
```bash
# Create conda environment
conda create -n FairMOT python=3.8
conda activate FairMOT

# Install PyTorch for CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install requirements
pip install cython
pip install -r requirements.txt

# Install ffmpeg
brew install ffmpeg
```

### 2. Setup DCNv2
```bash
# Clone and build DCNv2 for CPU
https://github.com/yytang2012/DCNv2.git 
cd DCNv2
export FORCE_CPU=1
python setup.py build_ext --inplace
cd ..
```

### 3. Download Model
```bash
# Create models directory
mkdir -p models

# Download the model file from Google Drive
# Link: https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view
# Save as models/fairmot_dla34.pth
```

## Running Examples

### Track MOT20 Test Set
```bash
cd src
python track.py mot --test_mot20 True --load_model ../models/fairmot_dla34.pth --device cpu --conf_thres 0.6
```

### Track MOT20 Validation Set
```bash
cd src
python track.py mot --val_mot20 True --load_model ../models/fairmot_dla34.pth --device cpu --conf_thres 0.6
```

### Process Your Own Video
```bash
cd src
python demo.py mot --load_model ../models/fairmot_dla34.pth --device cpu --conf_thres 0.4 --input-video your_video.mp4
```

### Test Detection Only
```bash
cd src
python test_det.py mot --load_model ../models/fairmot_dla34.pth --device cpu
```

## Device Options

The system automatically detects if CUDA is available:

```bash
# Automatic detection (recommended)
python track.py mot --test_mot20 True --load_model ../models/fairmot_dla34.pth

# Force CPU usage
python track.py mot --test_mot20 True --load_model ../models/fairmot_dla34.pth --device cpu

# Try CUDA (will fallback to CPU if unavailable)
python track.py mot --test_mot20 True --load_model ../models/fairmot_dla34.pth --device cuda
```

## Common Issues

### DCNv2 Build Error
```bash
export FORCE_CPU=1
export MACOSX_DEPLOYMENT_TARGET=10.15
cd DCNv2
python setup.py clean
python setup.py build_ext --inplace
```

### Missing Dependencies
```bash
pip install opencv-python pillow matplotlib
brew install ffmpeg
```

### Dataset Path Issues
Make sure your MOT20 dataset is in the correct location or update the path in `src/lib/opts.py`:
```python
self.parser.add_argument('--data_dir', type=str, default='/path/to/your/MOT20/dataset')
```

That's it! The system will automatically use CPU and provide clear feedback about device selection.