# Sample Videos

This directory contains sample videos for testing FairMOT tracking.

## Current Samples

- `MOT16-03.mp4` - Sample video from MOT16 dataset for testing tracking algorithms

## Usage

Use these sample videos with the demo script:

```bash
# Quick preview (10 seconds)
python3 scripts/demo.py --input_video samples/MOT16-03.mp4 --preview_mode

# Full processing
python3 scripts/demo.py --input_video samples/MOT16-03.mp4 --output_format video
```

## Adding Your Own Videos

You can add your own video files to this directory or use absolute paths:

```bash
# Using your own video
python3 scripts/demo.py --input_video /path/to/your/video.mp4 --preview_mode
```

## Supported Formats

- MP4 (recommended)
- AVI
- MOV
- Other formats supported by OpenCV