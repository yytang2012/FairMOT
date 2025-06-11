# FairMOT Tools

This directory contains various utilities and tools for FairMOT development and data processing.

## Directory Structure

```
tools/
├── data_preparation/       # Dataset preparation scripts
│   ├── create_mot20_mini.py    # Create mini MOT20 dataset
│   ├── gen_labels_20_mini.py   # Generate labels for mini dataset
│   ├── gen_labels_*.py         # Label generation for various datasets
│   └── gen_data_path.py        # Generate data path files
├── demo/                   # Demo and inference scripts
│   ├── demo.py                 # General demo script
│   └── detection_demo.py       # Detection-specific demo
├── testing/                # Testing utilities
│   ├── test_det.py            # Detection testing
│   └── test_emb.py            # Embedding testing
└── README.md              # This file
```

## Quick Start

### Data Preparation

Create a mini MOT20 dataset for quick testing:
```bash
cd data_preparation
python create_mot20_mini.py
python gen_labels_20_mini.py
```

### Running Demos

```bash
cd demo
python demo.py --input_video /path/to/video.mp4
```

### Testing

```bash
cd testing
python test_det.py --model_path /path/to/model.pth
```

## Tool Categories

### Data Preparation Tools
- **create_mot20_mini.py**: Create subset of MOT20 for quick testing
- **gen_labels_*.py**: Generate training labels for various datasets
- **gen_data_path.py**: Create data path configuration files

### Demo Tools
- **demo.py**: General tracking demonstration
- **detection_demo.py**: Object detection visualization

### Testing Tools
- **test_det.py**: Test detection performance
- **test_emb.py**: Test embedding/ReID performance

## Migration from Old Structure

The tools have been reorganized from the old flat structure in `src/` to a more organized hierarchy.

### Script Locations
- Data preparation: `tools/data_preparation/`
- Demo scripts: `tools/demo/`
- Testing utilities: `tools/testing/`

## Adding New Tools

When adding new tools:

1. Choose appropriate subdirectory based on tool purpose
2. Include comprehensive docstrings and help text
3. Add command-line argument parsing with `argparse`
4. Follow existing naming conventions
5. Update this README and relevant documentation

## Dependencies

Most tools rely on the main FairMOT dependencies. Specific tools may have additional requirements documented in their individual files.

For more detailed information about specific tools, see their individual documentation or use the `--help` flag.