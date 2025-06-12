#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from pathlib import Path


def get_data_dir():
    """
    Get FairMOT dataset root directory, compatible with ByteTrack data organization
    
    Priority:
    1. Environment variable DATA_ROOT
    2. Environment variable FAIRMOT_DATADIR (legacy compatibility)
    3. Environment variable YOLOX_DATADIR (ByteTrack compatibility)
    4. Default to datasets folder in project root directory
    
    Returns:
        str: Dataset root directory path
    """
    # First check general environment variable DATA_ROOT
    datadir = os.getenv("DATA_ROOT", None)
    
    # If not set, try FairMOT specific environment variable
    if datadir is None:
        datadir = os.getenv("FAIRMOT_DATADIR", None)
    
    # If not set, try ByteTrack environment variable
    if datadir is None:
        datadir = os.getenv("YOLOX_DATADIR", None)
    
    # Finally use default path
    if datadir is None:
        # Get project root directory (go up three levels from src/lib/utils/)
        project_root = Path(__file__).parent.parent.parent.parent
        datadir = os.path.join(project_root, "datasets")
    
    # Ensure directory exists
    os.makedirs(datadir, exist_ok=True)
    return datadir


def get_mot_datadir():
    """
    Get MOT dataset directory (MOT17 data is placed in datasets/mot/)
    
    Returns:
        str: MOT dataset directory path
    """
    return os.path.join(get_data_dir(), "mot")


def get_mot20_datadir():
    """
    Get MOT20 dataset directory
    
    Returns:
        str: MOT20 dataset directory path
    """
    return os.path.join(get_data_dir(), "MOT20")


def get_crowdhuman_datadir():
    """
    Get CrowdHuman dataset directory
    
    Returns:
        str: CrowdHuman dataset directory path
    """
    return os.path.join(get_data_dir(), "crowdhuman")


def get_cityscapes_datadir():
    """
    Get Cityscapes dataset directory
    
    Returns:
        str: Cityscapes dataset directory path
    """
    return os.path.join(get_data_dir(), "Cityscapes")


def get_ethz_datadir():
    """
    Get ETHZ dataset directory
    
    Returns:
        str: ETHZ dataset directory path
    """
    return os.path.join(get_data_dir(), "ETHZ")


def create_bytetrack_compatible_structure():
    """
    Create ByteTrack compatible dataset directory structure
    
    Directory structure:
    datasets/
    ├── mot/
    │   ├── train/
    │   └── test/
    ├── crowdhuman/
    │   ├── Crowdhuman_train/
    │   ├── Crowdhuman_val/
    │   ├── annotation_train.odgt
    │   └── annotation_val.odgt
    ├── MOT20/
    │   ├── train/
    │   └── test/
    ├── Cityscapes/
    │   ├── images/
    │   └── labels_with_ids/
    └── ETHZ/
        ├── eth01/
        ├── eth02/
        ├── ...
        └── eth07/
    """
    datadir = get_data_dir()
    
    # Create MOT17 directory structure (under mot folder)
    mot_dirs = [
        "mot/train",
        "mot/test"
    ]
    
    # Create MOT20 directory structure
    mot20_dirs = [
        "MOT20/train", 
        "MOT20/test"
    ]
    
    # Create CrowdHuman directory structure
    crowdhuman_dirs = [
        "crowdhuman/Crowdhuman_train",
        "crowdhuman/Crowdhuman_val"
    ]
    
    # Create Cityscapes directory structure
    cityscapes_dirs = [
        "Cityscapes/images",
        "Cityscapes/labels_with_ids"
    ]
    
    # Create ETHZ directory structure
    ethz_dirs = [f"ETHZ/eth{i:02d}" for i in range(1, 8)]
    
    all_dirs = mot_dirs + mot20_dirs + crowdhuman_dirs + cityscapes_dirs + ethz_dirs
    
    for dir_path in all_dirs:
        full_path = os.path.join(datadir, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    # Create CrowdHuman annotation file placeholders
    crowdhuman_annotations = [
        "crowdhuman/annotation_train.odgt",
        "crowdhuman/annotation_val.odgt"
    ]
    
    for annotation_file in crowdhuman_annotations:
        full_path = os.path.join(datadir, annotation_file)
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f:
                pass  # Create empty file
    
    print(f"ByteTrack compatible dataset directory structure created at: {datadir}")
    print("\nDirectory structure:")
    print("datasets/")
    print("├── mot/")
    print("│   ├── train/")
    print("│   └── test/")
    print("├── crowdhuman/")
    print("│   ├── Crowdhuman_train/")
    print("│   ├── Crowdhuman_val/")
    print("│   ├── annotation_train.odgt")
    print("│   └── annotation_val.odgt")
    print("├── MOT20/")
    print("│   ├── train/")
    print("│   └── test/")
    print("├── Cityscapes/")
    print("│   ├── images/")
    print("│   └── labels_with_ids/")
    print("└── ETHZ/")
    print("    ├── eth01/")
    print("    ├── ...") 
    print("    └── eth07/")
    
    return datadir


def get_conversion_scripts_info():
    """
    Return dataset conversion scripts information
    
    Returns:
        dict: Conversion scripts information
    """
    scripts_info = {
        "mot17": "tools/convert_mot17_to_coco.py",
        "mot20": "tools/convert_mot20_to_coco.py", 
        "crowdhuman": "tools/convert_crowdhuman_to_coco.py",
        "cityperson": "tools/convert_cityperson_to_coco.py",
        "ethz": "tools/convert_ethz_to_coco.py"
    }
    
    return scripts_info


def print_setup_instructions():
    """
    Print dataset setup instructions
    """
    print("\n" + "="*60)
    print("FairMOT Dataset Setup Instructions (ByteTrack Compatible)")
    print("="*60)
    print("\n1. Download datasets and place them in corresponding directories:")
    print("   - MOT17: datasets/mot/")
    print("   - MOT20: datasets/MOT20/")
    print("   - CrowdHuman: datasets/crowdhuman/")
    print("   - Cityscapes: datasets/Cityscapes/")
    print("   - ETHZ: datasets/ETHZ/")
    
    print("\n2. Convert datasets to COCO format:")
    scripts = get_conversion_scripts_info()
    for dataset, script in scripts.items():
        print(f"   python3 {script}")
    
    print("\n3. Environment variable setup (optional):")
    print("   export DATA_ROOT=/path/to/your/datasets")
    print("   export FAIRMOT_DATADIR=/path/to/your/datasets  # Legacy compatibility")
    print("   export YOLOX_DATADIR=/path/to/your/datasets  # ByteTrack compatibility")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Create directory structure
    create_bytetrack_compatible_structure()
    
    # Print setup instructions
    print_setup_instructions()