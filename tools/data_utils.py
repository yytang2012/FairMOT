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
    
    # Only return the path, don't create directory
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


def scan_dataset_structure():
    """
    Scan data root directory and check for valid dataset structure
    
    Returns:
        dict: Dataset statistics and validation information
    """
    datadir = get_data_dir()
    
    # Define expected dataset structure
    expected_structure = {
        "mot": {
            "dirs": ["mot/train", "mot/test"],
            "files": []
        },
        "MOT20": {
            "dirs": ["MOT20/train", "MOT20/test"],
            "files": []
        },
        "crowdhuman": {
            "dirs": ["crowdhuman/Crowdhuman_train", "crowdhuman/Crowdhuman_val"],
            "files": ["crowdhuman/annotation_train.odgt", "crowdhuman/annotation_val.odgt"]
        },
        "Cityscapes": {
            "dirs": ["Cityscapes/images", "Cityscapes/labels_with_ids"],
            "files": []
        },
        "ETHZ": {
            "dirs": [f"ETHZ/eth{i:02d}" for i in range(1, 8)],
            "files": []
        }
    }
    
    stats = {
        "data_root": datadir,
        "data_root_exists": os.path.exists(datadir),
        "datasets": {}
    }
    
    if not stats["data_root_exists"]:
        return stats
    
    # Check each dataset
    for dataset_name, structure in expected_structure.items():
        dataset_stats = {
            "available": False,
            "directories": {},
            "files": {},
            "total_size": 0,
            "file_count": 0
        }
        
        # Check directories
        all_dirs_exist = True
        for dir_path in structure["dirs"]:
            full_path = os.path.join(datadir, dir_path)
            exists = os.path.exists(full_path) and os.path.isdir(full_path)
            dataset_stats["directories"][dir_path] = {
                "exists": exists,
                "path": full_path
            }
            
            if exists:
                # Count files and calculate size
                try:
                    for root, dirs, files in os.walk(full_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                file_size = os.path.getsize(file_path)
                                dataset_stats["total_size"] += file_size
                                dataset_stats["file_count"] += 1
                            except OSError:
                                pass
                except OSError:
                    pass
            else:
                all_dirs_exist = False
        
        # Check files
        for file_path in structure["files"]:
            full_path = os.path.join(datadir, file_path)
            exists = os.path.exists(full_path) and os.path.isfile(full_path)
            dataset_stats["files"][file_path] = {
                "exists": exists,
                "path": full_path
            }
            
            if exists:
                try:
                    file_size = os.path.getsize(full_path)
                    dataset_stats["total_size"] += file_size
                    dataset_stats["file_count"] += 1
                except OSError:
                    pass
            else:
                all_dirs_exist = False
        
        # Dataset is available if all expected dirs/files exist
        dataset_stats["available"] = all_dirs_exist
        stats["datasets"][dataset_name] = dataset_stats
    
    return stats


def print_dataset_statistics(stats):
    """
    Print formatted dataset statistics
    
    Args:
        stats (dict): Dataset statistics from scan_dataset_structure()
    """
    def format_size(size_bytes):
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.1f} {units[unit_index]}"
    
    print("\n" + "="*70)
    print("Dataset Structure Scan Results")
    print("="*70)
    
    print(f"\nData Root: {stats['data_root']}")
    print(f"Data Root Exists: {'✓' if stats['data_root_exists'] else '✗'}")
    
    if not stats['data_root_exists']:
        print("\nData root directory does not exist. No datasets found.")
        return
    
    print(f"\nDataset Summary:")
    print("-" * 50)
    
    total_datasets = len(stats['datasets'])
    available_datasets = sum(1 for ds in stats['datasets'].values() if ds['available'])
    
    print(f"Total datasets checked: {total_datasets}")
    print(f"Available datasets: {available_datasets}")
    print(f"Missing datasets: {total_datasets - available_datasets}")
    
    print(f"\nDetailed Dataset Information:")
    print("-" * 50)
    
    for dataset_name, dataset_info in stats['datasets'].items():
        status = "✓ Available" if dataset_info['available'] else "✗ Missing"
        size = format_size(dataset_info['total_size'])
        file_count = dataset_info['file_count']
        
        print(f"\n{dataset_name}: {status}")
        print(f"  Files: {file_count:,}")
        print(f"  Size: {size}")
        
        # Show directory status
        if dataset_info['directories']:
            print("  Directories:")
            for dir_path, dir_info in dataset_info['directories'].items():
                dir_status = "✓" if dir_info['exists'] else "✗"
                print(f"    {dir_status} {dir_path}")
        
        # Show file status
        if dataset_info['files']:
            print("  Files:")
            for file_path, file_info in dataset_info['files'].items():
                file_status = "✓" if file_info['exists'] else "✗"
                print(f"    {file_status} {file_path}")
    
    print("\n" + "="*70)


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
    # Scan dataset structure and print statistics
    stats = scan_dataset_structure()
    print_dataset_statistics(stats)
    
    # Print setup instructions
    print_setup_instructions()