#!/usr/bin/env python3
"""
FairMOT Simple Configuration
Set FAIRMOT_DATA_ROOT environment variable or use default path
"""
import os
import sys

def get_data_root():
    """Get the data root directory from environment variable or use current dataset path"""
    return os.environ.get('FAIRMOT_DATA_ROOT', '/path/to/your/datasets')

def check_config():
    """Check current data root configuration"""
    data_root = get_data_root()
    print("=== FairMOT Configuration ===")
    print(f"Data Root: {data_root}")
    
    # Check if datasets exist
    datasets = ['MOT17', 'MOT20']
    found_datasets = []
    
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        if os.path.exists(dataset_path):
            found_datasets.append(dataset)
            print(f"✓ {dataset}: {dataset_path}")
        else:
            print(f"✗ {dataset}: {dataset_path} (not found)")
    
    if not found_datasets:
        print("\nNo datasets found!")
        print("To configure your data path:")
        print("export FAIRMOT_DATA_ROOT=/path/to/your/datasets")
    else:
        print(f"\nFound {len(found_datasets)} dataset(s). Ready to use!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_config()
    else:
        print("FairMOT Configuration")
        print("Usage: python -m lib.config check")
        print()
        check_config()