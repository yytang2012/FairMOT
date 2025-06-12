#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import argparse

# Add src path for importing utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lib.utils.data_utils import get_data_dir, print_setup_instructions


def convert_all_datasets():
    """
    Convert all supported datasets to COCO format
    """
    print("Starting dataset conversion to COCO format...")
    
    # Import conversion scripts
    try:
        from convert_mot17_to_coco import convert_mot17_to_coco
        print("\n=== Converting MOT17 Dataset ===")
        convert_mot17_to_coco()
    except Exception as e:
        print(f"MOT17 conversion failed: {e}")
    
    try:
        from convert_mot20_to_coco import convert_mot20_to_coco
        print("\n=== Converting MOT20 Dataset ===")
        convert_mot20_to_coco()
    except Exception as e:
        print(f"MOT20 conversion failed: {e}")
    
    # TODO: Add other dataset conversions
    # try:
    #     from convert_crowdhuman_to_coco import convert_crowdhuman_to_coco
    #     print("\n=== Converting CrowdHuman Dataset ===")
    #     convert_crowdhuman_to_coco()
    # except Exception as e:
    #     print(f"CrowdHuman conversion failed: {e}")
    
    print("\nDataset conversion completed!")


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to COCO format')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'mot17', 'mot20', 'crowdhuman', 'cityperson', 'ethz'],
                        help='Select dataset to convert')
    parser.add_argument('--show-setup', action='store_true',
                        help='Show dataset setup instructions')
    
    args = parser.parse_args()
    
    if args.show_setup:
        print_setup_instructions()
        return
    
    print(f"Dataset root directory: {get_data_dir()}")
    
    if args.dataset == 'all':
        convert_all_datasets()
    elif args.dataset == 'mot17':
        from convert_mot17_to_coco import convert_mot17_to_coco
        convert_mot17_to_coco()
    elif args.dataset == 'mot20':
        from convert_mot20_to_coco import convert_mot20_to_coco
        convert_mot20_to_coco()
    else:
        print(f"Conversion script for dataset {args.dataset} is not yet implemented")


if __name__ == '__main__':
    main()