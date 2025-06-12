#!/usr/bin/env python3
"""
Create a mini MOT20 dataset for quick testing and validation.
This script extracts a subset of sequences from the full MOT20 dataset
with configurable parameters for frame sampling.
"""

import os
import shutil
import numpy as np
import json
import argparse
from pathlib import Path

def create_mini_dataset(
    source_root=None,
    output_root=None,
    sequences=["MOT20-01", "MOT20-02"], 
    frame_limit=100,
    skip_frames=5
):
    """
    Create a mini MOT20 dataset by extracting subset of sequences and frames.
    
    Args:
        source_root (str): Path to the original MOT20 dataset
        output_root (str): Path where mini dataset will be created
        sequences (list): List of sequence names to include
        frame_limit (int): Maximum number of frames per sequence
        skip_frames (int): Frame sampling interval (take every Nth frame)
    """
    
    print(f"Creating MOT20 mini dataset...")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Sequences: {sequences}")
    print(f"Frame limit: {frame_limit}, Skip frames: {skip_frames}")
    
    # Create output directory structure
    os.makedirs(output_root, exist_ok=True)
    
    # Process training data only
    for phase in ["train"]:
        source_images = os.path.join(source_root, "images", phase)
        source_labels = os.path.join(source_root, "labels_with_ids", phase)
        
        output_images = os.path.join(output_root, "images", phase)
        output_labels = os.path.join(output_root, "labels_with_ids", phase)
        
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)
        
        for seq_name in sequences:
            if not os.path.exists(os.path.join(source_images, seq_name)):
                print(f"Warning: Sequence {seq_name} not found in {source_images}")
                continue
                
            print(f"Processing sequence: {seq_name}")
            
            # Create sequence directories
            seq_img_dir = os.path.join(output_images, seq_name, "img1")
            seq_label_dir = os.path.join(output_labels, seq_name, "img1")
            os.makedirs(seq_img_dir, exist_ok=True)
            os.makedirs(seq_label_dir, exist_ok=True)
            
            # Get list of source images
            source_img_dir = os.path.join(source_images, seq_name, "img1")
            if not os.path.exists(source_img_dir):
                print(f"Warning: Image directory not found: {source_img_dir}")
                continue
                
            img_files = sorted([f for f in os.listdir(source_img_dir) if f.endswith('.jpg')])
            
            # Select frames to copy based on sampling parameters
            selected_frames = img_files[::skip_frames][:frame_limit]
            
            print(f"  Selected {len(selected_frames)} frames from {len(img_files)} total frames")
            
            # Copy images and corresponding labels
            for img_file in selected_frames:
                # Copy image file
                src_img = os.path.join(source_img_dir, img_file)
                dst_img = os.path.join(seq_img_dir, img_file)
                shutil.copy2(src_img, dst_img)
                
                # Copy corresponding label file (if exists)
                label_file = img_file.replace('.jpg', '.txt')
                src_label = os.path.join(source_labels, seq_name, "img1", label_file)
                dst_label = os.path.join(seq_label_dir, label_file)
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    # Create empty label file if original doesn't exist
                    open(dst_label, 'a').close()
            
            # Copy sequence info file if it exists
            seqinfo_src = os.path.join(source_images, seq_name, "seqinfo.ini")
            if os.path.exists(seqinfo_src):
                seqinfo_dst = os.path.join(output_images, seq_name, "seqinfo.ini")
                shutil.copy2(seqinfo_src, seqinfo_dst)
                
                # Update frame count in seqinfo.ini
                update_seqinfo(seqinfo_dst, len(selected_frames))
    
    # Create data path files for training
    create_data_files(output_root, sequences)
    
    print(f"Mini dataset created successfully at: {output_root}")

def update_seqinfo(seqinfo_path, frame_count):
    """
    Update the frame count in seqinfo.ini file.
    
    Args:
        seqinfo_path (str): Path to seqinfo.ini file
        frame_count (int): New frame count to set
    """
    if not os.path.exists(seqinfo_path):
        return
        
    with open(seqinfo_path, 'r') as f:
        content = f.read()
    
    # Update seqLength parameter
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('seqLength='):
            lines[i] = f'seqLength={frame_count}'
            break
    
    with open(seqinfo_path, 'w') as f:
        f.write('\n'.join(lines))

def create_data_files(output_root, sequences):
    """
    Create training data path files for the mini dataset.
    
    Args:
        output_root (str): Root path of mini dataset
        sequences (list): List of sequence names
    """
    data_dir = os.path.join(output_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create training data path file
    train_file = os.path.join(data_dir, "mot20_mini.train")
    with open(train_file, 'w') as f:
        for seq_name in sequences:
            seq_path = os.path.join(output_root, "images", "train", seq_name, "img1")
            if os.path.exists(seq_path):
                f.write(f"{seq_path}\n")
    
    print(f"Created training data file: {train_file}")

def create_config_file(output_root):
    """
    Create JSON configuration file for the mini dataset.
    
    Args:
        output_root (str): Root path of mini dataset
        
    Returns:
        str: Path to created config file
    """
    config = {
        "root": output_root,
        "train": {
            "mot20_mini": "./data/mot20_mini.train"
        },
        "test_emb": {
            "mot20_mini": "./data/mot20_mini.train"
        },
        "test": {
            "mot20_mini": "./data/mot20_mini.train"
        }
    }
    
    config_path = os.path.join(output_root, "mot20_mini.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created config file: {config_path}")
    return config_path

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Create MOT20 mini dataset for quick testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create mini dataset with default settings
  python create_mot20_mini.py
  
  # Create with custom sequences and frame limits
  python create_mot20_mini.py --sequences MOT20-01 MOT20-03 --frames 50 --skip 3
  
  # Create with custom output path
  python create_mot20_mini.py --output /path/to/custom/output
        '''
    )
    
    parser.add_argument('--source', default=None,
                       help='Source MOT20 dataset path')
    parser.add_argument('--output', default=None,
                       help='Output mini dataset path')
    parser.add_argument('--sequences', nargs='+', default=["MOT20-01", "MOT20-02"],
                       help='Sequences to include (default: MOT20-01 MOT20-02)')
    parser.add_argument('--frames', type=int, default=100,
                       help='Maximum frames per sequence (default: 100)')
    parser.add_argument('--skip', type=int, default=5,
                       help='Skip frames interval - take every Nth frame (default: 5)')
    
    args = parser.parse_args()
    
    # Get data root from environment variable or command line
    data_root = os.getenv('FAIRMOT_DATA_ROOT')
    
    # Set default paths if not provided
    if args.source is None:
        if data_root:
            args.source = os.path.join(data_root, 'MOT20')
        else:
            print("Error: No source path specified.")
            print("Please either:")
            print("  1. Use --source to specify the MOT20 dataset path")
            print("  2. Set FAIRMOT_DATA_ROOT environment variable")
            return 1
            
    if args.output is None:
        if data_root:
            args.output = os.path.join(data_root, 'MOT20_mini')
        else:
            print("Error: No output path specified.")
            print("Please either:")
            print("  1. Use --output to specify the output path")
            print("  2. Set FAIRMOT_DATA_ROOT environment variable")
            return 1
    
    # Validate source dataset exists
    if not os.path.exists(args.source):
        print(f"Error: Source dataset not found: {args.source}")
        print("Please ensure the MOT20 dataset is available at the specified path.")
        return 1
    
    # Create mini dataset
    try:
        create_mini_dataset(
            source_root=args.source,
            output_root=args.output,
            sequences=args.sequences,
            frame_limit=args.frames,
            skip_frames=args.skip
        )
        
        # Create configuration file
        config_path = create_config_file(args.output)
        
        print("\n" + "="*60)
        print("Mini dataset creation completed successfully!")
        print("="*60)
        print(f"Dataset location: {args.output}")
        print(f"Config file: {config_path}")
        print("\nTo use this mini dataset:")
        print(f"1. Set environment variable: export MOT20_ROOT={args.output}")
        print("2. Or update the config file path in your training script")
        print("3. Use gen_labels_20_mini.py to generate training labels")
        
        return 0
        
    except Exception as e:
        print(f"Error creating mini dataset: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())