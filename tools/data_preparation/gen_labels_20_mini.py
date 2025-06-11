#!/usr/bin/env python3
"""
Generate training labels for MOT20 mini dataset.
This script processes ground truth annotations and creates YOLO-format labels
that correspond to the frames in the mini dataset.
"""

import os.path as osp
import os
import numpy as np
import argparse
import sys

def mkdirs(d):
    """Create directory if it doesn't exist."""
    if not osp.exists(d):
        os.makedirs(d)

def generate_labels_for_mini_dataset(dataset_root="/Users/yutao/Dataset/MOT/JDE/MOT20_mini"):
    """
    Generate YOLO-format labels for MOT20 mini dataset.
    
    This function reads ground truth annotations from the original MOT20Labels
    and creates corresponding label files for frames that exist in the mini dataset.
    
    Args:
        dataset_root (str): Path to the MOT20 mini dataset
    """
    
    # Use MOT20Labels as the source for ground truth data
    mot20_labels_root = "/Users/yutao/Dataset/MOT/JDE/MOT20Labels"
    
    seq_root = osp.join(dataset_root, "images", "train")
    label_root = osp.join(dataset_root, "labels_with_ids", "train")
    
    if not osp.exists(seq_root):
        print(f"Error: Sequence root not found: {seq_root}")
        print("Please run create_mot20_mini.py first to create the mini dataset")
        return False
    
    if not osp.exists(mot20_labels_root):
        print(f"Error: MOT20Labels not found: {mot20_labels_root}")
        print("Please ensure MOT20Labels is available for ground truth data")
        return False
    
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root) if osp.isdir(osp.join(seq_root, s))]
    
    print(f"Found sequences: {seqs}")
    print(f"Ground truth source: {mot20_labels_root}")
    
    tid_curr = 0
    tid_last = -1
    total_annotations = 0
    
    for seq in seqs:
        print(f"Processing sequence: {seq}")
        
        # Read sequence information for image dimensions
        seqinfo_path = osp.join(seq_root, seq, 'seqinfo.ini')
        if not osp.exists(seqinfo_path):
            print(f"Warning: seqinfo.ini not found for {seq}")
            continue
            
        seq_info = open(seqinfo_path).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        
        print(f"  Image dimensions: {seq_width}x{seq_height}")
        
        # Load ground truth annotations from MOT20Labels
        gt_txt = osp.join(mot20_labels_root, "train", seq, 'gt', 'gt.txt')
        if not osp.exists(gt_txt):
            print(f"Warning: Ground truth file not found: {gt_txt}")
            continue
            
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        
        # Create label directory for this sequence
        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)
        
        # Get list of images that actually exist in mini dataset
        img_dir = osp.join(seq_root, seq, 'img1')
        existing_images = set()
        if osp.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.endswith('.jpg'):
                    frame_id = int(img_file.replace('.jpg', ''))
                    existing_images.add(frame_id)
        
        print(f"  Found {len(existing_images)} images in mini dataset")
        
        # Process ground truth annotations
        seq_annotations = 0
        for fid, tid, x, y, w, h, mark, label, _ in gt:
            # Filter out invalid annotations
            # mark=0 means the object is ignored
            # label!=1 means it's not a pedestrian
            if mark == 0 or not label == 1:
                continue
                
            fid = int(fid)
            # Only process frames that exist in mini dataset
            if fid not in existing_images:
                continue
                
            tid = int(tid)
            # Assign continuous track IDs
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
                
            # Convert bounding box to center coordinates
            x += w / 2
            y += h / 2
            
            # Create label file path
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            
            # Format: class_id track_id center_x center_y width height (normalized)
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            
            # Append to label file (multiple objects per frame possible)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            
            seq_annotations += 1
            
        print(f"  Generated {seq_annotations} annotations")
        total_annotations += seq_annotations
    
    print(f"\nLabel generation completed!")
    print(f"Total annotations: {total_annotations}")
    return True

def validate_mini_dataset(dataset_root):
    """
    Validate that the mini dataset has the expected structure.
    
    Args:
        dataset_root (str): Path to mini dataset
        
    Returns:
        bool: True if dataset structure is valid
    """
    required_dirs = [
        osp.join(dataset_root, "images", "train"),
        osp.join(dataset_root, "labels_with_ids", "train")
    ]
    
    for dir_path in required_dirs:
        if not osp.exists(dir_path):
            print(f"Missing required directory: {dir_path}")
            return False
    
    # Check if there are any sequences
    train_dir = osp.join(dataset_root, "images", "train")
    sequences = [s for s in os.listdir(train_dir) if osp.isdir(osp.join(train_dir, s))]
    
    if not sequences:
        print("No sequences found in training directory")
        return False
    
    print(f"Dataset validation passed. Found sequences: {sequences}")
    return True

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate training labels for MOT20 mini dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script generates YOLO-format training labels for the MOT20 mini dataset.
It reads ground truth annotations from MOT20Labels and creates corresponding
label files for frames that exist in the mini dataset.

The generated labels have the format:
class_id track_id center_x center_y width height

Where coordinates are normalized to [0,1] range.

Examples:
  # Generate labels with default mini dataset path
  python gen_labels_20_mini.py
  
  # Generate labels for custom mini dataset location
  python gen_labels_20_mini.py --dataset_root /path/to/custom/mini/dataset
        '''
    )
    
    parser.add_argument('--dataset_root', default="/Users/yutao/Dataset/MOT/JDE/MOT20_mini",
                       help='MOT20 mini dataset root path (default: /Users/yutao/Dataset/MOT/JDE/MOT20_mini)')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate dataset structure without generating labels')
    
    args = parser.parse_args()
    
    # Validate dataset exists and has correct structure
    if not osp.exists(args.dataset_root):
        print(f"Error: Dataset root not found: {args.dataset_root}")
        print("Please run create_mot20_mini.py first to create the mini dataset")
        return 1
    
    if not validate_mini_dataset(args.dataset_root):
        print("Dataset validation failed")
        return 1
    
    if args.validate:
        print("Dataset validation completed successfully")
        return 0
    
    # Generate labels
    try:
        success = generate_labels_for_mini_dataset(args.dataset_root)
        if success:
            print(f"\nLabels generated successfully at: {osp.join(args.dataset_root, 'labels_with_ids')}")
            return 0
        else:
            print("Label generation failed")
            return 1
            
    except Exception as e:
        print(f"Error generating labels: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())