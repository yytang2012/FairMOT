#!/usr/bin/env python3
"""
Argument utilities for FairMOT scripts - independent from global opts
Provides reusable argument groups for demo, track, train and other scripts
"""

import argparse
import os
import sys
import torch
from datetime import datetime


def add_device_args(parser):
    """Add device-related arguments that are commonly used"""
    parser.add_argument('--device', default='auto', 
                       help='Device to use: "cpu", "cuda", "auto", "cuda:0", etc.')
    parser.add_argument('--gpus', default='0',
                       help='GPU ids to use (comma separated)')


def add_model_args(parser):
    """Add model-related arguments for inference"""
    parser.add_argument('--load_model', default='',
                       help='Path to pretrained model')
    parser.add_argument('--arch', default='dla_34',
                       help='Model architecture (dla_34, hrnet_18, etc.)')
    parser.add_argument('--head_conv', type=int, default=-1,
                       help='Conv layer channels for output head')
    parser.add_argument('--down_ratio', type=int, default=4,
                       help='Output stride')


def add_input_args(parser):
    """Add input processing arguments"""
    parser.add_argument('--input_h', type=int, default=608,
                       help='Input height')
    parser.add_argument('--input_w', type=int, default=1088,
                       help='Input width')
    parser.add_argument('--input_res', type=int, default=-1,
                       help='Input resolution (overrides input_h/input_w if set)')


def add_tracking_args(parser):
    """Add tracking-specific arguments"""
    parser.add_argument('--conf_thres', type=float, default=0.4,
                       help='Confidence threshold for tracking')
    parser.add_argument('--det_thres', type=float, default=0.3,
                       help='Confidence threshold for detection')
    parser.add_argument('--nms_thres', type=float, default=0.4,
                       help='IoU threshold for NMS')
    parser.add_argument('--track_buffer', type=int, default=30,
                       help='Tracking buffer size')
    parser.add_argument('--min_box_area', type=float, default=100,
                       help='Filter out tiny boxes')


def add_dataset_args(parser):
    """Add dataset-related arguments"""
    parser.add_argument('--data_dir', type=str, default='',
                       help='Dataset root directory')
    parser.add_argument('--dataset', type=str, default='MOT16',
                       choices=['MOT15', 'MOT16', 'MOT17', 'MOT20', 'MOT20_mini'],
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test', 'train_half', 'val_half'],
                       help='Dataset split: train/val/test (traditional), train_half/val_half (ByteTrack style)')
    parser.add_argument('--seqs', type=str, default='',
                       help='Comma-separated sequence names (overrides default sequences)')


def add_demo_args(parser):
    """Add demo-specific arguments"""
    parser.add_argument('--input_video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='',
                       help='Output directory (default: auto-generated in outputs/)')
    parser.add_argument('--output_format', type=str, default='video',
                       choices=['video', 'frames', 'text'],
                       help='Output format: video (mp4), frames (images), or text (MOT format)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual tracking frames')
    parser.add_argument('--show_fps', action='store_true',
                       help='Display FPS in output video')
    parser.add_argument('--quality', type=str, default='high',
                       choices=['low', 'medium', 'high'],
                       help='Output video quality')
    
    # Processing limits for faster preview/testing
    parser.add_argument('--max_duration', type=float, default=0,
                       help='Maximum duration to process in seconds (0 = process entire video)')
    parser.add_argument('--max_frames', type=int, default=0,
                       help='Maximum number of frames to process (0 = process all frames)')
    parser.add_argument('--start_time', type=float, default=0,
                       help='Start time in seconds (skip initial frames)')
    parser.add_argument('--preview_mode', action='store_true',
                       help='Quick preview mode: process only first 10 seconds')


def add_track_args(parser):
    """Add tracking evaluation specific arguments"""
    parser.add_argument('--exp_id', type=str, default='default',
                       help='Experiment ID')
    parser.add_argument('--output_root', type=str, default='',
                       help='Output root directory for results')
    parser.add_argument('--save_videos', action='store_true',
                       help='Save output videos of tracking results')
    parser.add_argument('--save_images', action='store_true',
                       help='Save individual frame images')


def create_demo_parser():
    """Create argument parser for demo script"""
    parser = argparse.ArgumentParser(
        description='FairMOT Demo - Multi-Object Tracking on Video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add argument groups
    add_demo_args(parser)
    add_tracking_args(parser)
    add_model_args(parser)
    add_input_args(parser)
    add_device_args(parser)
    
    return parser


def create_track_parser():
    """Create argument parser for track evaluation script"""
    parser = argparse.ArgumentParser(
        description='FairMOT Tracking Evaluation - Evaluate on MOT datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add argument groups
    add_track_args(parser)
    add_dataset_args(parser)
    add_tracking_args(parser)
    add_model_args(parser)
    add_input_args(parser)
    add_device_args(parser)
    
    return parser


def process_common_args(args):
    """Process common arguments shared across scripts"""
    # Handle device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = f'cuda:{torch.cuda.current_device()}'
            args.use_cuda = True
            print(f"AUTO: CUDA detected, using {args.device}")
        else:
            args.device = 'cpu'
            args.use_cuda = False
            print("AUTO: CUDA not available, using CPU")
    elif args.device == 'cpu':
        args.use_cuda = False
    elif args.device.startswith('cuda'):
        if torch.cuda.is_available():
            args.use_cuda = True
        else:
            print(f"WARNING: CUDA not available, falling back to CPU")
            args.device = 'cpu'
            args.use_cuda = False
    
    # Parse GPU IDs
    if args.use_cuda:
        args.gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
        args.gpus_str = args.gpus
    else:
        args.gpus = [-1]
        args.gpus_str = '-1'
    
    # Set up input resolution
    if args.input_res > 0:
        args.input_h = args.input_res
        args.input_w = args.input_res
    
    args.img_size = (args.input_w, args.input_h)
    args.output_h = args.input_h // args.down_ratio
    args.output_w = args.input_w // args.down_ratio
    
    # Set up model parameters
    if args.head_conv == -1:
        args.head_conv = 256 if 'dla' in args.arch else 256
    
    # Set up model heads for MOT
    args.heads = {
        'hm': 1,  # num_classes
        'wh': 4,  # bbox regression
        'id': 128,  # reid_dim
        'reg': 2  # offset regression
    }
    
    # Additional MOT-specific parameters
    args.num_classes = 1
    args.reid_dim = 128
    args.nID = 14455
    args.mean = [0.408, 0.447, 0.470]
    args.std = [0.289, 0.274, 0.278]
    args.ltrb = True
    args.reg_offset = True
    args.vis_thresh = 0.5
    args.K = 500  # max_per_image parameter for tracking
    
    # Set up data directory (try to get from config if not provided)
    if hasattr(args, 'data_dir') and not args.data_dir:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))
            from config import get_data_root
            args.data_dir = get_data_root()
        except:
            # Fallback to default
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            args.data_dir = os.path.join(script_dir, 'datasets')
    
    return args


def process_demo_args(args):
    """Process and validate demo arguments"""
    args = process_common_args(args)
    
    # Set default model if none specified
    if not args.load_model:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model = os.path.join(script_dir, 'models', 'fairmot_dla34.pth')
        if os.path.exists(default_model):
            args.load_model = default_model
            print(f"Using default model: {args.load_model}")
        else:
            print("Warning: No model specified and default model not found")
            print("Please specify a model with --load_model or ensure models/fairmot_dla34.pth exists")
    
    # Handle preview mode
    if args.preview_mode:
        args.max_duration = 3.0  # 10 seconds preview
        print("Preview mode enabled: processing first 10 seconds only")
    
    # Validate processing limits
    if args.max_duration < 0:
        raise ValueError("max_duration must be non-negative")
    if args.max_frames < 0:
        raise ValueError("max_frames must be non-negative") 
    if args.start_time < 0:
        raise ValueError("start_time must be non-negative")
    
    # Demo-specific processing
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(args.input_video))[0]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_root = os.path.join(os.path.dirname(script_dir), 'outputs')
        
        # Add preview suffix to output directory name
        suffix = "_preview" if args.preview_mode or args.max_duration > 0 or args.max_frames > 0 else ""
        args.output_dir = os.path.join(outputs_root, f"{input_name}_{timestamp}{suffix}")
    
    # Validate input video
    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    
    return args


def process_track_args(args):
    """Process and validate track evaluation arguments"""
    args = process_common_args(args)
    
    # Track-specific processing
    if not args.output_root:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output_root = os.path.join(script_dir, 'results')
    
    # Set up experiment directory
    args.exp_dir = os.path.join(args.output_root, args.exp_id)
    args.save_dir = args.exp_dir
    
    return args


def parse_demo_args():
    """Main function to parse demo arguments"""
    parser = create_demo_parser()
    args = parser.parse_args()
    args = process_demo_args(args)
    return args


def parse_track_args():
    """Main function to parse track evaluation arguments"""
    parser = create_track_parser()
    args = parser.parse_args()
    args = process_track_args(args)
    return args


if __name__ == '__main__':
    # Test the argument parsers
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'track':
        # Remove 'track' from args for testing
        sys.argv.pop(1)
        args = parse_track_args()
        print("Track arguments parsed successfully:")
    else:
        args = parse_demo_args()
        print("Demo arguments parsed successfully:")
    
    for key, value in vars(args).items():
        print(f"  {key}: {value}")