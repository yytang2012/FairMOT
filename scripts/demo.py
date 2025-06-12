import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))
import os.path as osp
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
from arg_utils import parse_demo_args


logger.setLevel(logging.INFO)


class LimitedVideoLoader:
    """
    Wrapper for video dataloader with frame/duration limits for faster preview
    """
    def __init__(self, dataloader, max_frames=0, max_duration=0, start_time=0):
        """
        Args:
            dataloader: Original video dataloader
            max_frames: Maximum number of frames to process (0 = no limit)
            max_duration: Maximum duration in seconds (0 = no limit)
            start_time: Start time in seconds (skip initial frames)
        """
        self.dataloader = dataloader
        self.max_frames = max_frames
        self.max_duration = max_duration
        self.start_time = start_time
        self.frame_rate = dataloader.frame_rate
        
        # Calculate frame limits
        self.start_frame = int(start_time * self.frame_rate) if start_time > 0 else 0
        
        if max_duration > 0:
            max_frames_from_duration = int(max_duration * self.frame_rate)
            if max_frames > 0:
                self.total_frames = min(max_frames, max_frames_from_duration)
            else:
                self.total_frames = max_frames_from_duration
        else:
            self.total_frames = max_frames if max_frames > 0 else len(dataloader)
        
        self.current_frame = 0
        self.processed_frames = 0
        
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            # Skip frames before start_time
            if i < self.start_frame:
                continue
                
            # Check if we've reached the frame limit
            if self.total_frames > 0 and self.processed_frames >= self.total_frames:
                break
                
            self.processed_frames += 1
            yield data
    
    def __len__(self):
        """Return the effective length considering limits"""
        original_len = len(self.dataloader)
        effective_start = min(self.start_frame, original_len)
        remaining_frames = original_len - effective_start
        
        if self.total_frames > 0:
            return min(self.total_frames, remaining_frames)
        return remaining_frames


def demo(args):
    """
    Run FairMOT demo on input video
    
    Args:
        args: Parsed arguments from demo_args.parse_demo_args()
    """
    # Create output directory
    result_root = args.output_dir
    mkdir_if_missing(result_root)
    
    logger.info('Starting FairMOT tracking demo...')
    logger.info(f'Input video: {args.input_video}')
    logger.info(f'Output directory: {result_root}')
    logger.info(f'Output format: {args.output_format}')
    logger.info(f'Model architecture: {args.arch}')
    logger.info(f'Using device: {args.device}')
    
    # Load video
    original_dataloader = datasets.LoadVideo(args.input_video, args.img_size)
    frame_rate = original_dataloader.frame_rate
    
    # Apply processing limits if specified
    has_limits = args.max_duration > 0 or args.max_frames > 0 or args.start_time > 0
    if has_limits:
        dataloader = LimitedVideoLoader(
            original_dataloader,
            max_frames=args.max_frames,
            max_duration=args.max_duration,
            start_time=args.start_time
        )
        
        # Log processing limits
        total_frames = len(original_dataloader)
        limited_frames = len(dataloader)
        
        logger.info(f'📊 Video info: {total_frames} total frames @ {frame_rate:.1f} FPS')
        if args.start_time > 0:
            logger.info(f'⏭️ Starting from: {args.start_time:.1f}s (frame {int(args.start_time * frame_rate)})')
        if args.max_duration > 0:
            logger.info(f'⏱️ Max duration: {args.max_duration:.1f}s')
        if args.max_frames > 0:
            logger.info(f'🎞️ Max frames: {args.max_frames}')
        logger.info(f'🎯 Processing: {limited_frames} frames ({limited_frames/frame_rate:.1f}s)')
        
        if args.preview_mode:
            logger.info('👀 Preview mode: Quick 10-second demo for testing')
    else:
        dataloader = original_dataloader
        logger.info(f'📊 Processing entire video: {len(dataloader)} frames @ {frame_rate:.1f} FPS')
    
    result_filename = os.path.join(result_root, 'results.txt')
    
    # Set up frame output directory for visualization
    frame_dir = None
    if args.output_format in ['video', 'frames'] or args.save_frames:
        frame_dir = osp.join(result_root, 'frames')
        mkdir_if_missing(frame_dir)
    
    # Run tracking
    logger.info('🚀 Starting tracking...')
    start_time = time.time()
    eval_seq(args, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=args.use_cuda)
    
    processing_time = time.time() - start_time
    logger.info(f'Tracking completed in {processing_time:.2f} seconds')
    
    # Generate output video if requested
    input_name = osp.splitext(osp.basename(args.input_video))[0]
    if args.output_format == 'video' and frame_dir and osp.exists(frame_dir):
        output_video_path = osp.join(result_root, f'{input_name}_tracking_result.mp4')
        logger.info(f'Creating output video: {output_video_path}')
        
        # Quality settings
        quality_settings = {
            'low': '-crf 28 -preset fast',
            'medium': '-crf 23 -preset medium', 
            'high': '-crf 18 -preset slow'
        }
        quality = quality_settings.get(args.quality, quality_settings['high'])
        
        # Use a more robust ffmpeg command
        cmd_str = f'ffmpeg -y -framerate {frame_rate} -i {frame_dir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p {quality} "{output_video_path}"'
        
        exit_code = os.system(cmd_str)
        if exit_code == 0:
            logger.info(f'Video saved successfully: {output_video_path}')
        else:
            logger.warning(f'Failed to create video. Exit code: {exit_code}')
            logger.info(f'Individual frames are available in: {frame_dir}')
    
    # Print summary
    print("\n" + "="*60)
    print("FAIRMOT DEMO RESULTS SUMMARY")
    print("="*60)
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {result_root}")
    
    # Processing information
    processed_frames = len(dataloader)
    processed_duration = processed_frames / frame_rate
    print(f"Processed: {processed_frames} frames ({processed_duration:.1f}s)")
    
    if has_limits:
        total_frames = len(original_dataloader)
        total_duration = total_frames / frame_rate
        print(f"Original video: {total_frames} frames ({total_duration:.1f}s)")
        if args.preview_mode:
            print("Mode: Preview (first 10 seconds)")
        elif args.max_duration > 0 or args.max_frames > 0:
            print("Mode: Limited processing")
    
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {processed_frames / processing_time:.2f}")
    
    if osp.exists(result_filename):
        print(f"Tracking results: {result_filename}")
    
    if args.output_format == 'video':
        output_video_path = osp.join(result_root, f'{input_name}_tracking_result.mp4')
        if osp.exists(output_video_path):
            print(f"Output video: {output_video_path}")
        elif frame_dir and osp.exists(frame_dir):
            print(f"Frames directory: {frame_dir}")
    elif frame_dir and osp.exists(frame_dir):
        print(f"Frames directory: {frame_dir}")
    
    print("="*60)
    
    return result_root




if __name__ == '__main__':
    # Parse arguments using independent argument parser
    args = parse_demo_args()
    
    # Set CUDA device visibility
    if hasattr(args, 'gpus_str'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus_str)
    
    # Run demo
    output_dir = demo(args)
    print(f"\nDemo completed successfully!")
    print(f"Results saved to: {output_dir}")
