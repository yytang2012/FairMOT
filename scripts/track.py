from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))
import os.path as osp
import cv2
import logging
import motmetrics as mm
import numpy as np
import torch
from datetime import datetime

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
# Use original FairMOT dataset format (not ByteTrack format)
import datasets.dataset.jde as datasets
# Alternative: import datasets.dataset.jde_yolov5 as datasets  # for ByteTrack format

from tracking_utils.utils import mkdir_if_missing
from arg_utils import parse_track_args


class HalfSplitDataloader:
    """Dataloader for half-split data (train_half or val_half) following ByteTrack's approach"""
    
    def __init__(self, img_path, img_size, half_type):
        """
        Args:
            img_path: Path to img1 directory
            img_size: (width, height) tuple
            half_type: 'train_half' or 'val_half'
        """
        import glob
        
        self.img_size = img_size
        self.half_type = half_type
        
        # Get all image files
        image_files = glob.glob(os.path.join(img_path, '*.[jJ][pP][gG]'))
        image_files.extend(glob.glob(os.path.join(img_path, '*.[pP][nN][gG]')))
        self.all_files = sorted(image_files)
        
        total_frames = len(self.all_files)
        
        # ByteTrack's splitting logic
        if half_type == 'train_half':
            # First half: frames 0 to total_frames//2
            self.selected_files = self.all_files[:total_frames // 2]
        elif half_type == 'val_half':
            # Second half: frames total_frames//2+1 to end  
            self.selected_files = self.all_files[total_frames // 2:]
        else:
            raise ValueError(f"Invalid half_type: {half_type}. Use 'train_half' or 'val_half'")
        
        logger.info(f'Half split ({half_type}): using {len(self.selected_files)}/{total_frames} frames')
        
    def __iter__(self):
        for img_path in self.selected_files:
            yield self._load_image(img_path)
            
    def __len__(self):
        return len(self.selected_files)
        
    def _load_image(self, img_path):
        """Load and preprocess image using FairMOT's approach"""
        from datasets.dataset.jde import letterbox
        
        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.img_size[1], width=self.img_size[0])

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))



def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True, half_split=None):
    logger.setLevel(logging.INFO)
    # Output results to FairMOT main directory instead of data directory
    fairmot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_root = os.path.join(fairmot_root, 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(fairmot_root, 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        
        # Handle half split data loading
        if half_split:
            logger.info(f'Using {half_split} of sequence: {seq}')
            dataloader = HalfSplitDataloader(osp.join(data_root, seq, 'img1'), opt.img_size, half_split)
            result_filename = os.path.join(result_root, f'{seq}_{half_split}.txt')
        else:
            dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
            result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # Use the new device selection logic
        use_cuda = getattr(opt, 'use_cuda', opt.gpus[0] >= 0 if len(opt.gpus) > 0 else False)
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, use_cuda=use_cuda)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        if half_split:
            gt_path = os.path.join(data_root, seq, 'gt', f'gt_{half_split}.txt')
        else:
            gt_path = os.path.join(data_root, seq, 'gt', 'gt.txt')
            
        if os.path.exists(gt_path):
            logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))
        else:
            if half_split:
                logger.info(f'No ground truth found for {seq} {half_split} - ensure ByteTrack conversion was run')
            else:
                logger.info('No ground truth found for {} - skipping evaluation (test set)'.format(seq))
            accs.append(None)
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary (only for sequences with ground truth)
    valid_accs = [acc for acc in accs if acc is not None]
    valid_seqs = [seq for seq, acc in zip(seqs, accs) if acc is not None]
    
    if valid_accs:
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(valid_accs, valid_seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
    else:
        print("No ground truth available for evaluation (test set). Results saved to:")
        for seq in seqs:
            result_file = os.path.join(result_root, '{}.txt'.format(seq))
            if os.path.exists(result_file):
                print(f"  {seq}: {result_file}")
        print(f"All results saved in: {result_root}")


def get_dataset_config(args):
    """Get dataset configuration based on arguments"""
    
    # Define dataset configurations
    dataset_configs = {
        'MOT15': {
            'val': {
                'seqs': ['Venice-2', 'KITTI-13', 'KITTI-17', 'ETH-Bahnhof', 'ETH-Sunnyday', 
                        'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ADL-Rundle-6', 
                        'ADL-Rundle-8', 'ETH-Pedcross2'],
                'path': 'MOT15/images/train'
            },
            'test': {
                'seqs': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing',
                        'ETH-Jelmoli', 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 
                        'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
                'path': 'MOT15/images/test'
            }
        },
        'MOT16': {
            'val': {
                'seqs': ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'],
                'path': 'MOT16/train'
            },
            'test': {
                'seqs': ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'],
                'path': 'MOT16/test'
            },
            'train_half': {
                'seqs': ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'],
                'path': 'MOT16/train',
                'half_split': 'train_half'
            },
            'val_half': {
                'seqs': ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'],
                'path': 'MOT16/train',
                'half_split': 'val_half'
            }
        },
        'MOT17': {
            'val': {
                'seqs': ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 
                        'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP'],
                'path': 'MOT17/train'
            },
            'test': {
                'seqs': ['MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP',
                        'MOT17-08-SDP', 'MOT17-12-SDP', 'MOT17-14-SDP'],
                'path': 'MOT17/test'
            }
        },
        'MOT20': {
            'val': {
                'seqs': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
                'path': 'MOT20/train'
            },
            'test': {
                'seqs': ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
                'path': 'MOT20/test'
            },
            'train_half': {
                'seqs': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
                'path': 'MOT20/train',
                'half_split': 'train_half'
            },
            'val_half': {
                'seqs': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
                'path': 'MOT20/train',
                'half_split': 'val_half'
            }
        },
        'MOT20_mini': {
            'val': {
                'seqs': ['MOT20-01', 'MOT20-02'],
                'path': 'MOT20_mini/images/train'
            },
            'test': {
                'seqs': ['MOT20-01', 'MOT20-02'],
                'path': 'MOT20_mini/images/train'
            }
        }
    }
    
    # Handle custom sequences
    if args.seqs:
        seqs = [seq.strip() for seq in args.seqs.split(',')]
        # Use dataset path or default to data_dir
        if args.dataset in dataset_configs and args.split in dataset_configs[args.dataset]:
            config = dataset_configs[args.dataset][args.split]
            data_root = os.path.join(args.data_dir, config['path'])
            half_split = config.get('half_split', None)
        else:
            data_root = args.data_dir
            half_split = None
        dataset_name = f"{args.dataset}_{args.split}_custom"
        return data_root, seqs, dataset_name, half_split
    
    # Get standard dataset configuration
    if args.dataset not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from: {list(dataset_configs.keys())}")
    
    if args.split not in dataset_configs[args.dataset]:
        available_splits = list(dataset_configs[args.dataset].keys())
        raise ValueError(f"Unsupported split '{args.split}' for {args.dataset}. Available: {available_splits}")
    
    config = dataset_configs[args.dataset][args.split]
    data_root = os.path.join(args.data_dir, config['path'])
    seqs = config['seqs']
    dataset_name = f"{args.dataset}_{args.split}"
    half_split = config.get('half_split', None)
    
    return data_root, seqs, dataset_name, half_split


if __name__ == '__main__':
    # Parse arguments using independent argument parser
    args = parse_track_args()
    
    # Set CUDA device visibility
    if hasattr(args, 'gpus_str'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus_str)
    
    try:
        # Get dataset configuration
        data_root, seqs, dataset_name, half_split = get_dataset_config(args)
        
        # Generate experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{dataset_name}_{timestamp}" if dataset_name != "CUSTOM" else f"custom_{timestamp}"
        
        print(f"\n🚀 Starting FairMOT tracking evaluation")
        print(f"📊 Dataset: {dataset_name}")
        print(f"📁 Data root: {data_root}")
        print(f"🎯 Sequences: {', '.join(seqs)}")
        print(f"📝 Experiment: {exp_name}")
        print(f"🔧 Device: {args.device}")
        print("="*60)
        
        # Run evaluation
        main(args,
             data_root=data_root,
             seqs=seqs,
             exp_name=exp_name,
             show_image=False,
             save_images=args.save_images,
             save_videos=args.save_videos,
             half_split=half_split)
             
        print(f"\n✅ Tracking evaluation completed successfully!")
        print(f"📁 Results saved in: {args.exp_dir}")
        
    except Exception as e:
        print(f"\n❌ Tracking evaluation failed with error: {e}")
        sys.exit(1)
