#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import json
import cv2

# Add src path for importing utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lib.utils.data_utils import get_mot20_datadir


def convert_mot20_to_coco():
    """
    Convert MOT20 dataset to COCO format
    Compatible with ByteTrack conversion script
    """
    # Use unified data directory management
    DATA_PATH = get_mot20_datadir()
    OUT_PATH = os.path.join(DATA_PATH, 'annotations')
    SPLITS = ['train_half', 'val_half', 'train', 'test']
    HALF_VIDEO = True
    CREATE_SPLITTED_ANN = True
    CREATE_SPLITTED_DET = True

    print(f"MOT20 data path: {DATA_PATH}")
    print(f"Output annotation path: {OUT_PATH}")

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test')
        else:
            data_path = os.path.join(DATA_PATH, 'train')
            
        if not os.path.exists(data_path):
            print(f"Warning: Data path does not exist: {data_path}")
            continue
            
        out_path = os.path.join(OUT_PATH, f'{split}.json')
        out = {
            'images': [], 
            'annotations': [], 
            'videos': [],
            'categories': [{'id': 1, 'name': 'pedestrian'}]
        }
        
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
                
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            
            if not os.path.exists(img_path):
                print(f"Warning: Image path does not exist: {img_path}")
                continue
                
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])

            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                    
                img_file = os.path.join(data_path, f'{seq}/img1/{i + 1:06d}.jpg')
                if not os.path.exists(img_file):
                    continue
                    
                img = cv2.imread(img_file)
                height, width = img.shape[:2]
                image_info = {
                    'file_name': f'{seq}/img1/{i + 1:06d}.jpg',
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1 - image_range[0],
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height, 
                    'width': width
                }
                out['images'].append(image_info)
                
            print(f'{seq}: {num_images} images')
            
            if split != 'test':
                det_path = os.path.join(seq_path, 'det/det.txt')
                
                if os.path.exists(ann_path):
                    anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                    
                    if CREATE_SPLITTED_ANN and ('half' in split):
                        anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                             if int(anns[i][0]) - 1 >= image_range[0] and
                                             int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                        anns_out[:, 0] -= image_range[0]
                        gt_out = os.path.join(seq_path, f'gt/gt_{split}.txt')
                        os.makedirs(os.path.dirname(gt_out), exist_ok=True)
                        with open(gt_out, 'w') as fout:
                            for o in anns_out:
                                fout.write(f'{int(o[0])},{int(o[1])},{int(o[2])},{int(o[3])},{int(o[4])},{int(o[5])},{int(o[6])},{int(o[7])},{o[8]:.6f}\n')

                    if os.path.exists(det_path):
                        dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                        if CREATE_SPLITTED_DET and ('half' in split):
                            dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                                 if int(dets[i][0]) - 1 >= image_range[0] and
                                                 int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                            dets_out[:, 0] -= image_range[0]
                            det_out = os.path.join(seq_path, f'det/det_{split}.txt')
                            os.makedirs(os.path.dirname(det_out), exist_ok=True)
                            with open(det_out, 'w') as dout:
                                for o in dets_out:
                                    dout.write(f'{int(o[0])},{int(o[1])},{o[2]:.1f},{o[3]:.1f},{o[4]:.1f},{o[5]:.1f},{o[6]:.6f}\n')

                    print(f'{int(anns[:, 0].max())} annotation images')
                    for i in range(anns.shape[0]):
                        frame_id = int(anns[i][0])
                        if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                            continue
                        track_id = int(anns[i][1])
                        cat_id = int(anns[i][7])
                        ann_cnt += 1
                        
                        # MOT20 specific filtering conditions
                        if not (int(anns[i][6]) == 1):  # whether ignore.
                            continue
                        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
                            category_id = -1
                        else:
                            category_id = 1  # pedestrian(non-static)
                            if not track_id == tid_last:
                                tid_curr += 1
                                tid_last = track_id
                                
                        ann = {
                            'id': ann_cnt,
                            'category_id': category_id,
                            'image_id': image_cnt + frame_id,
                            'track_id': tid_curr,
                            'bbox': anns[i][2:6].tolist(),
                            'conf': float(anns[i][6]),
                            'iscrowd': 0,
                            'area': float(anns[i][4] * anns[i][5])
                        }
                        out['annotations'].append(ann)
                        
            image_cnt += num_images
            print(f'tid_curr: {tid_curr}, tid_last: {tid_last}')
            
        print(f'Loaded {split} for {len(out["images"])} images and {len(out["annotations"])} samples')
        
        with open(out_path, 'w') as f:
            json.dump(out, f)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    convert_mot20_to_coco()