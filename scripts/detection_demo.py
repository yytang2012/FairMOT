from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import sys

# Add src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import datasets.dataset.jde as datasets
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process


class DetectionDemo:
    def __init__(self, weights_path='models/mot20_fairmot.pth', data_path='datasets/MOT20/test/MOT20-01/img1'):
        self.max_per_image = 500
        self.num_classes = 1
        self.img_size = (640, 640)
        self.gpu = True
        self.reid_dim = 128
        self.arch = 'dla_34'
        self.ltrb = True
        self.reg_offset = True
        self.conf_thres = 0.3
        self.Kt = 500
        self.heads = {'hm': self.num_classes, 'wh': 2 if not self.ltrb else 4, 'id': self.reid_dim, 'reg': 2}
        self.head_conv = 256
        self.down_ratio = 4
        self.weights_path = weights_path
        self.data_path = data_path
        
        if self.gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.current_index = 0
        self.image_paths = []
        
        self._load_model()
        self._load_images()
        
    def _load_model(self):
        print('Creating model...')
        print(f'arch: {self.arch}, heads: {self.heads}, head_conv: {self.head_conv}, device: {self.device}')
        self.model = create_model(self.arch, self.heads, self.head_conv)
        
        if os.path.exists(self.weights_path):
            self.model = load_model(self.model, self.weights_path)
            print(f'Model loaded from: {self.weights_path}')
        else:
            print(f'Warning: Weights file not found at {self.weights_path}')
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _load_images(self):
        if os.path.exists(self.data_path):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(self.data_path, ext)))
            self.image_paths.sort()
            print(f'Found {len(self.image_paths)} images in {self.data_path}')
        else:
            print(f'Warning: Data path not found: {self.data_path}')
            
    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], 
                                meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results
        
    def detect_image(self, img_path):
        if not os.path.exists(img_path):
            return None, 0
            
        # Load and preprocess image
        dataloader = datasets.LoadImages(os.path.dirname(img_path), self.img_size)
        
        # Find the specific image in dataloader
        for path, img, img0 in dataloader:
            if path == img_path:
                person_count = 0
                im_blob = torch.from_numpy(img).to(self.device).unsqueeze(0)
                width = img0.shape[1]
                height = img0.shape[0]
                inp_height = im_blob.shape[2]
                inp_width = im_blob.shape[3]
                c = np.array([width / 2., height / 2.], dtype=np.float32)
                s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
                meta = {'c': c, 's': s, 'out_height': inp_height // self.down_ratio, 
                       'out_width': inp_width // self.down_ratio}

                # Network forward
                with torch.no_grad():
                    output = self.model(im_blob)[-1]
                    hm = output['hm'].sigmoid_()
                    wh = output['wh']
                    id_feature = output['id']
                    id_feature = F.normalize(id_feature, dim=1)

                    reg = output['reg'] if self.reg_offset else None
                    dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.ltrb, K=self.Kt)
                    id_feature = _tranpose_and_gather_feat(id_feature, inds)
                    id_feature = id_feature.squeeze(0)
                    id_feature = id_feature.cpu().numpy()

                dets = self.post_process(dets, meta)
                dets = self.merge_outputs([dets])[1]
                remain_inds = dets[:, 4] > self.conf_thres
                dets = dets[remain_inds]
                id_feature = id_feature[remain_inds]

                # Draw bounding boxes
                person_count = len(dets)
                for i in range(0, dets.shape[0]):
                    bbox = dets[i][0:4].astype(int)
                    score = dets[i][4]
                    cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(img0, f'{score:.2f}', (bbox[0], bbox[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                return img0, person_count
        
        return None, 0
        
    def run_demo(self):
        if not self.image_paths:
            print("No images found!")
            return
            
        print("Navigation: Left/Right arrow keys to navigate, 'q' to quit")
        
        while True:
            if 0 <= self.current_index < len(self.image_paths):
                img_path = self.image_paths[self.current_index]
                img_result, person_count = self.detect_image(img_path)
                
                if img_result is not None:
                    # Add image info text
                    filename = os.path.basename(img_path)
                    info_text = f'Image: {filename} ({self.current_index + 1}/{len(self.image_paths)}) - Detections: {person_count}'
                    print(info_text)
                    
                    # Add text overlay
                    cv2.putText(img_result, info_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.namedWindow('Detection Demo', cv2.WINDOW_NORMAL)
                    cv2.imshow('Detection Demo', img_result)
                else:
                    print(f"Failed to process image: {img_path}")
            
            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == 81 or key == 2:  # Left arrow key
                self.current_index = max(0, self.current_index - 1)
                print(f"Previous image ({self.current_index + 1}/{len(self.image_paths)})")
            elif key == 83 or key == 3:  # Right arrow key
                self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
                print(f"Next image ({self.current_index + 1}/{len(self.image_paths)})")
                
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='FairMOT Detection Demo')
    parser.add_argument('--weights', type=str, default='./models/mot20_fairmot.pth',
                       help='path to model weights')
    parser.add_argument('--data_path', type=str, default='./datasets/MOT20/test/MOT20-04/img1',
                       help='path to test images directory')
    parser.add_argument('--conf_thres', type=float, default=0.3,
                       help='confidence threshold')
    parser.add_argument('--img_size', type=int, default=640,
                       help='input image size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    demo = DetectionDemo(
        weights_path=args.weights,
        data_path=args.data_path
    )
    demo.conf_thres = args.conf_thres
    demo.img_size = (args.img_size, args.img_size)
    
    demo.run_demo()