from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import numpy as np
from torchvision import transforms
import glob







class InteractiveSegmentationDataset(Dataset):
    def __init__(self, data_dir, num_classes, transform=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = transform
        
        self.samples = []
        image_names = os.listdir(os.path.join(data_dir, "rgb"))
        for image_name in image_names:
            gt_pattern = f"{os.path.splitext(image_name)[0]}_msk*.png"
            gt_paths = glob.glob(os.path.join(data_dir, f"{num_classes}_classes", "mask", gt_pattern))
            
            srb_pattern = f"{os.path.splitext(image_name)[0]}_scr*.png"
            srb_paths = glob.glob(os.path.join(data_dir, f"{num_classes}_classes", "scribble", srb_pattern))
            
            for gt_path, srb_path in zip(gt_paths, srb_paths):
                self.samples.append((image_name, gt_path, srb_path))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_name, gt_path, srb_path = self.samples[idx]
        
        # Load RGB image
        rgb_path = os.path.join(self.data_dir, "rgb", image_name)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        

        
        # Load scribble mask
        srb = cv2.imread(srb_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations if provided
        if self.transform:
            rgb = self.transform(rgb)
            gt = torch.from_numpy(gt).unsqueeze(0)
            srb = torch.from_numpy(srb).unsqueeze(0)
        
        # Create data dictionary
        data = {
            'rgb': rgb,
            'gt': gt,
            'srb': srb,
            'info': image_name
        }
        
        return data