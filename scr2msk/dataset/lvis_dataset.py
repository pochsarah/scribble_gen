import torch
from torch.utils.data.dataset import Dataset
import os
import cv2
import glob
from tqdm import tqdm

class InteractiveSegmentationDataset(Dataset):
    def __init__(self, data_dir, num_classes, transform=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = transform
        
        self.samples = []
        self.image_names = os.listdir(os.path.join(data_dir, "rgb"))
        
        for i, image_name in tqdm(enumerate(self.image_names), total=len(self.image_names)):
            gt_pattern = f"{os.path.splitext(image_name)[0]}_msk*.png"
            gt_paths = glob.glob(os.path.join(data_dir, f"{num_classes}_classes", "mask", gt_pattern))
            
            srb_pattern = f"{os.path.splitext(image_name)[0]}_scr*.png"
            srb_paths = glob.glob(os.path.join(data_dir, f"{num_classes}_classes", "scribble", srb_pattern))
            
            for gt_path, srb_path in zip(gt_paths, srb_paths):
                self.samples.append((image_name, gt_path, srb_path))

        print(f"Number of samples for {self.num_classes} classes : {len(self.samples)}")
        
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
        
        cls_gt = (gt>0.5).long().squeeze(0) # WARNING obligé de mettre ca ???


        # Create data dictionary
        data = {
            'rgb': rgb,
            'gt': gt,
            'srb': srb,
            'cls_gt': cls_gt, 
            'info': image_name
        }
        
        return data