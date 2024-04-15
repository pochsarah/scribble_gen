# External library imports for image processing and manipulation
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import time

from tqdm import tqdm
import yaml
import cv2
import numpy as np
# PyTorch related imports
import torch

from utils.dataloaders import CustomScribbleDataset
from torch.utils.data import DataLoader
import visualize
import scribbles


def generate_scribble_mask(params, mask):
        # Generate scribbles inside contours
        internal_scribbles = scribbles.generate_scribbles_inside_contours(mask, **params["intern"])

        # Generate border scribbles
        border_scribbles = scribbles.generate_border_multiclass_scribbles(mask, **params["border"])

        # Combine scribbles
        combined_scribbles = np.maximum(internal_scribbles, border_scribbles)

        return combined_scribbles

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--config', default='config/params.yaml')
    parser.add_argument('--project', default='./results/')
    ARGS = parser.parse_args()

    mask_paths = glob("/home/spoch/Documents/private/termatics/gtFine_trainvaltest/gtFine/train/*/*_color.png", recursive=True)
    mask_paths.sort()
    
    #save_dir = "/home/spoch/Documents/private/termatics/scribbles/"
    with open(ARGS.config) as f:
        scribble_params = yaml.load(f, Loader=yaml.SafeLoader) 

    colors = {label: np.random.randint(0, 256, 3) for label in range(256)}
    save_dir = Path(ARGS.project)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0= time.time()
    for i in tqdm(range(len(mask_paths))):

        basename = str(Path(mask_paths[i]).name)
        gray_image = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)

        combined = generate_scribble_mask(scribble_params, gray_image)
        visualize.save_multiclass_scribbles(combined, str(save_dir)+basename, colors)

    print(f'Done. ({time.time() - t0:.3f}s)')

