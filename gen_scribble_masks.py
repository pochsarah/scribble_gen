import os
import cv2
import yaml
import time
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import visualize
import scribbles

#-------------------
# Faire en sorte d'avoir les mêmes couleurs pour les classes entre le mask et la couche scribble
#-------------------


def generate_scribble_mask(params: dict, mask: np.array):
        # Generate scribbles inside contours
        internal_scribbles = scribbles.generate_scribbles_inside_contours(mask, **params["intern"])

        # Generate border scribbles
        border_scribbles = scribbles.generate_border_multiclass_scribbles(mask, **params["border"])

        # Combine scribbles
        return np.maximum(internal_scribbles, border_scribbles)

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--config', type=str, default='./config/params.yaml')
    parser.add_argument('--project', type=str, default='./results/')
    ARGS = parser.parse_args()

    with open(ARGS.config) as f:
        scribble_params = yaml.load(f, Loader=yaml.SafeLoader)

    if Path(ARGS.source).is_file():
         mask_paths=[ARGS.source]
    else:
        mask_paths = glob(ARGS.source, recursive=True)
        
    save_dir = Path(ARGS.project)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0= time.time()
    for i in tqdm(range(len(mask_paths))):
        basename = Path(mask_paths[i]).name
        gray_image = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)

        label, index = np.unique(gray_image, return_index=True)

        color_img = cv2.imread(mask_paths[i], cv2.IMREAD_COLOR).reshape(-1, 3)
        lab_col = {label[i]: np.array(color_img[index[i]]) for i in range(len(label))}

        combined = generate_scribble_mask(scribble_params, gray_image)
        visualize.save_multiclass_scribbles(combined, str(os.path.join(save_dir, basename)), lab_col)

    print(f'Done. ({time.time() - t0:.3f}s)')
