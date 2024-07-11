import os
import shutil
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import scribbles
from empatches import EMPatches
import argparse
import yaml 


"""
TODO : 
    - add the patching in the CustomScribbleGenerator for more efficiency
    x put all the scribbles params in a config file for more easy management
        - randomyze some of the values ? 
    
"""

class CustomScribbleGenerator:
    def __init__(self, image_paths, mask_paths, output_dir, n_binary, n_subclasses, border_scribble_params, internal_scribble_params, patch_size=None, patch_images=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_dir = output_dir
        self.n_binary = n_binary
        self.n_subclasses = n_subclasses
        self.border_scribble_params = border_scribble_params
        self.internal_scribble_params = internal_scribble_params
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.patch_size = patch_size
        # self.patch_images = patch_images

    def generate_and_save(self):
        rgb_dir = os.path.join(self.output_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        for idx in tqdm(range(len(self.image_paths))):
            # Load image and mask
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

            # Move the original RGB image to the rgb folder
            rgb_name = os.path.basename(self.image_paths[idx])
            rgb_path = os.path.join(rgb_dir, rgb_name)
            shutil.copy(self.image_paths[idx], rgb_path)

            # Normalize and convert the image to tensor
            image_tensor = transforms.ToTensor()(image)
            image_tensor = self.normalize(image_tensor)

            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes != 0]
            
            # Binary classification for random classes
            if self.n_binary > 0 and len(unique_classes) > 1:
                for i in range(self.n_binary):
                    cls = np.random.choice(unique_classes)
                    binary_mask = np.where(mask == cls, cls, 0)
                    binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)
                    scribble_mask_tensor = self._generate_scribble_mask(binary_mask_tensor)
                    self._save_files(rgb_name, scribble_mask_tensor, binary_mask_tensor, 1, idx, i)
            
            # Generate multiclass scenarios
            if self.n_subclasses > 0 and len(unique_classes) > 1:
                n_subclasses = len(unique_classes) if len(unique_classes) <= self.n_subclasses else self.n_subclasses
                for i in range(self.n_subclasses):
                    num_classes = np.random.randint(2, len(unique_classes))
                    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)

                    # Create a multiclass mask with original class labels
                    sub_mask = np.zeros_like(mask)
                    for cls in selected_classes:
                        sub_mask[mask == cls] = cls

                    sub_mask_tensor = torch.tensor(sub_mask, dtype=torch.int64).unsqueeze(0)
                    scribble_mask_tensor = self._generate_scribble_mask(sub_mask_tensor)
                    self._save_files(rgb_name, scribble_mask_tensor, sub_mask_tensor, num_classes, idx, i)

            # # Patch and save images if chosen
            # if self.patch_images and self.patch_size is not None:
            #     self._patch_and_save_images(image, mask, rgb_name, idx)

    def _generate_scribble_mask(self, mask_tensor):
        # Convert tensor to numpy array and remove channel dimension
        mask = mask_tensor.squeeze().numpy().astype(np.uint8)

        # Generate scribbles inside contours
        internal_scribbles = scribbles.generate_scribbles_inside_contours(mask, **self.internal_scribble_params)

        # Generate border scribbles
        border_scribbles = scribbles.generate_border_multiclass_scribbles(mask, **self.border_scribble_params)

        # Combine scribbles
        combined_scribbles = np.maximum(internal_scribbles, border_scribbles)

        # Convert combined scribbles to tensor
        scribble_tensor = torch.tensor(combined_scribbles, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return scribble_tensor

    def _save_files(self, rgb_name, scribble_mask_tensor, mask_tensor, num_classes, idx, i):
        # Create directory for the number of classes if it doesn't exist
        class_dir = os.path.join(self.output_dir, f"{num_classes}_classes")
        os.makedirs(class_dir, exist_ok=True)

        # Create subfolders for scribble and mask
        scr_dir = os.path.join(class_dir, "scribble")
        msk_dir = os.path.join(class_dir, "mask")
        os.makedirs(scr_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        # Save scribble mask
        scr_name = f"{os.path.splitext(rgb_name)[0]}_scr_{idx}_{i}.png"
        scr_path = os.path.join(scr_dir, scr_name)
        scr_image = transforms.ToPILImage()(scribble_mask_tensor)
        scr_image.save(scr_path)

        # Save ground truth mask
        msk_name = f"{os.path.splitext(rgb_name)[0]}_msk_{idx}_{i}.png"
        msk_path = os.path.join(msk_dir, msk_name)
        msk_image = transforms.ToPILImage()(mask_tensor.to(torch.uint8))
        msk_image.save(msk_path)

    # def _patch_and_save_images(self, image, mask, rgb_name, idx):
    #     # Create directory for patched images and masks
    #     patched_dir = os.path.join(self.output_dir, "patched")
    #     patched_rgb_dir = os.path.join(patched_dir, "rgb")
    #     patched_mask_dir = os.path.join(patched_dir, "mask")
    #     os.makedirs(patched_rgb_dir, exist_ok=True)
    #     os.makedirs(patched_mask_dir, exist_ok=True)

    #     # Calculate the number of patches in each dimension
    #     num_patches_x = image.shape[1] // self.patch_size
    #     num_patches_y = image.shape[0] // self.patch_size

    #     # Iterate over each patch
    #     for i in range(num_patches_y):
    #         for j in range(num_patches_x):
    #             # Extract the patch from the image and mask
    #             start_y = i * self.patch_size
    #             end_y = start_y + self.patch_size
    #             start_x = j * self.patch_size
    #             end_x = start_x + self.patch_size
    #             image_patch = image[start_y:end_y, start_x:end_x]
    #             mask_patch = mask[start_y:end_y, start_x:end_x]

    #             # Save the patched image
    #             patched_rgb_name =f"{os.path.splitext(rgb_name)[0]}patch{idx}{i}{j}.png"
    #             patched_rgb_path = os.path.join(patched_rgb_dir, patched_rgb_name)
    #             cv2.imwrite(patched_rgb_path, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
    #                         # Save the patched mask
    #             patched_mask_name = f"{os.path.splitext(rgb_name)[0]}_patch_mask_{idx}_{i}_{j}.png"
    #             patched_mask_path = os.path.join(patched_mask_dir, patched_mask_name)
    #             cv2.imwrite(patched_mask_path, mask_patch)
  

if __name__=='__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgs", type=str, help="Path of the image file or directory")
    parser.add_argument("--masks", type=str, help="Path of the masks file or directory")
    parser.add_argument("--save", type=str, default="./scribbled", help="Path for saving scribbles")
    parser.add_argument("--patch", type=str, default="./patched", help="Path for saving patched images")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path of the configuration file")
    parser.add_argument("--patch-size", type=int, default=512, help="Size in pixels for the square patches")

    PARAMS = parser.parse_args()    
        
    with open(PARAMS.config, 'r') as f: 
        config = yaml.safe_load(f)    
    
    image_paths, mask_paths = glob.glob(PARAMS.imgs), glob.glob(PARAMS.masks)
    image_paths.sort()
    mask_paths.sort()   

    generator = CustomScribbleGenerator(image_paths, mask_paths, PARAMS.save, config["n_binary"], config["n_subclasses"],
                                        config["border_scribble_params"], config["internal_scribble_params"])
    generator.generate_and_save()

    rgb = glob.glob(PARAMS.save+'/rgb/*')
    scr = glob.glob(PARAMS.save+'/*/scribble/*')
    msk = glob.glob(PARAMS.save+'/*/mask/*')

    for p in tqdm(rgb + scr + msk):

        img = cv2.imread(p, cv2.IMREAD_COLOR)

        empatch = EMPatches()
        patches = empatch.extract_patches(img, patchsize=PARAMS.patch_size, overlap=0.2)

        directory =  os.path.dirname(p).replace(PARAMS.save, PARAMS.patch)
        os.makedirs(directory, exist_ok=True)

        for i, file_path in enumerate(patches.imgs):
            if p in rgb: 
                new_name = f"{os.path.splitext(os.path.basename(p))[0]}_{i}.png"
            elif p in scr: 
                name, ext = os.path.splitext(os.path.basename(p))
                parts = name.split('_scr')
                new_name = f"{parts[0]}_{i}_scr{parts[1]}{ext}"
            elif p in msk: 
                name, ext = os.path.splitext(os.path.basename(p))
                parts = name.split('_msk')
                new_name = f"{parts[0]}_{i}_msk{parts[1]}{ext}"

            new_file_path = os.path.join(directory, new_name)
            cv2.imwrite(new_file_path, file_path)