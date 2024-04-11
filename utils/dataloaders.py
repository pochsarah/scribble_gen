
import cv2
import torch 
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import IterableDataset


import scribbles


class CustomScribbleDataset(IterableDataset):
    def __init__(self, image_paths, mask_paths, n_binary, n_subclasses, border_scribble_params, internal_scribble_params):
        self.image_paths = image_paths #chemin img
        self.mask_paths = mask_paths # chemin du mask 
        self.n_binary = n_binary
        self.n_subclasses = n_subclasses
        self.border_scribble_params = border_scribble_params #paramètres des grib de front
        self.internal_scribble_params = internal_scribble_params # param grib internes 
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __iter__(self):
        for idx in tqdm(range(len(self.image_paths))):
            # Load image and mask
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            basename = str(Path(self.image_paths[idx]).name)

            # Normalize and convert the image to tensor
            image_tensor = transforms.ToTensor()(image)
            image_tensor = self.normalize(image_tensor)

            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes != 0]
            
            # Binary classification for random classes
            if self.n_binary > 0 and len(unique_classes) > 1:
                for _ in range(self.n_binary):
                    cls = np.random.choice(unique_classes)
                    binary_mask = np.where(mask == cls, cls, 0)
                    print(np.unique(binary_mask))
                    binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)
                    scribble_mask_tensor = self._generate_scribble_mask(binary_mask_tensor)
                    yield image_tensor, scribble_mask_tensor,binary_mask_tensor
            
            # Generate multiclass scenarios
            if self.n_subclasses > 0 and len(unique_classes) > 1:
                n_subclasses = len(unique_classes) if len(unique_classes) <= self.n_subclasses else self.n_subclasses
                for _ in range(self.n_subclasses):
                    num_classes = np.random.randint(2, len(unique_classes))
                    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)
                    #print(selected_classes)

                    # Create a multiclass mask with original class labels
                    sub_mask = np.zeros_like(mask)
                    for cls in selected_classes:
                        sub_mask[mask == cls] = cls

                    sub_mask_tensor = torch.tensor(sub_mask, dtype=torch.int64).unsqueeze(0)
                    scribble_mask_tensor = self._generate_scribble_mask(sub_mask_tensor)
                    yield image_tensor, scribble_mask_tensor, sub_mask_tensor, basename

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
