
from typing import List, Tuple

# External library imports for image processing and manipulation
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_image_as_nparray(file_path:str):
    """
    Load an image from a file path as a NumPy array.
    """
    image = Image.open(file_path)
    return np.array(image)

def plot_images_side_by_side(image1:np.ndarray, image2:np.ndarray, title1:str="Original", title2:str="Grayscale"):
    """
    Plot two images side by side for comparison.
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray' if len(image1.shape) == 2 else None)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.show()

def plot_overlay_multiclass_scribbles(base_image:np.ndarray, scribble_mask:np.ndarray,plot_size:Tuple[int, int] = (20, 20)):
    """
    Overlay a base image (grayscale or RGB) with its generated multiclass scribbles.
    The base image can be either grayscale or RGB, but the scribble mask is always grayscale.
    Random colors are generated for each class.
    """
    # Check if the base image is grayscale; if so, convert it to RGB
    if len(base_image.shape) == 2 or (len(base_image.shape) == 3 and base_image.shape[2] == 1):
        # Convert the single-channel image to a three-channel image
        overlay_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        # The base image is already in RGB format
        overlay_image = base_image.copy()

    unique_classes = np.unique(scribble_mask)[1:]  # Exclude background (0)

    # Generate random colors for each class
    colors = {label: np.random.randint(0, 256, 3) for label in unique_classes}

    # Overlay each class's scribbles onto the base image
    for class_label, color in colors.items():
        # Create a mask for the current class
        class_mask = scribble_mask == class_label

        # Overlay the color on the base image
        overlay_image[class_mask] = overlay_image[class_mask] * 0.5 + color * 0.5

    plt.figure(figsize=plot_size)  # Increase the size as needed
    # Plot the result with matplotlib
    plt.imshow(overlay_image)
    plt.axis('off')  # Hide the axes
    plt.title('Multiclass Scribbles Overlay on Image')
    plt.show()