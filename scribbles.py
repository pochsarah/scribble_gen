# Standard library imports
import glob
import os
import random
import subprocess
from typing import List, Tuple

# External library imports for image processing and manipulation
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# PyTorch related imports
import torch
import torch.nn.functional as F
from torchvision import transforms



def draw_scribble_line(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], thickness: int, epsilon: float = 1.0):
    """
    Draws a scribbled line between two points on the image using intermediate points with added random noise and simplifies it using the Ramer-Douglas-Peucker algorithm.

    :param img: The image on which to draw the scribble line.
    :param p1: The starting point (x, y) of the scribble line.
    :param p2: The ending point (x, y) of the scribble line.
    :param color: The color of the scribble line.
    :param thickness: The thickness of the scribble line.
    :param epsilon: The approximation accuracy of the Ramer-Douglas-Peucker algorithm; smaller values more closely approximate the original points.
    """
    # Number of intermediate points
    num_points = random.randint(3, 10)
    # Interpolate points between p1 and p2 and add random noise
    points = np.linspace(np.array(p1), np.array(p2), num_points)
    noise_scale = 0.1
    noise = np.random.normal(0, noise_scale, points.shape)
    points = points + noise
    
    # Ensure the points are 32-bit floats, as required by approxPolyDP
    points = points.astype(np.float32)
    
    # Simplify the curve with Ramer-Douglas-Peucker algorithm
    points = np.expand_dims(points, 1)  # Needs to be 3-dimensional for approxPolyDP
    simplified_points = cv2.approxPolyDP(points, epsilon, False)
    simplified_points = np.squeeze(simplified_points)  # Convert back to 2D

    # Draw lines between the simplified intermediate points
    for i in range(len(simplified_points) - 1):
        pt1 = tuple(simplified_points[i].astype(int))
        pt2 = tuple(simplified_points[i + 1].astype(int))
        cv2.line(img, pt1, pt2, color, thickness)

def generate_contour_chunks(binary_mask: np.ndarray, max_chunk_length: int, min_chunk_length: int,thickness: int, border_margin: int, epsilon: float = 1.0, inclusion_probability: float = 0.5):
    """
    Extracts random chunks of contours from a binary mask and applies a scribble effect to them. Chunks are of random lengths and start at random points within the contours.

    :param binary_mask: The binary mask from which to extract contour chunks.
    :param max_chunk_length: The maximum length of a contour chunk.
    :param min_chunk_length: The minimum length required to consider a contour chunk.
    :param border_margin: The margin from the border within which contour points will be ignored.
    :param epsilon: The approximation accuracy of the Ramer-Douglas-Peucker algorithm used in drawing scribbled lines.
    :param inclusion_probability: The probability of including a random contour chunk in the output.
    """    # Ensure the binary mask is of type np.uint8
    binary_mask = np.uint8(binary_mask)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an empty image with the same dimensions as the binary mask to place random contour chunks
    contour_chunks_image = np.zeros_like(binary_mask)

    # Check if any contours were found
    if not contours:
        return contour_chunks_image  # Return the empty image if no contours were found

    # Loop over each contour to create random contour chunks
    for contour in contours:
        # Randomly decide whether to include this contour, based on the inclusion_probability
        if random.random() > inclusion_probability:
            continue

        # Ensure the contour is long enough for a chunk
        contour = np.squeeze(contour)  # Remove redundant dimensions and convert to NumPy array
        if contour.shape[0] >= min_chunk_length:
            # Filter out contour points that are too close to the border
            contour = np.array([pt for pt in contour if (pt[0] >= border_margin and
                                                         pt[0] <= binary_mask.shape[1] - border_margin and
                                                         pt[1] >= border_margin and
                                                         pt[1] <= binary_mask.shape[0] - border_margin)])
            if contour.shape[0] < min_chunk_length:  # If the contour is too short after filtering, skip it
                continue

            # Choose a random starting point for the chunk
            start_index = random.randint(0, contour.shape[0] - 1)
            # Choose a random length for the chunk, not exceeding the max_chunk_length and considering the contour's length
            chunk_length = random.randint(min_chunk_length, min(contour.shape[0], max_chunk_length))
            # Define the ending index based on the starting index and chunk_length
            end_index = start_index + chunk_length
            # If the end index exceeds the contour length, wrap the selection to the start of the contour
            end_index = end_index if end_index <= contour.shape[0] else contour.shape[0]

            # Grab a continuous segment of the contour
            contour_chunk = contour[start_index:end_index]

            # Draw the random contour chunk onto the mask with scribble effect
            for i in range(len(contour_chunk) - 1):
                draw_scribble_line(contour_chunks_image,
                                   tuple(contour_chunk[i]),
                                   tuple(contour_chunk[i + 1]),
                                   color=255,
                                   thickness=thickness,
                                   epsilon=epsilon)

    return contour_chunks_image



def generate_multiclass_scribbles_(multiclass_image: np.ndarray, 
                                  max_chunk_length: int, 
                                  min_chunk_length: int, 
                                  border_margin: int, 
                                  epsilon: float, 
                                  inclusion_probability: float, 
                                  thickness: int) -> np.ndarray:
    """
    Generate scribbles for each class in a multiclass image. This includes generating separation lines between regions.
    The scribbles are generated by extracting contour chunks from binary masks for each class and applying a scribble effect.

    :param multiclass_image: A 2D numpy array where each pixel's value corresponds to a class label.
    :param max_chunk_length: The maximum length of any contour chunk for the scribble.
    :param min_chunk_length: The minimum length a contour must have to be considered for a scribble chunk.
    :param border_margin: The margin from the borders of the image within which contours will be ignored.
    :param epsilon: The approximation accuracy of the Ramer-Douglas-Peucker algorithm used in drawing scribbled lines.
    :param inclusion_probability: The probability that a found contour will be included as a chunk in the scribbles image.
    :param thickness: The thickness of the scribbles.
    :return: A 2D numpy array of the same shape as `multiclass_image`, where each pixel's value corresponds to a class label, with scribbles overlaid.
    """
    # Initialize the scribbles image with zeros (assuming background class is represented by 0)
    scribbles_image = np.zeros_like(multiclass_image, dtype=np.uint8)

    # Get unique class labels from the multiclass image, excluding the background if necessary
    class_labels = np.unique(multiclass_image)

    # Iterate over each unique class label to generate scribbles
    for class_label in class_labels:
        # Create a binary mask for the current class
        binary_mask = (multiclass_image == class_label).astype(np.uint8)
        # Generate positive scribbles for the current class
        # The function generate_contour_chunks needs to be defined previously
        positive_scribbles = generate_contour_chunks(binary_mask, max_chunk_length, min_chunk_length, thickness, border_margin, epsilon, inclusion_probability)
        
        # Merge the current class scribbles with the scribbles image
        scribbles_image[positive_scribbles > 0] = class_label

        # Find edges using Canny edge detection with fixed thresholds
        edges = cv2.Canny(binary_mask, 1, 1)

        # Create a mask where the edges are
        edge_mask = edges.astype(bool)

        # Set the pixels on the edge in the scribbles image to zero to create separation lines
        scribbles_image[edge_mask] = 0

    return scribbles_image


def generate_border_multiclass_scribbles(multiclass_image: np.ndarray, 
                                  border_scribble_max_chunk_length: int, 
                                  border_scribble_min_chunk_length: int, 
                                  border_scribble_border_margin: int, 
                                  border_scribble_epsilon: float, 
                                  border_scribble_inclusion_probability: float, 
                                  border_scribble_thickness: int,
                                  border_scribble_erode_iterations: int) -> np.ndarray:
    """
    Generate scribbles for each class in a multiclass image, including thinning line-like elements. 
    The scribbles are generated by extracting contour chunks from binary masks for each class 
    and applying a scribble effect, followed by thinning the lines using erosion.

    :param multiclass_image: A 2D numpy array where each pixel's value corresponds to a class label.
    :param max_chunk_length: The maximum length of any contour chunk for the scribble.
    :param min_chunk_length: The minimum length a contour must have to be considered for a scribble chunk.
    :param border_margin: The margin from the borders of the image within which contours will be ignored.
    :param epsilon: The approximation accuracy of the Ramer-Douglas-Peucker algorithm used in drawing scribbled lines.
    :param inclusion_probability: The probability that a found contour will be included as a chunk in the scribbles image.
    :param thickness: The thickness of the scribbles.
    :param erode_iterations: Number of iterations for the erosion process to thin the lines.
    :return: A 2D numpy array of the same shape as `multiclass_image`, where each pixel's value corresponds to a class label, with scribbles and thinned lines overlaid.
    """
    # Initialize the scribbles image with zeros (assuming background class is represented by 0)
    scribbles_image = np.zeros_like(multiclass_image, dtype=np.uint8)

    # Get unique class labels from the multiclass image, excluding the background if necessary
    class_labels = np.unique(multiclass_image)

    # Define the structuring element for erosion
    kernel = np.ones((3,3), np.uint8)

    # Iterate over each unique class label to generate scribbles
    for class_label in class_labels:
        # Create a binary mask for the current class
        binary_mask = (multiclass_image == class_label).astype(np.uint8)
        
        # Erode the binary mask to thin the lines
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=border_scribble_erode_iterations)

        # Generate positive scribbles for the current class
        # The function generate_contour_chunks needs to be defined previously
        positive_scribbles = generate_contour_chunks(eroded_mask, border_scribble_max_chunk_length, border_scribble_min_chunk_length, border_scribble_thickness, border_scribble_border_margin, border_scribble_epsilon, border_scribble_inclusion_probability)
        
        # Merge the current class scribbles with the scribbles image
        scribbles_image[positive_scribbles > 0] = class_label

        # Find edges using Canny edge detection with fixed thresholds
        edges = cv2.Canny(eroded_mask, 1, 1)

        # Create a mask where the edges are
        edge_mask = edges.astype(bool)

        # Set the pixels on the edge in the scribbles image to zero to create separation lines
        scribbles_image[edge_mask] = 0

    return scribbles_image

def generate_scribbles_inside_contours(
    multiclass_image: np.ndarray, 
    intern_scribble_num_scribbles_per_contour: int, 
    intern_scribble_max_scribble_length: int, 
    intern_scribble_thickness_range: Tuple[int, int],
    intern_scribble_edge_thickness: int
) -> np.ndarray:
    """
    Generates scribbles within the contours of class regions in a mask image and applies a buffer zone to the edges.

    Args:
        mask (np.ndarray): An image mask with different classes labeled as different integer values.
        num_scribbles_per_contour (int): The number of scribbles to generate per contour.
        max_scribble_length (int): The maximum length of each scribble.
        thickness_range (Tuple[int, int]): A tuple specifying the minimum and maximum thickness of the scribbles.
        edge_thickness (int): The thickness of the buffer zone around the edges of the contours.

    Returns:
        np.ndarray: An image array of the same shape as the input mask with scribbles drawn inside the contours 
        of the mask classes and a buffer around the edges of these contours.
    """
    scribbles = np.zeros_like(multiclass_image, dtype=np.uint8)
    classes = np.unique(multiclass_image)

    for class_id in classes:
        if class_id == 0:  # Skip background
            continue

        class_mask = (multiclass_image == class_id).astype(np.uint8)
        filled_contours = np.zeros_like(multiclass_image, dtype=np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled_contours, contours, -1, color=1, thickness=cv2.FILLED)

        for contour in contours:
            for _ in range(intern_scribble_num_scribbles_per_contour):
                # Generate random points inside the contour
                possible_points = np.argwhere(filled_contours == 1)
                if len(possible_points) > 0:
                    start_point_idx = random.choice(range(len(possible_points)))
                    start_point = tuple(possible_points[start_point_idx])

                    # Generate a random angle and length for the scribble
                    angle = random.uniform(0, 2 * np.pi)
                    length = random.randint(1, intern_scribble_max_scribble_length)
                    
                    # Calculate end point
                    end_x = int(start_point[1] + length * np.cos(angle))
                    end_y = int(start_point[0] + length * np.sin(angle))
                    end_point = (end_x, end_y)

                    # Check if the end point is inside the contour
                    if 0 <= end_x < multiclass_image.shape[1] and 0 <= end_y < multiclass_image.shape[0] and filled_contours[end_y, end_x] == 1:
                        # Draw the scribble
                        thickness = random.randint(*intern_scribble_thickness_range)
                        color  = int(class_id) 
                        cv2.line(scribbles, start_point[::-1], end_point, color, thickness)
                        
        edges = cv2.Canny(multiclass_image, 1, 1)
        dilated_edges = cv2.dilate(edges, np.ones((intern_scribble_edge_thickness, intern_scribble_edge_thickness), np.uint8))

    # Assign background color to the dilated edge areas in the scribbles image
    scribbles[dilated_edges > 0] = 0
    scribbles = np.where(multiclass_image == scribbles, scribbles, 0)

    return scribbles