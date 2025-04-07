"""
Functions for computing and manipulating normal maps
"""

import os
import numpy as np
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QCoreApplication

from core.photometric_stereo import PhotometricStereo
from utils.image_processing import normalize_normals, lay_normals


def compute_normal_maps(config, progress_callback=None):
    """
    Compute normal maps from input images.
    Equivalent to PSBoxComputeMaps1.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
            - input_images: List of file paths to the input images
            - light_directions: ndarray of light directions (3 x num_images)
            - mask: Binary mask of valid pixels (optional)
    progress_callback : function, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    dict
        Dictionary containing the computed maps:
            - normals: Normal map as ndarray (height, width, 3)
            - normals_rgb: Normal map converted to RGB for visualization
            - albedo: Albedo map as ndarray (height, width)
    """
    # Update progress
    if progress_callback:
        progress_callback(0)
        QCoreApplication.processEvents()
    
    # Load images
    images = load_images(config['input_images'])
    
    if progress_callback:
        progress_callback(10)
        QCoreApplication.processEvents()
    
    # Get light directions
    light_directions = config['light_directions']
    
    # Create mask if specified
    mask = config.get('mask', None)
    
    # Process with photometric stereo
    ps = PhotometricStereo(images, light_directions, mask)
    albedo, normals = ps.process()
    
    if progress_callback:
        progress_callback(60)
        QCoreApplication.processEvents()
    
    # Post-process normals
    # Convert from (3, height, width) to (height, width, 3) format
    normals_hwc = np.transpose(normals, (1, 2, 0))
    
    # Apply any normal map corrections if needed
    normals_hwc = lay_normals(normals_hwc)
    
    if progress_callback:
        progress_callback(80)
        QCoreApplication.processEvents()
    
    # Convert normals to RGB for visualization
    normals_rgb = normals_to_rgb(normals_hwc)
    
    if progress_callback:
        progress_callback(100)
    
    # Return results
    return {
        'normals': normals_hwc,
        'normals_rgb': normals_rgb,
        'albedo': albedo
    }


def load_images(image_paths):
    """
    Load images from file paths.
    
    Parameters
    ----------
    image_paths : list
        List of file paths to the input images
        
    Returns
    -------
    ndarray
        Stacked images, shape (height, width, num_images)
    """
    import cv2
    
    num_images = len(image_paths)
    
    # Load the first image to get dimensions
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    
    # Convert BGR to RGB if it's a color image
    if len(first_image.shape) == 3 and first_image.shape[2] == 3:
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    
    height, width = first_image.shape[:2]
    
    # Determine if images are color or grayscale
    if len(first_image.shape) == 2:
        # Grayscale
        images = np.zeros((height, width, num_images), dtype=np.float32)
        images[:, :, 0] = first_image.astype(np.float32)
    else:
        # Color - convert to grayscale for PS calculation
        images = np.zeros((height, width, num_images), dtype=np.float32)
        if first_image.shape[2] == 3:
            images[:, :, 0] = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            images[:, :, 0] = cv2.cvtColor(first_image[:,:,:3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Load remaining images
    for i in range(1, num_images):
        img = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
        
        # Check if image has the same dimensions
        if img.shape[:2] != (height, width):
            raise ValueError(f"Image {image_paths[i]} has dimensions {img.shape[:2]}, "
                            f"but expected {(height, width)}")
        
        # Convert to grayscale if needed
        if len(img.shape) == 2:
            images[:, :, i] = img.astype(np.float32)
        else:
            if img.shape[2] == 3:
                images[:, :, i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                images[:, :, i] = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Normalize images to [0, 1] range
    max_val = np.max(images)
    if max_val > 255:  # 16-bit images
        images /= 65535.0
    else:  # 8-bit images
        images /= 255.0
    
    return images


def normals_to_rgb(normals):
    """
    Convert normal vectors to RGB colors for visualization.
    
    Parameters
    ----------
    normals : ndarray
        Normal map, shape (height, width, 3) with values in [-1, 1]
        
    Returns
    -------
    ndarray
        RGB image, shape (height, width, 3) with values in [0, 255]
    """
    # Scale normals from [-1, 1] to [0, 1]
    rgb = (normals + 1) / 2.0
    
    # Scale to [0, 255] range and convert to uint8
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def compute_normal_maps(images, light_directions, method="standard"):
    """
    Compute normal map from images using photometric stereo
    
    Args:
        images: List of input images
        light_directions: List of light direction vectors
        method: Method to use for normal map computation
        
    Returns:
        tuple: (normal_map, albedo)
    """
    # Placeholder implementation
    if not images or not light_directions:
        raise ValueError("Images and light directions must not be empty")
        
    height, width = images[0].shape[:2]
    normal_map = np.zeros((height, width, 3))
    albedo = np.zeros((height, width))
    
    # In a real implementation, this would compute the normal map using photometric stereo
    
    return normal_map, albedo