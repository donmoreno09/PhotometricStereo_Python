# tests/conftest.py
"""
Common test fixtures and utilities for photometric-stereo-py.
"""

import os
import pytest
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def data_dir():
    """
    Returns the path to the test data directory.
    Creates it if it doesn't exist.
    """
    test_dir = Path(__file__).parent.resolve()
    data_dir = test_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture
def sample_normal_map():
    """
    Create a sample normal map for testing.
    
    Returns:
        np.ndarray: Normal map with shape (100, 100, 3)
    """
    # Create a hemisphere normal map
    h, w = 100, 100
    y, x = np.mgrid[:h, :w]
    x = (x - w/2) / (w/2)
    y = (y - h/2) / (h/2)
    z = np.sqrt(np.maximum(0, 1 - x*x - y*y))
    
    # Stack x, y, z into a normal map
    normal_map = np.stack([x, y, z], axis=2)
    
    # Normalize to unit length
    norms = np.sqrt(np.sum(normal_map**2, axis=2))
    normal_map[:,:,0] /= norms
    normal_map[:,:,1] /= norms
    normal_map[:,:,2] /= norms
    
    # Set NaNs to zero
    normal_map = np.nan_to_num(normal_map)
    
    return normal_map


@pytest.fixture
def sample_depth_map():
    """
    Create a sample depth map for testing.
    
    Returns:
        np.ndarray: Depth map with shape (100, 100)
    """
    # Create a parabolic depth map
    h, w = 100, 100
    y, x = np.mgrid[:h, :w]
    x = (x - w/2) / (w/2)
    y = (y - h/2) / (h/2)
    
    # z = 1 - x^2 - y^2
    depth_map = 1 - x*x - y*y
    
    return depth_map


@pytest.fixture
def sample_images():
    """
    Create a set of sample images for testing.
    
    Returns:
        list: List of numpy arrays, each representing an image
    """
    # Create 4 simple gradient images
    h, w = 100, 100
    images = []
    
    # Image 1: Gradient from left to right
    x = np.linspace(0, 1, w)
    img1 = np.tile(x, (h, 1))
    images.append(img1)
    
    # Image 2: Gradient from top to bottom
    y = np.linspace(0, 1, h)
    img2 = np.tile(y.reshape(-1, 1), (1, w))
    images.append(img2)
    
    # Image 3: Gradient from top-left to bottom-right
    img3 = (img1 + img2) / 2
    images.append(img3)
    
    # Image 4: Radial gradient from center
    y, x = np.mgrid[:h, :w]
    cx, cy = w/2, h/2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    img4 = 1 - np.clip(dist / (w/2), 0, 1)
    images.append(img4)
    
    return images


@pytest.fixture
def sample_light_directions():
    """
    Create a set of sample light directions for testing.
    
    Returns:
        np.ndarray: Light directions with shape (4, 3)
    """
    # Four light directions (normalized)
    directions = np.array([
        [1.0, 0.0, 1.0],  # From right
        [0.0, 1.0, 1.0],  # From bottom
        [1.0, 1.0, 1.0],  # From bottom-right
        [0.0, 0.0, 1.0]   # From directly above
    ])
    
    # Normalize
    norms = np.sqrt(np.sum(directions**2, axis=1))
    directions /= norms[:, np.newaxis]
    
    return directions


def are_arrays_close(a, b, rtol=1e-5, atol=1e-8):
    """
    Check if two arrays are close (similar to np.allclose but with more info).
    
    Returns:
        tuple: (is_close, abs_diff, rel_diff)
    """
    abs_diff = np.abs(a - b)
    abs_diff_max = np.max(abs_diff)
    
    # Calculate relative difference where values are significant
    significant = np.abs(b) > atol
    if np.any(significant):
        rel_diff = np.abs(a - b)[significant] / np.abs(b)[significant]
        rel_diff_max = np.max(rel_diff)
    else:
        rel_diff_max = 0
    
    is_close = np.allclose(a, b, rtol=rtol, atol=atol)
    
    return is_close, abs_diff_max, rel_diff_max
