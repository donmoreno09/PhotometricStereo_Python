"""
Tests for image processing utilities.
"""

import pytest
import numpy as np
import cv2
from utils.image_processing import (
    decode_rgb_to_normals,
    compute_images_mean, 
    check_bit_depth,
    normalize_normal, 
    specularize_x,
    remap_range,
    rgb_from_normals
)


def test_decode_rgb_to_normals():
    """Test decoding RGB to normals."""
    # Create a simple RGB normal map
    h, w = 10, 10
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Let's set different regions with different normals
    # RGB = [128, 128, 255] -> Normal = [0, 0, 1] (up)
    rgb_image[:5, :5, 0] = 128
    rgb_image[:5, :5, 1] = 128
    rgb_image[:5, :5, 2] = 255
    
    # RGB = [255, 128, 128] -> Normal = [1, 0, 0.5] (right and up)
    rgb_image[:5, 5:, 0] = 255
    rgb_image[:5, 5:, 1] = 128
    rgb_image[:5, 5:, 2] = 128
    
    # RGB = [128, 255, 128] -> Normal = [0, 1, 0.5] (down and up)
    rgb_image[5:, :5, 0] = 128
    rgb_image[5:, :5, 1] = 255
    rgb_image[5:, :5, 2] = 128
    
    # RGB = [0, 0, 0] -> Normal = [-1, -1, 0] (left and up)
    rgb_image[5:, 5:, 0] = 0
    rgb_image[5:, 5:, 1] = 0
    rgb_image[5:, 5:, 2] = 0
    
    # Decode to normals
    normals = decode_rgb_to_normals(rgb_image)
    
    # Check dimensions
    assert normals.shape == (h, w, 3)
    
    # Check values
    # For [128, 128, 255] -> [0, 0, 1]
    assert np.allclose(normals[0, 0], [0, 0, 1], atol=0.1)
    
    # For [255, 128, 128] -> Normalized [1, 0, 0.5]
    normal = np.array([1, 0, 0.5])
    normal = normal / np.linalg.norm(normal)
    assert np.allclose(normals[0, 9], normal, atol=0.2)
    
    # For [128, 255, 128] -> Normalized [0, 1, 0.5]
    normal = np.array([0, 1, 0.5])
    normal = normal / np.linalg.norm(normal)
    assert np.allclose(normals[9, 0], normal, atol=0.2)
    
    # For [0, 0, 0] -> Normalized [-1, -1, 0]
    # In this case, all components are 0, so the vector can't be normalized
    assert np.allclose(normals[9, 9], [0, 0, 0], atol=0.1)


def test_compute_images_mean():
    """Test computing mean of images."""
    # Create a few test images
    h, w = 5, 5
    images = []
    
    # Image 1: All ones
    images.append(np.ones((h, w)))
    
    # Image 2: All twos
    images.append(np.ones((h, w)) * 2)
    
    # Image 3: All threes
    images.append(np.ones((h, w)) * 3)
    
    # Compute mean without weights
    mean_image = compute_images_mean(images)
    assert np.allclose(mean_image, 2)  # (1 + 2 + 3) / 3 = 2
    
    # Compute mean with weights
    weights = [1, 2, 1]  # Emphasize the middle image
    mean_image = compute_images_mean(images, weights)
    assert np.allclose(mean_image, 2)  # (1*1 + 2*2 + 1*3) / 4 = 2
    
    # Test with color images
    color_images = []
    color_images.append(np.ones((h, w, 3)))  # [1, 1, 1]
    color_images.append(np.array([2, 0, 0]).reshape(1, 1, 3) * np.ones((h, w, 1)))  # [2, 0, 0]
    color_images.append(np.array([0, 3, 0]).reshape(1, 1, 3) * np.ones((h, w, 1)))  # [0, 3, 0]
    
    # Compute color mean without weights
    mean_color = compute_images_mean(color_images)
    assert np.allclose(mean_color, np.array([1, 4/3, 1/3]))  # Mean of each channel
    
    # Compute color mean with weights
    weights = [1, 2, 3]
    mean_color = compute_images_mean(color_images, weights)
    expected = (1*np.array([1, 1, 1]) + 2*np.array([2, 0, 0]) + 3*np.array([0, 3, 0])) / 6
    assert np.allclose(mean_color, expected)


def test_check_bit_depth():
    """Test bit depth checking."""
    # Binary image
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[5:, 5:] = 1
    assert check_bit_depth(binary) == 1
    
    # 8-bit image
    img_8bit = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    assert check_bit_depth(img_8bit) == 8
    
    # 16-bit image
    img_16bit = np.random.randint(0, 65536, size=(10, 10), dtype=np.uint16)
    assert check_bit_depth(img_16bit) == 16
    
    # Floating point normalized image
    img_float = np.random.random((10, 10)).astype(np.float32)
    assert check_bit_depth(img_float) == 8
    
    # Floating point high dynamic range image
    img_hdr = (np.random.random((10, 10)) * 10).astype(np.float32)
    assert check_bit_depth(img_hdr) == 16


def test_normalize_normal():
    """Test normal vector normalization."""
    # Create some test normal vectors
    normals = np.array([
        [1.0, 0.0, 0.0],    # Already unit length
        [2.0, 0.0, 0.0],    # Length 2
        [0.0, 0.0, 0.0],    # Zero vector
        [3.0, 4.0, 0.0]     # Length 5
    ])
    
    # Normalize
    normalized = normalize_normal(normals)
    
    # Check results
    assert np.allclose(normalized[0], [1, 0, 0])  # Already normalized
    assert np.allclose(normalized[1], [1, 0, 0])  # Should be normalized
    assert np.allclose(normalized[2], [0, 0, 0])  # Zero stays zero
    assert np.allclose(normalized[3], [0.6, 0.8, 0])  # 3/5, 4/5, 0


def test_specularize_x():
    """Test horizontal flipping."""
    # Create a test array
    arr = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # Flip horizontally
    flipped = specularize_x(arr)
    
    # Expected result
    expected = np.array([
        [3, 2, 1],
        [6, 5, 4],
        [9, 8, 7]
    ])
    
    # Check result
    assert np.array_equal(flipped, expected)


def test_remap_range():
    """Test range remapping."""
    # Create test array
    arr = np.array([-1.0, 0.0, 1.0, 2.0])
    
    # Remap from [-1, 2] to [0, 1]
    remapped = remap_range(arr, -1, 2, 0, 1)
    expected = np.array([0.0, 1/3, 2/3, 1.0])
    assert np.allclose(remapped, expected)
    
    # Remap to [0, 255] (uint8)
    remapped = remap_range(arr, -1, 2, 0, 255, dtype=np.uint8)
    expected = np.array([0, 85, 170, 255], dtype=np.uint8)
    assert np.array_equal(remapped, expected)


def test_rgb_from_normals():
    """Test conversion from normals to RGB."""
    # Create a normal map
    normals = np.zeros((3, 3, 3))
    
    # Up: [0, 0, 1] -> [128, 128, 255]
    normals[0, 0] = [0, 0, 1]
    
    # Right: [1, 0, 0] -> [255, 128, 0]
    normals[0, 1] = [1, 0, 0]
    
    # Down: [0, 1, 0] -> [128, 255, 0]
    normals[0, 2] = [0, 1, 0]
    
    # Left: [-1, 0, 0] -> [0, 128, 0]
    normals[1, 0] = [-1, 0, 0]
    
    # Mixed directions
    normals[1, 1] = [0.707, 0.707, 0]  # 45 degrees
    normals[1, 2] = [0.577, 0.577, 0.577]  # Equal in all directions
    
    # Convert to RGB
    rgb = rgb_from_normals(normals)
    
    # Check dimensions and type
    assert rgb.shape == (3, 3, 3)
    assert rgb.dtype == np.uint8
    
    # Check some values
    # For [0, 0, 1] -> [128, 128, 255]
    assert np.allclose(rgb[0, 0], [128, 128, 255], atol=2)
    
    # For [1, 0, 0] -> [255, 128, 0]
    assert np.allclose(rgb[0, 1], [255, 128, 0], atol=2)
    
    # For [0, 1, 0] -> [128, 255, 0]
    assert np.allclose(rgb[0, 2], [128, 255, 0], atol=2)
    
    # For [-1, 0, 0] -> [0, 128, 0]
    assert np.allclose(rgb[1, 0], [0, 128, 0], atol=2)