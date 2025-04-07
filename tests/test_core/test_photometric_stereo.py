"""
Tests for the photometric stereo algorithm.
Port of PSBoxTest.m
"""

import pytest
import numpy as np
from core.photometric_stereo import photometric_stereo


def test_photometric_stereo_simple():
    """Test photometric stereo with a simple setup."""
    # Create simple test images
    h, w = 10, 10
    num_images = 4
    
    # Generate a hemisphere normal map
    y, x = np.mgrid[:h, :w]
    x = (x - w/2) / (w/2)  # Map to [-1, 1]
    y = (y - h/2) / (h/2)  # Map to [-1, 1]
    
    # For points inside unit circle, compute z for a hemisphere
    r_squared = x**2 + y**2
    normals_true = np.zeros((h, w, 3))
    mask = r_squared <= 1.0
    
    # Normal vectors for a hemisphere
    normals_true[mask, 0] = x[mask]
    normals_true[mask, 1] = y[mask]
    normals_true[mask, 2] = np.sqrt(1.0 - r_squared[mask])
    
    # Normalize to unit length
    norms = np.sqrt(np.sum(normals_true**2, axis=2))
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals_true[:,:,0] /= norms
    normals_true[:,:,1] /= norms
    normals_true[:,:,2] /= norms
    
    # Uniform albedo in the valid region
    albedo_true = np.zeros((h, w))
    albedo_true[mask] = 1.0
    
    # Light directions from different angles
    light_dirs = np.array([
        [0.0, 0.0, 1.0],     # From above
        [0.5, 0.0, 0.866],   # From right
        [0.0, 0.5, 0.866],   # From bottom
        [-0.5, 0.0, 0.866]   # From left
    ])
    
    # Generate images based on Lambertian model: I = albedo * max(0, n·l)
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals_true * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo_true * np.maximum(0, n_dot_l)
    
    # Run photometric stereo
    result = photometric_stereo(images, light_dirs, mask=mask)
    
    # Check that we get a result
    assert len(result) >= 2
    normal_map = result["normal_map"]
    albedo = result["albedo"]
    
    # Check normal map dimensions
    assert normal_map.shape == (h, w, 3)
    
    # Check albedo dimensions
    assert albedo.shape == (h, w)
    
    # Check normal map values within mask
    # Note: We take absolute values because normals might be flipped depending on implementation
    normal_aligned = np.abs(normals_true * np.sign(normals_true[:,:,2:3]))
    normal_result_aligned = np.abs(normal_map * np.sign(normal_map[:,:,2:3]))
    assert np.allclose(normal_result_aligned[mask], normal_aligned[mask], atol=0.1)
    
    # Check albedo values within mask
    # Adjust for any scaling differences
    albedo_scale = np.mean(albedo[mask]) / np.mean(albedo_true[mask])
    assert np.allclose(albedo[mask] / albedo_scale, albedo_true[mask], atol=0.1)


def test_photometric_stereo_with_noise():
    """Test photometric stereo with noisy images."""
    # Create simple test images
    h, w = 10, 10
    num_images = 4
    
    # Generate a simple normal map (all normals point up)
    normals_true = np.zeros((h, w, 3))
    normals_true[:,:,2] = 1.0
    
    # Uniform albedo
    albedo_true = np.ones((h, w))
    
    # Light directions from different angles
    light_dirs = np.array([
        [0.0, 0.0, 1.0],     # From above
        [0.5, 0.0, 0.866],   # From right
        [0.0, 0.5, 0.866],   # From bottom
        [-0.5, 0.0, 0.866]   # From left
    ])
    
    # Generate images with noise
    np.random.seed(42)  # For reproducibility
    images = np.zeros((h, w, num_images))
    noise_level = 0.1
    
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals_true * light_dirs[i].reshape(1, 1, 3), axis=2)
        clean_image = albedo_true * np.maximum(0, n_dot_l)
        noise = np.random.normal(0, noise_level, size=(h, w))
        images[:, :, i] = np.maximum(0, clean_image + noise)
    
    # Run photometric stereo
    result = photometric_stereo(images, light_dirs)
    
    # Check that we get a result
    assert len(result) >= 2
    normal_map = result["normal_map"]
    albedo = result["albedo"]
    
    # Check normal map dimensions
    assert normal_map.shape == (h, w, 3)
    
    # Check that the normal vectors are close to [0,0,1]
    # Due to noise, we use a relaxed tolerance
    assert np.allclose(normal_map[:,:,0], 0.0, atol=0.2)
    assert np.allclose(normal_map[:,:,1], 0.0, atol=0.2)
    assert np.allclose(normal_map[:,:,2], 1.0, atol=0.2)
    
    # Check that albedo is close to 1.0
    assert np.allclose(albedo, 1.0, atol=0.2)


def test_photometric_stereo_robust():
    """Test robust photometric stereo with outliers."""
    # Create simple test images
    h, w = 10, 10
    num_images = 5  # Use 5 images, one will be an outlier
    
    # Generate a simple normal map (all normals point up)
    normals_true = np.zeros((h, w, 3))
    normals_true[:,:,2] = 1.0
    
    # Uniform albedo
    albedo_true = np.ones((h, w))
    
    # Light directions from different angles
    light_dirs = np.array([
        [0.0, 0.0, 1.0],     # From above
        [0.5, 0.0, 0.866],   # From right
        [0.0, 0.5, 0.866],   # From bottom
        [-0.5, 0.0, 0.866],  # From left
        [0.0, -0.5, 0.866]   # From top
    ])
    
    # Generate clean images for 4 directions
    images = np.zeros((h, w, num_images))
    for i in range(num_images - 1):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals_true * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo_true * np.maximum(0, n_dot_l)
    
    # Last image is an outlier (e.g., strong specular highlight)
    images[:, :, -1] = 2.0  # Much brighter than expected
    
    # Run standard photometric stereo
    result_standard = photometric_stereo(images, light_dirs, robust=False)
    
    # Run robust photometric stereo
    result_robust = photometric_stereo(images, light_dirs, robust=True)
    
    # The robust estimation should be closer to the true normals
    normal_standard = result_standard["normal_map"]
    normal_robust = result_robust["normal_map"]
    
    # Calculate error for standard and robust methods
    error_standard = np.mean((normal_standard[:,:,2] - normals_true[:,:,2])**2)
    error_robust = np.mean((normal_robust[:,:,2] - normals_true[:,:,2])**2)
    
    # Robust method should have lower error
    assert error_robust < error_standard