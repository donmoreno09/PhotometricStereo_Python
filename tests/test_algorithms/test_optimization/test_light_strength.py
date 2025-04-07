"""
Tests for light strength estimation.
Port of PSEstimateLightStrengthCostTest.m
"""

import pytest
import numpy as np
from algorithms.optimization.light_strength import ps_estimate_light_strength_cost
from core.light_direction import ps_estimate_light_strength


def test_ps_estimate_light_strength_cost():
    """Test the cost function for light strength estimation."""
    # Create simple test setup
    h, w = 5, 5
    num_images = 3
    
    # Simple normal map (all normals point up)
    normals = np.zeros((h, w, 3))
    normals[:,:,2] = 1.0
    
    # Uniform albedo
    albedo = np.ones((h, w))
    
    # Light directions (all normalized)
    light_dirs = np.array([
        [0.0, 0.0, 1.0],    # From above
        [0.5, 0.0, 0.866],  # From right
        [0.0, 0.5, 0.866]   # From below
    ])
    
    # True light strengths
    true_strengths = np.array([1.0, 0.8, 0.6])
    
    # Generate images: I = albedo * (n·l) * light_strength
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo * np.maximum(0, n_dot_l) * true_strengths[i]
    
    # Test the cost function
    # Start with all strengths = 1.0
    test_strengths = np.ones(num_images)
    error = ps_estimate_light_strength_cost(test_strengths, images, normals, albedo, light_dirs)
    
    # Check that error has the correct shape
    expected_shape = (h * w * num_images,)
    assert error.shape == expected_shape
    
    # The error should be zero for test_strengths == true_strengths
    error_true = ps_estimate_light_strength_cost(true_strengths, images, normals, albedo, light_dirs)
    assert np.allclose(error_true, 0.0, atol=1e-10)


def test_ps_estimate_light_strength():
    """Test the light strength estimation function."""
    # Create simple test setup
    h, w = 8, 8
    num_images = 3
    
    # Generate a hemisphere normal map
    y, x = np.mgrid[:h, :w]
    x = (x - w/2) / (w/2)  # Map to [-1, 1]
    y = (y - h/2) / (h/2)  # Map to [-1, 1]
    
    # For points inside unit circle, compute z for a hemisphere
    r_squared = x**2 + y**2
    normals = np.zeros((h, w, 3))
    mask = r_squared <= 1.0
    
    # Normal vectors for a hemisphere
    normals[mask, 0] = x[mask]
    normals[mask, 1] = y[mask]
    normals[mask, 2] = np.sqrt(1.0 - r_squared[mask])
    
    # Normalize to unit length
    norms = np.sqrt(np.sum(normals**2, axis=2))
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals[:,:,0] /= norms
    normals[:,:,1] /= norms
    normals[:,:,2] /= norms
    
    # Uniform albedo in the valid region
    albedo = np.zeros((h, w))
    albedo[mask] = 1.0
    
    # Light directions (all normalized)
    light_dirs = np.array([
        [0.0, 0.0, 1.0],    # From above
        [0.5, 0.0, 0.866],  # From right
        [0.0, 0.5, 0.866]   # From below
    ])
    
    # True light strengths
    true_strengths = np.array([1.0, 0.8, 0.6])
    
    # Generate images: I = albedo * (n·l) * light_strength
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo * np.maximum(0, n_dot_l) * true_strengths[i]
    
    # Test the light strength estimation
    estimated_strengths = ps_estimate_light_strength(images, normals, albedo, light_dirs, mask)
    
    # Check that estimated strengths are close to true strengths
    assert np.allclose(estimated_strengths, true_strengths, atol=0.05)
    
    # Test without mask
    estimated_strengths_no_mask = ps_estimate_light_strength(images, normals, albedo, light_dirs)
    assert np.allclose(estimated_strengths_no_mask, true_strengths, atol=0.1)