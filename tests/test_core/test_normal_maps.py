"""
Tests for normal map computation.
Based on PSBoxComputeMaps1.m and related functions
"""

import pytest
import numpy as np
from core.normal_maps import compute_normal_map, compute_albedo


def test_compute_normal_map_flat():
    """Test normal map computation with a flat surface."""
    # Create images of a flat surface with different lighting
    h, w = 10, 10
    num_images = 3
    
    # All normals point up [0, 0, 1]
    # Light directions from different angles
    light_dirs = np.array([
        [0.0, 0.0, 1.0],     # From above
        [0.5, 0.0, 0.866],   # From right
        [0.0, 0.5, 0.866]    # From below
    ])
    
    # Uniform albedo
    albedo_true = np.ones((h, w))
    
    # Generate images
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # For flat surface with normal [0,0,1], intensity is dot product with light
        images[:, :, i] = albedo_true * np.dot([0, 0, 1], light_dirs[i])
    
    # Compute normal map
    normal_map = compute_normal_map(images, light_dirs)
    
    # Check that all normals point up
    assert np.allclose(normal_map[:,:,0], 0, atol=1e-10)  # x component
    assert np.allclose(normal_map[:,:,1], 0, atol=1e-10)  # y component
    assert np.allclose(normal_map[:,:,2], 1, atol=1e-10)  # z component


def test_compute_normal_map_sphere():
    """Test normal map computation with a spherical surface."""
    # Create images of a hemisphere with different lighting
    h, w = 20, 20
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
    
    # Light directions from different angles
    light_dirs = np.array([
        [0.0, 0.0, 1.0],     # From above
        [0.7, 0.0, 0.7],     # From right
        [0.0, 0.7, 0.7],     # From below
        [-0.7, 0.0, 0.7]     # From left
    ])
    
    # Uniform albedo
    albedo_true = np.ones((h, w))
    albedo_true[~mask] = 0   # Zero albedo outside the hemisphere
    
    # Generate images
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals_true * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo_true * np.maximum(0, n_dot_l)
    
    # Compute normal map
    normal_map = compute_normal_map(images, light_dirs, mask=mask)
    
    # Check results only in masked region
    # We take absolute values of both normals to account for possible sign flips
    assert np.allclose(np.abs(normal_map[mask]), np.abs(normals_true[mask]), atol=0.1)


def test_compute_albedo():
    """Test albedo computation."""
    # Create simple test scenario
    h, w = 10, 10
    
    # Define normals (all pointing up)
    normals = np.zeros((h, w, 3))
    normals[:,:,2] = 1.0
    
    # Define light directions
    light_dirs = np.array([
        [0.0, 0.0, 1.0],    # From above
        [0.5, 0.0, 0.866],  # From right
        [0.0, 0.5, 0.866]   # From below
    ])
    
    # True albedo varies linearly from left to right
    albedo_true = np.tile(np.linspace(0.5, 1.5, w), (h, 1))
    
    # Generate images
    num_images = len(light_dirs)
    images = np.zeros((h, w, num_images))
    
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo_true * n_dot_l
    
    # Compute albedo
    albedo = compute_albedo(images, normals, light_dirs)
    
    # Check results
    assert np.allclose(albedo, albedo_true, atol=1e-5)