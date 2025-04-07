"""
Tests for light direction refinement.
Port of PSRefineLightCostTest.m
"""

import pytest
import numpy as np
from algorithms.optimization.light_refinement import ps_refine_light, estimate_normals_albedo
from algorithms.optimization.nonlinear_least_squares import nonlinear_least_squares


def test_estimate_normals_albedo():
    """Test estimation of normals and albedo."""
    # Create simple test images
    h, w = 10, 10
    num_images = 3
    
    # Generate a simple normal map (all normals point up - [0,0,1])
    normals_true = np.zeros((h, w, 3))
    normals_true[:,:,2] = 1.0  # All normals point in z direction
    
    # Uniform albedo
    albedo_true = np.ones((h, w))
    
    # Simple light directions
    light_dirs = np.array([
        [0.0, 0.0, 1.0],    # From above
        [0.5, 0.0, 0.866],  # From right
        [0.0, 0.5, 0.866]   # From below
    ])
    
    # Generate images based on Lambertian model: I = albedo * max(0, n·l)
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals_true * light_dirs[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo_true * np.maximum(0, n_dot_l)
    
    # Call the function
    normal_map, albedo = estimate_normals_albedo(images, light_dirs)
    
    # Check results
    assert np.allclose(normal_map, normals_true, atol=1e-5)
    assert np.allclose(albedo, albedo_true, atol=1e-5)


def test_ps_refine_light():
    """Test light direction refinement."""
    # Create simple test images
    h, w = 10, 10
    num_images = 3
    
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
    
    # True light directions
    light_dirs_true = np.array([
        [0.0, 0.0, 1.0],    # From above
        [0.5, 0.0, 0.866],  # From right
        [0.0, 0.5, 0.866]   # From below
    ])
    
    # Generate images based on Lambertian model: I = albedo * max(0, n·l)
    images = np.zeros((h, w, num_images))
    for i in range(num_images):
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals_true * light_dirs_true[i].reshape(1, 1, 3), axis=2)
        images[:, :, i] = albedo_true * np.maximum(0, n_dot_l)
    
    # Perturb light directions as initial estimate
    np.random.seed(42)  # For reproducibility
    perturbation = np.random.normal(0, 0.1, light_dirs_true.shape)
    light_dirs_init = light_dirs_true + perturbation
    
    # Normalize perturbed directions
    for i in range(num_images):
        light_dirs_init[i] = light_dirs_init[i] / np.linalg.norm(light_dirs_init[i])
    
    # Call the light refinement function
    refined_light_dirs, refined_normals, refined_albedo = ps_refine_light(
        images, light_dirs_init, mask=mask, robust=False
    )
    
    # Check that refined directions are close to true directions
    for i in range(num_images):
        # We need to check for both possible orientations since the sign might flip
        alignment = np.abs(np.dot(refined_light_dirs[i], light_dirs_true[i]))
        assert alignment > 0.95  # At least 95% alignment
    
    # Check normals and albedo reconstruction
    # Only compare in the masked region
    assert np.allclose(np.abs(refined_normals[mask]), np.abs(normals_true[mask]), atol=0.2)
    assert np.allclose(refined_albedo[mask], albedo_true[mask], atol=0.2)