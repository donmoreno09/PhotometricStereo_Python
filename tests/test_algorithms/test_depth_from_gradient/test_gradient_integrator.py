"""
Tests for depth from gradient integration.
Port of DepthFromGradientTest.m and DfGBoxTest.m
"""

import pytest
import numpy as np
from algorithms.depth_from_gradient.gradient_integrator import depth_from_gradient


def test_depth_from_gradient_simple():
    """Test depth from gradient on a simple case."""
    # Create a simple 5x5 gradient field
    h, w = 5, 5
    
    # Gradient components (constant gradients)
    grad_x = np.ones((h, w)) * 0.1  # Constant slope in x
    grad_y = np.ones((h, w)) * 0.2  # Constant slope in y
    
    # Expected depth: z = 0.1*x + 0.2*y + c
    y, x = np.mgrid[:h, :w]
    expected_depth = 0.1 * x + 0.2 * y
    
    # Compute depth from gradient
    result = depth_from_gradient(grad_x, grad_y)
    
    # Remove mean to match expected depth (constant offset doesn't matter)
    result = result - np.mean(result) + np.mean(expected_depth)
    
    # Check results
    assert np.allclose(result, expected_depth, atol=1e-10)


def test_depth_from_gradient_circle():
    """Test depth from gradient on a hemisphere."""
    # Create a hemisphere gradient field
    h, w = 20, 20
    y, x = np.mgrid[:h, :w]
    
    # Center coordinates
    cx, cy = w/2 - 0.5, h/2 - 0.5
    
    # Normalized coordinates [-1, 1]
    x_norm = (x - cx) / (w/2)
    y_norm = (y - cy) / (h/2)
    
    # Radius from center
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Generate a hemisphere
    # z = sqrt(1 - x^2 - y^2) for points inside the unit circle
    z = np.zeros((h, w))
    mask = r <= 1
    z[mask] = np.sqrt(1 - x_norm[mask]**2 - y_norm[mask]**2)
    
    # Compute analytical gradients
    grad_x = np.zeros((h, w))
    grad_y = np.zeros((h, w))
    
    # dz/dx = -x / sqrt(1 - x^2 - y^2)
    # dz/dy = -y / sqrt(1 - x^2 - y^2)
    non_zero = mask & (z > 0)
    grad_x[non_zero] = -x_norm[non_zero] / z[non_zero]
    grad_y[non_zero] = -y_norm[non_zero] / z[non_zero]
    
    # Compute depth from gradient
    reconstructed_depth = depth_from_gradient(grad_x, grad_y)
    
    # Remove mean to match expected depth
    reconstructed_depth = reconstructed_depth - np.mean(reconstructed_depth[mask]) + np.mean(z[mask])
    
    # Check results (only for points inside the unit circle)
    assert np.allclose(reconstructed_depth[mask], z[mask], atol=0.1)


def test_depth_from_gradient_with_mask():
    """Test depth from gradient with a mask."""
    # Create a simple gradient field
    h, w = 10, 10
    grad_x = np.ones((h, w)) * 0.1
    grad_y = np.ones((h, w)) * 0.2
    
    # Create a mask (only use central region)
    mask = np.zeros((h, w), dtype=bool)
    mask[2:8, 2:8] = True
    
    # Expected depth in the masked region
    y, x = np.mgrid[2:8, 2:8]
    expected_depth_masked = 0.1 * x + 0.2 * y
    
    # Compute depth from gradient with mask
    result = depth_from_gradient(grad_x, grad_y, mask=mask)
    
    # Extract masked region
    result_masked = result[2:8, 2:8]
    
    # Remove mean to match expected depth
    result_masked = result_masked - np.mean(result_masked) + np.mean(expected_depth_masked)
    
    # Check results
    assert np.allclose(result_masked, expected_depth_masked, atol=1e-10)


def test_depth_from_gradient_methods():
    """Test different integration methods."""
    # Create a simple gradient field
    h, w = 10, 10
    grad_x = np.ones((h, w)) * 0.1
    grad_y = np.ones((h, w)) * 0.2
    
    # Expected depth: z = 0.1*x + 0.2*y + c
    y, x = np.mgrid[:h, :w]
    expected_depth = 0.1 * x + 0.2 * y
    
    # Test different methods
    methods = ['poisson', 'frankot-chellappa']
    
    for method in methods:
        # Compute depth from gradient
        result = depth_from_gradient(grad_x, grad_y, method=method)
        
        # Remove mean to match expected depth
        result = result - np.mean(result) + np.mean(expected_depth)
        
        # Check results
        assert np.allclose(result, expected_depth, atol=0.1)