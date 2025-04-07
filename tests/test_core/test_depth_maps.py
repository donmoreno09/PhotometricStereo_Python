"""
Tests for depth map reconstruction from normal maps.
"""

import pytest
import numpy as np
from core.depth_maps import compute_depth_map
from utils.image_processing import poly_correction


def test_compute_depth_map_simple():
    """Test depth map computation with a simple gradient."""
    # Create a simple 10x10 normal map
    h, w = 10, 10
    
    # Normals for a plane with constant gradient
    normals = np.zeros((h, w, 3))
    normals[:,:,0] = -0.1  # dz/dx = -0.1
    normals[:,:,1] = -0.2  # dz/dy = -0.2
    normals[:,:,2] = 1.0   # Pointing mostly up
    
    # Normalize normal vectors
    norms = np.sqrt(np.sum(normals**2, axis=2))
    normals[:,:,0] /= norms
    normals[:,:,1] /= norms
    normals[:,:,2] /= norms
    
    # Expected depth: z = 0.1*x + 0.2*y + constant
    y, x = np.mgrid[:h, :w]
    expected_depth = 0.1 * x + 0.2 * y
    expected_depth -= np.mean(expected_depth)  # Center around zero
    
    # Compute depth map
    depth_map = compute_depth_map(normals)
    
    # Remove any constant offset (integration constant)
    depth_map -= np.mean(depth_map)
    
    # Check results
    assert np.allclose(depth_map, expected_depth, atol=0.05)


def test_compute_depth_map_sphere():
    """Test depth map computation with a hemisphere."""
    # Create a 20x20 normal map for a hemisphere
    h, w = 20, 20
    
    # Generate normals for a hemisphere
    y, x = np.mgrid[:h, :w]
    x = (x - w/2) / (w/2)  # Map to [-1, 1]
    y = (y - h/2) / (h/2)  # Map to [-1, 1]
    
    # For points inside unit circle, compute hemisphere normals
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
    
    # Expected depth for hemisphere: z = sqrt(1 - x^2 - y^2)
    expected_depth = np.zeros((h, w))
    expected_depth[mask] = np.sqrt(1.0 - r_squared[mask])
    
    # Compute depth map
    depth_map = compute_depth_map(normals, mask=mask)
    
    # Scale to match expected depth (since integration might introduce scaling)
    if np.sum(depth_map[mask]) > 0:  # Avoid division by zero
        scale = np.sum(expected_depth[mask]) / np.sum(depth_map[mask])
        depth_map *= scale
    
    # Check results only in the masked region
    assert np.allclose(depth_map[mask], expected_depth[mask], atol=0.2)


def test_poly_correction():
    """Test polynomial correction for depth maps."""
    # Create a depth map with a global trend
    h, w = 20, 20
    y, x = np.mgrid[:h, :w]
    
    # Base depth map: a paraboloid z = x^2 + y^2
    # Center coordinates
    cx, cy = w/2, h/2
    
    # Normalized coordinates [-1, 1]
    x_norm = (x - cx) / (w/2)
    y_norm = (y - cy) / (h/2)
    
    # Create depth map with quadratic trend
    depth_map = x_norm**2 + y_norm**2
    
    # Local feature (a small bump)
    bump_x, bump_y = 12, 12
    radius = 3
    mask = ((x - bump_x)**2 + (y - bump_y)**2) <= radius**2
    depth_map[mask] += 0.3 * np.exp(-((x[mask] - bump_x)**2 + (y[mask] - bump_y)**2) / (radius/2)**2)
    
    # Apply polynomial correction (order 2 to remove quadratic trend)
    corrected_depth = poly_correction(depth_map, order=2)
    
    # In the corrected map, the quadratic trend should be gone
    # Only the bump should remain
    
    # Test if the overall trend is flat (mean close to zero)
    assert np.isclose(np.mean(corrected_depth), 0, atol=0.05)
    
    # Test if the bump is still there
    # Find max location in corrected depth
    max_idx = np.unravel_index(np.argmax(corrected_depth), corrected_depth.shape)
    bump_found = np.sqrt((max_idx[1] - bump_x)**2 + (max_idx[0] - bump_y)**2) <= radius
    assert bump_found
    
    # Test if the planar correction works too
    planar_trend = 0.1 * x_norm + 0.2 * y_norm
    depth_with_plane = depth_map + planar_trend
    
    corrected_depth_plane = poly_correction(depth_with_plane, order=1)
    
    # The result should be similar to the quadratic case
    # since we're only removing the linear trend
    assert np.isclose(np.mean(corrected_depth_plane), 0, atol=0.05)