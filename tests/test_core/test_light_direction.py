"""
Tests for light direction computation.
Port of psBoxComputeLights.m and FindLightDirectionFromChromeSphere.m tests
"""

import pytest
import numpy as np
import cv2
from core.light_direction import (
    ps_compute_lights, 
    find_light_direction_from_chrome_sphere,
    fit_light_probe_circle
)


def create_chrome_sphere_image(light_dir, center=(50, 50), radius=40):
    """Create a synthetic image of a chrome sphere with highlight."""
    # Normalize light direction
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Create a black image
    h, w = 100, 100
    image = np.zeros((h, w), dtype=np.float32)
    
    # Create a meshgrid for pixel coordinates
    y, x = np.mgrid[:h, :w]
    
    # Convert center to numpy array
    cx, cy = center
    
    # Calculate distance from center for each pixel
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Mask for pixels inside the sphere
    mask = dist <= radius
    
    # For each point inside the sphere, calculate the normal vector
    normals = np.zeros((h, w, 3))
    normals[mask, 0] = (x[mask] - cx) / radius
    normals[mask, 1] = (y[mask] - cy) / radius
    
    # Calculate z-component of the normal vector
    normals[mask, 2] = np.sqrt(1.0 - normals[mask, 0]**2 - normals[mask, 1]**2)
    
    # Add base intensity for the sphere
    image[mask] = 0.2
    
    # Find the highlight
    highlight_x, highlight_y = 0, 0
    max_dot = -1
    
    # For each point, find where normal aligns with light direction
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                # Calculate dot product with light direction
                dot = np.dot(normals[i, j], light_dir)
                if dot > max_dot:
                    max_dot = dot
                    highlight_y, highlight_x = i, j
    
    # Add highlight
    cv2.circle(image, (highlight_x, highlight_y), 3, 1.0, -1)
    
    return image


def test_find_light_direction_from_chrome_sphere():
    """Test finding light direction from a chrome sphere image."""
    # Create a synthetic sphere image with known light direction
    light_dir_true = np.array([0.5, 0.3, 0.8])
    light_dir_true = light_dir_true / np.linalg.norm(light_dir_true)
    
    center = (50, 50)
    radius = 40
    
    image = create_chrome_sphere_image(light_dir_true, center, radius)
    
    # Find light direction
    light_dir = find_light_direction_from_chrome_sphere(image, center, radius)
    
    # Check that the direction is close to the true direction
    # (dot product should be close to 1)
    alignment = np.abs(np.dot(light_dir, light_dir_true))
    assert alignment > 0.95  # At least 95% alignment


def test_fit_light_probe_circle():
    """Test fitting a circle to a chrome sphere in an image."""
    # Create a synthetic sphere image
    light_dir = np.array([0, 0, 1])  # Light from directly above
    center_true = (50, 50)
    radius_true = 40
    
    image = create_chrome_sphere_image(light_dir, center_true, radius_true)
    
    # Add some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Convert to uint8 for circle detection
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Fit circle
    center, radius = fit_light_probe_circle(image_uint8)
    
    # Check that the circle is close to the true circle
    assert abs(center[0] - center_true[0]) < 3
    assert abs(center[1] - center_true[1]) < 3
    assert abs(radius - radius_true) < 3


def test_ps_compute_lights():
    """Test computing light directions for multiple images."""
    # Create multiple synthetic sphere images with different light directions
    light_dirs_true = [
        [0, 0, 1],      # From directly above
        [0.5, 0, 0.866],  # From the right
        [0, 0.5, 0.866]   # From below
    ]
    
    center = (50, 50)
    radius = 40
    
    images = [
        create_chrome_sphere_image(np.array(ldir), center, radius)
        for ldir in light_dirs_true
    ]
    
    # Convert to uint8 for processing
    images_uint8 = [(img * 255).astype(np.uint8) for img in images]
    
    # Compute light directions
    light_dirs = ps_compute_lights(images_uint8, center, radius)
    
    # Check that each computed direction is close to the true direction
    for i, true_dir in enumerate(light_dirs_true):
        # Normalize true direction
        true_dir = np.array(true_dir) / np.linalg.norm(true_dir)
        
        # Calculate cosine similarity
        similarity = np.abs(np.dot(light_dirs[i], true_dir))
        
        # Should be close to 1 (vectors pointing in same direction)
        assert similarity > 0.95