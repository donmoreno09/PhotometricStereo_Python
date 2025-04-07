"""
Tests for normal map exporting.
Port of writeNormals.m tests and related functions
"""

import pytest
import numpy as np
import tempfile
import os
import cv2
from export.normals_exporter import (
    write_normals,
    write_reflection_map,
    write_albedo
)


def test_write_normals(tmp_path):
    """Test writing normal maps to file."""
    # Create a simple normal map
    h, w = 20, 20
    normal_map = np.zeros((h, w, 3))
    
    # Gradient in x and y
    y, x = np.mgrid[:h, :w]
    normal_map[:,:,0] = (x - w/2) / (w/2)  # -1 to 1 in x
    normal_map[:,:,1] = (y - h/2) / (h/2)  # -1 to 1 in y
    
    # Complete the normal vectors to unit length
    normal_map[:,:,2] = np.sqrt(np.maximum(0, 1 - normal_map[:,:,0]**2 - normal_map[:,:,1]**2))
    
    # Normalize
    norms = np.sqrt(np.sum(normal_map**2, axis=2))
    norms[norms == 0] = 1.0  # Avoid division by zero
    for i in range(3):
        normal_map[:,:,i] /= norms
    
    # Define output path
    output_path = tmp_path / "normal_map.png"
    
    # Write normal map
    write_normals(normal_map, output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the image
    img = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (h, w, 3)
    
    # In the PNG, normals should be mapped to [0, 255]
    # Check general pattern: x component increases left to right
    assert np.mean(img[h//2, 0, 0]) < np.mean(img[h//2, w-1, 0])
    
    # y component increases top to bottom
    assert np.mean(img[0, w//2, 1]) < np.mean(img[h-1, w//2, 1])


def test_write_reflection_map(tmp_path):
    """Test writing reflection maps to file."""
    # Create a simple normal map
    h, w = 20, 20
    normal_map = np.zeros((h, w, 3))
    
    # All normals point up
    normal_map[:,:,2] = 1.0
    
    # Define output path
    output_path = tmp_path / "reflection_map.png"
    
    # Write reflection map
    # We'll test with a default environment map
    write_reflection_map(normal_map, output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the image
    img = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (h, w, 3)
    
    # For uniform normal map, the reflection should also be relatively uniform
    # Check variance is low
    assert np.var(img.astype(float)) < 100


def test_write_albedo(tmp_path):
    """Test writing albedo maps to file."""
    # Create a simple albedo map with gradient
    h, w = 20, 20
    
    # Create gradient albedo
    y, x = np.mgrid[:h, :w]
    albedo = np.sqrt(((x - w/2) / (w/2))**2 + ((y - h/2) / (h/2))**2)
    # Normalize to [0, 1]
    albedo = albedo / np.max(albedo)
    
    # Define output path
    output_path = tmp_path / "albedo.png"
    
    # Write albedo
    write_albedo(albedo, output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the image
    img = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape == (h, w)
    
    # Check gradient pattern
    # Center should be darker than edges
    assert np.mean(img[h//2-2:h//2+2, w//2-2:w//2+2]) < np.mean(img)