"""
Tests for depth map exporting.
Port of writeDepthMap.m test functionality
"""

import pytest
import numpy as np
import tempfile
import os
from export.depth_exporter import write_depth_map


def test_write_depth_map_png(tmp_path):
    """Test writing depth maps to PNG file."""
    # Create a simple depth map
    h, w = 20, 20
    
    # Create a parabolic depth map
    y, x = np.mgrid[:h, :w]
    depth_map = ((x - w/2) / (w/2))**2 + ((y - h/2) / (h/2))**2
    # Normalize to [0, 1]
    depth_map = depth_map / np.max(depth_map)
    
    # Define output path
    output_path = tmp_path / "depth_map.png"
    
    # Write depth map
    write_depth_map(depth_map, output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # For PNG, just check if file is created successfully
    # Reading back would require handling value normalization


def test_write_depth_map_exr(tmp_path):
    """Test writing depth maps to EXR file."""
    # Skip if OpenEXR is not available
    try:
        import OpenEXR
        import Imath
    except ImportError:
        pytest.skip("OpenEXR is not installed, skipping EXR tests")
    
    # Create a simple depth map
    h, w = 20, 20
    
    # Create a parabolic depth map
    y, x = np.mgrid[:h, :w]
    depth_map = ((x - w/2) / (w/2))**2 + ((y - h/2) / (h/2))**2
    
    # Define output path
    output_path = tmp_path / "depth_map.exr"
    
    # Write depth map
    write_depth_map(depth_map, output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)


def test_write_depth_map_16bit(tmp_path):
    """Test writing depth maps to 16-bit PNG."""
    # Create a simple depth map
    h, w = 20, 20
    
    # Create a parabolic depth map
    y, x = np.mgrid[:h, :w]
    depth_map = ((x - w/2) / (w/2))**2 + ((y - h/2) / (h/2))**2
    # Scale to wider range
    depth_map = depth_map * 1000
    
    # Define output path
    output_path = tmp_path / "depth_map_16bit.png"
    
    # Write depth map with 16-bit option
    write_depth_map(depth_map, output_path, bit_depth=16)
    
    # Check that file was created
    assert os.path.exists(output_path)


def test_write_depth_map_obj(tmp_path):
    """Test writing depth maps to OBJ file."""
    # Create a simple depth map
    h, w = 10, 10  # Smaller for OBJ to keep test fast
    
    # Create a parabolic depth map
    y, x = np.mgrid[:h, :w]
    depth_map = ((x - w/2) / (w/2))**2 + ((y - h/2) / (h/2))**2
    # Scale for better visualization
    depth_map = depth_map * 2
    
    # Define output path
    output_path = tmp_path / "depth_map.obj"
    
    # Write depth map as OBJ
    write_depth_map(depth_map, output_path, format='obj')
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file and check if it contains vertices and faces
    with open(output_path, 'r') as f:
        content = f.read()
        
        # Check if file contains vertices
        assert 'v ' in content
        
        # Check if file contains faces
        assert 'f ' in content
        
        # Check number of vertices (should be h*w)
        vertex_count = content.count('v ')
        assert vertex_count >= h*w