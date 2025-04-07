"""
Tests for STL file export.
Port of surf2stl.m test functionality
"""

import pytest
import numpy as np
import os
import struct
from export.stl_writer import write_stl


def test_write_stl_simple_mesh(tmp_path):
    """Test writing a simple mesh to STL file."""
    # Create a simple depth map for a pyramid
    h, w = 10, 10
    depth_map = np.zeros((h, w))
    
    # Create a pyramid
    for i in range(h):
        for j in range(w):
            depth_map[i, j] = min(i, j, h-i-1, w-j-1) / 4
    
    # Define output path
    output_path = tmp_path / "simple_mesh.stl"
    
    # Write STL file
    write_stl(output_path, depth_map)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file to check format (binary STL)
    with open(output_path, 'rb') as f:
        # Read header (80 bytes)
        header = f.read(80)
        # Read number of triangles (4 bytes)
        triangle_count = struct.unpack('I', f.read(4))[0]
        
        # Check that there are triangles
        assert triangle_count > 0
        
        # Each triangle is 50 bytes (12 for normal, 36 for vertices, 2 for attribute)
        # Read first triangle
        normal = struct.unpack('fff', f.read(12))
        vertex1 = struct.unpack('fff', f.read(12))
        vertex2 = struct.unpack('fff', f.read(12))
        vertex3 = struct.unpack('fff', f.read(12))
        _ = f.read(2)  # attribute byte count
        
        # Check that normal is valid
        assert not all(np.isclose(n, 0) for n in normal)
        
        # Check that vertices form a valid triangle
        assert not np.allclose(vertex1, vertex2)
        assert not np.allclose(vertex1, vertex3)
        assert not np.allclose(vertex2, vertex3)


def test_write_stl_ascii(tmp_path):
    """Test writing an ASCII STL file."""
    # Create a simple depth map
    h, w = 5, 5
    depth_map = np.zeros((h, w))
    
    # Create a simple gradient
    for i in range(h):
        for j in range(w):
            depth_map[i, j] = i + j
    
    # Define output path
    output_path = tmp_path / "ascii_mesh.stl"
    
    # Write STL file in ASCII mode
    write_stl(output_path, depth_map, mode='ascii')
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.read()
    
    # Check that it's ASCII format
    assert 'solid ' in content
    assert 'facet normal ' in content
    assert 'outer loop' in content
    assert 'vertex ' in content
    assert 'endloop' in content
    assert 'endfacet' in content
    assert 'endsolid ' in content


def test_write_stl_with_mask(tmp_path):
    """Test writing an STL file with a mask."""
    # Create a simple depth map
    h, w = 10, 10
    depth_map = np.zeros((h, w))
    
    # Create a gradient
    for i in range(h):
        for j in range(w):
            depth_map[i, j] = i + j
    
    # Create a mask (central square)
    mask = np.zeros((h, w), dtype=bool)
    mask[2:8, 2:8] = True
    
    # Define output path
    output_path = tmp_path / "masked_mesh.stl"
    
    # Write STL file with mask
    write_stl(output_path, depth_map, mask=mask)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file (binary STL)
    with open(output_path, 'rb') as f:
        # Read header (80 bytes)
        header = f.read(80)
        # Read number of triangles (4 bytes)
        triangle_count = struct.unpack('I', f.read(4))[0]
        
        # Since we've masked out a lot of the mesh, there should be fewer triangles
        # Based on understanding the mesh generation algorithm for a 10x10 grid:
        # - Full mesh would have 2*(9*9) triangles
        # - Masked mesh would have 2*(6*6) triangles
        assert triangle_count < 2 * 9 * 9