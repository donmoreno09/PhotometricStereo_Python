"""
Tests for PLY file export.
"""

import pytest
import numpy as np
import os
from export.ply_writer import write_ply


def test_write_ply_simple_mesh(tmp_path):
    """Test writing a simple mesh to PLY file."""
    # Create a simple depth map
    h, w = 10, 10
    
    # Create a dome-shaped depth map
    y, x = np.mgrid[:h, :w]
    depth_map = 1 - ((x - w/2) / (w/2))**2 - ((y - h/2) / (h/2))**2
    depth_map = np.maximum(0, depth_map)
    
    # Define output path
    output_path = tmp_path / "simple_mesh.ply"
    
    # Write PLY file
    write_ply(output_path, depth_map)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.readlines()
    
    # Check PLY header
    assert 'ply' in content[0]
    assert 'format ascii' in content[1]
    
    # Check that vertex and face elements are present
    header = ''.join(content[:20])  # Just check the first 20 lines for header
    assert 'element vertex' in header
    assert 'element face' in header
    
    # Count vertices and faces in the file
    vertex_count = 0
    face_count = 0
    
    for line in content:
        if 'element vertex' in line:
            vertex_count = int(line.strip().split()[-1])
        if 'element face' in line:
            face_count = int(line.strip().split()[-1])
    
    # Verify counts
    assert vertex_count == h * w
    # Face count depends on algorithm, but should be positive
    assert face_count > 0


def test_write_ply_with_vertex_colors(tmp_path):
    """Test writing a PLY file with vertex colors."""
    # Create a simple depth map
    h, w = 5, 5
    
    # Create a simple gradient depth map
    y, x = np.mgrid[:h, :w]
    depth_map = (x + y) / (h + w - 2)  # Normalize to [0, 1]
    
    # Create vertex colors (gradient from red to blue)
    colors = np.zeros((h, w, 3), dtype=np.uint8)
    colors[:,:,0] = np.linspace(0, 255, h)[:, np.newaxis]  # Red gradient
    colors[:,:,2] = np.linspace(255, 0, w)[np.newaxis, :]  # Blue gradient
    
    # Define output path
    output_path = tmp_path / "colored_mesh.ply"
    
    # Write PLY file with colors
    write_ply(output_path, depth_map, vertex_colors=colors)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.readlines()
    
    # Check that color properties are in header
    header = ''.join(content[:20])  # Check first 20 lines
    assert 'property uchar red' in header
    assert 'property uchar green' in header
    assert 'property uchar blue' in header
    
    # Check that vertex data includes color values
    # Find end of header
    data_start = 0
    for i, line in enumerate(content):
        if line.strip() == 'end_header':
            data_start = i + 1
            break
    
    # Check first vertex line
    if data_start > 0:
        vertex_line = content[data_start].strip().split()
        # Format should be: x y z r g b
        assert len(vertex_line) >= 6
        # Check that color values are integers
        assert all(c.isdigit() for c in vertex_line[3:6])


def test_write_ply_with_mask(tmp_path):
    """Test writing a PLY file with a mask."""
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
    output_path = tmp_path / "masked_mesh.ply"
    
    # Write PLY file with mask
    write_ply(output_path, depth_map, mask=mask)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.readlines()
    
    # Count vertices in the file
    vertex_count = 0
    for line in content:
        if 'element vertex' in line:
            vertex_count = int(line.strip().split()[-1])
    
    # With mask, we should have fewer vertices than full grid
    masked_points = np.sum(mask)
    assert vertex_count == masked_points