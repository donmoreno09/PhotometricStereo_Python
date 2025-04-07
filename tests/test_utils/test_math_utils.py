"""
Tests for math utilities.
Port of computeNodesAndFaces.m and checkDecimation.m test functionality
"""

import pytest
import numpy as np
from utils.math_utils import compute_nodes_and_faces, check_decimation


def test_compute_nodes_and_faces_simple():
    """Test computing nodes and faces from a simple depth map."""
    # Create a simple 3x3 depth map
    depth_map = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    # Compute nodes and faces
    nodes, faces = compute_nodes_and_faces(depth_map)
    
    # Check nodes shape
    assert nodes.shape == (9, 3)  # 3x3 grid = 9 points
    
    # Check nodes values
    # Each node should have x, y coordinates matching grid
    # and z coordinate matching depth
    for i in range(3):
        for j in range(3):
            node_idx = i * 3 + j
            # Check x, y coordinates
            assert nodes[node_idx, 0] == j
            assert nodes[node_idx, 1] == i
            # Check z coordinate
            assert nodes[node_idx, 2] == depth_map[i, j]
    
    # Check faces
    # For 3x3 grid, we should have 8 triangles (2 per grid cell)
    assert faces.shape[0] == 8
    assert faces.shape[1] == 3  # 3 vertices per face


def test_compute_nodes_and_faces_with_mask():
    """Test computing nodes and faces with a mask."""
    # Create a simple 4x4 depth map
    depth_map = np.zeros((4, 4))
    
    # Create a mask (exclude corners)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = mask[0, 3] = mask[3, 0] = mask[3, 3] = False
    
    # Compute nodes and faces
    nodes, faces = compute_nodes_and_faces(depth_map, mask=mask)
    
    # Check that we have fewer nodes with mask
    assert nodes.shape[0] < 16  # 4x4 grid would normally have 16 nodes
    
    # Check faces are valid (all indices should be within range)
    assert np.all(faces < nodes.shape[0])


def test_compute_nodes_and_faces_decimation():
    """Test computing nodes and faces with decimation."""
    # Create a larger depth map
    depth_map = np.zeros((10, 10))
    
    # Compute nodes and faces with decimation
    nodes_full, faces_full = compute_nodes_and_faces(depth_map, decimation=1)
    nodes_dec, faces_dec = compute_nodes_and_faces(depth_map, decimation=2)
    
    # Check node counts
    assert nodes_full.shape[0] == 10 * 10  # No decimation
    assert nodes_dec.shape[0] == 5 * 5     # Decimation by 2
    
    # Check face counts
    # For a 10x10 grid with no decimation:
    # - 9x9 cells, each with 2 triangles
    # - Total: 2*9*9 = 162 triangles
    assert faces_full.shape[0] == 2 * 9 * 9
    
    # For a 5x5 grid (after decimation by 2):
    # - 4x4 cells, each with 2 triangles
    # - Total: 2*4*4 = 32 triangles
    assert faces_dec.shape[0] == 2 * 4 * 4


def test_check_decimation():
    """Test checking decimation factor."""
    # Create depth maps of various sizes
    depth_small = np.zeros((10, 10))
    depth_medium = np.zeros((100, 100))
    depth_large = np.zeros((500, 500))
    
    # Check decimation factors
    assert check_decimation(depth_small) == 1   # Small map, no decimation
    assert check_decimation(depth_medium) > 1   # Medium map, some decimation
    assert check_decimation(depth_large) > 1    # Large map, more decimation
    
    # Check with override
    assert check_decimation(depth_large, override=1) == 1  # Force no decimation
    assert check_decimation(depth_small, override=2) == 2  # Force decimation
    
    # Check with maximum limit
    max_dec = 5
    assert check_decimation(depth_large, max_decimation=max_dec) <= max_dec