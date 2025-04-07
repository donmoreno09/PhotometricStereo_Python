"""
Tests for OBJ file export.
Port of write_wobj.m and obj_write.m test functionality
"""

import pytest
import numpy as np
import os
import tempfile
from export.obj_writer import write_obj


def test_write_obj_simple_mesh(tmp_path):
    """Test writing a simple mesh to OBJ file."""
    # Create a simple 3x3 grid
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 1], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0]
    ])
    
    # Create faces (1-indexed for OBJ format)
    faces = np.array([
        [1, 2, 5], [1, 5, 4],
        [2, 3, 6], [2, 6, 5],
        [4, 5, 8], [4, 8, 7],
        [5, 6, 9], [5, 9, 8]
    ])
    
    # Define output path
    output_path = tmp_path / "simple_mesh.obj"
    
    # Write OBJ file
    write_obj(output_path, vertices, faces)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.readlines()
    
    # Check that it contains all vertices
    vertex_lines = [line for line in content if line.startswith('v ')]
    assert len(vertex_lines) == len(vertices)
    
    # Check that it contains all faces
    face_lines = [line for line in content if line.startswith('f ')]
    assert len(face_lines) == len(faces)
    
    # Check a few vertices
    # v 0 0 0
    assert 'v 0 0 0' in [line.strip() for line in vertex_lines]
    # v 1 1 1
    assert 'v 1 1 1' in [line.strip() for line in vertex_lines]
    
    # Check a face (allowing for whitespace variations)
    # f 1 2 5
    face_indices = []
    for line in face_lines:
        # Extract numbers from face line
        parts = line.strip().split()[1:]
        # Convert to integers, handling possible vertex/texture/normal format (v/vt/vn)
        indices = [int(p.split('/')[0]) for p in parts]
        face_indices.append(indices)
    
    # Check that all faces are present
    for face in faces:
        assert face.tolist() in face_indices


def test_write_obj_with_normals(tmp_path):
    """Test writing a mesh with normals to OBJ file."""
    # Create a simple 2x2 grid
    vertices = np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 1, 0], [1, 1, 0]
    ])
    
    # Create faces
    faces = np.array([
        [1, 2, 4], [1, 4, 3]
    ])
    
    # Create normals (all pointing up)
    normals = np.array([
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]
    ])
    
    # Define output path
    output_path = tmp_path / "mesh_with_normals.obj"
    
    # Write OBJ file
    write_obj(output_path, vertices, faces, normals=normals)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.readlines()
    
    # Check that it contains all vertices
    vertex_lines = [line for line in content if line.startswith('v ')]
    assert len(vertex_lines) == len(vertices)
    
    # Check that it contains all normals
    normal_lines = [line for line in content if line.startswith('vn ')]
    assert len(normal_lines) == len(normals)
    
    # Check that faces reference normals
    face_lines = [line for line in content if line.startswith('f ')]
    assert len(face_lines) == len(faces)
    
    # Check that faces include normal indices (format f v/vt/vn)
    for line in face_lines:
        parts = line.strip().split()[1:]
        # Each part should have the format v//vn
        for part in parts:
            assert '//' in part
            v_str, vn_str = part.split('//')
            # Check that both vertex and normal indices are present
            assert v_str.isdigit()
            assert vn_str.isdigit()


def test_write_obj_from_depth_map(tmp_path):
    """Test writing a mesh from a depth map to OBJ file."""
    # Create a simple depth map
    h, w = 10, 10
    
    # Create a dome-shaped depth map
    y, x = np.mgrid[:h, :w]
    depth_map = 1 - ((x - w/2) / (w/2))**2 - ((y - h/2) / (h/2))**2
    depth_map = np.maximum(0, depth_map)
    
    # Define output path
    output_path = tmp_path / "depth_mesh.obj"
    
    # Call the simplified interface
    write_obj(output_path, depth_map)
    
    # Check that file was created
    assert os.path.exists(output_path)
    
    # Read back the file
    with open(output_path, 'r') as f:
        content = f.readlines()
    
    # Check that it contains vertices
    vertex_lines = [line for line in content if line.startswith('v ')]
    assert len(vertex_lines) > 0
    
    # Check that it contains faces
    face_lines = [line for line in content if line.startswith('f ')]
    assert len(face_lines) > 0
    
    # Number of vertices should match depth map size
    assert len(vertex_lines) == h*w