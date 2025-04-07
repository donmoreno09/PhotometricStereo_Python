"""
Functions for exporting PLY 3D models.
"""

import os
import numpy as np
import struct

from utils.file_io import compute_path_out
from utils.math_utils import check_decimation, compute_nodes_and_faces


def write_ply(config, vertices=None, faces=None, normals=None, colors=None):
    """
    Write PLY file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    vertices : ndarray, optional
        Vertices, shape (num_vertices, 3)
    faces : ndarray, optional
        Faces, shape (num_faces, 3)
    normals : ndarray, optional
        Normal vectors, shape (num_vertices, 3)
    colors : ndarray, optional
        Colors, shape (num_vertices, 3)
        
    Returns
    -------
    str
        Output file path
    """
    # Compute output path
    out_path = compute_path_out(config, 'ply')
    
    # If vertices and faces aren't provided, create from depth map
    if vertices is None or faces is None:
        if 'depth_map' in config:
            vertices, faces = create_mesh_from_depth_map(config)
        else:
            raise ValueError("No vertices, faces, or depth map provided")
    
    # Generate colors from albedo if available and not provided
    if colors is None and 'albedo' in config:
        albedo = config['albedo']
        
        # Create colors array
        num_vertices = vertices.shape[0]
        colors = np.zeros((num_vertices, 3), dtype=np.uint8)
        
        # Map albedo values to vertices
        height, width = albedo.shape
        
        for i in range(num_vertices):
            # Get x,y coordinates from vertex
            x = int(vertices[i, 0])
            y = int(vertices[i, 1])
            
            # Ensure coordinates are in bounds
            if 0 <= x < width and 0 <= y < height:
                # Scale albedo to [0, 255] and use as grayscale color
                color_val = int(albedo[y, x] * 255)
                colors[i, :] = [color_val, color_val, color_val]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Write PLY file
    write_ply_file(out_path, vertices, faces, normals, colors)
    
    return out_path


def create_mesh_from_depth_map(config):
    """
    Create mesh vertices and faces from depth map.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    tuple
        (vertices, faces)
    """
    # Check if we have a depth map
    if 'depth_map' not in config:
        raise ValueError("No depth map found in config")
    
    depth_map = config['depth_map']
    
    # Get decimation factors
    decimation_x = config.get('mesh_decimation_x', 1)
    decimation_y = config.get('mesh_decimation_y', 1)
    
    # Check and adjust decimation factors
    decimation_y, decimation_x = check_decimation(config, decimation_y, decimation_x)
    
    # Compute nodes and faces
    from utils.math_utils import compute_nodes_and_faces
    vertices, faces = compute_nodes_and_faces(depth_map, decimation_y, decimation_x)
    
    return vertices, faces


def write_ply_file(filename, vertices, faces, normals=None, colors=None, binary=True):
    """
    Write PLY file in ASCII or binary format.
    
    Parameters
    ----------
    filename : str
        Output file path
    vertices : ndarray
        Vertices, shape (num_vertices, 3)
    faces : ndarray
        Faces, shape (num_faces, 3)
    normals : ndarray, optional
        Normal vectors, shape (num_vertices, 3)
    colors : ndarray, optional
        Colors, shape (num_vertices, 3)
    binary : bool, optional
        Whether to write in binary format
        
    Returns
    -------
    bool
        True if successful
    """
    try:
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]
        
        # Start writing the file
        with open(filename, 'wb' if binary else 'w') as f:
            # Write header
            f.write(b"ply\n" if binary else "ply\n")
            f.write(b"format binary_little_endian 1.0\n" if binary else 
                   "format ascii 1.0\n")
            f.write(b"comment Created by Photometric Stereo Py\n" if binary else 
                   "comment Created by Photometric Stereo Py\n")
            f.write(f"element vertex {num_vertices}\n".encode() if binary else 
                   f"element vertex {num_vertices}\n")
            f.write(b"property float x\n" if binary else "property float x\n")
            f.write(b"property float y\n" if binary else "property float y\n")
            f.write(b"property float z\n" if binary else "property float z\n")
            
            # Add normal properties if present
            if normals is not None:
                f.write(b"property float nx\n" if binary else "property float nx\n")
                f.write(b"property float ny\n" if binary else "property float ny\n")
                f.write(b"property float nz\n" if binary else "property float nz\n")
            
            # Add color properties if present
            if colors is not None:
                f.write(b"property uchar red\n" if binary else "property uchar red\n")
                f.write(b"property uchar green\n" if binary else "property uchar green\n")
                f.write(b"property uchar blue\n" if binary else "property uchar blue\n")
            
            # Add face element
            f.write(f"element face {num_faces}\n".encode() if binary else 
                   f"element face {num_faces}\n")
            f.write(b"property list uchar int vertex_indices\n" if binary else 
                   "property list uchar int vertex_indices\n")
            f.write(b"end_header\n" if binary else "end_header\n")
            
            # Write vertices
            if binary:
                for i in range(num_vertices):
                    # Write vertex coordinates
                    f.write(struct.pack('<fff', 
                                       float(vertices[i, 0]), 
                                       float(vertices[i, 1]), 
                                       float(vertices[i, 2])))
                    
                    # Write normals if present
                    if normals is not None:
                        f.write(struct.pack('<fff',
                                           float(normals[i, 0]),
                                           float(normals[i, 1]),
                                           float(normals[i, 2])))
                    
                    # Write colors if present
                    if colors is not None:
                        f.write(struct.pack('<BBB',
                                           int(colors[i, 0]),
                                           int(colors[i, 1]),
                                           int(colors[i, 2])))
            else:
                for i in range(num_vertices):
                    line = f"{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}"
                    
                    if normals is not None:
                        line += f" {normals[i, 0]} {normals[i, 1]} {normals[i, 2]}"
                    
                    if colors is not None:
                        line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
                    
                    f.write(line + "\n")
            
            # Write faces
            if binary:
                for i in range(num_faces):
                    # Number of vertices per face
                    f.write(struct.pack('<B', 3))
                    # Vertex indices
                    f.write(struct.pack('<III',
                                       int(faces[i, 0]),
                                       int(faces[i, 1]),
                                       int(faces[i, 2])))
            else:
                for i in range(num_faces):
                    f.write(f"3 {int(faces[i, 0])} {int(faces[i, 1])} {int(faces[i, 2])}\n")
        
        return True
    
    except Exception as e:
        print(f"Error writing PLY file: {str(e)}")
        return False