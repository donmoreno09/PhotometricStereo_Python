"""
Functions for exporting OBJ 3D models.
Python implementation of write_wobj.m and obj_write.m
"""

import os
import numpy as np

from utils.file_io import compute_path_out
from utils.math_utils import check_decimation, compute_nodes_and_faces


def write_obj(config, vertices=None, faces=None, normals=None, texture_coords=None):
    """
    Write 3D model to OBJ file.
    Python implementation of write_wobj.m
    
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
    texture_coords : ndarray, optional
        Texture coordinates, shape (num_vertices, 2)
        
    Returns
    -------
    str
        Output file path
    """
    # Compute output path
    out_path = compute_path_out(config, 'obj')
    
    # If vertices and faces aren't provided, create from depth map
    if vertices is None or faces is None:
        vertices, faces = create_mesh_from_depth_map(config)
    
    # Check if we have valid data
    if vertices is None or faces is None:
        raise ValueError("No valid mesh data provided or created")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Write OBJ file
    with open(out_path, 'w') as f:
        # Write header
        f.write("# OBJ file created by Photometric Stereo Py\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write texture coordinates if available
        if texture_coords is not None:
            for tc in texture_coords:
                f.write(f"vt {tc[0]} {tc[1]}\n")
        
        # Write normals if available
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        # Write faces
        # OBJ uses 1-based indexing
        if texture_coords is not None and normals is not None:
            # Write faces with texture and normal indices
            for face in faces:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")
        elif texture_coords is not None:
            # Write faces with texture indices
            for face in faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
        elif normals is not None:
            # Write faces with normal indices
            for face in faces:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        else:
            # Write faces with only vertex indices
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
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
    vertices, faces = compute_nodes_and_faces(depth_map, decimation_y, decimation_x)
    
    return vertices, faces


def obj_write(filepath, v, f, vn=None, vt=None, comments=None):
    """
    Write OBJ file with more control over formatting.
    Python implementation of obj_write.m
    
    Parameters
    ----------
    filepath : str
        Output file path
    v : ndarray
        Vertices, shape (num_vertices, 3)
    f : ndarray
        Faces, shape (num_faces, 3)
    vn : ndarray, optional
        Normal vectors, shape (num_vertices, 3)
    vt : ndarray, optional
        Texture coordinates, shape (num_vertices, 2)
    comments : list, optional
        List of comment strings to add to header
        
    Returns
    -------
    bool
        True if successful
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as fid:
            # Write header comments
            fid.write("# OBJ file created by Photometric Stereo Py\n")
            
            if comments is not None:
                for comment in comments:
                    fid.write(f"# {comment}\n")
            
            # Write vertices
            for i in range(v.shape[0]):
                fid.write(f"v {v[i, 0]:.6f} {v[i, 1]:.6f} {v[i, 2]:.6f}\n")
            
            # Write texture coordinates if available
            if vt is not None:
                for i in range(vt.shape[0]):
                    fid.write(f"vt {vt[i, 0]:.6f} {vt[i, 1]:.6f}\n")
            
            # Write normals if available
            if vn is not None:
                for i in range(vn.shape[0]):
                    fid.write(f"vn {vn[i, 0]:.6f} {vn[i, 1]:.6f} {vn[i, 2]:.6f}\n")
            
            # Write faces
            # OBJ uses 1-based indexing
            for i in range(f.shape[0]):
                if vn is not None and vt is not None:
                    # Faces with vertex, texture, and normal indices
                    fid.write(f"f {f[i, 0]+1}/{f[i, 0]+1}/{f[i, 0]+1} {f[i, 1]+1}/{f[i, 1]+1}/{f[i, 1]+1} {f[i, 2]+1}/{f[i, 2]+1}/{f[i, 2]+1}\n")
                elif vt is not None:
                    # Faces with vertex and texture indices
                    fid.write(f"f {f[i, 0]+1}/{f[i, 0]+1} {f[i, 1]+1}/{f[i, 1]+1} {f[i, 2]+1}/{f[i, 2]+1}\n")
                elif vn is not None:
                    # Faces with vertex and normal indices
                    fid.write(f"f {f[i, 0]+1}//{f[i, 0]+1} {f[i, 1]+1}//{f[i, 1]+1} {f[i, 2]+1}//{f[i, 2]+1}\n")
                else:
                    # Faces with only vertex indices
                    fid.write(f"f {f[i, 0]+1} {f[i, 1]+1} {f[i, 2]+1}\n")
        
        return True
    
    except Exception as e:
        print(f"Error writing OBJ file: {str(e)}")
        return False