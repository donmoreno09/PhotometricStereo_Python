"""
Functions for exporting STL 3D models.
Python implementation of surf2stl.m
"""

import os
import numpy as np
import struct

from utils.file_io import compute_path_out
from utils.math_utils import check_decimation


def surf2stl(config, depth_map=None, filename=None):
    """
    Convert depth map to STL file.
    Python implementation of surf2stl.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    depth_map : ndarray, optional
        Depth map, shape (height, width)
        If None, use depth_map from config
    filename : str, optional
        Output file path
        If None, compute from config
        
    Returns
    -------
    str
        Output file path
    """
    # Use provided depth map or get from config
    if depth_map is None:
        if 'depth_map' not in config:
            raise ValueError("No depth map provided or found in config")
        depth_map = config['depth_map']
    
    # Compute output path if not provided
    if filename is None:
        filename = compute_path_out(config, 'stl')
    
    # Get decimation factors
    decimation_x = config.get('mesh_decimation_x', 1)
    decimation_y = config.get('mesh_decimation_y', 1)
    
    # Check and adjust decimation factors
    decimation_y, decimation_x = check_decimation(config, decimation_y, decimation_x)
    
    # Extract coordinates
    y, x = np.mgrid[0:depth_map.shape[0]:decimation_y, 0:depth_map.shape[1]:decimation_x]
    z = depth_map[0:depth_map.shape[0]:decimation_y, 0:depth_map.shape[1]:decimation_x]
    
    # Check if we have valid data
    if z.size == 0:
        raise ValueError("No valid data after decimation")
    
    # Create triangulation
    ny, nx = z.shape
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write STL file
    write_stl_binary(filename, x, y, z)
    
    return filename


def write_stl_binary(filename, x, y, z):
    """
    Write binary STL file.
    
    Parameters
    ----------
    filename : str
        Output file path
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    z : ndarray
        Z coordinates
        
    Returns
    -------
    bool
        True if successful
    """
    try:
        # Get dimensions
        ny, nx = z.shape
        
        # Count triangles (two per grid cell)
        n_triangles = 2 * (ny - 1) * (nx - 1)
        
        with open(filename, 'wb') as f:
            # Write header (80 bytes)
            header = "STL binary file created by Photometric Stereo Py".ljust(80, '\0')
            f.write(header.encode())
            
            # Write number of triangles
            f.write(struct.pack('I', n_triangles))
            
            # Write triangles
            for i in range(ny - 1):
                for j in range(nx - 1):
                    # First triangle (lower-left half of the quad)
                    p1 = [x[i, j], y[i, j], z[i, j]]
                    p2 = [x[i+1, j], y[i+1, j], z[i+1, j]]
                    p3 = [x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]]
                    
                    # Calculate normal vector
                    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
                    v2 = np.array([p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]])
                    normal = np.cross(v1, v2)
                    
                    # Normalize
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    
                    # Write normal and vertices
                    f.write(struct.pack('fff', *normal))
                    f.write(struct.pack('fff', *p1))
                    f.write(struct.pack('fff', *p2))
                    f.write(struct.pack('fff', *p3))
                    
                    # Attribute byte count (unused)
                    f.write(struct.pack('H', 0))
                    
                    # Second triangle (upper-right half of the quad)
                    p1 = [x[i, j], y[i, j], z[i, j]]
                    p2 = [x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]]
                    p3 = [x[i, j+1], y[i, j+1], z[i, j+1]]
                    
                    # Calculate normal vector
                    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
                    v2 = np.array([p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]])
                    normal = np.cross(v1, v2)
                    
                    # Normalize
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    
                    # Write normal and vertices
                    f.write(struct.pack('fff', *normal))
                    f.write(struct.pack('fff', *p1))
                    f.write(struct.pack('fff', *p2))
                    f.write(struct.pack('fff', *p3))
                    
                    # Attribute byte count (unused)
                    f.write(struct.pack('H', 0))
        
        return True
    
    except Exception as e:
        print(f"Error writing STL file: {str(e)}")
        return False