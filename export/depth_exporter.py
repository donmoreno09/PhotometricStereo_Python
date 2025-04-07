"""
Functions for exporting depth maps.
Python implementation of writeDepthMap.m
"""

import os
import numpy as np
import cv2

from utils.file_io import compute_path_out
from utils.math_utils import check_decimation
from export.stl_writer import surf2stl
from export.ply_writer import write_ply


def write_depth_map(config, depth_map=None, format="ply"):
    """
    Write depth map to 3D file format.
    Python implementation of writeDepthMap.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    depth_map : ndarray, optional
        Depth map to export, shape (height, width)
        If None, use depth_map from config
    format : str, optional
        Output format: 'ply', 'stl', or 'obj'
        
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
    
    # Check depth map dimensions
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be a 2D array with shape (height, width)")
    
    # Create output filename based on format
    format = format.lower()
    if format not in ['ply', 'stl', 'obj']:
        raise ValueError(f"Unsupported format: {format}. Use 'ply', 'stl', or 'obj'.")
    
    # Check if we have normal maps
    has_normals = 'normal_map' in config
    
    # Check if we have albedo
    has_albedo = 'albedo' in config
    
    # Export based on format
    if format == 'ply':
        # Use the ply_writer module
        from export.ply_writer import write_ply
        return write_ply(config)
    
    elif format == 'stl':
        # Use the stl_writer module
        from export.stl_writer import surf2stl
        return surf2stl(config)
    
    elif format == 'obj':
        # Use the obj_writer module
        from export.obj_writer import write_obj
        return write_obj(config)
    
    return None


def write_depth_image(config, depth_map=None, min_depth=None, max_depth=None):
    """
    Write depth map as grayscale image.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    depth_map : ndarray, optional
        Depth map to export, shape (height, width)
        If None, use depth_map from config
    min_depth : float, optional
        Minimum depth value for normalization
        If None, use minimum value in depth map
    max_depth : float, optional
        Maximum depth value for normalization
        If None, use maximum value in depth map
        
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
    
    # Check depth map dimensions
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be a 2D array with shape (height, width)")
    
    # Compute output path
    out_path = compute_path_out(config, 'depth')
    
    # Normalize depth to [0, 1]
    if min_depth is None:
        min_depth = np.nanmin(depth_map)
    if max_depth is None:
        max_depth = np.nanmax(depth_map)
    
    depth_range = max_depth - min_depth
    if depth_range > 0:
        depth_norm = (depth_map - min_depth) / depth_range
    else:
        depth_norm = np.zeros_like(depth_map)
    
    # Handle NaN/Inf values
    depth_norm = np.nan_to_num(depth_norm, nan=0, posinf=1, neginf=0)
    
    # Clip to [0, 1]
    depth_norm = np.clip(depth_norm, 0, 1)
    
    # Scale to 8-bit or 16-bit range based on config
    bit_depth = config.get('export', {}).get('bit_depth', 8)
    
    if bit_depth == 16:
        depth_out = (depth_norm * 65535).astype(np.uint16)
    else:
        depth_out = (depth_norm * 255).astype(np.uint8)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(out_path, depth_out)
    
    return out_path