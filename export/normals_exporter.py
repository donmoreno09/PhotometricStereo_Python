"""
Functions for exporting normal maps and related outputs.
Python implementation of writeNormals.m and writeReflectionMap.m
"""

import os
import numpy as np
import cv2

from utils.file_io import compute_path_out
from utils.exif_tools import add_exif
from utils.image_processing import render_normals_in_rgb


def write_normals(config, normal_map=None):
    """
    Write normal map to file.
    Python implementation of writeNormals.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    normal_map : ndarray, optional
        Normal map to export, shape (height, width, 3)
        If None, use normal_map from config
        
    Returns
    -------
    str
        Output file path
    """
    # Use provided normal map or get from config
    if normal_map is None:
        if 'normal_map' not in config:
            raise ValueError("No normal map provided or found in config")
        normal_map = config['normal_map']
    
    # Check normal map dimensions
    if normal_map.ndim != 3 or normal_map.shape[2] != 3:
        raise ValueError("Normal map must be a 3D array with shape (height, width, 3)")
    
    # Compute output path
    out_path = compute_path_out(config, 'normal')
    
    # Convert normal map to RGB representation
    rgb_normals = render_normals_in_rgb(normal_map)
    
    # Save image
    # RGB order for OpenCV is BGR, so we need to convert
    rgb_normals_bgr = cv2.cvtColor(rgb_normals, cv2.COLOR_RGB2BGR)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(out_path, rgb_normals_bgr)
    
    # Add color profile if requested
    color_space = config.get('export', {}).get('color_profile', 'none')
    if color_space.lower() != 'none':
        # We need a source image for profile copying - use the input image if available
        if 'all_lights_on_image_path' in config:
            in_path = config['all_lights_on_image_path']
        else:
            in_path = out_path  # Use output as input (no profile copy)
        
        # Add EXIF data and color profile
        add_exif(in_path, out_path, color_space)
    
    return out_path


def write_reflection_map(config, normal_map=None, light_dir=None):
    """
    Write reflection map to file.
    Python implementation of writeReflectionMap.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    normal_map : ndarray, optional
        Normal map, shape (height, width, 3)
        If None, use normal_map from config
    light_dir : ndarray, optional
        Light direction vector, shape (3,)
        If None, use a default direction [0, 0, 1]
        
    Returns
    -------
    str
        Output file path
    """
    # Use provided normal map or get from config
    if normal_map is None:
        if 'normal_map' not in config:
            raise ValueError("No normal map provided or found in config")
        normal_map = config['normal_map']
    
    # Check normal map dimensions
    if normal_map.ndim != 3 or normal_map.shape[2] != 3:
        raise ValueError("Normal map must be a 3D array with shape (height, width, 3)")
    
    # Use provided light direction or default to [0, 0, 1]
    if light_dir is None:
        light_dir = np.array([0, 0, 1])
    
    # Normalize light direction
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Compute reflection map (dot product of normals and light direction)
    reflection = np.zeros(normal_map.shape[:2], dtype=np.float32)
    
    # Compute dot product for each pixel
    for i in range(3):
        reflection += normal_map[:, :, i] * light_dir[i]
    
    # Clamp negative values to 0
    reflection = np.maximum(reflection, 0)
    
    # Normalize to [0, 1]
    reflection_max = np.max(reflection)
    if reflection_max > 0:
        reflection /= reflection_max
    
    # Scale to 8-bit range
    reflection_8bit = (reflection * 255).astype(np.uint8)
    
    # Compute output path
    out_path = compute_path_out(config, 'reflection')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(out_path, reflection_8bit)
    
    # Add color profile if requested
    color_space = config.get('export', {}).get('color_profile', 'none')
    if color_space.lower() != 'none':
        # We need a source image for profile copying - use the input image if available
        if 'all_lights_on_image_path' in config:
            in_path = config['all_lights_on_image_path']
        else:
            in_path = out_path  # Use output as input (no profile copy)
        
        # Add EXIF data and color profile
        add_exif(in_path, out_path, color_space)
    
    return out_path


def write_ambient_occlusion(config, normal_map=None, samples=32):
    """
    Write ambient occlusion map to file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    normal_map : ndarray, optional
        Normal map, shape (height, width, 3)
        If None, use normal_map from config
    samples : int, optional
        Number of hemisphere samples for AO calculation
        
    Returns
    -------
    str
        Output file path
    """
    # Use provided normal map or get from config
    if normal_map is None:
        if 'normal_map' not in config:
            raise ValueError("No normal map provided or found in config")
        normal_map = config['normal_map']
    
    # Check normal map dimensions
    if normal_map.ndim != 3 or normal_map.shape[2] != 3:
        raise ValueError("Normal map must be a 3D array with shape (height, width, 3)")
    
    # Generate hemisphere samples
    hemisphere_samples = _generate_hemisphere_samples(samples)
    
    # Calculate ambient occlusion
    ao = np.zeros(normal_map.shape[:2], dtype=np.float32)
    
    # For each sample direction
    for sample_dir in hemisphere_samples:
        # Compute dot product between normal and sample direction
        dot_product = np.zeros(normal_map.shape[:2], dtype=np.float32)
        for i in range(3):
            dot_product += normal_map[:, :, i] * sample_dir[i]
        
        # Add contribution to AO map
        ao += np.maximum(dot_product, 0)
    
    # Normalize by number of samples
    ao /= len(hemisphere_samples)
    
    # Convert to 8-bit
    ao_8bit = (ao * 255).astype(np.uint8)
    
    # Compute output path
    out_path = compute_path_out(config, 'ao')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(out_path, ao_8bit)
    
    return out_path


def write_albedo(config, albedo=None):
    """
    Write albedo map to file.
    Python implementation of writeAlbedo.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    albedo : ndarray, optional
        Albedo map to export, shape (height, width)
        If None, use albedo from config
        
    Returns
    -------
    str
        Output file path
    """
    # Use provided albedo or get from config
    if albedo is None:
        if 'albedo' not in config:
            raise ValueError("No albedo provided or found in config")
        albedo = config['albedo']
    
    # Check albedo dimensions
    if albedo.ndim != 2:
        raise ValueError("Albedo must be a 2D array with shape (height, width)")
    
    # Compute output path
    out_path = compute_path_out(config, 'albedo')
    
    # Normalize albedo to [0, 1]
    albedo_max = np.max(albedo)
    if albedo_max > 0:
        albedo_norm = albedo / albedo_max
    else:
        albedo_norm = albedo
    
    # Scale to 8-bit or 16-bit range based on config
    bit_depth = config.get('export', {}).get('bit_depth', 8)
    
    if bit_depth == 16:
        albedo_out = (albedo_norm * 65535).astype(np.uint16)
    else:
        albedo_out = (albedo_norm * 255).astype(np.uint8)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(out_path, albedo_out)
    
    # Add color profile if requested
    color_space = config.get('export', {}).get('color_profile', 'none')
    if color_space.lower() != 'none':
        # Add EXIF data and color profile
        if 'all_lights_on_image_path' in config:
            in_path = config['all_lights_on_image_path']
        else:
            in_path = out_path
            
        add_exif(in_path, out_path, color_space)
    
    return out_path


def _generate_hemisphere_samples(num_samples):
    """
    Generate uniformly distributed samples on the upper hemisphere.
    
    Parameters
    ----------
    num_samples : int
        Number of samples
        
    Returns
    -------
    ndarray
        Sample directions, shape (num_samples, 3)
    """
    samples = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(num_samples):
        # Fibonacci lattice on sphere
        phi = 2 * np.pi * i / golden_ratio
        cosTheta = 1 - (2 * i + 1) / (2 * num_samples)
        sinTheta = np.sqrt(1 - cosTheta * cosTheta)
        
        x = np.cos(phi) * sinTheta
        y = np.sin(phi) * sinTheta
        z = cosTheta
        
        # Only keep points in upper hemisphere
        if z > 0:
            samples.append([x, y, z])
    
    return np.array(samples)