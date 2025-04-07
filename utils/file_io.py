"""
File input/output utilities.
Python implementations of:
- getImageNameOnly.m
- computePathOut.m
"""

import os
import glob
import numpy as np
import cv2


def get_image_name_only(config):
    """
    Extract image name without extension from path.
    Python implementation of getImageNameOnly.m
    
    Parameters
    ----------
    config : dict
        Configuration containing 'all_lights_on_image_path'
        
    Returns
    -------
    str
        Image filename without extension
    """
    if 'all_lights_on_image_path' not in config:
        raise ValueError("Config must contain 'all_lights_on_image_path'")
    
    # Get filename without extension
    file_path = config['all_lights_on_image_path']
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    
    return file_name


def compute_path_out(config, file_type):
    """
    Compute output file path.
    Python implementation of computePathOut.m
    
    Parameters
    ----------
    config : dict
        Configuration containing output settings
    file_type : str
        Type of output file ('normal', 'depth', 'albedo', 'obj', 'ply', 'stl')
        
    Returns
    -------
    str
        Output file path
    """
    if 'output_directory' not in config:
        raise ValueError("Config must contain 'output_directory'")
    
    output_dir = config['output_directory']
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get base filename from all_lights_on_image if available
    if 'all_lights_on_image_path' in config:
        base_name = get_image_name_only(config)
    else:
        base_name = "output"
    
    # Add appropriate suffix and extension based on file type
    if file_type == 'normal':
        output_file = f"{base_name}_normal.tif"
    elif file_type == 'depth':
        output_file = f"{base_name}_depth.tif"
    elif file_type == 'albedo':
        output_file = f"{base_name}_albedo.tif"
    elif file_type == 'obj':
        output_file = f"{base_name}.obj"
    elif file_type == 'ply':
        output_file = f"{base_name}.ply"
    elif file_type == 'stl':
        output_file = f"{base_name}.stl"
    else:
        output_file = f"{base_name}_{file_type}.tif"
    
    # Combine directory and filename
    full_path_out = os.path.join(output_dir, output_file)
    
    return full_path_out


def load_image(file_path, convert_to_grayscale=False):
    """
    Load image from file.
    
    Parameters
    ----------
    file_path : str
        Path to image file
    convert_to_grayscale : bool, optional
        Whether to convert color images to grayscale
        
    Returns
    -------
    ndarray
        Loaded image
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Load image
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    # Check if image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    
    # Convert BGR to RGB for color images
    if len(image.shape) == 3 and image.shape[2] == 3 and not convert_to_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale if requested
    if convert_to_grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def save_image(image, file_path, compression='lzw'):
    """
    Save image to file.
    
    Parameters
    ----------
    image : ndarray
        Image to save
    file_path : str
        Output file path
    compression : str, optional
        Compression type for TIFF images ('lzw', 'none', etc.)
        
    Returns
    -------
    bool
        True if image was saved successfully
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Convert to 8-bit if necessary
    if image.dtype != np.uint8 and image.dtype != np.uint16:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_save = image
    
    # Save based on file type
    if ext in ['.tif', '.tiff']:
        # For TIFF, we can use compression
        if compression == 'lzw':
            success = cv2.imwrite(file_path, image_save, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        else:
            success = cv2.imwrite(file_path, image_save)
    else:
        # For other formats
        success = cv2.imwrite(file_path, image_save)
    
    return success


def load_light_directions(file_path):
    """
    Load light directions from file.
    
    Parameters
    ----------
    file_path : str
        Path to light directions file
        
    Returns
    -------
    ndarray
        Light directions, shape (3, num_lights)
    """
    try:
        # Try loading as NumPy array first
        light_dirs = np.loadtxt(file_path)
        
        # Check shape
        if light_dirs.ndim == 1:
            # Single light direction
            if len(light_dirs) == 3:
                light_dirs = light_dirs.reshape(3, 1)
            else:
                raise ValueError(f"Invalid light direction data in {file_path}")
        elif light_dirs.ndim == 2:
            # Multiple light directions
            if light_dirs.shape[0] == 3:
                # Already in correct format (3 x num_lights)
                pass
            elif light_dirs.shape[1] == 3:
                # Transpose to get (3 x num_lights)
                light_dirs = light_dirs.T
            else:
                raise ValueError(f"Invalid light direction data in {file_path}")
        else:
            raise ValueError(f"Invalid light direction data in {file_path}")
        
    except Exception as e:
        raise ValueError(f"Error loading light directions from {file_path}: {str(e)}")
    
    return light_dirs


def save_light_directions(light_dirs, file_path):
    """
    Save light directions to file.
    
    Parameters
    ----------
    light_dirs : ndarray
        Light directions, shape (3, num_lights)
    file_path : str
        Output file path
        
    Returns
    -------
    bool
        True if saved successfully
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save as text file
        np.savetxt(file_path, light_dirs.T, fmt='%.6f')
        return True
        
    except Exception as e:
        print(f"Error saving light directions to {file_path}: {str(e)}")
        return False