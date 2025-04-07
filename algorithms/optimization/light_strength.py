"""
Light strength estimation for photometric stereo.
Python implementation of PSEstimateLightStrength.m and related functions
"""

import numpy as np


def ps_estimate_light_strength_cost(params, images, normals, albedo, light_dirs, mask=None):
    """
    Cost function for light strength estimation.
    Python implementation of PSEstimateLightStrengthCost.m
    
    Parameters
    ----------
    params : ndarray
        Light strengths
    images : ndarray
        Stacked images, shape (h, w, num_images)
    normals : ndarray
        Normal map, shape (h, w, 3)
    albedo : ndarray
        Albedo map, shape (h, w)
    light_dirs : ndarray
        Light directions, shape (num_images, 3)
    mask : ndarray, optional
        Binary mask for valid pixels
        
    Returns
    -------
    ndarray
        Error vector
    """
    # Get dimensions
    h, w, num_images = images.shape
    
    # Create default mask if none provided
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    
    # Reshape light strengths
    light_strengths = params.reshape(-1)
    
    # Initialize error vector
    num_valid_pixels = np.sum(mask)
    error = np.zeros(num_valid_pixels * num_images)
    
    # Compute errors for each image
    idx = 0
    for img_idx in range(num_images):
        # Get light direction and strength
        light_dir = light_dirs[img_idx]
        light_strength = light_strengths[img_idx]
        
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals * light_dir.reshape(1, 1, 3), axis=2)
        
        # Compute estimated image: albedo * (n·l) * light_strength
        est_image = albedo * np.maximum(0, n_dot_l) * light_strength
        
        # Compute error for valid pixels
        img_error = images[:, :, img_idx] - est_image
        error[idx:idx+num_valid_pixels] = img_error[mask]
        
        idx += num_valid_pixels
    
    return error


def ps_estimate_light_strength_jacobian(params, images, normals, albedo, light_dirs, mask=None):
    """
    Jacobian for light strength estimation.
    Companion to PSEstimateLightStrengthCost.m
    
    Parameters
    ----------
    params : ndarray
        Light strengths
    images : ndarray
        Stacked images, shape (h, w, num_images)
    normals : ndarray
        Normal map, shape (h, w, 3)
    albedo : ndarray
        Albedo map, shape (h, w)
    light_dirs : ndarray
        Light directions, shape (num_images, 3)
    mask : ndarray, optional
        Binary mask for valid pixels
        
    Returns
    -------
    ndarray
        Jacobian matrix
    """
    # Get dimensions
    h, w, num_images = images.shape
    
    # Create default mask if none provided
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    
    # Initialize Jacobian matrix
    num_valid_pixels = np.sum(mask)
    J = np.zeros((num_valid_pixels * num_images, num_images))
    
    # Compute Jacobian entries
    idx = 0
    for img_idx in range(num_images):
        # Get light direction
        light_dir = light_dirs[img_idx]
        
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals * light_dir.reshape(1, 1, 3), axis=2)
        
        # Derivative of error w.r.t. light strength is -albedo * (n·l)
        derivative = -albedo * np.maximum(0, n_dot_l)
        
        # Fill Jacobian for valid pixels
        J[idx:idx+num_valid_pixels, img_idx] = derivative[mask]
        
        idx += num_valid_pixels
    
    return J