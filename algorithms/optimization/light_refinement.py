"""
Light direction refinement for photometric stereo.
Python implementation of PSRefineLight.m and related functions
"""

import numpy as np
from algorithms.optimization.nonlinear_least_squares import nonlinear_least_squares


def ps_refine_light(images, light_dirs_init, mask=None, robust=False, options=None):
    """
    Refine light directions using photometric stereo optimization.
    Python implementation of PSRefineLight.m
    
    Parameters
    ----------
    images : ndarray
        Images tensor, shape (height, width, num_images)
    light_dirs_init : ndarray
        Initial light directions, shape (num_images, 3)
    mask : ndarray, optional
        Binary mask for valid pixels
    robust : bool, optional
        Whether to use robust estimation
    options : dict, optional
        Optimization options
        
    Returns
    -------
    tuple
        (refined_light_dirs, normal_map, albedo)
    """
    # Default options
    default_options = {
        'max_iterations': 100,
        'tolerance': 1e-6,
        'step_size': 0.1,
        'verbose': False
    }
    
    if options is None:
        options = default_options
    else:
        # Merge with defaults
        for key, value in default_options.items():
            if key not in options:
                options[key] = value
    
    # Get dimensions
    h, w, num_images = images.shape
    
    # Create default mask if none provided
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    
    # Flatten and normalize light directions
    light_dirs_flat = light_dirs_init.flatten()
    
    # Create cost function for optimization
    def cost_function(params):
        # Reshape parameters to light directions
        light_dirs = params.reshape(-1, 3)
        
        # Normalize light directions
        for i in range(light_dirs.shape[0]):
            light_dirs[i] = light_dirs[i] / np.linalg.norm(light_dirs[i])
        
        # Compute normals and albedo using current light directions
        normal_map, albedo = estimate_normals_albedo(images, light_dirs, mask, robust)
        
        # Compute reconstruction error
        errors = np.zeros_like(images)
        for i in range(num_images):
            # Compute estimated image: albedo * (n·l)
            n_dot_l = np.sum(normal_map * light_dirs[i].reshape(1, 1, 3), axis=2)
            est_image = albedo * np.maximum(0, n_dot_l)
            
            # Compute error
            errors[:, :, i] = images[:, :, i] - est_image
        
        # Return flattened errors for all valid pixels
        return errors[mask].flatten()
    
    # Run optimization
    refined_params = nonlinear_least_squares(cost_function, light_dirs_flat, options=options)
    
    # Reshape result to light directions
    refined_light_dirs = refined_params.reshape(-1, 3)
    
    # Normalize light directions
    for i in range(refined_light_dirs.shape[0]):
        refined_light_dirs[i] = refined_light_dirs[i] / np.linalg.norm(refined_light_dirs[i])
    
    # Compute final normals and albedo
    normal_map, albedo = estimate_normals_albedo(images, refined_light_dirs, mask, robust)
    
    return refined_light_dirs, normal_map, albedo


def estimate_normals_albedo(images, light_dirs, mask=None, robust=False):
    """
    Estimate normal map and albedo from images and light directions.
    
    Parameters
    ----------
    images : ndarray
        Images tensor, shape (height, width, num_images)
    light_dirs : ndarray
        Light directions, shape (num_images, 3)
    mask : ndarray, optional
        Binary mask for valid pixels
    robust : bool, optional
        Whether to use robust estimation
        
    Returns
    -------
    tuple
        (normal_map, albedo)
    """
    # Get dimensions
    h, w, num_images = images.shape
    
    # Create default mask if none provided
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    
    # Initialize outputs
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    albedo = np.zeros((h, w), dtype=np.float32)
    
    # Create light direction matrix
    light_matrix = light_dirs.copy()
    
    # Process each pixel
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            
            # Get intensities for this pixel
            intensities = images[y, x, :].flatten()
            
            if robust:
                # Robust estimation using iteratively reweighted least squares
                weights = np.ones(num_images)
                for _ in range(3):  # 3 iterations of IRLS
                    # Weighted least squares
                    weighted_light = light_matrix * weights[:, np.newaxis]
                    weighted_intensities = intensities * weights
                    
                    # Solve system
                    g, residuals, rank, s = np.linalg.lstsq(weighted_light, weighted_intensities, rcond=None)
                    
                    # Update weights based on residuals
                    residual = intensities - light_matrix @ g
                    weights = 1.0 / (np.abs(residual) + 1e-6)
                    weights = weights / np.sum(weights) * num_images
            else:
                # Standard least squares
                g, residuals, rank, s = np.linalg.lstsq(light_matrix, intensities, rcond=None)
            
            # Extract normal and albedo
            normal_length = np.linalg.norm(g)
            if normal_length > 0:
                normal_map[y, x] = g / normal_length
                albedo[y, x] = normal_length
    
    return normal_map, albedo