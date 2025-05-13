"""
Functions for computing normal maps from images.
Python implementation of PSBoxComputeMaps1.m
"""

import os
import numpy as np
import cv2
from scipy import ndimage
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QCoreApplication
from scipy.spatial import Delaunay

from core.photometric_stereo import PhotometricStereo
from utils.image_processing import normalize_normals, lay_normals, poly_correction
from utils.depth_estimation import depth_from_gradient
from utils.math_utils import check_bit_depth, check_decimation


def compute_normal_maps(config, progress_callback=None):
    """
    Compute normal maps from input images.
    Equivalent to PSBoxComputeMaps1.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
            - input_images: List of file paths to the input images
            - light_directions: ndarray of light directions (3 x num_images)
            - lights_order: Light order configuration
            - light_strength_path: Path to light strength file
            - default_black: Path to black error image for step 1
            - default_black_2: Path to black error image for step 2
            - downsample: Whether to downsample images
            - strel_size: Size of structuring element for erosion
            - shadow_threshold: Threshold for shadow mask
            - all_lights_on_image_path: Path to image with all lights on
    
    Returns
    -------
    dict
        Dictionary containing the computed maps:
            - normals: Normal map as ndarray (height, width, 3)
            - normals_rgb: Normal map converted to RGB for visualization
            - albedo: Albedo map as ndarray (height, width)
            - depth: Depth map as ndarray (height, width)
    """
    # Update progress
    if progress_callback:
        progress_callback(0)
        QCoreApplication.processEvents()
    
    print('Loading data...')
    
    # Load light directions
    light_directions = config['light_directions']
    
    # Reorder lights if needed
    if config.get('lights_order', '') != '45N ...15W':
        print('Reordering light directions...')
        cL = light_directions.copy()
        light_directions[:, 0] = cL[:, 3]
        light_directions[:, 1] = cL[:, 0]
        light_directions[:, 2] = cL[:, 1]
        light_directions[:, 3] = cL[:, 2]
        light_directions[:, 4] = cL[:, 7]
        light_directions[:, 5] = cL[:, 4]
        light_directions[:, 6] = cL[:, 5]
        light_directions[:, 7] = cL[:, 6]
    
    # Load images
    images = load_processed_images(
        config['input_images'], 
        options={
            'image_channel': config.get('image_channel', 0),
            'resample': config.get('downsample', False),
            'normalize_percentile': config.get('normalize_percentile', None)
        }
    )
    num_images = images.shape[2]
    
    # Create shadow mask
    shadow_threshold = config.get('shadow_threshold', 0.01)
    shadow_mask = (images > shadow_threshold)
    
    # Erode shadow mask
    strel_size = config.get('strel_size', 3)
    for i in range(num_images):
        # Create disk-shaped structuring element
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strel_size, strel_size))
        shadow_mask[:, :, i] = cv2.erode(shadow_mask[:, :, i].astype(np.uint8), selem).astype(bool)
    
    if progress_callback:
        progress_callback(20)
        QCoreApplication.processEvents()
    
    # First photometric stereo without light strength estimation
    print('Estimating normal vectors and albedo (without light strength estimation) ...')
    ps = PhotometricStereo(images, light_directions, shadow_mask)
    albedo, normals = ps.process()

    # Convert from (3, height, width) to (height, width, 3) format if needed
    if normals.shape[0] == 3:
        normals = np.transpose(normals, (1, 2, 0))

    # Evaluate normal estimate by intensity error
    eval_opts = {'display': True}
    Ierr = eval_n_estimate_by_i_error(albedo, normals, images, shadow_mask, light_directions, eval_opts)

    if progress_callback:
        progress_callback(40)
        QCoreApplication.processEvents()

    # Using light strength estimation if available
    if 'light_strength_path' in config and config['light_strength_path']:
        print('Estimating normal vectors and albedo (with light strength estimation) ...')
        try:
            lambda_values = np.loadtxt(config['light_strength_path'])
            # Apply light strength to each light direction
            light_directions_scaled = light_directions.copy()
            for i in range(light_directions.shape[1]):
                light_directions_scaled[:, i] *= lambda_values[i]
            
            # Run photometric stereo with scaled light directions
            ps = PhotometricStereo(images, light_directions_scaled, shadow_mask)
            albedo, normals = ps.process()
            
            # Convert from (3, height, width) to (height, width, 3) format if needed
            if normals.shape[0] == 3:
                normals = np.transpose(normals, (1, 2, 0))
            
            # Evaluate normal estimate by intensity error - commented out as in MATLAB
            # eval_opts = {'display': True}
            # Ierr = eval_n_estimate_by_i_error(albedo, normals, images, shadow_mask, light_directions_scaled, eval_opts)
            
        except Exception as e:
            print(f"Error loading light strength file: {e}")
    
    if progress_callback:
        progress_callback(60)
        QCoreApplication.processEvents()
    
    # Straighten normals position
    normals = lay_normals(normals)
    
    # Process in two steps (like PSBoxComputeMaps2)
    # Step 1
    normals_step1 = compute_maps_step(config, normals, albedo, 1, progress_callback)
    
    if progress_callback:
        progress_callback(80)
        QCoreApplication.processEvents()
    
    # Step 2
    results = compute_maps_step(config, normals_step1, albedo, 2, progress_callback)
    
    if progress_callback:
        progress_callback(100)
        QCoreApplication.processEvents()
    
    print('*** Done. ***')
    return results


def compute_maps_step(config, normals, albedo, step, progress_callback=None):
    """
    Equivalent to PSBoxComputeMaps2 function.
    Process normals and albedo in two steps.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    normals : ndarray
        Normal map, shape (height, width, 3)
    albedo : ndarray
        Albedo map, shape (height, width)
    step : int
        Step number (1 or 2)
    progress_callback : function, optional
        Progress callback function
        
    Returns
    -------
    dict
        Dictionary with processed maps
    """
    downsample = config.get('downsample', False)
    
    # Subtract black error from normals
    black_error_path = config.get('default_black') if step == 1 else config.get('default_black_2')
    if black_error_path and os.path.exists(black_error_path):
        err_image = cv2.imread(black_error_path, cv2.IMREAD_UNCHANGED)
        if downsample:
            err_image = cv2.resize(err_image, (normals.shape[1], normals.shape[0]))
        
        # Determine bit depth for normalization
        divider = check_bit_depth(err_image)
        
        # Apply error correction
        err_image_float = err_image.astype(np.float32) / divider
        if len(err_image_float.shape) == 2:
            # If grayscale, apply to all channels
            normals = normals - err_image_float[:, :, np.newaxis]
        else:
            normals = normals - err_image_float
                                                                                                                                                                                                                                                                                                                                                              
    # Estimate depth map from normal vectors
    print('Estimating normal vectors...')
    # Calculate p, q (surface gradients)
    p = -normals[:, :, 0] / normals[:, :, 2]
    q = -normals[:, :, 1] / normals[:, :, 2]
    
    # Replace NaN values
    p[np.isnan(p)] = 0
    q[np.isnan(q)] = 0
    
    # Apply decimation if configured
    # This would be equivalent to checkDecimation in MATLAB
    p, q = check_decimation(config, p, q)
    
    # Compute depth from gradient
    Z = depth_from_gradient(p, q)
    
    # Level Z to the lowest value
    offset = np.min(Z)
    Z = Z - offset
    
    # Set depth to 0 for invalid normal vectors
    mask = np.logical_or.reduce((np.isnan(normals[:, :, 0]),
                                np.isnan(normals[:, :, 1]),
                                np.isnan(normals[:, :, 2])))
    Z[mask] = 0
    
    # Results dictionary
    results = {
        'normals': normals,
        'albedo': albedo,
        'depth': Z
    }
    
    if step == 1:
        # Apply polynomial correction
        # In MATLAB this was polyCorrection(config, Z, step)
        Z = poly_correction(Z, config=config, step=step, mask=~mask)
        
        # Recalculate gradients and normals from corrected depth map
        dz_dx, dz_dy = np.gradient(Z)
        h, w = Z.shape
        normals_corrected = np.zeros((h, w, 3), dtype=np.float32)
        normals_corrected[:, :, 0] = -dz_dx
        normals_corrected[:, :, 1] = -dz_dy
        normals_corrected[:, :, 2] = np.ones_like(Z)
        
        # Normalize normals
        norm_magnitudes = np.sqrt(normals_corrected[:, :, 0]**2 +
                                  normals_corrected[:, :, 1]**2 +
                                  normals_corrected[:, :, 2]**2)
        normals = normals_corrected / norm_magnitudes[:, :, np.newaxis]
        
        # Convert normals to RGB for visualization
        normals_rgb = normals_to_rgb(normals)
        results['normals'] = normals
        results['normals_rgb'] = normals_rgb
        
        # Save normals if configured
        if 'output_dir' in config:
            write_normals(normals_rgb, os.path.join(config['output_dir'], 'normals.png'))
            
        return normals  # Return normals for step 2
    
    elif step == 2:
        # Process all lights on image if available
        if 'all_lights_on_image_path' in config and os.path.exists(config['all_lights_on_image_path']):
            all_lights_on_image = cv2.imread(config['all_lights_on_image_path'], cv2.IMREAD_UNCHANGED)
            if config.get('downsample', False):
                all_lights_on_image = cv2.resize(all_lights_on_image, (albedo.shape[1], albedo.shape[0]))
            
            # Save albedo if configured
            if 'output_dir' in config:
                write_albedo(all_lights_on_image, albedo, 
                             os.path.join(config['output_dir'], 'albedo.png'))
                
                # Save reflection map
                write_reflection_map(all_lights_on_image, albedo,
                                    os.path.join(config['output_dir'], 'reflection_map.png'))
        
        # Compute mesh and save depth map
        if 'output_dir' in config:
            # Compute nodes and faces for mesh
            nodes, faces = compute_nodes_and_faces(Z)
            
            # Save depth map
            write_depth_map({'nodes': nodes, 'faces': faces},
                           os.path.join(config['output_dir'], 'depth_map.obj'))
            
            results['mesh'] = {'nodes': nodes, 'faces': faces}
    
    return results


def write_normals(normals_rgb, output_path):
    """
    Write normals RGB image to file
    Equivalent to writeNormals function in MATLAB
    
    Parameters
    ----------
    normals_rgb : ndarray
        RGB representation of normal map
    output_path : str
        Path to save the normal map image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved normal map to {output_path}")


def write_albedo(all_lights_on_image, albedo, output_path):
    """
    Write albedo map to file
    Equivalent to writeAlbedo function in MATLAB
    
    Parameters
    ----------
    all_lights_on_image : ndarray
        Image with all lights on
    albedo : ndarray
        Albedo map
    output_path : str
        Path to save the albedo map image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize albedo
    albedo_normalized = albedo / np.max(albedo)
    
    # Convert to 8-bit
    albedo_image = (albedo_normalized * 255).astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, albedo_image)
    print(f"Saved albedo map to {output_path}")


def write_reflection_map(all_lights_on_image, albedo, output_path):
    """
    Write reflection map to file
    Equivalent to writeReflectionMap function in MATLAB
    
    Parameters
    ----------
    all_lights_on_image : ndarray
        Image with all lights on
    albedo : ndarray
        Albedo map
    output_path : str
        Path to save the reflection map image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert all_lights_on_image to grayscale if it's color
    if len(all_lights_on_image.shape) == 3:
        grayscale = cv2.cvtColor(all_lights_on_image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = all_lights_on_image
    
    # Normalize grayscale image
    max_val = np.max(grayscale)
    divider = 65535.0 if max_val > 255 else 255.0
    grayscale = grayscale.astype(np.float32) / divider
    
    # Calculate reflection map (dividing grayscale by albedo)
    # Avoid division by zero
    epsilon = 1e-8
    reflection_map = np.zeros_like(grayscale)
    valid_mask = albedo > epsilon
    reflection_map[valid_mask] = grayscale[valid_mask] / (albedo[valid_mask] + epsilon)
    
    # Normalize reflection map
    reflection_map = reflection_map / np.max(reflection_map)
    
    # Convert to 8-bit
    reflection_image = (reflection_map * 255).astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, reflection_image)
    print(f"Saved reflection map to {output_path}")


def compute_nodes_and_faces(depth_map):
    """
    Compute mesh nodes and faces from depth map
    Equivalent to computeNodesAndFaces function in MATLAB
    
    Parameters
    ----------
    depth_map : ndarray
        Depth map
        
    Returns
    -------
    tuple
        (nodes, faces) - nodes are 3D points, faces are triangle indices
    """
    h, w = depth_map.shape
    
    # Create coordinate grids
    y, x = np.mgrid[:h, :w]
    
    # Create nodes (x, y, z coordinates)
    nodes = np.zeros((h * w, 3))
    nodes[:, 0] = x.flatten()
    nodes[:, 1] = y.flatten()
    nodes[:, 2] = depth_map.flatten()
    
    # Create faces (triangular mesh)
    points = np.column_stack((x.flatten(), y.flatten()))
    tri = Delaunay(points)
    faces = tri.simplices
    
    return nodes, faces


def write_depth_map(mesh, output_path):
    """
    Write depth map as OBJ file
    Equivalent to writeDepthMap function in MATLAB
    
    Parameters
    ----------
    mesh : dict
        Dictionary with nodes and faces
    output_path : str
        Path to save the OBJ file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nodes = mesh['nodes']
    faces = mesh['faces']
    
    # Write OBJ file
    with open(output_path, 'w') as f:
        # Write header
        f.write("# OBJ file created by normal_maps.py\n")
        
        # Write vertices
        for node in nodes:
            f.write(f"v {node[0]} {node[1]} {node[2]}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Saved depth map to {output_path}")


def load_processed_images(image_paths, options=None):
    """
    Load and process input images for photometric stereo.
    Python equivalent of PSLoadProcessedImagesTheMatlabWay.m
    
    Parameters
    ----------
    image_paths : list
        List of file paths to the input images
    options : dict
        Options for image loading:
            - image_channel: The image channel to use (0, 1, or 2)
            - resample: Whether to downsample images by factor of 10
            - normalize_percentile: If provided, images will be normalized to this percentile
        
    Returns
    -------
    ndarray
        Processed images as ndarray (height, width, num_images)
    """
    if options is None:
        options = {}
    
    # Set default options
    image_channel = options.get('image_channel', 0)  # Default to first channel (0 in Python)
    resample = options.get('resample', False)
    
    # Get image names and dimension
    num_images = len(image_paths)
    
    # Load first image to get dimensions
    img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    if resample:
        img = cv2.resize(img, None, fx=0.1, fy=0.1)
    
    # Get dimensions
    if len(img.shape) == 2:
        height, width = img.shape
        depth = 1
    else:
        height, width, depth = img.shape
    
    # Initialize output array
    I = np.zeros((height, width, num_images), dtype=np.float32)
    
    # Load images
    for i in range(num_images):
        # Read image and convert to double
        img_tmp = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
        img_tmp = img_tmp.astype(np.float32)
        
        # Determine divider based on bit depth
        if np.max(img_tmp) > 1000:
            divider = 65536.0
        else:
            divider = 255.0
        
        # Normalize to [0, 1]
        img_tmp = img_tmp / divider
        
        # Check if we need to select a specific channel
        if depth > 1 and image_channel < depth:
            # The MATLAB code does img_tmp(end:-1:1, :, options.ImageChannel)
            # This flips the image vertically and selects a specific channel
            img_tmp = np.flip(img_tmp, axis=0)
            img_tmp = img_tmp[:, :, image_channel]
        
        # Resize if requested
        if resample:
            img_tmp = cv2.resize(img_tmp, (width, height))
        
        # Apply percentile normalization if requested
        if 'normalize_percentile' in options:
            pct = np.percentile(img_tmp, options['normalize_percentile'])
            if pct > 0:  # Avoid division by zero
                img_tmp = img_tmp / pct
        
        # Store the processed image
        I[:, :, i] = img_tmp
    
    return I


def normals_to_rgb(normals):
    """
    Convert normal vectors to RGB colors for visualization.
    
    Parameters
    ----------
    normals : ndarray
        Normal map, shape (height, width, 3) with values in [-1, 1]
        
    Returns
    -------
    ndarray
        RGB image, shape (height, width, 3) with values in [0, 255]
    """
    # Scale normals from [-1, 1] to [0, 1]
    rgb = (normals + 1) / 2.0
    
    # Scale to [0, 255] range and convert to uint8
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def compute_normal_maps(images, light_directions, method="standard"):
    """
    Compute normal map from images using photometric stereo
    
    Args:
        images: List of input images
        light_directions: List of light direction vectors
        method: Method to use for normal map computation
        
    Returns:
        tuple: (normal_map, albedo)
    """
    # Placeholder implementation
    if not images or not light_directions:
        raise ValueError("Images and light directions must not be empty")
        
    height, width = images[0].shape[:2]
    normal_map = np.zeros((height, width, 3))
    albedo = np.zeros((height, width))
    
    # In a real implementation, this would compute the normal map using photometric stereo
    
    return normal_map, albedo


import numpy as np
import cv2
import os

def load_processed_images(image_paths, options=None):
    """
    Load and process input images for photometric stereo.
    Python equivalent of PSLoadProcessedImagesTheMatlabWay.m
    
    Parameters
    ----------
    image_paths : list
        List of file paths to the input images
    options : dict
        Options for image loading:
            - image_channel: The image channel to use (0, 1, or 2)
            - resample: Whether to downsample images by factor of 10
            - normalize_percentile: If provided, images will be normalized to this percentile
        
    Returns
    -------
    ndarray
        Processed images as ndarray (height, width, num_images)
    """
    if options is None:
        options = {}
    
    # Set default options
    image_channel = options.get('image_channel', 0)  # Default to first channel (0 in Python)
    resample = options.get('resample', False)
    
    # Get image names and dimension
    num_images = len(image_paths)
    
    # Load first image to get dimensions
    img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    if resample:
        img = cv2.resize(img, None, fx=0.1, fy=0.1)
    
    # Get dimensions
    if len(img.shape) == 2:
        height, width = img.shape
        depth = 1
    else:
        height, width, depth = img.shape
    
    # Initialize output array
    I = np.zeros((height, width, num_images), dtype=np.float32)
    
    # Load images
    for i in range(num_images):
        # Read image and convert to double
        img_tmp = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
        img_tmp = img_tmp.astype(np.float32)
        
        # Determine divider based on bit depth
        if np.max(img_tmp) > 1000:
            divider = 65536.0
        else:
            divider = 255.0
        
        # Normalize to [0, 1]
        img_tmp = img_tmp / divider
        
        # Check if we need to select a specific channel
        if depth > 1 and image_channel < depth:
            # The MATLAB code does img_tmp(end:-1:1, :, options.ImageChannel)
            # This flips the image vertically and selects a specific channel
            img_tmp = np.flip(img_tmp, axis=0)
            img_tmp = img_tmp[:, :, image_channel]
        
        # Resize if requested
        if resample:
            img_tmp = cv2.resize(img_tmp, (width, height))
        
        # Apply percentile normalization if requested
        if 'normalize_percentile' in options:
            pct = np.percentile(img_tmp, options['normalize_percentile'])
            if pct > 0:  # Avoid division by zero
                img_tmp = img_tmp / pct
        
        # Store the processed image
        I[:, :, i] = img_tmp
    
    return I


def eval_n_estimate_by_i_error(rho, n, I, mask, light_directions, options=None):
    """
    Evaluate scaled normal estimation by intensity error.
    Python equivalent of EvalNEstimateByIError.m
    
    Parameters
    ----------
    rho : ndarray
        Albedo map, shape (height, width)
    n : ndarray
        Normal map, shape (height, width, 3) or (3, height, width)
    I : ndarray
        Input images, shape (height, width, num_images)
    mask : ndarray
        Shadow mask, shape (height, width, num_images)
    light_directions : ndarray
        Light directions, shape (3, num_images)
    options : dict, optional
        Options:
        - display: whether to print error statistics
        
    Returns
    -------
    ndarray
        Error map, shape (height, width)
    """
    if options is None:
        options = {'display': False}
    
    # Handle different dimension layouts for n
    if n.shape[0] == 3:
        n = np.transpose(n, (1, 2, 0))
    
    # Get dimensions
    height, width, num_images = I.shape
    N = height * width
    
    # Resize (vectorize) the input
    I_reshaped = I.reshape(N, num_images)
    mask_reshaped = mask.reshape(N, num_images)
    n_reshaped = n.reshape(N, 3)
    
    # Calculate b = rho * n
    rho_expanded = np.repeat(rho.reshape(N, 1), 3, axis=1)
    b = rho_expanded * n_reshaped
    
    # Compute error map
    Ierr = np.zeros(N)
    for i in range(num_images):
        Ierr_i = I_reshaped[:, i] - np.dot(b, light_directions[:, i])
        Ierr_i[~mask_reshaped[:, i]] = 0
        Ierr = Ierr + Ierr_i**2
    
    # Compute RMS and reshape
    mask_sum = np.sum(mask_reshaped, axis=1)
    
    # Avoid division by zero
    mask_sum[mask_sum == 0] = np.nan
    Ierr = np.sqrt(Ierr / mask_sum)
    Ierr = Ierr.reshape(height, width)
    
    # Print error statistics
    if options.get('display', False):
        Ierr_valid = Ierr[np.isfinite(Ierr)]
        print('Evaluate scaled normal estimation by intensity error:')
        print(f'  RMS = {np.sqrt(np.mean(Ierr_valid**2)):.4f}')
        print(f'  Mean = {np.mean(Ierr_valid):.4f}')
        print(f'  Median = {np.median(Ierr_valid):.4f}')
        print(f'  90 percentile = {np.percentile(Ierr_valid, 90):.4f}')
        print(f'  Max = {np.max(Ierr_valid):.4f}')
    
    return Ierr