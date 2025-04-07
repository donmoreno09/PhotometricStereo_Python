"""
Image processing utilities.
Python implementation of various image manipulation MATLAB scripts:
- specularizeX.m
- imNrm.m
- reRange.m
- renderNormalsInRGB.m
- normalizzazioneNormale.m
- layNormals.m
"""

import numpy as np
import cv2


def specularize_x(matrix_in):
    """
    Flip matrix vertically (along x-axis).
    Python implementation of specularizeX.m
    
    Parameters
    ----------
    matrix_in : ndarray
        Input matrix or image
        
    Returns
    -------
    ndarray
        Vertically flipped matrix or image
    """
    # Check number of dimensions
    if len(matrix_in.shape) == 3:
        # For RGB images
        return matrix_in[::-1, :, :]
    else:
        # For grayscale images or 2D matrices
        return matrix_in[::-1, :]


def normalize_image(image_in):
    """
    Normalize image to [0, 1] range.
    Python implementation of imNrm.m
    
    Parameters
    ----------
    image_in : ndarray
        Input image
        
    Returns
    -------
    ndarray
        Normalized image in [0, 1] range
    """
    image_out = image_in.astype(np.float64)
    max_val = np.max(image_out)
    
    if max_val > 0:
        image_out /= max_val
    
    return image_out


def re_range(image_in):
    """
    Bring image to [0, 1] range without rescaling.
    Python implementation of reRange.m
    
    Parameters
    ----------
    image_in : ndarray
        Input image
        
    Returns
    -------
    ndarray
        Image with values clamped to [0, 1] range
    """
    # Convert to float
    image_out = image_in.astype(np.float64)
    
    # Determine bit depth (8-bit or 16-bit)
    max_val = np.max(image_out)
    if max_val > 255:
        image_out /= 65535.0  # 16-bit image
    else:
        image_out /= 255.0  # 8-bit image
    
    # Clamp to [0, 1]
    image_out = np.clip(image_out, 0.0, 1.0)
    
    return image_out


def render_normals_in_rgb(normals):
    """
    Render normal vectors as RGB colors.
    Python implementation of renderNormalsInRGB.m
    
    Parameters
    ----------
    normals : ndarray
        Normal vectors, either a single vector [x, y, z] or
        an image of normal vectors (height, width, 3)
        
    Returns
    -------
    ndarray
        RGB representation of normal vectors in uint8 format
    """
    # Check if input is a single vector
    if normals.ndim == 1:
        if len(normals) != 3:
            raise ValueError("Single normal vector must have 3 components")
        
        rgb = np.zeros(3, dtype=np.uint8)
        
        # Map [-1, 1] to [0, 255]
        rgb[0] = np.round((normals[0] + 1) * 127.5)
        rgb[1] = np.round((normals[1] + 1) * 127.5)
        rgb[2] = np.round((normals[2]) * 255)  # Z is [0, 1]
        
        return rgb
    
    # Input is an image of normal vectors
    elif normals.ndim == 3 and normals.shape[2] == 3:
        rgb = np.zeros_like(normals, dtype=np.uint8)
        
        # Map [-1, 1] to [0, 255] for X and Y
        rgb[:, :, 0] = np.round((normals[:, :, 0] + 1) * 127.5)
        rgb[:, :, 1] = np.round((normals[:, :, 1] + 1) * 127.5)
        # Map [0, 1] to [0, 255] for Z
        rgb[:, :, 2] = np.round(normals[:, :, 2] * 255)
        
        return rgb
    
    else:
        raise ValueError("Input must be a single normal vector or an image of normal vectors")


def normalize_normals(normals, leave_b_at_full_range=False):
    """
    Normalize normal vectors to unit length.
    Python implementation of normalizzazioneNormale.m
    
    Parameters
    ----------
    normals : ndarray
        Normal vectors, either a single vector [x, y, z] or
        an image of normal vectors (height, width, 3)
    leave_b_at_full_range : bool, optional
        If True, doesn't normalize the Z component
        
    Returns
    -------
    ndarray
        Normalized normal vectors
    """
    # Check if input is a single vector
    if normals.ndim == 1:
        normalized = normals.copy().astype(np.float64)
        norm = np.sqrt(normalized[0]**2 + normalized[1]**2 + (0 if leave_b_at_full_range else normalized[2]**2))
        
        if norm > 0:
            normalized[0] /= norm
            normalized[1] /= norm
            if not leave_b_at_full_range:
                normalized[2] /= norm
        
        return normalized
    
    # Input is an image of normal vectors
    elif normals.ndim == 3 and normals.shape[2] == 3:
        normalized = normals.copy().astype(np.float64)
        
        # Calculate norm for each pixel
        if leave_b_at_full_range:
            # Only use X and Y for normalization
            norm = np.sqrt(normalized[:, :, 0]**2 + normalized[:, :, 1]**2)
        else:
            # Use all components for normalization
            norm = np.sqrt(np.sum(normalized**2, axis=2))
        
        # Avoid division by zero
        mask = norm > 0
        
        # Normalize each component
        normalized[:, :, 0][mask] = normalized[:, :, 0][mask] / norm[mask]
        normalized[:, :, 1][mask] = normalized[:, :, 1][mask] / norm[mask]
        
        if not leave_b_at_full_range:
            normalized[:, :, 2][mask] = normalized[:, :, 2][mask] / norm[mask]
        
        return normalized
    
    else:
        raise ValueError("Input must be a single normal vector or an image of normal vectors")


def lay_normals(normals):
    """
    Adjust normal vectors by adding a mean component.
    Python implementation of layNormals.m
    
    Parameters
    ----------
    normals : ndarray
        Image of normal vectors (height, width, 3)
        
    Returns
    -------
    ndarray
        Adjusted normal vectors
    """
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError("Input must be an image of normal vectors")
    
    # Calculate mean values for each channel
    mean_normals = np.mean(normals, axis=(0, 1))
    
    # Adjust the Z component
    adjusted = normals.copy()
    adjusted[:, :, 2] = adjusted[:, :, 2] + mean_normals[2] * 0.1
    
    # Re-normalize
    return normalize_normals(adjusted)


def compute_images_mean(array_of_images):
    """
    Compute the mean of multiple images.
    Python implementation of computeImagesMean.m
    
    Parameters
    ----------
    array_of_images : list or ndarray
        List of images or 3D array with images stacked along 3rd dimension
        
    Returns
    -------
    ndarray
        Mean image
    """
    if isinstance(array_of_images, list):
        n_img = len(array_of_images)
        if n_img == 0:
            return None
        
        # Initialize with first image
        image_out = array_of_images[0].astype(np.float64)
        
        # Add remaining images
        for i in range(1, n_img):
            image_out += array_of_images[i]
        
        # Compute mean
        image_out /= n_img
        
    else:
        # 3D array with images stacked along 3rd dimension
        if array_of_images.ndim != 3:
            raise ValueError("Input array must be 3D with images stacked along 3rd dimension")
        
        # Compute mean along 3rd dimension
        image_out = np.mean(array_of_images, axis=2)
    
    return image_out


def check_bit_depth(image):
    """
    Determine the bit depth of an image.
    Python implementation of checkBithdepth.m
    
    Parameters
    ----------
    image : ndarray
        Input image
        
    Returns
    -------
    int
        Divider for normalization: 255 for 8-bit, 65535 for 16-bit
    """
    max_val = np.max(image)
    
    if max_val > 255:
        return 65535  # 16-bit image
    else:
        return 255  # 8-bit image
    
def poly_correction(depth_map, order=2, mask=None):
    """
    Correct depth map with polynomial fitting to remove global shape bias.
    Python implementation of polyCorrection.m
    
    Parameters
    ----------
    depth_map : ndarray
        Input depth map
    order : int, optional
        Order of polynomial fit (1=planar, 2=quadratic)
    mask : ndarray, optional
        Binary mask indicating valid pixels
        
    Returns
    -------
    ndarray
        Corrected depth map
    """
    # Check input
    if depth_map is None:
        raise ValueError("Depth map is None")
    
    # If no mask provided, use all pixels
    if mask is None:
        mask = np.ones_like(depth_map, dtype=bool)
    
    # Get image dimensions
    h, w = depth_map.shape
    
    # Create x, y meshgrid for polynomial fitting
    y, x = np.mgrid[:h, :w]
    
    # Normalize coordinates to [-1, 1] range for numerical stability
    x_norm = (x / w * 2) - 1
    y_norm = (y / h * 2) - 1
    
    # Get masked coordinates and depth values
    valid_mask = np.logical_and(mask, np.isfinite(depth_map))
    x_valid = x_norm[valid_mask]
    y_valid = y_norm[valid_mask]
    z_valid = depth_map[valid_mask]
    
    # Construct design matrix for polynomial fitting
    if order == 1:
        # Planar fit: z = a*x + b*y + c
        A = np.column_stack([x_valid, y_valid, np.ones_like(x_valid)])
    elif order == 2:
        # Quadratic fit: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
        A = np.column_stack([
            x_valid**2, y_valid**2, x_valid*y_valid,
            x_valid, y_valid, np.ones_like(x_valid)
        ])
    else:
        raise ValueError(f"Unsupported polynomial order: {order}")
    
    # Solve least squares problem
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_valid, rcond=None)
    
    # Create polynomial surface for the entire image
    if order == 1:
        # Planar surface
        poly_surface = (
            coeffs[0] * x_norm +
            coeffs[1] * y_norm +
            coeffs[2]
        )
    elif order == 2:
        # Quadratic surface
        poly_surface = (
            coeffs[0] * x_norm**2 +
            coeffs[1] * y_norm**2 +
            coeffs[2] * x_norm * y_norm +
            coeffs[3] * x_norm +
            coeffs[4] * y_norm +
            coeffs[5]
        )
    
    # Subtract polynomial surface from depth map
    corrected_depth = depth_map - poly_surface
    
    # Restore original mean height
    corrected_depth = corrected_depth + np.mean(z_valid)
    
    return corrected_depth


def decode_rgb_to_normals(rgb):
    """
    Convert RGB image to normal vectors.
    Python implementation of decodeRGBToNormals.m
    
    Parameters
    ----------
    rgb : ndarray
        RGB image, either a single pixel [r, g, b] or
        an image of pixels (height, width, 3)
        
    Returns
    -------
    ndarray
        Normal vectors
    """
    # Check if input is a single pixel
    if rgb.ndim == 1:
        if len(rgb) != 3:
            raise ValueError("Single pixel must have 3 components")
        
        nn = np.zeros(3, dtype=np.float64)
        
        # Map [0, 255] to [-1, 1] for X and Y
        nn[0] = (rgb[0] / 255.0) * 2 - 1
        nn[1] = (rgb[1] / 255.0) * 2 - 1
        # Map [0, 255] to [0, 1] for Z
        nn[2] = rgb[2] / 255.0
        
        return nn
    
    # Input is an RGB image
    elif rgb.ndim == 3 and rgb.shape[2] == 3:
        nn = np.zeros_like(rgb, dtype=np.float64)
        
        # Map [0, 255] to [-1, 1] for X and Y
        nn[:, :, 0] = (rgb[:, :, 0] / 255.0) * 2 - 1
        nn[:, :, 1] = (rgb[:, :, 1] / 255.0) * 2 - 1
        # Map [0, 255] to [0, 1] for Z
        nn[:, :, 2] = rgb[:, :, 2] / 255.0
        
        return nn
    
    else:
        raise ValueError("Input must be a single RGB pixel or an RGB image")


"""
Utility functions for image processing
"""
import numpy as np

def specularize_x(image):
    """
    Placeholder function for image processing
    
    Args:
        image: Input image
        
    Returns:
        processed_image: Processed image
    """
    # Return a copy of the input image
    return np.copy(image) if image is not None else None