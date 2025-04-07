"""
Functions for finding light directions from chrome sphere images.
Python implementation of FindLightDirectionFromChromeSphere.m
"""

import numpy as np
import cv2
from scipy.optimize import least_squares

from algorithms.sphere_detection.circle_fitting import fit_circle


def find_light_direction_from_chrome_sphere(image, circle=None, threshold=None, options=None):
    """
    Find light direction from a chrome sphere image.
    
    Parameters
    ----------
    image : ndarray
        Chrome sphere image, shape (height, width) or (height, width, 3)
    circle : tuple, optional
        Circle parameters (x, y, r) of the chrome sphere
    threshold : float, optional
        Threshold for highlight detection
    options : dict, optional
        Additional options for light direction finding
        
    Returns
    -------
    ndarray
        Light direction vector, shape (3,)
    """
    # Convert color image to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    # Find circle if not provided
    if circle is None:
        circle = detect_chrome_sphere(gray_image)
    
    # Extract circle parameters
    x_c, y_c, radius = circle
    
    # Determine threshold if not provided
    if threshold is None:
        # Use a simple method to estimate threshold
        # This could be improved with more sophisticated techniques
        highlight_region = extract_highlight_region(gray_image, circle)
        if highlight_region.size > 0:
            threshold = np.mean(highlight_region) * 0.9
        else:
            threshold = 0.9 * np.max(gray_image)
    
    # Find highlight points
    highlight_mask = gray_image > threshold
    y_indices, x_indices = np.where(highlight_mask)
    
    # Check if any highlights found
    if len(y_indices) == 0:
        raise ValueError("No highlight points found. Try reducing the threshold.")
    
    # Compute centroid of highlight region
    x_h = np.mean(x_indices)
    y_h = np.mean(y_indices)
    
    # Convert from image coordinates to sphere coordinates
    x = x_h - x_c
    y = y_h - y_c
    
    # Compute z coordinate on sphere
    z = np.sqrt(radius**2 - x**2 - y**2)
    
    # Normalize to get direction vector
    light_dir = np.array([x, y, z])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    return light_dir


def detect_chrome_sphere(image):
    """
    Detect chrome sphere in an image using circle detection.
    
    Parameters
    ----------
    image : ndarray
        Grayscale image, shape (height, width)
        
    Returns
    -------
    tuple
        Circle parameters (x, y, r)
    """
    # Use Hough circles to detect sphere
    circles = cv2.HoughCircles(
        image.astype(np.uint8), 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50,
        param1=50,  # Edge detection threshold
        param2=30,  # Circle detection threshold
        minRadius=20, 
        maxRadius=500
    )
    
    if circles is None or len(circles) == 0:
        raise ValueError("No chrome sphere detected.")
    
    # Get the most prominent circle
    circle = circles[0, 0]
    x_c, y_c, radius = circle
    
    return (x_c, y_c, radius)


def extract_highlight_region(image, circle):
    """
    Extract the highlight region within the circle.
    
    Parameters
    ----------
    image : ndarray
        Grayscale image, shape (height, width)
    circle : tuple
        Circle parameters (x, y, r)
        
    Returns
    -------
    ndarray
        Intensity values in the highlight region
    """
    height, width = image.shape
    x_c, y_c, radius = circle
    
    # Create a mask for the circle
    y_grid, x_grid = np.mgrid[:height, :width]
    mask = (x_grid - x_c)**2 + (y_grid - y_c)**2 <= radius**2
    
    # Extract pixel values inside the circle
    circle_values = image[mask]
    
    # Determine the highlight region (e.g., top 10% brightest pixels)
    if circle_values.size > 0:
        threshold = np.percentile(circle_values, 90)
        highlight = circle_values[circle_values > threshold]
    else:
        highlight = np.array([])
    
    return highlight


def fit_sphere_from_points(x, y, z):
    """
    Fit a sphere to 3D points.
    
    Parameters
    ----------
    x, y, z : ndarray
        Coordinates of points on the sphere surface
        
    Returns
    -------
    tuple
        Sphere parameters (x0, y0, z0, r) - center coordinates and radius
    """
    def sphere_error(params, x, y, z):
        x0, y0, z0, r = params
        return (x - x0)**2 + (y - y0)**2 + (z - z0)**2 - r**2
    
    # Initial guess: centroid and mean distance from centroid
    x0 = np.mean(x)
    y0 = np.mean(y)
    z0 = np.mean(z)
    r0 = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2))
    
    # Optimize sphere parameters using least squares
    params_init = [x0, y0, z0, r0]
    result = least_squares(sphere_error, params_init, args=(x, y, z))
    
    return result.x


def process_chrome_sphere_images(images, manual_selection=False):
    """
    Process multiple chrome sphere images to find light directions.
    
    Parameters
    ----------
    images : list
        List of chrome sphere images
    manual_selection : bool, optional
        Whether to use manual selection for chrome sphere detection
        
    Returns
    -------
    ndarray
        Light directions, shape (3, num_images)
    """
    num_images = len(images)
    light_directions = np.zeros((3, num_images))
    
    # Detect chrome sphere in first image
    if manual_selection:
        # In a real application, this would show the image and let the user select points
        # For this example, just use automatic detection
        circle = detect_chrome_sphere(images[0])
    else:
        circle = detect_chrome_sphere(images[0])
    
    # Find light direction for each image
    for i in range(num_images):
        try:
            light_directions[:, i] = find_light_direction_from_chrome_sphere(images[i], circle)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            # Use a default light direction as fallback
            light_directions[:, i] = np.array([0, 0, 1])
    
    return light_directions


def refine_light_directions(images, initial_light_dirs, mask=None):
    """
    Refine light directions using nonlinear optimization.
    
    Parameters
    ----------
    images : ndarray
        Input images stacked along 3rd dimension, shape (height, width, num_images)
    initial_light_dirs : ndarray
        Initial light directions, shape (3, num_images)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
        
    Returns
    -------
    ndarray
        Refined light directions, shape (3, num_images)
    """
    # This would be an implementation of PSRefineLight.m
    # For now, just return the initial light directions
    return initial_light_dirs

def ps_compute_lights(images, circle_center, circle_radius, refraction_index=None, epsilon=1e-6):
    """
    Compute light directions from sphere highlights.
    Python implementation of psBoxComputeLights.m
    
    Parameters
    ----------
    images : list
        List of images containing the chrome sphere
    circle_center : tuple
        (x, y) center of the chrome sphere
    circle_radius : float
        Radius of the chrome sphere
    refraction_index : float, optional
        Refraction index for the chrome sphere material
        If None, assume perfect reflection
    epsilon : float, optional
        Small value to avoid division by zero
        
    Returns
    -------
    ndarray
        Light directions, shape (num_images, 3)
    """
    num_images = len(images)
    light_directions = np.zeros((num_images, 3))
    
    # Convert center to numpy array
    center = np.array(circle_center)
    
    for i, image in enumerate(images):
        # Find highlight (brightest point) in the image
        # First, mask outside the sphere
        h, w = image.shape[:2]
        y, x = np.mgrid[:h, :w]
        mask = (x - center[0])**2 + (y - center[1])**2 <= circle_radius**2
        
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply mask
        masked = gray.copy()
        masked[~mask] = 0
        
        # Find brightest point
        max_loc = np.unravel_index(np.argmax(masked), masked.shape)
        highlight_y, highlight_x = max_loc
        
        # Compute light direction from highlight
        # First, calculate the normalized position on the sphere
        highlight_pos = np.array([highlight_x, highlight_y])
        
        # Vector from sphere center to highlight
        v = highlight_pos - center
        
        # Normalize to sphere radius
        v_norm = np.linalg.norm(v)
        if v_norm < epsilon:
            # Highlight at center, assume light from above
            light_dir = np.array([0, 0, 1])
        else:
            v = v / v_norm * circle_radius
            
            # Convert to sphere coordinates (assuming sphere is at origin)
            # We know x and y, need to compute z
            x = v[0]
            y = v[1]
            z = np.sqrt(max(0, circle_radius**2 - x**2 - y**2))
            
            # Normal vector at the highlight point
            normal = np.array([x, y, z]) / circle_radius
            
            if refraction_index is None:
                # Perfect reflection: Reflect view direction (0,0,1) around normal
                view_dir = np.array([0, 0, 1])
                light_dir = 2 * np.dot(normal, view_dir) * normal - view_dir
            else:
                # Account for refraction
                # This is a simplification - full refraction would require more complex calculation
                view_dir = np.array([0, 0, 1])
                # Approximate considering refraction
                light_dir = normal
            
            # Ensure unit length
            light_dir = light_dir / np.linalg.norm(light_dir)
        
        light_directions[i] = light_dir
    
    return light_directions


def ps_estimate_light_strength(images, normals, albedo, light_dirs, mask=None):
    """
    Estimate light strength from images, normals, and albedo.
    Python implementation of PSEstimateLightStrength.m
    
    Parameters
    ----------
    images : list or ndarray
        List of source images or stacked array (h, w, num_images)
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
        Light strengths, shape (num_images,)
    """
    # Check if images is a list or array
    if isinstance(images, list):
        # Convert list to array
        h, w = images[0].shape[:2]
        num_images = len(images)
        
        # Check if grayscale or color
        if len(images[0].shape) > 2:
            # Color images - convert to grayscale
            stacked_images = np.zeros((h, w, num_images), dtype=np.float32)
            for i, img in enumerate(images):
                stacked_images[:, :, i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            stacked_images = np.stack([img for img in images], axis=-1)
    else:
        # Already an array
        stacked_images = images
    
    # Get dimensions
    h, w, num_images = stacked_images.shape
    
    # Create default mask if none provided
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    
    # Create system of equations: I = albedo * (n·l) * light_strength
    # We'll solve for light_strength using least squares
    
    # Initialize matrix A and vector b
    num_valid_pixels = np.sum(mask)
    A = np.zeros((num_valid_pixels * num_images, num_images))
    b = np.zeros(num_valid_pixels * num_images)
    
    # Fill matrix A and vector b
    row_idx = 0
    for img_idx in range(num_images):
        # Get light direction for this image
        light_dir = light_dirs[img_idx]
        
        # Calculate n·l for each pixel
        n_dot_l = np.sum(normals * light_dir.reshape(1, 1, 3), axis=2)
        
        # Get intensities for this image
        img = stacked_images[:, :, img_idx]
        
        # Apply mask
        masked_n_dot_l = n_dot_l[mask]
        masked_albedo = albedo[mask]
        masked_img = img[mask]
        
        # Fill matrix A
        A[row_idx:row_idx+num_valid_pixels, img_idx] = masked_albedo * masked_n_dot_l
        
        # Fill vector b
        b[row_idx:row_idx+num_valid_pixels] = masked_img
        
        row_idx += num_valid_pixels
    
    # Solve system using least squares
    light_strengths, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    return light_strengths