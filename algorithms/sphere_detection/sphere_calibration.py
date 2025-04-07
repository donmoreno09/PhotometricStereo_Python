"""
Chrome sphere calibration for light direction estimation.
Python implementation of:
- FindLightDirectionFromChromeSphere.m
- PSFindLightDirection.m
"""

import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import convolve2d

from .circle_fitting import fit_circle


class ChromeSphereCalibration:
    """Chrome sphere calibration for light direction estimation"""
    
    def __init__(self, options=None):
        """
        Initialize chrome sphere calibration.
        
        Parameters
        ----------
        options : dict, optional
            Calibration options
        """
        self.options = {} if options is None else options
        self.circle = None  # (x, y, r)
    
    def detect_sphere(self, image):
        """
        Detect chrome sphere in an image.
        
        Parameters
        ----------
        image : ndarray
            Input image
            
        Returns
        -------
        tuple
            (x, y, r) - Circle parameters
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize image to [0, 255]
        if gray.dtype != np.uint8:
            gray = ((gray - np.min(gray)) / (np.max(gray) - np.min(gray)) * 255).astype(np.uint8)
        
        # Method from options
        method = self.options.get('detection_method', 'hough')
        
        if method == 'hough':
            # Use Hough transform for circle detection
            self.circle = self._detect_sphere_hough(gray)
        elif method == 'contour':
            # Use contour detection
            self.circle = self._detect_sphere_contour(gray)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        return self.circle
    
    def _detect_sphere_hough(self, gray):
        """
        Detect sphere using Hough circles.
        
        Parameters
        ----------
        gray : ndarray
            Grayscale image
            
        Returns
        -------
        tuple
            (x, y, r) - Circle parameters
        """
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50,
            param1=50,  # Edge detection threshold
            param2=30,  # Circle detection threshold
            minRadius=20, 
            maxRadius=int(min(gray.shape) / 2)
        )
        
        if circles is None or len(circles) == 0:
            raise ValueError("No chrome sphere detected using Hough transform.")
        
        # Get the most prominent circle
        circle = circles[0, 0]
        x, y, r = circle
        
        return x, y, r
    
    def _detect_sphere_contour(self, gray):
        """
        Detect sphere using contour detection.
        
        Parameters
        ----------
        gray : ndarray
            Grayscale image
            
        Returns
        -------
        tuple
            (x, y, r) - Circle parameters
        """
        # Apply threshold to separate potential sphere from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found for chrome sphere detection.")
        
        # Find the largest contour (likely the sphere)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit circle to contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        return x, y, radius
    
    def find_light_direction(self, image, circle=None, threshold=None):
        """
        Find light direction from chrome sphere image.
        
        Parameters
        ----------
        image : ndarray
            Chrome sphere image
        circle : tuple, optional
            Circle parameters (x, y, r)
        threshold : float, optional
            Threshold for highlight detection
            
        Returns
        -------
        ndarray
            Light direction vector (3,)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to [0, 1]
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float32) / 255
        elif gray.dtype == np.uint16:
            gray = gray.astype(np.float32) / 65535
        
        # Use provided circle or detect it
        if circle is None:
            if self.circle is None:
                circle = self.detect_sphere(image)
            else:
                circle = self.circle
        
        # Extract circle parameters
        x_c, y_c, radius = circle
        
        # Determine threshold if not provided
        if threshold is None:
            # Create a mask for the sphere
            height, width = gray.shape
            y_grid, x_grid = np.mgrid[:height, :width]
            mask = (x_grid - x_c)**2 + (y_grid - y_c)**2 <= radius**2
            
            # Extract pixel values inside the sphere
            sphere_values = gray[mask]
            
            # Use percentile for threshold
            percentile = self.options.get('highlight_percentile', 95)
            threshold = np.percentile(sphere_values, percentile)
        
        # Find highlight points
        highlight_mask = (gray > threshold) & ((x_grid - x_c)**2 + (y_grid - y_c)**2 <= radius**2)
        
        # Check if any highlights found
        highlight_points = np.where(highlight_mask)
        if len(highlight_points[0]) == 0:
            # If no highlights with current threshold, try a lower one
            threshold = np.percentile(gray[mask], 90)
            highlight_mask = (gray > threshold) & ((x_grid - x_c)**2 + (y_grid - y_c)**2 <= radius**2)
            highlight_points = np.where(highlight_mask)
            
            if len(highlight_points[0]) == 0:
                raise ValueError("No highlight points found. Try reducing the threshold.")
        
        # Compute centroid of highlight region
        y_h = np.mean(highlight_points[0])
        x_h = np.mean(highlight_points[1])
        
        # Convert from image coordinates to sphere coordinates
        x = x_h - x_c
        y = y_c - y_h  # Flip y-axis to match 3D coordinate system
        
        # Compute z coordinate on sphere
        z = np.sqrt(radius**2 - x**2 - y**2)
        
        # Create and normalize direction vector
        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def process_multiple_images(self, images):
        """
        Process multiple chrome sphere images to find light directions.
        
        Parameters
        ----------
        images : list
            List of chrome sphere images
            
        Returns
        -------
        ndarray
            Light directions, shape (3, num_images)
        """
        num_images = len(images)
        light_directions = np.zeros((3, num_images))
        
        # Detect chrome sphere in first image
        if self.circle is None:
            self.detect_sphere(images[0])
        
        # Find light direction for each image
        for i in range(num_images):
            try:
                light_directions[:, i] = self.find_light_direction(images[i])
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Use a default light direction as fallback
                light_directions[:, i] = np.array([0, 0, 1])
        
        return light_directions