"""
Core photometric stereo algorithm for normal and albedo calculation.
Python implementation of PhotometricStereo.m
"""

import numpy as np
from scipy import linalg


class PhotometricStereo:
    """
    Photometric stereo class for computing surface normals and albedo
    from multiple images taken under different lighting conditions.
    """
    
    def __init__(self, images, light_directions, mask=None):
        """
        Initialize the photometric stereo processor.
        
        Parameters
        ----------
        images : ndarray
            Input images stacked along 3rd dimension, shape (height, width, num_images)
        light_directions : ndarray
            Light directions, shape (3, num_images)
        mask : ndarray, optional
            Binary mask of valid pixels, shape (height, width)
        """
        self.images = images
        self.light_directions = light_directions
        
        # If no mask provided, use all pixels
        if mask is None:
            self.mask = np.ones((images.shape[0], images.shape[1]), dtype=bool)
        else:
            self.mask = mask
        
        self.height, self.width = images.shape[0], images.shape[1]
        self.num_images = images.shape[2]
        
        # Verify that we have the right number of light directions
        assert light_directions.shape[1] == self.num_images, (
            f"Number of light directions ({light_directions.shape[1]}) must match "
            f"number of images ({self.num_images})"
        )
        
        # Results computed by process()
        self.normals = None
        self.albedo = None
    
    def process(self):
        """
        Compute surface normals and albedo using photometric stereo.
        
        Returns
        -------
        tuple
            (albedo, normals) - albedo values and normal vectors
        """
        # Reshape images for processing
        # Original: I is (Height x Width x NumImages)
        # Reshaped: I_reshaped is (NumPixels x NumImages)
        num_pixels = self.height * self.width
        I_reshaped = self.images.reshape(num_pixels, self.num_images).T
        
        # Create mask for valid pixels
        mask_flat = self.mask.flatten()
        
        # Initialize results arrays for normals and albedo
        normals = np.zeros((3, num_pixels))
        albedo = np.zeros(num_pixels)
        
        # Process only pixels within mask
        valid_pixels = np.where(mask_flat)[0]
        
        # For each valid pixel, solve the linear system to find the surface normal
        # This is a vectorized version of the pixel-by-pixel computation
        I_valid = I_reshaped[:, valid_pixels]
        
        # Solve normal vectors using least squares for all pixels at once
        # Equivalent to: n_tilde = (L^T L)^(-1) L^T I for each pixel
        L = self.light_directions
        L_pseudo_inv = np.linalg.pinv(L)
        n_tilde = L_pseudo_inv @ I_valid
        
        # Compute albedo (norm of n_tilde)
        albedo_valid = np.linalg.norm(n_tilde, axis=0)
        
        # Normalize to get unit normal vectors
        # Avoid division by zero by setting a minimum albedo value
        min_albedo = 1e-8
        mask_non_zero = albedo_valid > min_albedo
        
        n_normalized = np.zeros_like(n_tilde)
        n_normalized[:, mask_non_zero] = n_tilde[:, mask_non_zero] / albedo_valid[mask_non_zero]
        
        # Assign results to output arrays
        normals[:, valid_pixels] = n_normalized
        albedo[valid_pixels] = albedo_valid
        
        # Reshape normals to image dimensions (3 x Height x Width)
        self.normals = normals.reshape((3, self.height, self.width))
        self.albedo = albedo.reshape((self.height, self.width))
        
        return self.albedo, self.normals
    
    def refine_light_directions(self):
        """
        Refine the light directions based on the computed normals and observed intensities.
        Equivalent to the PSRefineLight.m function.
        
        Returns
        -------
        ndarray
            Refined light directions
        """
        # Implementation of light direction refinement
        # This would be a port of the PSRefineLight.m algorithm
        # For now, return the original light directions
        return self.light_directions
    
    def evaluate_normals_by_image_error(self):
        """
        Evaluate the quality of normal estimation by reconstructing
        the images and computing the error.
        
        Returns
        -------
        float
            Mean square error between original and reconstructed images
        """
        if self.normals is None or self.albedo is None:
            raise ValueError("Normals and albedo must be computed first. Call process() method.")
        
        # Reshape for matrix operations
        normals_flat = self.normals.reshape(3, -1)
        albedo_flat = self.albedo.flatten()
        
        # Compute reconstructed images
        reconstructed_flat = albedo_flat * (self.light_directions.T @ normals_flat)
        
        # Original images in flat form
        original_flat = self.images.reshape(-1, self.num_images).T
        
        # Compute mean square error
        mask_flat = self.mask.flatten()
        error = np.mean(((original_flat[:, mask_flat] - reconstructed_flat[:, mask_flat]) ** 2))
        
        return error


def compute_photometric_stereo(images, light_directions, mask=None):
    """
    Simplified function to compute photometric stereo.
    
    Parameters
    ----------
    images : ndarray
        Input images stacked along 3rd dimension, shape (height, width, num_images)
    light_directions : ndarray
        Light directions, shape (3, num_images)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
        
    Returns
    -------
    tuple
        (albedo, normals) - albedo values and normal vectors
    """
    ps = PhotometricStereo(images, light_directions, mask)
    return ps.process()