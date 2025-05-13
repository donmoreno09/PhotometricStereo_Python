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
        Exact implementation of the MATLAB algorithm.
        """
        # Reshape images for processing
        [N1, N2, M] = self.images.shape
        N = N1 * N2
        I = np.reshape(self.images, (N, M)).T
        mask = np.reshape(self.mask, (N, M))
        
        # Create mask index for efficient computation
        mask_index = np.zeros(N, dtype=int)
        for i in range(M):
            mask_index = mask_index * 2 + mask[:, i]
        
        unique_mask_indices = np.unique(mask_index)
        
        # Estimate scaled normal vectors
        b = np.full((3, N), np.nan)
        for idx in unique_mask_indices:
            # Find all pixels with this index
            pixel_idx = np.where(mask_index == idx)[0]
            
            # Find all images that are active for this index
            image_tag = mask[pixel_idx[0], :]
            
            if np.sum(image_tag) < 3:
                continue
            
            # Create lighting matrix for active images
            Li = self.light_directions[:, image_tag]
            
            # Create intensity matrix for these pixels and active images
            Ii = I[image_tag, :][:, pixel_idx]
            
            # Compute scaled normal
            # Equivalent to MATLAB's Li' \ Ii
            b[:, pixel_idx] = np.linalg.lstsq(Li.T, Ii, rcond=None)[0]
        
        # Reshape and calculate albedo and unit normal vectors
        b = np.reshape(b.T, (N1, N2, 3))
        rho = np.sqrt(np.sum(b**2, axis=2))
        
        # Avoid division by zero
        n = np.zeros_like(b)
        valid_pixels = rho > 0
        for i in range(3):
            n_channel = b[:,:,i].copy()
            n_channel[valid_pixels] = n_channel[valid_pixels] / rho[valid_pixels]
            n[:,:,i] = n_channel
        
        self.normals = np.transpose(n, (2, 0, 1))  # Convert to (3, N1, N2)
        self.albedo = rho
        
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