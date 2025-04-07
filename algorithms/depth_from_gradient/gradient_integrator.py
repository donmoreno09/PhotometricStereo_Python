"""
Gradient integration methods for depth recovery.
Python implementation of DepthFromGradient.m
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.fftpack import dct, idct


class GradientIntegrator:
    """
    Class for integrating gradient fields to recover depth.
    Based on the DfGBox MATLAB toolbox.
    """
    
    def __init__(self, p, q, mask=None):
        """
        Initialize gradient integrator.
        
        Parameters
        ----------
        p : ndarray
            Gradient in x direction, shape (height, width)
        q : ndarray
            Gradient in y direction, shape (height, width)
        mask : ndarray, optional
            Binary mask of valid pixels, shape (height, width)
        """
        self.p = p
        self.q = q
        self.height, self.width = p.shape
        
        if mask is None:
            self.mask = np.ones((self.height, self.width), dtype=bool)
        else:
            self.mask = mask
        
        # Results
        self.Z = None
    
    def integrate(self, method='poisson', options=None):
        """
        Integrate gradient field using specified method.
        
        Parameters
        ----------
        method : str, optional
            Integration method: 'poisson', 'frankot', 'dct', or 'direct'
        options : dict, optional
            Additional options for integration
            
        Returns
        -------
        ndarray
            Integrated depth map
        """
        if options is None:
            options = {}
        
        if method == 'poisson':
            self.Z = self._integrate_poisson(options)
        elif method == 'frankot':
            self.Z = self._integrate_frankot_chellappa(options)
        elif method == 'dct':
            self.Z = self._integrate_dct(options)
        elif method == 'direct':
            self.Z = self._integrate_direct(options)
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        return self.Z
    
    def _integrate_poisson(self, options):
        """
        Poisson integration using sparse linear system.
        This is the most accurate but slowest method.
        
        Parameters
        ----------
        options : dict
            Additional options for integration
            
        Returns
        -------
        ndarray
            Integrated depth map
        """
        # Compute divergence of gradient field
        laplacian = self._compute_divergence()
        
        # Create linear system
        A, b = self._build_poisson_system(laplacian)
        
        # Solve linear system
        x = spsolve(A, b)
        
        # Reshape result to image dimensions
        Z = np.zeros((self.height, self.width))
        Z[self.mask] = x
        
        return Z
    
    def _integrate_frankot_chellappa(self, options):
        """
        Frankot-Chellappa integration using FFT.
        This method is fast but less accurate at boundaries.
        
        Parameters
        ----------
        options : dict
            Additional options for integration
            
        Returns
        -------
        ndarray
            Integrated depth map
        """
        # Apply mask to gradients
        p_masked = np.zeros_like(self.p)
        q_masked = np.zeros_like(self.q)
        p_masked[self.mask] = self.p[self.mask]
        q_masked[self.mask] = self.q[self.mask]
        
        # FFT of gradients
        p_fft = np.fft.fft2(p_masked)
        q_fft = np.fft.fft2(q_masked)
        
        # Wave numbers
        u = 2j * np.pi * np.fft.fftfreq(self.width)
        v = 2j * np.pi * np.fft.fftfreq(self.height)
        
        # Create meshgrid of wave numbers
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Avoid division by zero
        denom = u_grid**2 + v_grid**2
        denom[0, 0] = 1.0  # Avoid division by zero
        
        # Integration in frequency domain
        Z_fft = (u_grid * p_fft + v_grid * q_fft) / denom
        Z_fft[0, 0] = 0.0  # Set DC component to 0
        
        # Inverse FFT
        Z = np.real(np.fft.ifft2(Z_fft))
        
        # Apply mask
        Z_masked = Z.copy()
        Z_masked[~self.mask] = 0.0
        
        return Z_masked
    
    def _integrate_dct(self, options):
        """
        Integration using Discrete Cosine Transform.
        This method is fast and works well for natural images.
        
        Parameters
        ----------
        options : dict
            Additional options for integration
            
        Returns
        -------
        ndarray
            Integrated depth map
        """
        # Compute partial derivatives of gradients
        dx_p = np.zeros_like(self.p)
        dy_q = np.zeros_like(self.q)
        
        # Forward differences with boundary handling
        dx_p[:, :-1] = self.p[:, 1:] - self.p[:, :-1]
        dy_q[:-1, :] = self.q[1:, :] - self.q[:-1, :]
        
        # Compute Laplacian
        f = dx_p + dy_q
        
        # Apply DCT
        f_dct = dct(dct(f, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        # Prepare frequency grid
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        
        # Compute denominator in frequency domain
        denom = 2 * (np.cos(np.pi * X / self.width) + np.cos(np.pi * Y / self.height) - 2)
        denom[0, 0] = 1.0  # Avoid division by zero
        
        # Solve in frequency domain
        Z_dct = f_dct / denom
        Z_dct[0, 0] = 0.0  # Set DC component to 0
        
        # Apply inverse DCT
        Z = idct(idct(Z_dct, axis=1, norm='ortho'), axis=0, norm='ortho')
        
        # Apply mask
        Z_masked = Z.copy()
        Z_masked[~self.mask] = 0.0
        
        return Z_masked
    
    def _integrate_direct(self, options):
        """
        Direct integration by summing along paths.
        This method is simple but prone to error accumulation.
        
        Parameters
        ----------
        options : dict
            Additional options for integration
            
        Returns
        -------
        ndarray
            Integrated depth map
        """
        # Start from center of the image
        center_y, center_x = self.height // 2, self.width // 2
        
        # Make sure center is in mask
        if not self.mask[center_y, center_x]:
            # Find the nearest masked pixel
            valid_pixels = np.where(self.mask)
            if len(valid_pixels[0]) > 0:
                distances = (valid_pixels[0] - center_y)**2 + (valid_pixels[1] - center_x)**2
                nearest_idx = np.argmin(distances)
                center_y, center_x = valid_pixels[0][nearest_idx], valid_pixels[1][nearest_idx]
            else:
                # No valid pixels
                return np.zeros((self.height, self.width))
        
        # Initialize depth map
        Z = np.zeros((self.height, self.width))
        
        # Integrate horizontally from center
        for j in range(center_x+1, self.width):
            if self.mask[center_y, j] and self.mask[center_y, j-1]:
                Z[center_y, j] = Z[center_y, j-1] + self.p[center_y, j-1]
        
        for j in range(center_x-1, -1, -1):
            if self.mask[center_y, j] and self.mask[center_y, j+1]:
                Z[center_y, j] = Z[center_y, j+1] - self.p[center_y, j]
        
        # Integrate vertically from the horizontal line
        for i in range(center_y+1, self.height):
            for j in range(self.width):
                if self.mask[i, j] and self.mask[i-1, j]:
                    Z[i, j] = Z[i-1, j] + self.q[i-1, j]
        
        for i in range(center_y-1, -1, -1):
            for j in range(self.width):
                if self.mask[i, j] and self.mask[i+1, j]:
                    Z[i, j] = Z[i+1, j] - self.q[i, j]
        
        # Apply mask
        Z[~self.mask] = 0.0
        
        return Z
    
    def _compute_divergence(self):
        """
        Compute divergence of gradient field.
        
        Returns
        -------
        ndarray
            Divergence field (Laplacian of Z)
        """
        # Initialize divergence field
        div = np.zeros((self.height, self.width))
        
        # Compute divergence using central differences
        # dp/dx using central differences
        div[:, 1:-1] += (self.p[:, 2:] - self.p[:, :-2]) / 2.0
        
        # dq/dy using central differences
        div[1:-1, :] += (self.q[2:, :] - self.q[:-2, :]) / 2.0
        
        # Handle boundaries with forward/backward differences
        # Left and right boundaries
        div[:, 0] += self.p[:, 1] - self.p[:, 0]
        div[:, -1] += -self.p[:, -1] + self.p[:, -2]
        
        # Top and bottom boundaries
        div[0, :] += self.q[1, :] - self.q[0, :]
        div[-1, :] += -self.q[-1, :] + self.q[-2, :]
        
        return div
    
    def _build_poisson_system(self, laplacian):
        """
        Build sparse linear system for Poisson equation.
        
        Parameters
        ----------
        laplacian : ndarray
            Divergence field (Laplacian of Z)
            
        Returns
        -------
        tuple
            (A, b) - Sparse matrix and right-hand side vector
        """
        # Get valid pixels
        valid_pixels = np.where(self.mask)
        num_valid_pixels = len(valid_pixels[0])
        
        # Create mapping from (row, col) to linear index
        pixel_indices = -np.ones((self.height, self.width), dtype=int)
        pixel_indices[valid_pixels] = np.arange(num_valid_pixels)
        
        # Initialize arrays for sparse matrix construction
        row_indices = []
        col_indices = []
        values = []
        
        # Build sparse matrix
        for i in range(self.height):
            for j in range(self.width):
                if not self.mask[i, j]:
                    continue
                
                # Current pixel index
                idx = pixel_indices[i, j]
                
                # Initialize neighbor count
                neighbors = 0
                
                # Check each neighbor and add corresponding entries
                # Top neighbor (i-1, j)
                if i > 0 and self.mask[i-1, j]:
                    row_indices.append(idx)
                    col_indices.append(pixel_indices[i-1, j])
                    values.append(-1)
                    neighbors += 1
                
                # Bottom neighbor (i+1, j)
                if i < self.height-1 and self.mask[i+1, j]:
                    row_indices.append(idx)
                    col_indices.append(pixel_indices[i+1, j])
                    values.append(-1)
                    neighbors += 1
                
                # Left neighbor (i, j-1)
                if j > 0 and self.mask[i, j-1]:
                    row_indices.append(idx)
                    col_indices.append(pixel_indices[i, j-1])
                    values.append(-1)
                    neighbors += 1
                
                # Right neighbor (i, j+1)
                if j < self.width-1 and self.mask[i, j+1]:
                    row_indices.append(idx)
                    col_indices.append(pixel_indices[i, j+1])
                    values.append(-1)
                    neighbors += 1
                
                # Center pixel (i, j)
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(neighbors)
        
        # Create sparse matrix
        A = sparse.coo_matrix((values, (row_indices, col_indices)), 
                             shape=(num_valid_pixels, num_valid_pixels))
        A = A.tocsr()
        
        # Right-hand side vector
        b = laplacian[self.mask]
        
        return A, b