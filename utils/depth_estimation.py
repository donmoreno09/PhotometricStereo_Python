"""
Module for estimating depth from normal maps and gradients.
Equivalent to DepthFromGradient.m in MATLAB.
"""

import numpy as np


def depth_from_gradient(p, q, options=None):
    """
    Estimate a depth map from the given gradient field (p, q), such that
    dZ/dx = p, dZ/dy = q.
    
    Parameters
    ----------
    p : ndarray
        x-component of gradient (dZ/dx)
    q : ndarray
        y-component of gradient (dZ/dy)
    options : dict, optional
        Dictionary with options:
        - periodic: boolean indicating whether the output Z should be periodic
    
    Returns
    -------
    ndarray
        Estimated depth map
    """
    if options is None:
        options = {}
    
    # Parse options
    periodic = options.get('periodic', False)
    
    # Check input size
    assert p.shape == q.shape, "p and q must have the same size"
    assert len(p.shape) == 2, "p and q must be 2D arrays"
    
    M, N = p.shape
    
    # Perform copy-flip for non-periodic depth
    if not periodic:
        # Create flipped versions
        p_flipped_cols = -p[:, ::-1]
        p_flipped_rows = p[::-1, :]
        p_flipped_both = -p[::-1, ::-1]
        
        q_flipped_cols = q[:, ::-1]
        q_flipped_rows = -q[::-1, :]
        q_flipped_both = -q[::-1, ::-1]
        
        # Stack them together
        p = np.block([[p, p_flipped_cols], 
                      [p_flipped_rows, p_flipped_both]])
        q = np.block([[q, q_flipped_cols], 
                      [q_flipped_rows, q_flipped_both]])
        
        M, N = p.shape
    
    # Frequency indices
    half_M = (M - 1) / 2
    half_N = (N - 1) / 2
    
    u = np.fft.fftshift(np.arange(-np.ceil(half_N), np.floor(half_N) + 1))
    v = np.fft.fftshift(np.arange(-np.ceil(half_M), np.floor(half_M) + 1))
    
    u, v = np.meshgrid(u, v)
    
    # Shift back to match MATLAB's ifftshift
    u = np.fft.ifftshift(u)
    v = np.fft.ifftshift(v)
    
    # Compute the Fourier transform of p and q
    Fp = np.fft.fft2(p)
    Fq = np.fft.fft2(q)
    
    # Compute the Fourier transform of Z
    denominator = (u / N) ** 2 + (v / M) ** 2
    # Avoid division by zero
    denominator[0, 0] = 1.0
    
    Fz = -1j / (2 * np.pi) * (u * Fp / N + v * Fq / M) / denominator
    
    # Set DC component to 0 (mean of Z is arbitrary)
    Fz[0, 0] = 0
    
    # Recover depth Z
    Z = np.real(np.fft.ifft2(Fz))
    
    # Recover the non-periodic depth
    if not periodic:
        Z = Z[:M//2, :N//2]
    
    return Z


def depth_from_normals(normals, mask=None, options=None):
    """
    Estimate depth from surface normal vectors.
    
    Parameters
    ----------
    normals : ndarray
        Normal map (height, width, 3)
    mask : ndarray, optional
        Binary mask indicating valid pixels
    options : dict, optional
        Dictionary with options for depth_from_gradient
    
    Returns
    -------
    ndarray
        Estimated depth map
    """
    if options is None:
        options = {}
        
    # Extract height and width
    height, width = normals.shape[:2]
    
    # Extract gradient information from normals
    # p = dz/dx, q = dz/dy
    p = -normals[:, :, 0] / np.maximum(normals[:, :, 2], 1e-10)
    q = -normals[:, :, 1] / np.maximum(normals[:, :, 2], 1e-10)
    
    # Apply mask if provided
    if mask is not None:
        p = p * mask
        q = q * mask
    
    # Handle NaN and Inf values
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute depth using depth_from_gradient
    Z = depth_from_gradient(p, q, options)
    
    # Apply mask to result if provided
    if mask is not None:
        Z = Z * mask
    
    return Z