"""
Mathematical utilities for photometric stereo.
"""

import numpy as np
from scipy import linalg


def check_decimation(config, p, q):
    """
    Check and adjust decimation parameters for mesh generation.
    Python implementation of checkDecimation.m
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    p : int
        Initial decimation factor for rows
    q : int
        Initial decimation factor for columns
        
    Returns
    -------
    tuple
        (p, q) - Adjusted decimation factors
    """
    # Get maximum mesh size from config, default to 10000
    max_mesh_size = config.get('max_mesh_size', 10000)
    
    if 'normal_map' not in config:
        return p, q
    
    # Get dimensions of normal map
    height, width = config['normal_map'].shape[:2]
    
    # Calculate number of vertices based on current decimation
    num_vertices = (height // p) * (width // q)
    
    # If number of vertices is too large, increase decimation
    if num_vertices > max_mesh_size:
        # Calculate needed decimation factor
        decimation_factor = np.sqrt(num_vertices / max_mesh_size)
        
        # Increase decimation factors
        p = max(1, int(np.ceil(p * decimation_factor)))
        q = max(1, int(np.ceil(q * decimation_factor)))
    
    return p, q


def compute_nodes_and_faces(Z, p=1, q=1):
    """
    Compute nodes and faces for mesh generation from a depth map.
    Python implementation of computeNodesAndFaces.m
    
    Parameters
    ----------
    Z : ndarray
        Depth map, shape (height, width)
    p : int, optional
        Decimation factor for rows (default=1, no decimation)
    q : int, optional
        Decimation factor for columns (default=1, no decimation)
        
    Returns
    -------
    tuple
        (nodes, faces) - Mesh vertices and faces
    """
    rows, cols = Z.shape
    
    # Create grid coordinates
    y, x = np.mgrid[0:rows:p, 0:cols:q]
    
    # Extract Z values at grid points
    z = Z[0:rows:p, 0:cols:q]
    
    # Create vertices array [x, y, z]
    vertices = np.zeros((x.size, 3))
    vertices[:, 0] = x.flatten()
    vertices[:, 1] = y.flatten()
    vertices[:, 2] = z.flatten()
    
    # Create faces using Delaunay triangulation
    from scipy.spatial import Delaunay
    
    # Create 2D points for triangulation
    points = np.column_stack((x.flatten(), y.flatten()))
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    
    # Get faces (triangles)
    faces = tri.simplices
    
    return vertices, faces


def fit_circle(x, y):
    """
    Fit a circle to a set of 2D points.
    Python implementation of FitCircle.m
    
    Parameters
    ----------
    x : ndarray
        X coordinates of points
    y : ndarray
        Y coordinates of points
        
    Returns
    -------
    tuple
        (xc, yc, r) - Center coordinates and radius
    """
    # Mean of coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Shift coordinates to mean center
    u = x - x_mean
    v = y - y_mean
    
    # Square coordinates
    u_sq = u**2
    v_sq = v**2
    
    # Compute moments
    S_uv = np.sum(u * v)
    S_uu = np.sum(u_sq)
    S_vv = np.sum(v_sq)
    S_uuv = np.sum(u_sq * v)
    S_uvv = np.sum(u * v_sq)
    S_uuu = np.sum(u_sq * u)
    S_vvv = np.sum(v_sq * v)
    
    # Compute matrix elements
    A = np.array([
        [S_uu, S_uv],
        [S_uv, S_vv]
    ])
    
    B = np.array([
        [0.5 * (S_uuu + S_uvv)],
        [0.5 * (S_vvv + S_uuv)]
    ])
    
    # Solve system A*c = B for center
    try:
        c = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        # If singular matrix, try least squares solution
        c, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # Extract center in original coordinates
    xc = c[0] + x_mean
    yc = c[1] + y_mean
    
    # Compute radius
    r = np.sqrt(np.mean((x - xc)**2 + (y - yc)**2))
    
    return xc[0], yc[0], r


def progress(i, n_all, msg="Processing"):
    """
    Print progress message.
    Python implementation of ShowProgress.m
    
    Parameters
    ----------
    i : int
        Current iteration
    n_all : int
        Total number of iterations
    msg : str, optional
        Message prefix
    """
    # Only print progress at 10% intervals
    interval = max(1, n_all // 10)
    if i % interval == 0 or i == n_all:
        percent = 100.0 * i / n_all
        print(f"{msg}: {i}/{n_all} ({percent:.1f}%)")