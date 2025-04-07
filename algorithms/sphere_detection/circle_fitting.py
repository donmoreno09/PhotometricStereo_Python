"""
Circle fitting algorithms for chrome sphere detection.
Python implementation of:
- FitCircle.m
- circle_fit.m
"""

import numpy as np
from scipy.optimize import least_squares


def fit_circle(x, y, options=None):
    """
    Fit a circle to a set of 2D points.
    Python implementation of FitCircle.m
    
    Parameters
    ----------
    x : ndarray
        X coordinates of points
    y : ndarray
        Y coordinates of points
    options : dict, optional
        Fitting options
        
    Returns
    -------
    tuple
        (xc, yc, r) - Center coordinates and radius
    """
    if options is None:
        options = {}
    
    # Default method is algebraic
    method = options.get('method', 'algebraic')
    
    if method == 'algebraic':
        return fit_circle_algebraic(x, y)
    elif method == 'geometric':
        return fit_circle_geometric(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")


def fit_circle_algebraic(x, y):
    """
    Fit circle using algebraic method.
    Based on circle_fit.m
    
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


def fit_circle_geometric(x, y):
    """
    Fit circle using geometric method (minimizing sum of squared distances).
    
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
    # Use algebraic fit for initial guess
    xc_init, yc_init, r_init = fit_circle_algebraic(x, y)
    
    # Function to compute geometric distance
    def distance(params):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r
    
    # Perform least squares optimization
    params_init = [xc_init, yc_init, r_init]
    result = least_squares(distance, params_init)
    
    # Extract results
    xc, yc, r = result.x
    
    return xc, yc, r