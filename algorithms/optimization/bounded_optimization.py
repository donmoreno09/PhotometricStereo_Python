"""
Bounded optimization utilities.
Python implementation of:
- MapBoundedToUnconstrained.m
- MapUnconstrainedToBounded.m
- BoundedFcnToUnconstrainedFcn.m
"""

import numpy as np


def map_bounded_to_unconstrained(lb, ub):
    """
    Create mapping function from bounded to unconstrained space.
    Python implementation of MapBoundedToUnconstrained.m
    
    Parameters
    ----------
    lb : ndarray
        Lower bounds
    ub : ndarray
        Upper bounds
        
    Returns
    -------
    callable
        Mapping function
    """
    def mapping_function(x_bounded):
        """Map from bounded to unconstrained space"""
        # Convert to numpy array if needed
        x_bounded = np.asarray(x_bounded)
        shape = x_bounded.shape
        x_bounded = x_bounded.flatten()
        
        # Initialize output
        x_unbounded = np.zeros_like(x_bounded)
        
        for i in range(len(x_bounded)):
            if np.isfinite(lb[i]) and np.isfinite(ub[i]):
                # Bounded from both sides: use logit transform
                # Maps [lb, ub] to (-inf, inf)
                normalized = (x_bounded[i] - lb[i]) / (ub[i] - lb[i])
                if normalized <= 0:
                    x_unbounded[i] = -np.inf
                elif normalized >= 1:
                    x_unbounded[i] = np.inf
                else:
                    x_unbounded[i] = np.log(normalized / (1 - normalized))
            elif np.isfinite(lb[i]):
                # Only lower bound: use log transform
                # Maps [lb, inf) to (-inf, inf)
                if x_bounded[i] <= lb[i]:
                    x_unbounded[i] = -np.inf
                else:
                    x_unbounded[i] = np.log(x_bounded[i] - lb[i])
            elif np.isfinite(ub[i]):
                # Only upper bound: use negative log transform
                # Maps (-inf, ub] to (-inf, inf)
                if x_bounded[i] >= ub[i]:
                    x_unbounded[i] = np.inf
                else:
                    x_unbounded[i] = -np.log(ub[i] - x_bounded[i])
            else:
                # Unbounded: no transform needed
                x_unbounded[i] = x_bounded[i]
        
        # Return in original shape
        return x_unbounded.reshape(shape)
    
    return mapping_function


def map_unconstrained_to_bounded(lb, ub):
    """
    Create mapping function from unconstrained to bounded space.
    Python implementation of MapUnconstrainedToBounded.m
    
    Parameters
    ----------
    lb : ndarray
        Lower bounds
    ub : ndarray
        Upper bounds
        
    Returns
    -------
    callable
        Mapping function
    """
    def mapping_function(x_unbounded):
        """Map from unconstrained to bounded space"""
        # Convert to numpy array if needed
        x_unbounded = np.asarray(x_unbounded)
        shape = x_unbounded.shape
        x_unbounded = x_unbounded.flatten()
        
        # Initialize output
        x_bounded = np.zeros_like(x_unbounded)
        
        for i in range(len(x_unbounded)):
            if np.isfinite(lb[i]) and np.isfinite(ub[i]):
                # Bounded from both sides: use inverse logit transform
                # Maps (-inf, inf) to [lb, ub]
                sigmoid = 1.0 / (1.0 + np.exp(-x_unbounded[i]))
                x_bounded[i] = lb[i] + (ub[i] - lb[i]) * sigmoid
            elif np.isfinite(lb[i]):
                # Only lower bound: use inverse log transform
                # Maps (-inf, inf) to [lb, inf)
                if x_unbounded[i] < -709:  # Avoid underflow
                    x_bounded[i] = lb[i]
                else:
                    x_bounded[i] = lb[i] + np.exp(x_unbounded[i])
            elif np.isfinite(ub[i]):
                # Only upper bound: use inverse negative log transform
                # Maps (-inf, inf) to (-inf, ub]
                if x_unbounded[i] > 709:  # Avoid overflow
                    x_bounded[i] = ub[i]
                else:
                    x_bounded[i] = ub[i] - np.exp(-x_unbounded[i])
            else:
                # Unbounded: no transform needed
                x_bounded[i] = x_unbounded[i]
        
        # Return in original shape
        return x_bounded.reshape(shape)
    
    return mapping_function


def bounded_fcn_to_unconstrained_fcn(bounded_fcn, lb, ub):
    """
    Convert function in bounded space to function in unconstrained space.
    Python implementation of BoundedFcnToUnconstrainedFcn.m
    
    Parameters
    ----------
    bounded_fcn : callable
        Function in bounded space
    lb : ndarray
        Lower bounds
    ub : ndarray
        Upper bounds
        
    Returns
    -------
    callable
        Function in unconstrained space
    """
    # Create mapping functions
    to_bounded = map_unconstrained_to_bounded(lb, ub)
    
    def unconstrained_fcn(x_unbounded):
        """Function in unconstrained space"""
        # Map input to bounded space
        x_bounded = to_bounded(x_unbounded)
        
        # Evaluate bounded function
        if isinstance(bounded_fcn, tuple) and len(bounded_fcn) == 2:
            # Function and gradient provided
            f_fcn, g_fcn = bounded_fcn
            f = f_fcn(x_bounded)
            g = g_fcn(x_bounded)
            return f, g
        else:
            # Only function provided
            return bounded_fcn(x_bounded)
    
    return unconstrained_fcn