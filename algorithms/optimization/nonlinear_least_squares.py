"""
Nonlinear least squares optimization.
Python implementation of NonlinearLeastSquares.m
"""

import numpy as np
from scipy.optimize import least_squares, minimize


class NonlinearLeastSquares:
    """
    Class for solving nonlinear least squares problems.
    Based on the NLLSBox MATLAB toolbox.
    """
    
    def __init__(self):
        """Initialize nonlinear least squares solver"""
        self.options = {
            'max_iterations': 100,
            'tolerance': 1e-6,
            'verbose': False,
            'method': 'levenberg-marquardt'
        }
        self.iterations = 0
        self.residual_norm = None
        self.gradient_norm = None
    
    def solve(self, fcn, x0, lb=None, ub=None, options=None):
        """
        Solve nonlinear least squares problem.
        
        Parameters
        ----------
        fcn : callable
            Function handle returning residuals and optionally Jacobian
        x0 : ndarray
            Initial parameter vector
        lb : ndarray, optional
            Lower bounds on parameters
        ub : ndarray, optional
            Upper bounds on parameters
        options : dict, optional
            Solver options
            
        Returns
        -------
        tuple
            (x, fval, exitflag) - Solution, objective value, exit code
        """
        # Update options if provided
        if options is not None:
            self.options.update(options)
        
        # Extract options
        verbose = self.options['verbose']
        
        # Check if function provides Jacobian
        has_jacobian = self._check_jacobian_provided(fcn, x0)
        
        # Handle bounds
        if lb is None:
            lb = -np.inf * np.ones_like(x0)
        if ub is None:
            ub = np.inf * np.ones_like(x0)
        
        # If bounded and using Levenberg-Marquardt, switch to trust-region
        if (np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))) and \
           self.options['method'] == 'levenberg-marquardt':
            method = 'trf'  # Trust Region Reflective
        elif self.options['method'] == 'levenberg-marquardt':
            method = 'lm'   # Levenberg-Marquardt
        else:
            method = 'trf'  # Default to Trust Region Reflective
        
        # Function wrapper for scipy.optimize.least_squares
        if has_jacobian:
            # Function returns both residuals and Jacobian
            def residual_func(x):
                f, J = fcn(x)
                return f
            
            def jacobian_func(x):
                f, J = fcn(x)
                return J
            
            jac = jacobian_func
        else:
            # Function returns only residuals
            def residual_func(x):
                return fcn(x)
            
            jac = '2-point'  # Finite difference approximation
        
        # Print initial message
        if verbose:
            print("Starting nonlinear least squares optimization...")
            print(f"Initial parameters: {x0}")
        
        # Solve using scipy.optimize.least_squares
        result = least_squares(
            residual_func, 
            x0, 
            jac=jac,
            bounds=(lb, ub), 
            method=method,
            ftol=self.options['tolerance'],
            xtol=self.options['tolerance'],
            gtol=self.options['tolerance'],
            max_nfev=self.options['max_iterations'],
            verbose=2 if verbose else 0
        )
        
        # Extract results
        x = result.x
        if has_jacobian:
            f, J = fcn(x)
        else:
            f = residual_func(x)
        
        # Compute objective value (sum of squares)
        fval = np.sum(f**2)
        
        # Store iteration info
        self.iterations = result.nfev
        self.residual_norm = np.linalg.norm(f)
        if hasattr(result, 'jac'):
            self.gradient_norm = np.linalg.norm(result.jac.T @ f, np.inf)
        
        # Determine exit flag
        if result.status > 0:
            exitflag = 1  # Converged
        else:
            exitflag = 0  # Failed to converge
        
        # Print final message
        if verbose:
            print("Nonlinear least squares optimization finished.")
            print(f"Final objective value: {fval}")
            print(f"Number of iterations: {self.iterations}")
            print(f"Exit flag: {exitflag}")
        
        return x, fval, exitflag
    
    def _check_jacobian_provided(self, fcn, x0):
        """
        Check if the function provides a Jacobian.
        
        Parameters
        ----------
        fcn : callable
            Function handle
        x0 : ndarray
            Initial parameters
            
        Returns
        -------
        bool
            True if function provides Jacobian
        """
        try:
            result = fcn(x0)
            if isinstance(result, tuple) and len(result) == 2:
                # Function returns (f, J)
                return True
            else:
                # Function returns only f
                return False
        except Exception as e:
            print(f"Error checking function: {e}")
            return False


def curve_to_cost(fcn, x, y):
    """
    Convert curve fitting function to cost function.
    Python implementation of NLLSCurveToCost.m
    
    Parameters
    ----------
    fcn : callable
        Curve fitting function
    x : ndarray
        Independent variable
    y : ndarray
        Dependent variable (observations)
        
    Returns
    -------
    callable
        Cost function for nonlinear least squares
    """
    def cost_function(a):
        # Compute predicted y values
        if hasattr(fcn, '__code__') and fcn.__code__.co_argcount > 1:
            # Function accepts both x and a
            y_pred = fcn(x, a)
        else:
            # Function only accepts a (assumes x is captured in closure)
            y_pred = fcn(a)
        
        # Compute residuals
        residuals = y_pred - y
        
        return residuals
    
    return cost_function


def nlls_curve_fit(fcn, x, y, a0, lb=None, ub=None, options=None):
    """
    Fit curve to data using nonlinear least squares.
    
    Parameters
    ----------
    fcn : callable
        Curve fitting function
    x : ndarray
        Independent variable
    y : ndarray
        Dependent variable (observations)
    a0 : ndarray
        Initial parameter vector
    lb : ndarray, optional
        Lower bounds on parameters
    ub : ndarray, optional
        Upper bounds on parameters
    options : dict, optional
        Solver options
        
    Returns
    -------
    tuple
        (a, fval, exitflag) - Parameters, objective value, exit code
    """
    # Create cost function
    cost_fcn = curve_to_cost(fcn, x, y)
    
    # Create solver
    solver = NonlinearLeastSquares()
    
    # Solve
    a, fval, exitflag = solver.solve(cost_fcn, a0, lb, ub, options)
    
    return a, fval, exitflag