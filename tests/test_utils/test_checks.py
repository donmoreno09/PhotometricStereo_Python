"""
Test check utilities for numerical tests.
Ports of CheckNear.m, CheckNearRel.m, CheckNearAbs.m, etc.
"""

import pytest
import numpy as np


def check_near_abs(a, b, abs_tol=1e-9):
    """
    Check if arrays are close in absolute terms.
    Port of CheckNearAbs.m
    
    Parameters
    ----------
    a : ndarray
        First array
    b : ndarray
        Second array
    abs_tol : float
        Absolute tolerance
        
    Returns
    -------
    bool
        True if arrays are close
    """
    abs_diff = np.abs(a - b)
    max_diff = np.max(abs_diff)
    is_close = max_diff <= abs_tol
    
    return is_close, max_diff


def check_near_rel(a, b, rel_tol=1e-6):
    """
    Check if arrays are close in relative terms.
    Port of CheckNearRel.m
    
    Parameters
    ----------
    a : ndarray
        First array
    b : ndarray
        Second array
    rel_tol : float
        Relative tolerance
        
    Returns
    -------
    bool
        True if arrays are close
    """
    # Avoid division by zero
    mask = np.abs(b) > 1e-15
    
    if not np.any(mask):
        # If b is all zeros/tiny, use absolute difference
        abs_diff = np.abs(a - b)
        max_diff = np.max(abs_diff)
        is_close = max_diff <= rel_tol
        return is_close, max_diff
    
    # Calculate relative difference where b is significant
    rel_diff = np.abs(a - b)[mask] / np.abs(b[mask])
    max_diff = np.max(rel_diff)
    is_close = max_diff <= rel_tol
    
    return is_close, max_diff


def check_near(a, b, rel_tol=1e-6, abs_tol=1e-9):
    """
    Check if arrays are close using both absolute and relative tolerance.
    Port of CheckNear.m
    
    Parameters
    ----------
    a : ndarray
        First array
    b : ndarray
        Second array
    rel_tol : float
        Relative tolerance
    abs_tol : float
        Absolute tolerance
        
    Returns
    -------
    bool
        True if arrays are close
    """
    # Check absolute difference
    is_close_abs, abs_diff = check_near_abs(a, b, abs_tol)
    
    # Check relative difference
    is_close_rel, rel_diff = check_near_rel(a, b, rel_tol)
    
    # Arrays are close if either test passes
    is_close = is_close_abs or is_close_rel
    
    return is_close, abs_diff, rel_diff


def check_gradient(func, x, analytic_grad=None, delta=1e-7):
    """
    Check numerical gradient against analytical gradient.
    Port of CheckGradient.m
    
    Parameters
    ----------
    func : callable
        Function that returns a scalar value
    x : ndarray
        Point at which to evaluate the gradient
    analytic_grad : ndarray, optional
        Analytical gradient. If None, use finite differences only.
    delta : float
        Step size for finite differences
        
    Returns
    -------
    tuple
        (is_close, numerical_grad, analytic_grad, max_abs_err, max_rel_err)
    """
    x = np.asarray(x)
    n = len(x)
    
    # Compute numerical gradient
    numerical_grad = np.zeros_like(x)
    
    for i in range(n):
        # Forward step
        x_plus = x.copy()
        x_plus[i] += delta
        f_plus = func(x_plus)
        
        # Backward step
        x_minus = x.copy()
        x_minus[i] -= delta
        f_minus = func(x_minus)
        
        # Central difference approximation
        numerical_grad[i] = (f_plus - f_minus) / (2 * delta)
    
    # If analytical gradient is provided, compare
    if analytic_grad is not None:
        # Check if gradients are close
        is_close, abs_err, rel_err = check_near(numerical_grad, analytic_grad)
        return is_close, numerical_grad, analytic_grad, abs_err, rel_err
    else:
        return True, numerical_grad, None, 0, 0


def check_jacobian(func, x, analytic_jacobian=None, delta=1e-7):
    """
    Check numerical Jacobian against analytical Jacobian.
    Port of CheckJacobian.m
    
    Parameters
    ----------
    func : callable
        Function that returns a vector of values
    x : ndarray
        Point at which to evaluate the Jacobian
    analytic_jacobian : ndarray, optional
        Analytical Jacobian. If None, use finite differences only.
    delta : float
        Step size for finite differences
        
    Returns
    -------
    tuple
        (is_close, numerical_jac, analytic_jac, max_abs_err, max_rel_err)
    """
    x = np.asarray(x)
    n = len(x)
    
    # Evaluate function at current point to get output size
    f0 = func(x)
    m = len(f0)
    
    # Compute numerical Jacobian
    numerical_jac = np.zeros((m, n))
    
    for i in range(n):
        # Forward step
        x_plus = x.copy()
        x_plus[i] += delta
        f_plus = func(x_plus)
        
        # Backward step
        x_minus = x.copy()
        x_minus[i] -= delta
        f_minus = func(x_minus)
        
        # Central difference approximation
        numerical_jac[:, i] = (f_plus - f_minus) / (2 * delta)
    
    # If analytical Jacobian is provided, compare
    if analytic_jacobian is not None:
        # Check if Jacobians are close
        is_close, abs_err, rel_err = check_near(numerical_jac.flatten(), analytic_jacobian.flatten())
        return is_close, numerical_jac, analytic_jacobian, abs_err, rel_err
    else:
        return True, numerical_jac, None, 0, 0


def test_check_near_abs():
    """Test check_near_abs function."""
    # Test with values that are absolutely close
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0001, 2.0001, 3.0001])
    
    # Should be close with abs_tol = 0.001
    is_close, max_diff = check_near_abs(a, b, abs_tol=0.001)
    assert is_close
    assert max_diff <= 0.001
    
    # Should not be close with abs_tol = 0.00001
    is_close, max_diff = check_near_abs(a, b, abs_tol=0.00001)
    assert not is_close
    assert max_diff > 0.00001


def test_check_near_rel():
    """Test check_near_rel function."""
    # Test with values that are relatively close
    a = np.array([1000.0, 2000.0, 3000.0])
    b = np.array([1001.0, 2002.0, 3003.0])
    
    # Should be close with rel_tol = 0.001 (0.1%)
    is_close, max_diff = check_near_rel(a, b, rel_tol=0.001)
    assert is_close
    assert max_diff <= 0.001
    
    # Should not be close with rel_tol = 0.0001 (0.01%)
    is_close, max_diff = check_near_rel(a, b, rel_tol=0.0001)
    assert not is_close
    assert max_diff > 0.0001


def test_check_near():
    """Test check_near function."""
    # Test with absolutely close values
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.000001, 0.000001, 0.000001])
    
    # Should be close with abs_tol = 0.00001
    is_close, abs_diff, rel_diff = check_near(a, b, abs_tol=0.00001)
    assert is_close
    
    # Test with relatively close values
    a = np.array([1000.0, 2000.0, 3000.0])
    b = np.array([1001.0, 2002.0, 3003.0])
    
    # Should be close with rel_tol = 0.001 (0.1%)
    is_close, abs_diff, rel_diff = check_near(a, b, rel_tol=0.001)
    assert is_close


def test_check_gradient():
    """Test check_gradient function."""
    # Test function: f(x) = x^2
    def func(x):
        return np.sum(x**2)
    
    # Analytical gradient: df/dx = 2*x
    def grad(x):
        return 2 * x
    
    # Test at x = [1, 2, 3]
    x = np.array([1.0, 2.0, 3.0])
    analytic_grad_val = grad(x)
    
    is_close, numerical_grad, analytic_grad, abs_err, rel_err = check_gradient(func, x, analytic_grad_val)
    
    # Check that the gradients are close
    assert is_close
    assert np.allclose(numerical_grad, analytic_grad)


def test_check_jacobian():
    """Test check_jacobian function."""
    # Test function: f(x) = [x1^2, x1*x2, x2^2]
    def func(x):
        return np.array([x[0]**2, x[0]*x[1], x[1]**2])
    
    # Analytical Jacobian:
    # df1/dx1 = 2*x1, df1/dx2 = 0
    # df2/dx1 = x2,   df2/dx2 = x1
    # df3/dx1 = 0,    df3/dx2 = 2*x2
    def jacobian(x):
        return np.array([
            [2*x[0], 0],
            [x[1], x[0]],
            [0, 2*x[1]]
        ])
    
    # Test at x = [2, 3]
    x = np.array([2.0, 3.0])
    analytic_jac_val = jacobian(x)
    
    is_close, numerical_jac, analytic_jac, abs_err, rel_err = check_jacobian(func, x, analytic_jac_val)
    
    # Check that the Jacobians are close
    assert is_close
    assert np.allclose(numerical_jac, analytic_jac)