"""
Tests for bounded optimization.
Ports of MapUnconstrainedToBoundedTest.m and MapBoundedToUnconstrainedTest.m
"""

import pytest
import numpy as np
from algorithms.optimization.bounded_optimization import (
    map_unconstrained_to_bounded,
    map_bounded_to_unconstrained,
    bounded_func_to_unconstrained_func
)


def test_map_unconstrained_to_bounded_scalar():
    """Test unconstrained to bounded mapping for a scalar."""
    # Test inputs
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    a = 0.0
    b = 1.0
    
    # Expected outputs for sigmoid mapping
    expected = np.array([
        0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708
    ])
    
    # Apply mapping
    result = map_unconstrained_to_bounded(x, a, b)
    
    # Check results
    assert np.allclose(result, expected, atol=1e-7)
    
    # Check boundary behavior
    assert map_unconstrained_to_bounded(-np.inf, a, b) == a
    assert map_unconstrained_to_bounded(np.inf, a, b) == b


def test_map_unconstrained_to_bounded_vector():
    """Test unconstrained to bounded mapping for a vector."""
    # Test inputs
    x = np.array([[-2.0, -1.0], [1.0, 2.0]])
    a = np.array([[0.0, -1.0], [-2.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 2.0]])
    
    # Expected outputs for sigmoid mapping
    expected = np.array([
        [0.11920292, -0.73105858],
        [-1.26894142, 1.76159416]
    ])
    
    # Apply mapping
    result = map_unconstrained_to_bounded(x, a, b)
    
    # Check results
    assert np.allclose(result, expected, atol=1e-7)


def test_map_bounded_to_unconstrained_scalar():
    """Test bounded to unconstrained mapping for a scalar."""
    # Test inputs
    y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    a = 0.0
    b = 1.0
    
    # Expected outputs for logit mapping
    expected = np.array([
        -2.19722458, -0.84729786, 0.0, 0.84729786, 2.19722458
    ])
    
    # Apply mapping
    result = map_bounded_to_unconstrained(y, a, b)
    
    # Check results
    assert np.allclose(result, expected, atol=1e-7)
    
    # Check boundary behavior
    with pytest.warns(RuntimeWarning):
        assert np.isinf(map_bounded_to_unconstrained(a, a, b))
        assert np.isinf(map_bounded_to_unconstrained(b, a, b))


def test_map_bounded_to_unconstrained_vector():
    """Test bounded to unconstrained mapping for a vector."""
    # Test inputs
    y = np.array([[0.1, -0.8], [-1.5, 1.0]])
    a = np.array([[0.0, -1.0], [-2.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 2.0]])
    
    # Expected outputs for logit mapping
    expected = np.array([
        [-2.19722458, -1.38629436],
        [-1.38629436, np.inf]
    ])
    
    # Apply mapping
    with pytest.warns(RuntimeWarning):
        result = map_bounded_to_unconstrained(y, a, b)
    
    # Check results (excluding infinite value)
    mask = ~np.isinf(expected)
    assert np.allclose(result[mask], expected[mask], atol=1e-7)
    assert np.isinf(result[~mask])


def test_bounded_func_to_unconstrained_func():
    """Test conversion of bounded function to unconstrained function."""
    # Define a bounded function: f(x) = (x-0.5)^2, 0 <= x <= 1
    def bounded_func(x):
        return (x - 0.5)**2
    
    # Convert to unconstrained function
    lower_bound = 0.0
    upper_bound = 1.0
    unconstrained_func = bounded_func_to_unconstrained_func(bounded_func, lower_bound, upper_bound)
    
    # Test points
    u_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Expected values
    # First map u to x in [0,1] using sigmoid, then compute (x-0.5)^2
    x_values = map_unconstrained_to_bounded(u_values, lower_bound, upper_bound)
    expected = (x_values - 0.5)**2
    
    # Compute actual values
    result = unconstrained_func(u_values)
    
    # Check results
    assert np.allclose(result, expected)