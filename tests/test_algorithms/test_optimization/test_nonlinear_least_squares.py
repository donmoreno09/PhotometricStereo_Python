"""
Tests for nonlinear least squares optimization.
Port of NonlinearLeastSquaresTest.m and NLLSCurveToCostTest.m
"""

import pytest
import numpy as np
from algorithms.optimization.nonlinear_least_squares import nonlinear_least_squares, nlls_curve_to_cost


def test_nonlinear_least_squares_line_fit():
    """Test fitting a line with nonlinear least squares."""
    # Generate data points from a line: y = 2*x + 3 + noise
    x = np.linspace(0, 10, 100)
    y_true = 2 * x + 3
    np.random.seed(42)  # For reproducibility
    y = y_true + np.random.normal(0, 0.5, y_true.shape)
    
    # Initial guess [slope, intercept]
    initial_params = np.array([1.0, 1.0])
    
    # Cost function: difference between observed and predicted y values
    def cost_func(params):
        slope, intercept = params
        y_pred = slope * x + intercept
        return y - y_pred
    
    # Jacobian function: derivative of cost with respect to parameters
    def jacobian(params):
        slope, intercept = params
        J = np.zeros((len(x), 2))
        J[:, 0] = -x  # Derivative with respect to slope
        J[:, 1] = -1  # Derivative with respect to intercept
        return J
    
    # Optimize
    result = nonlinear_least_squares(cost_func, initial_params, jacobian_func=jacobian)
    
    # Check results
    assert result is not None
    assert len(result) == 2
    assert np.isclose(result[0], 2.0, atol=0.1)  # slope
    assert np.isclose(result[1], 3.0, atol=0.1)  # intercept


def test_nonlinear_least_squares_quadratic_fit():
    """Test fitting a quadratic curve with nonlinear least squares."""
    # Generate data points from a quadratic: y = a*x^2 + b*x + c + noise
    a_true, b_true, c_true = 0.5, -2.0, 3.0
    x = np.linspace(-5, 5, 100)
    y_true = a_true * x**2 + b_true * x + c_true
    np.random.seed(42)  # For reproducibility
    y = y_true + np.random.normal(0, 1.0, y_true.shape)
    
    # Initial guess [a, b, c]
    initial_params = np.array([0.1, -1.0, 1.0])
    
    # Cost function
    def cost_func(params):
        a, b, c = params
        y_pred = a * x**2 + b * x + c
        return y - y_pred
    
    # Jacobian function
    def jacobian(params):
        a, b, c = params
        J = np.zeros((len(x), 3))
        J[:, 0] = -x**2  # Derivative with respect to a
        J[:, 1] = -x     # Derivative with respect to b
        J[:, 2] = -1     # Derivative with respect to c
        return J
    
    # Optimize
    result = nonlinear_least_squares(cost_func, initial_params, jacobian_func=jacobian)
    
    # Check results
    assert result is not None
    assert len(result) == 3
    assert np.isclose(result[0], a_true, atol=0.1)
    assert np.isclose(result[1], b_true, atol=0.2)
    assert np.isclose(result[2], c_true, atol=0.2)


def test_nlls_curve_to_cost():
    """Test the curve-to-cost function converter."""
    # Generate data points for a line: y = 2*x + 3
    x = np.linspace(0, 10, 20)
    y_true = 2 * x + 3
    
    # Define a curve function (y = mx + b)
    def curve_func(params, x_data):
        m, b = params
        return m * x_data + b
    
    # Initial guess [slope, intercept]
    initial_params = np.array([1.0, 1.0])
    
    # Convert curve to cost function
    cost_func, jacobian_func = nlls_curve_to_cost(curve_func, x, y_true)
    
    # Optimize
    result = nonlinear_least_squares(cost_func, initial_params, jacobian_func=jacobian_func)
    
    # Check results
    assert result is not None
    assert len(result) == 2
    assert np.isclose(result[0], 2.0, atol=1e-5)  # slope
    assert np.isclose(result[1], 3.0, atol=1e-5)  # intercept