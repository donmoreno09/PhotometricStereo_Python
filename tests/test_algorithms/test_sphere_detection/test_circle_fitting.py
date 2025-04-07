"""
Tests for circle fitting algorithms.
Port of FitCircle.m tests and related functionality
"""

import pytest
import numpy as np
from algorithms.sphere_detection.circle_fitting import fit_circle


def test_fit_circle_exact():
    """Test circle fitting with exact circle points."""
    # Create points exactly on a circle
    center_true = np.array([5.0, 2.0])
    radius_true = 3.0
    
    # Generate points along the circle
    n_points = 20
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = center_true[0] + radius_true * np.cos(theta)
    y = center_true[1] + radius_true * np.sin(theta)
    points = np.column_stack((x, y))
    
    # Fit circle
    center, radius = fit_circle(points)
    
    # Check results
    assert np.allclose(center, center_true, atol=1e-10)
    assert np.isclose(radius, radius_true, atol=1e-10)


def test_fit_circle_with_noise():
    """Test circle fitting with noisy points."""
    # Create points with noise
    center_true = np.array([3.0, 4.0])
    radius_true = 2.5
    
    # Generate points along the circle with noise
    n_points = 30
    np.random.seed(42)  # For reproducibility
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    noise = np.random.normal(0, 0.1, size=(n_points, 2))  # Gaussian noise
    
    x = center_true[0] + radius_true * np.cos(theta) + noise[:, 0]
    y = center_true[1] + radius_true * np.sin(theta) + noise[:, 1]
    points = np.column_stack((x, y))
    
    # Fit circle
    center, radius = fit_circle(points)
    
    # Check results with some tolerance due to noise
    assert np.allclose(center, center_true, atol=0.15)
    assert np.isclose(radius, radius_true, atol=0.15)


def test_fit_circle_with_partial_arc():
    """Test circle fitting with points only on part of a circle."""
    # Create points on only a portion of a circle
    center_true = np.array([2.0, 3.0])
    radius_true = 4.0
    
    # Generate points along a partial arc (1/4 of a circle)
    n_points = 15
    theta = np.linspace(0, np.pi/2, n_points)
    x = center_true[0] + radius_true * np.cos(theta)
    y = center_true[1] + radius_true * np.sin(theta)
    points = np.column_stack((x, y))
    
    # Fit circle
    center, radius = fit_circle(points)
    
    # Check results with higher tolerance (partial arcs are harder to fit)
    assert np.allclose(center, center_true, atol=0.2)
    assert np.isclose(radius, radius_true, atol=0.2)


def test_fit_circle_with_outliers():
    """Test circle fitting with outliers."""
    # Create points with some outliers
    center_true = np.array([4.0, 3.0])
    radius_true = 3.0
    
    # Generate good points
    n_good = 25
    theta_good = np.linspace(0, 2*np.pi, n_good, endpoint=False)
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.05, size=(n_good, 2))
    
    x_good = center_true[0] + radius_true * np.cos(theta_good) + noise[:, 0]
    y_good = center_true[1] + radius_true * np.sin(theta_good) + noise[:, 1]
    
    # Generate outliers
    n_outliers = 5
    x_outliers = np.random.uniform(0, 8, n_outliers)
    y_outliers = np.random.uniform(0, 6, n_outliers)
    
    # Combine points
    x = np.concatenate([x_good, x_outliers])
    y = np.concatenate([y_good, y_outliers])
    points = np.column_stack((x, y))
    
    # Fit circle
    center, radius = fit_circle(points)
    
    # Check results with tolerance for outliers
    assert np.allclose(center, center_true, atol=0.3)
    assert np.isclose(radius, radius_true, atol=0.3)


def test_fit_circle_minimum_points():
    """Test circle fitting with minimal number of points (3)."""
    # Create exact circle with only 3 points
    # For a circle, three non-collinear points are enough to define it
    
    points = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ])
    
    # These points form a circle with:
    center_true = np.array([0.5, 0.5/np.sqrt(3)])
    radius_true = 1/np.sqrt(3)
    
    # Fit circle
    center, radius = fit_circle(points)
    
    # Check results
    assert np.allclose(center, center_true, atol=1e-10)
    assert np.isclose(radius, radius_true, atol=1e-10)