"""
Tests for chrome sphere selector widget.
Ports of ballSelection.m and ginputc.m test functionality
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from ui.widgets.chrome_sphere_selector import ChromeSphereSelector


def create_test_image():
    """Create a test image with a simulated chrome sphere."""
    # Create a black image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add a circle
    center = (100, 100)
    radius = 50
    cv2.circle(img, center, radius, (200, 200, 200), -1)
    
    # Add a highlight
    highlight = (80, 70)
    cv2.circle(img, highlight, 10, (255, 255, 255), -1)
    
    return img


def test_chrome_sphere_selector_initialization():
    """Test initialization of Chrome Sphere Selector."""
    # Create test image
    img = create_test_image()
    
    # Create selector
    selector = ChromeSphereSelector(img)
    
    # Check attributes
    assert selector.image is not None
    assert selector.image.shape == img.shape
    assert selector.center is None
    assert selector.radius is None


@pytest.mark.skipif(not hasattr(cv2, 'imshow'), reason="OpenCV GUI not available")
def test_chrome_sphere_selector_select_circle():
    """Test circle selection method."""
    # Skip if running in a headless environment
    
    # Create test image
    img = create_test_image()
    
    # Mock cv2.setMouseCallback to simulate user clicks
    # This is complex to test properly due to GUI interaction
    # For now, we'll just test that the method exists and can be called
    with patch('cv2.setMouseCallback') as mock_callback, \
         patch('cv2.imshow'), \
         patch('cv2.waitKey', return_value=13):  # Simulate Enter key
        
        # Create selector
        selector = ChromeSphereSelector(img)
        
        # Call select_circle
        # We're not actually testing the result, just that it runs
        selector.select_circle()
        
        # Check that setMouseCallback was called
        mock_callback.assert_called()


def test_chrome_sphere_selector_verify_selection():
    """Test selection verification method."""
    # Create test image
    img = create_test_image()
    
    # Create selector
    selector = ChromeSphereSelector(img)
    
    # Set center and radius
    selector.center = (100, 100)
    selector.radius = 50
    
    # Test verification
    # Mock the user input for verification
    with patch('cv2.imshow'), \
         patch('cv2.waitKey', return_value=13):  # Simulate Enter key
        
        # Call verify_selection
        selector.verify_selection()
        
        # We're not making assertions here since it's just UI interaction
        # The test passes if no exceptions are raised


def test_chrome_sphere_selector_automated_detection():
    """Test automated circle detection."""
    # Create test image
    img = create_test_image()
    
    # Create selector with automated detection
    selector = ChromeSphereSelector(img)
    
    # Call auto-detect
    success = selector.detect_circle_auto()
    
    # Check that detection was successful
    assert success is True
    
    # Check that center and radius are set
    assert selector.center is not None
    assert selector.radius is not None
    
    # Check if the detected center is close to expected
    assert abs(selector.center[0] - 100) < 10
    assert abs(selector.center[1] - 100) < 10
    
    # Check if the detected radius is close to expected
    assert abs(selector.radius - 50) < 10


def test_chrome_sphere_selector_no_circle_detected():
    """Test behavior when no circle is detected."""
    # Create a blank image (no circles)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Create selector
    selector = ChromeSphereSelector(img)
    
    # Call auto-detect
    success = selector.detect_circle_auto()
    
    # Check that detection failed
    assert success is False
    
    # Center and radius should remain None
    assert selector.center is None
    assert selector.radius is None