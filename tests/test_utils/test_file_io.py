"""
Tests for file I/O utilities.
"""

import os
import pytest
import numpy as np
from utils.file_io import get_image_name_only, compute_path_out


def test_get_image_name_only():
    """Test extraction of image name without extension."""
    # Test with simple paths
    assert get_image_name_only("image.jpg") == "image"
    assert get_image_name_only("image.png") == "image"
    assert get_image_name_only("image") == "image"  # No extension
    
    # Test with paths
    assert get_image_name_only("path/to/image.jpg") == "image"
    assert get_image_name_only("C:\\path\\to\\image.jpg") == "image"
    
    # Test with multiple dots
    assert get_image_name_only("image.with.dots.jpg") == "image.with.dots"
    
    # Test with uppercase extensions
    assert get_image_name_only("image.JPG") == "image"
    assert get_image_name_only("image.PNG") == "image"


def test_compute_path_out(tmp_path):
    """Test computation of output path."""
    # Create a temporary config for testing
    config = {
        'output_dir': str(tmp_path),
        'input_file': 'path/to/input/image.jpg',
        'export': {
            'format': 'png'
        }
    }
    
    # Test basic path computation
    path = compute_path_out(config, "normal")
    assert path.endswith(".png")
    assert "normal" in path
    assert "image" in path
    
    # Test path computation with different output type
    path = compute_path_out(config, "depth")
    assert path.endswith(".png")
    assert "depth" in path
    
    # Test with different format
    config['export']['format'] = 'jpg'
    path = compute_path_out(config, "albedo")
    assert path.endswith(".jpg")
    assert "albedo" in path
    
    # Test with different output directory
    new_dir = tmp_path / "output"
    os.makedirs(new_dir, exist_ok=True)
    config['output_dir'] = str(new_dir)
    path = compute_path_out(config, "normal")
    assert str(new_dir) in path
    
    # Test with specified base name
    path = compute_path_out(config, "normal", base_name="custom")
    assert "custom_normal" in path
    
    # Test with specified extension
    path = compute_path_out(config, "depth", extension="exr")
    assert path.endswith(".exr")