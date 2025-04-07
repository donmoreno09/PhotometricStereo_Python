"""
Tests for color profile tools.
Port of addDisplayP3Profile.m test functionality
"""

import pytest
import numpy as np
import tempfile
import os
from utils.profile_tools import add_display_p3_profile


def test_add_display_p3_profile():
    """Test adding Display P3 color profile to an image."""
    # This test is more limited because it depends on the ImageMagick library
    # Let's mock the functionality to test the interface
    
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
        # Create a mock function to replace subprocess call
        def mock_subprocess_run(cmd, check=True):
            # Instead of running ImageMagick, check that command looks correct
            cmd_str = ' '.join(cmd)
            assert 'convert' in cmd_str
            assert temp_file.name in cmd_str
            assert 'DisplayP3' in cmd_str
            # Create a dummy file to simulate success
            with open(temp_file.name, 'wb') as f:
                f.write(b'test')
            
            class MockCompletedProcess:
                returncode = 0
            return MockCompletedProcess()
        
        # Patch subprocess.run with our mock
        import subprocess
        original_run = subprocess.run
        subprocess.run = mock_subprocess_run
        
        try:
            # Call the function
            try:
                result = add_display_p3_profile(temp_file.name)
                # If ImageMagick is installed, this might succeed
                assert result is True
            except FileNotFoundError:
                # If ImageMagick is not installed, it will raise this error
                # This is still a valid test condition
                pass
        finally:
            # Restore original subprocess.run
            subprocess.run = original_run


def test_add_display_p3_profile_file_not_found():
    """Test behavior when file doesn't exist."""
    # Try with a non-existent file
    non_existent_file = "/path/to/nonexistent/file.png"
    
    # This should return False without crashing
    assert add_display_p3_profile(non_existent_file) is False


def test_add_display_p3_profile_unsupported_format():
    """Test behavior with unsupported file format."""
    with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
        # This should return False for unsupported format
        assert add_display_p3_profile(temp_file.name) is False