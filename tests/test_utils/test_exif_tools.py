"""
Tests for EXIF tools.
"""

import os
import pytest
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
from utils.exif_tools import check_if_exif_is_installed, add_exif


def test_check_if_exif_is_installed():
    """Test checking if exiftool is installed."""
    # Mock subprocess.check_output to simulate exiftool being installed
    with patch('subprocess.check_output') as mock_check_output:
        # Simulate success
        mock_check_output.return_value = b'exiftool version 12.30'
        assert check_if_exif_is_installed() is True
        
        # Simulate failure (command not found)
        mock_check_output.side_effect = subprocess.CalledProcessError(127, 'exiftool')
        assert check_if_exif_is_installed() is False
        
        # Simulate other error
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'exiftool')
        assert check_if_exif_is_installed() is False


def test_add_exif():
    """Test adding EXIF data to an image."""
    # Mock check_if_exif_is_installed to always return True
    with patch('utils.exif_tools.check_if_exif_is_installed', return_value=True), \
         patch('subprocess.run') as mock_run:
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            # Test with minimal metadata
            metadata = {
                'Artist': 'Test User',
                'Copyright': 'Test Copyright'
            }
            
            # Call add_exif
            add_exif(temp_file.name, metadata)
            
            # Check that subprocess.run was called correctly
            mock_run.assert_called_once()
            
            # Check that the command contains the metadata entries
            args = mock_run.call_args[0][0]
            assert any('Artist=Test User' in arg for arg in args)
            assert any('Copyright=Test Copyright' in arg for arg in args)
            
            # Reset mock
            mock_run.reset_mock()
            
            # Test with more metadata
            metadata = {
                'Artist': 'Test User',
                'Copyright': 'Test Copyright',
                'ImageDescription': 'Test Description',
                'Software': 'Photometric Stereo Py'
            }
            
            # Call add_exif
            add_exif(temp_file.name, metadata)
            
            # Check that subprocess.run was called correctly
            mock_run.assert_called_once()
            
            # Check that the command contains all metadata entries
            args = mock_run.call_args[0][0]
            assert any('Artist=Test User' in arg for arg in args)
            assert any('Copyright=Test Copyright' in arg for arg in args)
            assert any('ImageDescription=Test Description' in arg for arg in args)
            assert any('Software=Photometric Stereo Py' in arg for arg in args)


def test_add_exif_exiftool_not_installed():
    """Test behavior when exiftool is not installed."""
    # Mock check_if_exif_is_installed to always return False
    with patch('utils.exif_tools.check_if_exif_is_installed', return_value=False), \
         patch('subprocess.run') as mock_run:
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            metadata = {'Artist': 'Test User'}
            
            # Call add_exif
            # Should return without calling subprocess.run
            add_exif(temp_file.name, metadata)
            
            # Check that subprocess.run was not called
            mock_run.assert_not_called()