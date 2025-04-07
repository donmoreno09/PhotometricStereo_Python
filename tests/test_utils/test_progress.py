"""
Tests for progress reporting utilities.
Port of ShowProgress.m test functionality
"""

import pytest
import time
from unittest.mock import patch
from utils.progress import show_progress


def test_show_progress_basic():
    """Test basic progress reporting functionality."""
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        # Call show_progress with different values
        show_progress(0, 100)
        show_progress(50, 100)
        show_progress(100, 100)
        
        # Check that print was called 3 times
        assert mock_print.call_count == 3
        
        # Check first call (0%)
        args = mock_print.call_args_list[0][0][0]
        assert '0%' in args
        
        # Check second call (50%)
        args = mock_print.call_args_list[1][0][0]
        assert '50%' in args
        
        # Check third call (100%)
        args = mock_print.call_args_list[2][0][0]
        assert '100%' in args


def test_show_progress_throttling():
    """Test throttling to avoid too frequent updates."""
    # Mock print and time to control timing
    with patch('builtins.print') as mock_print, \
         patch('time.time') as mock_time:
        
        # Mock time to simulate elapsed time
        mock_time.side_effect = [0.0, 0.1, 0.2]  # Less than default throttle time
        
        # Call show_progress in quick succession
        show_progress(0, 100)
        show_progress(10, 100)  # Should be throttled
        show_progress(20, 100)  # Should be throttled
        
        # Check that print was called only once (other calls throttled)
        assert mock_print.call_count == 1
        
        # Now simulate more time passing
        mock_time.side_effect = [0.0, 1.1]  # More than default throttle time
        
        # Reset mock_print
        mock_print.reset_mock()
        
        # Call show_progress again
        show_progress(0, 100)
        show_progress(50, 100)  # Should not be throttled
        
        # Check that print was called twice
        assert mock_print.call_count == 2


def test_show_progress_custom_message():
    """Test progress reporting with custom message."""
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        # Call show_progress with custom message
        show_progress(50, 100, message="Processing widgets")
        
        # Check that print was called
        mock_print.assert_called_once()
        
        # Check that the message appears in output
        args = mock_print.call_args[0][0]
        assert 'Processing widgets' in args
        assert '50%' in args


def test_show_progress_edge_cases():
    """Test edge cases like negative progress or total."""
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        # Negative progress
        show_progress(-10, 100)
        
        # Progress > total
        show_progress(150, 100)
        
        # Zero total
        show_progress(0, 0)
        
        # Check that print was called 3 times
        assert mock_print.call_count == 3
        
        # For negative progress, should show 0%
        args = mock_print.call_args_list[0][0][0]
        assert '0%' in args
        
        # For progress > total, should show 100%
        args = mock_print.call_args_list[1][0][0]
        assert '100%' in args