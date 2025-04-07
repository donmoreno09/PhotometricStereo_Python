"""
Tests for interactive table widget.
Port of interactiveTable.m test functionality
"""

import pytest
from unittest.mock import patch, MagicMock
from ui.widgets.interactive_table import InteractiveTable


def test_interactive_table_initialization():
    """Test initialization of Interactive Table."""
    # Create table with sample data
    headers = ['Column 1', 'Column 2', 'Column 3']
    data = [['A', 1, True], ['B', 2, False], ['C', 3, True]]
    
    table = InteractiveTable(headers, data)
    
    # Check attributes
    assert table.headers == headers
    assert table.data == data
    assert table.column_count == 3
    assert table.row_count == 3


def test_interactive_table_render():
    """Test rendering the table."""
    # Create a simple table
    headers = ['Name', 'Age']
    data = [['Alice', 25], ['Bob', 30]]
    
    table = InteractiveTable(headers, data)
    
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        # Render the table
        table.render()
        
        # Check number of print calls (header + divider + 2 rows)
        assert mock_print.call_count >= 4
        
        # Check headers are in the output
        header_call = mock_print.call_args_list[0][0][0]
        assert 'Name' in header_call
        assert 'Age' in header_call
        
        # Check data is in output
        row_calls = ''.join([str(call[0][0]) for call in mock_print.call_args_list])
        assert 'Alice' in row_calls
        assert '25' in row_calls
        assert 'Bob' in row_calls
        assert '30' in row_calls


def test_interactive_table_sort():
    """Test sorting the table."""
    # Create a simple table
    headers = ['Name', 'Age']
    data = [['Alice', 30], ['Bob', 25], ['Charlie', 35]]
    
    table = InteractiveTable(headers, data)
    
    # Sort by Name (ascending)
    table.sort(0, ascending=True)
    assert table.data[0][0] == 'Alice'
    assert table.data[1][0] == 'Bob'
    assert table.data[2][0] == 'Charlie'
    
    # Sort by Age (ascending)
    table.sort(1, ascending=True)
    assert table.data[0][0] == 'Bob'
    assert table.data[1][0] == 'Alice'
    assert table.data[2][0] == 'Charlie'
    
    # Sort by Age (descending)
    table.sort(1, ascending=False)
    assert table.data[0][0] == 'Charlie'
    assert table.data[1][0] == 'Alice'
    assert table.data[2][0] == 'Bob'


def test_interactive_table_filter():
    """Test filtering the table."""
    # Create a table with various data
    headers = ['Name', 'Department', 'Age']
    data = [
        ['Alice', 'HR', 30],
        ['Bob', 'IT', 25],
        ['Charlie', 'HR', 35],
        ['David', 'IT', 40],
        ['Eve', 'Finance', 28]
    ]
    
    table = InteractiveTable(headers, data)
    
    # Filter by Department = 'HR'
    filtered_data = table.filter(1, 'HR')
    assert len(filtered_data) == 2
    assert filtered_data[0][0] == 'Alice'
    assert filtered_data[1][0] == 'Charlie'
    
    # Filter by Age > 30
    filtered_data = table.filter(2, lambda x: x > 30)
    assert len(filtered_data) == 2
    assert filtered_data[0][0] == 'Charlie'
    assert filtered_data[1][0] == 'David'


def test_interactive_table_edit_cell():
    """Test editing a cell in the table."""
    # Create a simple table
    headers = ['Name', 'Age']
    data = [['Alice', 25], ['Bob', 30]]
    
    table = InteractiveTable(headers, data)
    
    # Edit a cell
    table.edit_cell(0, 1, 26)  # Change Alice's age to 26
    
    # Check the change
    assert table.data[0][1] == 26
    assert table.data[1][1] == 30  # Bob's age unchanged