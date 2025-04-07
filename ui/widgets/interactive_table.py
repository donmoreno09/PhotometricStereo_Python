"""
Interactive table widget for managing light sources.
Python implementation of interactiveTable.m
"""

from PyQt6.QtWidgets import (QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QHeaderView, QAbstractItemView,
                             QCheckBox, QLabel, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor

import numpy as np


class InteractiveTable(QWidget):
    """
    Interactive table for editing light source positions.
    
    Features:
    - Edit light positions
    - Toggle light sources on/off
    - Import/export light positions
    - Auto-generate light positions
    """
    
    # Signals
    light_positions_changed = pyqtSignal(object)  # Emits numpy array of light positions
    light_states_changed = pyqtSignal(object)     # Emits list of boolean states (enabled/disabled)
    
    def __init__(self, parent=None, light_positions=None, enabled_states=None):
        """
        Initialize the interactive light table.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        light_positions : ndarray, optional
            Array of light positions shape (n, 3) where n is number of lights
        enabled_states : list, optional
            List of boolean states indicating if each light is enabled
        """
        super(InteractiveTable, self).__init__(parent)
        
        # Initialize data
        if light_positions is None:
            # Default: 4 light sources
            self.light_positions = np.array([
                [1.0, 0.0, 1.0],   # Front
                [0.0, 1.0, 1.0],   # Right
                [-1.0, 0.0, 1.0],  # Back
                [0.0, -1.0, 1.0]   # Left
            ], dtype=np.float32)
            
            # Normalize
            for i in range(self.light_positions.shape[0]):
                self.light_positions[i] = self.light_positions[i] / np.linalg.norm(self.light_positions[i])
        else:
            self.light_positions = light_positions
        
        # Initialize enabled states
        if enabled_states is None:
            self.enabled_states = [True] * self.light_positions.shape[0]
        else:
            self.enabled_states = enabled_states
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(5)  # Enable checkbox, X, Y, Z, Intensity
        self.table.setHorizontalHeaderLabels(['Enable', 'X', 'Y', 'Z', 'Intensity'])
        
        # Allow editing cells
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | 
                                  QAbstractItemView.EditTrigger.SelectedClicked)
        
        # Set column width behavior
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Checkbox
        for i in range(1, 5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)  # Coordinates and intensity
        
        layout.addWidget(self.table)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        # Add buttons
        self.add_button = QPushButton("Add Light")
        self.add_button.clicked.connect(self.add_light)
        
        self.remove_button = QPushButton("Remove Light")
        self.remove_button.clicked.connect(self.remove_light)
        
        self.normalize_button = QPushButton("Normalize")
        self.normalize_button.clicked.connect(self.normalize_lights)
        
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.import_lights)
        
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_lights)
        
        # Add buttons to layout
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.normalize_button)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
        # Populate table with initial data
        self.populate_table()
        
        # Connect signals
        self.table.cellChanged.connect(self.on_cell_changed)
    
    def populate_table(self):
        """Populate the table with light position data."""
        self.table.setRowCount(self.light_positions.shape[0])
        
        # Temporarily disconnect signal to avoid triggering while populating
        self.table.cellChanged.disconnect(self.on_cell_changed)
        
        for row in range(self.light_positions.shape[0]):
            # Add enable checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(self.enabled_states[row])
            checkbox.stateChanged.connect(lambda state, r=row: self.on_checkbox_change(r, state))
            self.table.setCellWidget(row, 0, checkbox)
            
            # Add X, Y, Z values
            for col in range(3):
                item = QTableWidgetItem(f"{self.light_positions[row, col]:.4f}")
                self.table.setItem(row, col + 1, item)
            
            # Add intensity (initially 1.0)
            intensity_item = QTableWidgetItem("1.0000")
            self.table.setItem(row, 4, intensity_item)
            
        # Reconnect signal
        self.table.cellChanged.connect(self.on_cell_changed)
    
    def add_light(self):
        """Add a new light to the table."""
        # Add a new default light position
        new_light = np.array([[0.0, 0.0, 1.0]])
        self.light_positions = np.vstack([self.light_positions, new_light])
        
        # Add to enabled states
        self.enabled_states.append(True)
        
        # Add row to table
        row = self.light_positions.shape[0] - 1
        self.table.setRowCount(row + 1)
        
        # Add checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state, r=row: self.on_checkbox_change(r, state))
        self.table.setCellWidget(row, 0, checkbox)
        
        # Add X, Y, Z values
        for col in range(3):
            item = QTableWidgetItem(f"{self.light_positions[row, col]:.4f}")
            self.table.setItem(row, col + 1, item)
        
        # Add intensity
        intensity_item = QTableWidgetItem("1.0000")
        self.table.setItem(row, 4, intensity_item)
        
        # Notify of change
        self.light_positions_changed.emit(self.light_positions)
        self.light_states_changed.emit(self.enabled_states)
    
    def remove_light(self):
        """Remove the selected light from the table."""
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a row to remove.")
            return
        
        # Remove in reverse order to avoid index issues
        for row in reversed(selected_rows):
            self.table.removeRow(row)
            self.light_positions = np.delete(self.light_positions, row, axis=0)
            del self.enabled_states[row]
        
        # Notify of change
        self.light_positions_changed.emit(self.light_positions)
        self.light_states_changed.emit(self.enabled_states)
    
    def normalize_lights(self):
        """Normalize all light vectors to unit length."""
        # Disconnect to avoid triggering while updating
        self.table.cellChanged.disconnect(self.on_cell_changed)
        
        # Normalize each light vector
        for row in range(self.light_positions.shape[0]):
            norm = np.linalg.norm(self.light_positions[row])
            
            if norm > 0:
                self.light_positions[row] = self.light_positions[row] / norm
                
                # Update table
                for col in range(3):
                    item = QTableWidgetItem(f"{self.light_positions[row, col]:.4f}")
                    self.table.setItem(row, col + 1, item)
        
        # Reconnect signal
        self.table.cellChanged.connect(self.on_cell_changed)
        
        # Notify of change
        self.light_positions_changed.emit(self.light_positions)
    
    def import_lights(self):
        """Import light positions from a file."""
        from PyQt6.QtWidgets import QFileDialog
        import os
        
        # Open file dialog
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Light Positions", "", 
            "Text Files (*.txt);;All Files (*)", options=options)
        
        if not filename:
            return
        
        try:
            # Read light positions from file
            lights = np.loadtxt(filename)
            
            # Check if file contains valid data
            if len(lights.shape) != 2 or lights.shape[1] != 3:
                if len(lights.shape) == 1 and lights.size == 3:
                    # Single light position
                    lights = lights.reshape(1, 3)
                else:
                    raise ValueError("Invalid light position data format")
            
            # Update light positions
            self.light_positions = lights
            
            # Reset enabled states
            self.enabled_states = [True] * self.light_positions.shape[0]
            
            # Update table
            self.populate_table()
            
            # Notify of change
            self.light_positions_changed.emit(self.light_positions)
            self.light_states_changed.emit(self.enabled_states)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import light positions: {str(e)}")
    
    def export_lights(self):
        """Export light positions to a file."""
        from PyQt6.QtWidgets import QFileDialog
        
        # Open file dialog
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Light Positions", "", 
            "Text Files (*.txt);;All Files (*)", options=options)
        
        if not filename:
            return
        
        try:
            # Add .txt extension if not present
            if not filename.endswith('.txt'):
                filename += '.txt'
                
            # Save light positions to file
            np.savetxt(filename, self.light_positions, fmt='%.6f')
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Light positions exported to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export light positions: {str(e)}")
    
    def on_cell_changed(self, row, col):
        """Handle changes to table cells."""
        if col >= 1 and col <= 4:  # X, Y, Z or Intensity columns
            try:
                # Get updated value
                value = float(self.table.item(row, col).text())
                
                if col <= 3:  # X, Y, Z coordinates
                    # Update light position
                    self.light_positions[row, col - 1] = value
                    
                    # Notify of change
                    self.light_positions_changed.emit(self.light_positions)
            
            except ValueError:
                # Reset to previous value
                if col <= 3:
                    self.table.item(row, col).setText(f"{self.light_positions[row, col - 1]:.4f}")
                else:
                    self.table.item(row, col).setText("1.0000")
    
    def on_checkbox_change(self, row, state):
        """Handle checkbox state changes."""
        self.enabled_states[row] = (state == Qt.CheckState.Checked)
        
        # Highlight/dim the row based on state
        for col in range(1, 5):
            item = self.table.item(row, col)
            if item:
                if state == Qt.CheckState.Checked:
                    item.setForeground(QColor(0, 0, 0))  # Black for enabled
                else:
                    item.setForeground(QColor(150, 150, 150))  # Gray for disabled
        
        # Notify of change
        self.light_states_changed.emit(self.enabled_states)
    
    def get_light_positions(self):
        """Get the current light positions."""
        return self.light_positions
    
    def get_enabled_states(self):
        """Get the enabled states for all lights."""
        return self.enabled_states
    
    def get_enabled_light_positions(self):
        """Get only the enabled light positions."""
        enabled_positions = []
        for i, enabled in enumerate(self.enabled_states):
            if enabled:
                enabled_positions.append(self.light_positions[i])
        
        return np.array(enabled_positions)

"""
Interactive table widget for managing light directions
"""
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt

class LightDirectionTable(QTableWidget):
    """
    Table widget for displaying and editing light directions
    """
    def __init__(self):
        super().__init__()
        
        # Set up table with 3 columns for X, Y, Z coordinates
        self.setColumnCount(3)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(["X", "Y", "Z"])
        
        # Configure table appearance
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setMinimumHeight(200)
        
    def set_light_directions(self, directions):
        """
        Set the light directions in the table
        
        Args:
            directions: List of (x, y, z) tuples representing light directions
        """
        self.setRowCount(len(directions))
        
        for i, (x, y, z) in enumerate(directions):
            self.setItem(i, 0, QTableWidgetItem(str(x)))
            self.setItem(i, 1, QTableWidgetItem(str(y)))
            self.setItem(i, 2, QTableWidgetItem(str(z)))
            
    def get_light_directions(self):
        """
        Get the light directions from the table
        
        Returns:
            List of (x, y, z) tuples representing light directions
        """
        directions = []
        for row in range(self.rowCount()):
            try:
                x = float(self.item(row, 0).text())
                y = float(self.item(row, 1).text())
                z = float(self.item(row, 2).text())
                directions.append((x, y, z))
            except (ValueError, AttributeError):
                # Skip invalid entries
                continue
                
        return directions
        
    def add_light(self, x=0, y=0, z=1):
        """Add a new light direction to the table"""
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(x)))
        self.setItem(row, 1, QTableWidgetItem(str(y)))
        self.setItem(row, 2, QTableWidgetItem(str(z)))
        
    def remove_selected_light(self):
        """Remove the currently selected light direction"""
        current_row = self.currentRow()
        if current_row >= 0:
            self.removeRow(current_row)