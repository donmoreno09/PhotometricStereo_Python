"""
Custom UI widgets for the photometric stereo application.
"""

from PyQt5.QtWidgets import (QLabel, QWidget, QVBoxLayout, QProgressBar,
                             QHBoxLayout, QPushButton, QSizePolicy, QScrollArea,
                             QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor, QMouseEvent

import numpy as np
import cv2


class ImageLabel(QLabel):
    """
    Custom QLabel for displaying images with zoom and pan functionality.
    """
    
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.pan_position = QPoint(0, 0)
        self.panning = False
        self.last_pan_pos = None
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Enable focus to receive key events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Set alignment
        self.setAlignment(Qt.AlignCenter)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def set_image(self, image):
        """
        Set the image to display.
        
        Parameters
        ----------
        image : ndarray
            Image (BGR format for OpenCV) or path to image
        """
        if isinstance(image, str):
            # Load from file
            self.original_pixmap = QPixmap(image)
        else:
            # Convert OpenCV BGR to RGB for Qt
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Convert grayscale to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w
            
            # Create QImage from numpy array
            q_image = QImage(rgb_image.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Create pixmap from QImage
            self.original_pixmap = QPixmap.fromImage(q_image)
        
        # Reset zoom and pan
        self.zoom_factor = 1.0
        self.pan_position = QPoint(0, 0)
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the displayed image with current zoom and pan."""
        if self.original_pixmap is None:
            return
        
        # Start with original pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * self.zoom_factor),
            int(self.original_pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Create a pixmap large enough for the scaled image plus pan
        display_pixmap = QPixmap(self.width(), self.height())
        display_pixmap.fill(Qt.black)
        
        # Calculate position for centered pixmap
        center_x = (self.width() - scaled_pixmap.width()) / 2
        center_y = (self.height() - scaled_pixmap.height()) / 2
        
        # Apply pan
        pos_x = center_x + self.pan_position.x()
        pos_y = center_y + self.pan_position.y()
        
        # Create painter
        painter = QPainter(display_pixmap)
        
        # Draw scaled pixmap
        painter.drawPixmap(int(pos_x), int(pos_y), scaled_pixmap)
        
        # End painting
        painter.end()
        
        # Set pixmap
        self.setPixmap(display_pixmap)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if self.original_pixmap is None:
            return
        
        # Get delta
        delta = event.angleDelta().y()
        
        # Calculate zoom factor
        zoom_delta = 1.1 if delta > 0 else 0.9
        new_zoom = self.zoom_factor * zoom_delta
        
        # Limit zoom
        if 0.1 <= new_zoom <= 10.0:
            self.zoom_factor = new_zoom
            
            # Update display
            self.update_display()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for panning."""
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.last_pan_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for panning."""
        if event.button() == Qt.LeftButton:
            self.panning = False
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for panning."""
        if self.panning and self.last_pan_pos is not None:
            # Calculate delta
            delta = event.pos() - self.last_pan_pos
            self.last_pan_pos = event.pos()
            
            # Update pan position
            self.pan_position += delta
            
            # Update display
            self.update_display()
    
    def reset_view(self):
        """Reset zoom and pan to original state."""
        self.zoom_factor = 1.0
        self.pan_position = QPoint(0, 0)
        
        # Update display
        self.update_display()


class ClickableLabel(QLabel):
    """
    Label that emits a signal when clicked.
    """
    
    # Signal emitted when label is clicked
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        
        # Call parent method
        super(ClickableLabel, self).mousePressEvent(event)


class ProgressBar(QProgressBar):
    """
    Custom progress bar with percentage text.
    """
    
    def __init__(self, parent=None):
        super(ProgressBar, self).__init__(parent)
        
        # Set format string to show percentage
        self.setFormat("%p%")
        
        # Set text visibility
        self.setTextVisible(True)
        
        # Set initial range and value
        self.setRange(0, 100)
        self.setValue(0)


class CollapsibleBox(QWidget):
    """
    Collapsible box/section widget.
    """
    
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toggle button
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("text-align: left; padding: 5px;")
        self.toggle_button.clicked.connect(self.on_toggle)
        
        # Content widget
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        
        # Add widgets to layout
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content)
        
        # Hide content initially
        self.content.setVisible(False)
    
    def on_toggle(self, checked):
        """Handle toggle button click."""
        self.content.setVisible(checked)
    
    def setContentLayout(self, layout):
        """Set the content layout."""
        # Clear existing layout
        QWidget().setLayout(self.content.layout())
        
        # Set new layout
        self.content.setLayout(layout)
    
    def addWidget(self, widget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)