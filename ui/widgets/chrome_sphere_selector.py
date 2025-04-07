"""
Chrome sphere selection tool for light direction calibration.
Python implementation of ballSelection.m and ginputc.m
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QApplication, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage

import numpy as np
import cv2


class ChromeSphereSelector(QWidget):
    """
    Widget for selecting and fitting a circle to a chrome sphere in an image.
    Used for calibrating light directions.
    """
    
    # Signals
    circle_fit_complete = pyqtSignal(object)  # Emits dict with center and radius
    
    def __init__(self, parent=None, image=None):
        """
        Initialize the chrome sphere selector.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        image : ndarray, optional
            Image containing the chrome sphere (BGR format for OpenCV)
        """
        super(ChromeSphereSelector, self).__init__(parent)
        
        # Initialize data
        self.image = image
        self.points = []
        self.circle_center = None
        self.circle_radius = None
        self.min_points = 3  # Minimum points needed for circle fitting
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray")
        layout.addWidget(self.image_label)
        
        # Instructions
        instructions = QLabel("Click on points around the chrome sphere boundary. Click at least 3 points.")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Add buttons
        self.fit_button = QPushButton("Fit Circle")
        self.fit_button.clicked.connect(self.fit_circle)
        self.fit_button.setEnabled(False)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_selection)
        self.reset_button.setEnabled(False)
        
        self.close_button = QPushButton("Done")
        self.close_button.clicked.connect(self.accept_selection)
        self.close_button.setEnabled(False)
        
        # Add buttons to layout
        button_layout.addWidget(self.fit_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Set up image if provided
        if self.image is not None:
            self.set_image(self.image)
    
    def set_image(self, image):
        """
        Set the image to display.
        
        Parameters
        ----------
        image : ndarray
            Image containing the chrome sphere (BGR format for OpenCV)
        """
        self.image = image
        
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
        self.pixmap = QPixmap.fromImage(q_image)
        
        # Display image
        self.image_label.setPixmap(self.pixmap)
        
        # Reset points and circle
        self.reset_selection()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for point selection."""
        if self.image is None:
            return
        
        # Check if click is inside the image label
        if self.image_label.geometry().contains(event.pos()):
            # Convert global position to local coordinates
            local_pos = self.image_label.mapFromParent(event.pos())
            
            # Handle scaling and centering
            label_size = self.image_label.size()
            pixmap_size = self.pixmap.size()
            
            # Calculate offset for centered pixmap
            x_offset = (label_size.width() - pixmap_size.width()) / 2
            y_offset = (label_size.height() - pixmap_size.height()) / 2
            
            # Calculate actual position on the pixmap
            pixmap_x = local_pos.x() - x_offset
            pixmap_y = local_pos.y() - y_offset
            
            # Ensure click is within the pixmap
            if 0 <= pixmap_x < pixmap_size.width() and 0 <= pixmap_y < pixmap_size.height():
                # Add point
                self.points.append([pixmap_x, pixmap_y])
                
                # Update display
                self.update_display()
                
                # Enable fit and reset buttons if we have enough points
                if len(self.points) >= self.min_points:
                    self.fit_button.setEnabled(True)
                
                self.reset_button.setEnabled(True)
    
    def update_display(self):
        """Update the display with points and circle."""
        if self.image is None:
            return
        
        # Create a copy of the pixmap
        display_pixmap = self.pixmap.copy()
        
        # Create painter
        painter = QPainter(display_pixmap)
        
        # Set pen for points
        point_pen = QPen(Qt.red)
        point_pen.setWidth(3)
        painter.setPen(point_pen)
        
        # Draw points
        for pt in self.points:
            painter.drawPoint(int(pt[0]), int(pt[1]))
        
        # Draw circle if fit is complete
        if self.circle_center is not None and self.circle_radius is not None:
            circle_pen = QPen(Qt.green)
            circle_pen.setWidth(2)
            painter.setPen(circle_pen)
            
            # Draw circle
            center_x, center_y = self.circle_center
            painter.drawEllipse(
                int(center_x - self.circle_radius), 
                int(center_y - self.circle_radius),
                int(self.circle_radius * 2),
                int(self.circle_radius * 2)
            )
        
        # End painting
        painter.end()
        
        # Update display
        self.image_label.setPixmap(display_pixmap)
    
    def fit_circle(self):
        """Fit a circle to the selected points."""
        if len(self.points) < self.min_points:
            QMessageBox.warning(
                self, "Not Enough Points", 
                f"Please select at least {self.min_points} points."
            )
            return
        
        try:
            # Convert points to numpy array
            points = np.array(self.points)
            
            # Import circle fitting function
            from algorithms.sphere_detection.circle_fitting import fit_circle
            
            # Fit circle
            center, radius = fit_circle(points)
            
            # Store results
            self.circle_center = center
            self.circle_radius = radius
            
            # Enable done button
            self.close_button.setEnabled(True)
            
            # Update display
            self.update_display()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Circle Fitting Error", 
                f"Failed to fit circle: {str(e)}"
            )
    
    def reset_selection(self):
        """Reset the point selection and circle fit."""
        self.points = []
        self.circle_center = None
        self.circle_radius = None
        
        # Disable buttons
        self.fit_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.close_button.setEnabled(False)
        
        # Update display
        if self.image is not None:
            self.image_label.setPixmap(self.pixmap)
    
    def accept_selection(self):
        """Accept the circle fit and emit signal."""
        if self.circle_center is not None and self.circle_radius is not None:
            # Create result dictionary
            result = {
                'center': self.circle_center,
                'radius': self.circle_radius
            }
            
            # Emit signal
            self.circle_fit_complete.emit(result)
    
    def get_circle_parameters(self):
        """Get the circle parameters."""
        if self.circle_center is not None and self.circle_radius is not None:
            return {
                'center': self.circle_center,
                'radius': self.circle_radius
            }
        else:
            return None