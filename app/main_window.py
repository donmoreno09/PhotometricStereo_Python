"""
Main application window for the Photometric Stereo application
This implements the main GUI previously defined in the NLights MATLAB class
"""

import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                            QLabel, QPushButton, QFileDialog, QMessageBox,
                            QTabWidget, QScrollArea, QFrame, QGridLayout,
                            QGroupBox, QComboBox, QCheckBox, QSpinBox,
                            QDoubleSpinBox, QProgressBar, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QDir
from PyQt6.QtGui import QPixmap, QImage

from core.photometric_stereo import PhotometricStereo
from core.normal_maps import compute_normal_maps
from core.depth_maps import compute_depth_maps
from ui.widgets.interactive_table import LightDirectionTable
from utils.image_processing import specularize_x


class ProcessingThread(QThread):
    """Worker thread to handle image processing operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            # This would be called with all the processing steps
            result = {}
            
            # Signal progress as processing continues
            for i in range(101):
                self.progress.emit(i)
                self.msleep(20)  # Just for demonstration
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for Photometric Stereo"""
    
    def __init__(self):
        super().__init__()
        
        self.config = {
            'light_directions': None,
            'input_images': [],
            'working_directory': '',
            'output_directory': '',
            'all_lights_on_image': None,
            'chrome_sphere_image': None,
            'normal_map': None,
            'depth_map': None,
            'albedo': None
        }
        
        self.processing_thread = None
        
        self.init_ui()
        self.setWindowTitle("Photometric Stereo")
        self.resize(1200, 800)
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different sections
        tab_widget = QTabWidget()
        
        # Create tabs
        input_tab = self.create_input_tab()
        processing_tab = self.create_processing_tab()
        results_tab = self.create_results_tab()
        export_tab = self.create_export_tab()
        
        # Add tabs to the tab widget
        tab_widget.addTab(input_tab, "Input Images")
        tab_widget.addTab(processing_tab, "Processing")
        tab_widget.addTab(results_tab, "Results")
        tab_widget.addTab(export_tab, "Export")
        
        # Add tab widget to the main layout
        main_layout.addWidget(tab_widget)
        
        # Add status bar for messages
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Set the central widget
        self.setCentralWidget(central_widget)
    
    def create_input_tab(self):
        """Create the input images tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create group for selecting input directory
        input_group = QGroupBox("Input Images")
        input_layout = QVBoxLayout(input_group)
        
        # Add buttons for selecting input directory and files
        select_dir_button = QPushButton("Select Image Directory...")
        select_dir_button.clicked.connect(self.select_input_directory)
        input_layout.addWidget(select_dir_button)
        
        # Add container for image previews
        self.image_preview_layout = QGridLayout()
        input_layout.addLayout(self.image_preview_layout)
        
        # Add group for light direction settings
        light_group = QGroupBox("Light Directions")
        light_layout = QVBoxLayout(light_group)
        
        # Add options for loading/configuring light directions
        light_methods = QComboBox()
        light_methods.addItems(["From Chrome Sphere", "From Configuration File", "Manual Entry"])
        light_layout.addWidget(light_methods)
        
        # Chrome sphere selection
        self.select_sphere_button = QPushButton("Select Chrome Sphere Image...")
        self.select_sphere_button.clicked.connect(self.select_chrome_sphere)
        light_layout.addWidget(self.select_sphere_button)
        
        # Add the light direction table
        self.light_table = LightDirectionTable()
        light_layout.addWidget(self.light_table)
        
        # Add load/save buttons for light configurations
        light_buttons_layout = QHBoxLayout()
        load_light_button = QPushButton("Load Configuration...")
        save_light_button = QPushButton("Save Configuration...")
        light_buttons_layout.addWidget(load_light_button)
        light_buttons_layout.addWidget(save_light_button)
        light_layout.addLayout(light_buttons_layout)
        
        # Add groups to main layout
        layout.addWidget(input_group)
        layout.addWidget(light_group)
        
        return tab
    
    def create_processing_tab(self):
        """Create the processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout(options_group)
        
        # Add various processing options
        options_layout.addWidget(QLabel("Normal Map Generation:"), 0, 0)
        normal_method = QComboBox()
        normal_method.addItems(["Standard Photometric Stereo", "Advanced (NLLS)"])
        options_layout.addWidget(normal_method, 0, 1)
        
        options_layout.addWidget(QLabel("Depth Map Generation:"), 1, 0)
        depth_method = QComboBox()
        depth_method.addItems(["Gradient Integration", "Poisson"])
        options_layout.addWidget(depth_method, 1, 1)
        
        options_layout.addWidget(QLabel("Polynomial Correction:"), 2, 0)
        poly_correction = QCheckBox("Apply polynomial correction")
        options_layout.addWidget(poly_correction, 2, 1)
        
        # Progress indicator
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        # Process button
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        
        # Add all widgets to layout
        layout.addWidget(options_group)
        layout.addWidget(progress_group)
        layout.addWidget(self.process_button)
        
        return tab
    
    def create_results_tab(self):
        """Create the results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create splitter for main sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Results selection panel (left side)
        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)
        
        selection_layout.addWidget(QLabel("<b>Available Results:</b>"))
        
        # Buttons for different result types
        self.normal_map_button = QPushButton("Normal Map")
        self.normal_map_button.clicked.connect(lambda: self.show_result("normal_map"))
        
        self.depth_map_button = QPushButton("Depth Map")
        self.depth_map_button.clicked.connect(lambda: self.show_result("depth_map"))
        
        self.albedo_button = QPushButton("Albedo")
        self.albedo_button.clicked.connect(lambda: self.show_result("albedo"))
        
        selection_layout.addWidget(self.normal_map_button)
        selection_layout.addWidget(self.depth_map_button)
        selection_layout.addWidget(self.albedo_button)
        selection_layout.addStretch()
        
        # Results display panel (right side)
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        self.result_label = QLabel("No results to display yet")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        display_layout.addWidget(self.result_label)
        
        # Add widgets to splitter
        splitter.addWidget(selection_widget)
        splitter.addWidget(display_widget)
        splitter.setSizes([200, 600])  # Initial sizes
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_export_tab(self):
        """Create the export tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Output directory selection
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        self.output_dir_label = QLabel("No directory selected")
        select_output_dir = QPushButton("Select Output Directory...")
        select_output_dir.clicked.connect(self.select_output_directory)
        
        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(select_output_dir)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout(export_group)
        
        # Normal map export
        export_layout.addWidget(QLabel("Normal Map:"), 0, 0)
        self.export_normal_checkbox = QCheckBox("Export Normal Map")
        self.export_normal_checkbox.setChecked(True)
        export_layout.addWidget(self.export_normal_checkbox, 0, 1)
        
        # Depth map export
        export_layout.addWidget(QLabel("Depth Map:"), 1, 0)
        self.export_depth_checkbox = QCheckBox("Export Depth Map")
        self.export_depth_checkbox.setChecked(True)
        export_layout.addWidget(self.export_depth_checkbox, 1, 1)
        
        # 3D model export
        export_layout.addWidget(QLabel("3D Model:"), 2, 0)
        self.export_3d_checkbox = QCheckBox("Export 3D Model")
        self.export_3d_checkbox.setChecked(True)
        export_layout.addWidget(self.export_3d_checkbox, 2, 1)
        
        # 3D format selection
        export_layout.addWidget(QLabel("3D Format:"), 3, 0)
        self.export_3d_format = QComboBox()
        self.export_3d_format.addItems(["OBJ", "STL", "PLY"])
        export_layout.addWidget(self.export_3d_format, 3, 1)
        
        # Albedo export
        export_layout.addWidget(QLabel("Albedo:"), 4, 0)
        self.export_albedo_checkbox = QCheckBox("Export Albedo")
        self.export_albedo_checkbox.setChecked(True)
        export_layout.addWidget(self.export_albedo_checkbox, 4, 1)
        
        # ICC profile
        export_layout.addWidget(QLabel("Color Profile:"), 5, 0)
        self.icc_profile_combo = QComboBox()
        self.icc_profile_combo.addItems(["None", "sRGB", "Display P3"])
        export_layout.addWidget(self.icc_profile_combo, 5, 1)
        
        # Export button
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        
        # Add all widgets to layout
        layout.addWidget(dir_group)
        layout.addWidget(export_group)
        layout.addWidget(export_button)
        layout.addStretch()
        
        return tab
    
    def select_input_directory(self):
        """Open a dialog to select the input directory containing images"""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Image Directory")
        if directory:
            self.config['working_directory'] = directory
            self.load_images_from_directory(directory)
    
    def load_images_from_directory(self, directory):
        """Load images from the selected directory"""
        # Clear current images
        self.clear_layout(self.image_preview_layout)
        self.config['input_images'] = []
        
        # Find image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(QDir(directory).entryList([ext], QDir.Filter.Files))
        
        # Display preview of found images
        if image_files:
            # Store image paths
            self.config['input_images'] = [os.path.join(directory, f) for f in image_files]
            
            # Create image previews (up to 16)
            max_preview = min(16, len(image_files))
            rows = (max_preview + 3) // 4  # Ceil division for rows
            
            for i in range(max_preview):
                image_path = os.path.join(directory, image_files[i])
                preview = QLabel()
                pixmap = QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                preview.setPixmap(pixmap)
                preview.setToolTip(image_files[i])
                
                row = i // 4
                col = i % 4
                self.image_preview_layout.addWidget(preview, row, col)
            
            # Update status
            self.status_bar.showMessage(f"Loaded {len(image_files)} images from {directory}")
        else:
            QMessageBox.warning(self, "No Images Found", 
                               f"No image files found in the selected directory: {directory}")
    
    def select_chrome_sphere(self):
        """Open dialog to select a chrome sphere image for light direction calculation"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Chrome Sphere Image", "", "Image Files (*.jpg *.jpeg *.png *.tif *.tiff)")
        
        if file_path:
            self.config['chrome_sphere_image'] = file_path
            # Here you would call the chrome sphere processing function
            # and populate the light direction table with the results
            self.status_bar.showMessage(f"Chrome sphere image selected: {os.path.basename(file_path)}")
    
    def select_output_directory(self):
        """Open a dialog to select the output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.config['output_directory'] = directory
            self.output_dir_label.setText(directory)
    
    def start_processing(self):
        """Begin photometric stereo processing"""
        # Check if we have all necessary inputs
        if not self.config['input_images']:
            QMessageBox.warning(self, "Missing Input", "No input images selected")
            return
        
        if not self.config['light_directions'] and not self.config['chrome_sphere_image']:
            QMessageBox.warning(self, "Missing Light Directions", 
                               "Please select a chrome sphere image or load light directions")
            return
        
        # Update UI
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)
        self.progress_label.setText("Processing...")
        
        # Create worker thread
        self.processing_thread = ProcessingThread(self.config)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        
        # Start processing
        self.processing_thread.start()
    
    @pyqtSlot(int)
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    @pyqtSlot(dict)
    def processing_finished(self, results):
        """Handle completed processing"""
        # Update UI
        self.process_button.setEnabled(True)
        self.progress_label.setText("Processing completed")
        
        # Store results
        if 'normal_map' in results:
            self.config['normal_map'] = results['normal_map']
        
        if 'depth_map' in results:
            self.config['depth_map'] = results['depth_map']
        
        if 'albedo' in results:
            self.config['albedo'] = results['albedo']
        
        # Enable result buttons
        self.normal_map_button.setEnabled('normal_map' in results)
        self.depth_map_button.setEnabled('depth_map' in results)
        self.albedo_button.setEnabled('albedo' in results)
        
        # Show message
        self.status_bar.showMessage("Processing completed successfully")
    
    @pyqtSlot(str)
    def processing_error(self, error_msg):
        """Handle processing errors"""
        self.process_button.setEnabled(True)
        self.progress_label.setText("Error during processing")
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.status_bar.showMessage("Error during processing")
    
    def show_result(self, result_type):
        """Display the selected result type"""
        if result_type not in self.config or self.config[result_type] is None:
            self.result_label.setText(f"No {result_type.replace('_', ' ')} available")
            return
        
        # Convert the numpy array to a QImage and display it
        # This is a placeholder for the actual display logic
        self.result_label.setText(f"Displaying {result_type.replace('_', ' ')}")
    
    def export_results(self):
        """Export results to the selected output directory"""
        if not self.config['output_directory']:
            QMessageBox.warning(self, "No Output Directory", 
                               "Please select an output directory first")
            return
        
        # Check if we have results to export
        has_results = any(self.config.get(key) is not None for key in 
                          ['normal_map', 'depth_map', 'albedo'])
        
        if not has_results:
            QMessageBox.warning(self, "No Results", 
                               "No results available to export")
            return
        
        # Perform export (placeholder)
        self.status_bar.showMessage("Exporting results...")
        
        # In a real app, you would call your export functions here
        
        QMessageBox.information(self, "Export Complete", 
                               "Results exported successfully")
        self.status_bar.showMessage("Export completed")
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())