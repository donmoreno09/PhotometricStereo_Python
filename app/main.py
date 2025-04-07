"""
Main entry point for the Photometric Stereo application
"""

import sys
import os

# Set up path before imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDir

from main_window import MainWindow


def setup_environment():
    """Configure environment variables and paths"""
    # Configure default paths
    app_data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(app_data_dir):
        os.makedirs(app_data_dir)


def main():
    """Main entry point for the application"""
    setup_environment()
    
    app = QApplication(sys.argv)
    app.setApplicationName("Photometric Stereo")
    app.setOrganizationName("PhotometricStereo")
    
    # Set style sheet if needed
    # with open('ui/resources/styles/main_style.qss', 'r') as f:
    #     app.setStyleSheet(f.read())
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()