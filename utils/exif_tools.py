"""
EXIF metadata utilities.
Python implementation of:
- AddExif.m
- checkIfExifIsInstalled.m
"""

import os
import subprocess
import platform
import shutil


def check_if_exif_is_installed():
    """
    Check if exiftool is installed.
    Python implementation of checkIfExifIsInstalled.m
    
    Returns
    -------
    bool
        True if exiftool is installed
    """
    system = platform.system()
    
    if system == "Windows":
        # Check if exiftool.exe is in PATH
        exiftool_path = shutil.which("exiftool.exe")
        if exiftool_path:
            return True
        
        # Check if it's in the current directory
        if os.path.exists("exiftool.exe"):
            return True
        
        return False
    
    elif system == "Darwin":  # macOS
        # Check if exiftool is in PATH
        exiftool_path = shutil.which("exiftool")
        if exiftool_path:
            return True
        
        # Check common installation locations
        for path in ["/usr/local/bin/exiftool", "/opt/local/bin/exiftool", "/usr/bin/exiftool"]:
            if os.path.exists(path):
                return True
        
        return False
    
    else:  # Linux or other
        # Check if exiftool is in PATH
        exiftool_path = shutil.which("exiftool")
        if exiftool_path:
            return True
        
        # Check common installation locations
        for path in ["/usr/bin/exiftool", "/usr/local/bin/exiftool"]:
            if os.path.exists(path):
                return True
        
        return False


def add_exif(image_in, image_out, color_space):
    """
    Add EXIF metadata to image.
    Python implementation of AddExif.m
    
    Parameters
    ----------
    image_in : str
        Input image path
    image_out : str
        Output image path
    color_space : str
        Color space to set ('DISPLAY-P3', 'sRGB', or 'none')
        
    Returns
    -------
    bool
        True if successful
    """
    # Check if exiftool is installed
    if not check_if_exif_is_installed():
        print("Warning: exiftool is not installed. Cannot add EXIF metadata.")
        return False
    
    # Create temporary copy to preserve original
    temp_file = image_out + ".temp"
    shutil.copy2(image_in, temp_file)
    
    # Build exiftool command
    cmd = ["exiftool"]
    
    if color_space.upper() == "DISPLAY-P3":
        # Get path to Display P3 ICC profile
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icc_profile = os.path.join(script_dir, "profiles", "AppleDisplayP3.icc")
        
        if not os.path.exists(icc_profile):
            print(f"Warning: Display P3 ICC profile not found at {icc_profile}")
            
            # Try alternate locations
            for alt_name in ["DCI-P3.icc", "P3D65.icc"]:
                alt_path = os.path.join(script_dir, "profiles", alt_name)
                if os.path.exists(alt_path):
                    icc_profile = alt_path
                    break
        
        if os.path.exists(icc_profile):
            cmd.extend(["-icc_profile<=", icc_profile])
            cmd.extend(["-ColorSpace=Display P3"])
        else:
            print("Warning: No Display P3 ICC profile found. Using sRGB instead.")
            cmd.extend(["-ColorSpace=sRGB"])
    
    elif color_space.upper() == "SRGB":
        # Get path to sRGB ICC profile
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icc_profile = os.path.join(script_dir, "profiles", "sRGB_v4_ICC_preference.icc")
        
        if os.path.exists(icc_profile):
            cmd.extend(["-icc_profile<=", icc_profile])
        
        cmd.extend(["-ColorSpace=sRGB"])
    
    else:  # 'none' or any other value
        cmd.extend(["-ColorSpace="])
    
    # Set output file
    cmd.extend([temp_file])
    
    try:
        # Run exiftool
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Move temporary file to output file
        os.replace(temp_file, image_out)
        
        # Clean up backup file created by exiftool
        backup_file = temp_file + "_original"
        if os.path.exists(backup_file):
            os.remove(backup_file)
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running exiftool: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return False
    
    except Exception as e:
        print(f"Error adding EXIF metadata: {str(e)}")
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return False