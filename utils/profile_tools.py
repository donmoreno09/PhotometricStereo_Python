"""
Color profile utilities.
Python implementation of:
- addDisplayP3Profile.m
"""

import os
import shutil
import tempfile
import subprocess
import platform
from PIL import Image
import piexif


def add_display_p3_profile(image_containing_the_profile_to_copy, full_path_out):
    """
    Add Display P3 color profile to an image.
    Python implementation of addDisplayP3Profile.m
    
    Parameters
    ----------
    image_containing_the_profile_to_copy : str
        Path to image containing the profile to copy
    full_path_out : str
        Output image path
        
    Returns
    -------
    bool
        True if successful
    """
    is_profiled = False
    
    try:
        # Try to open the image and extract its profile
        with Image.open(image_containing_the_profile_to_copy) as img:
            # Check if the image has an ICC profile
            if 'icc_profile' in img.info:
                is_profiled = True
                icc_profile = img.info['icc_profile']
                
                # Try to determine the profile type
                profile_type = identify_profile_type(image_containing_the_profile_to_copy)
                
                # Apply the appropriate profile
                if profile_type == "DISPLAY-P3":
                    copy_and_add_profile(image_containing_the_profile_to_copy, full_path_out, "DISPLAY-P3")
                elif profile_type == "sRGB":
                    copy_and_add_profile(image_containing_the_profile_to_copy, full_path_out, "sRGB")
                else:
                    copy_and_add_profile(image_containing_the_profile_to_copy, full_path_out, "none")
            else:
                print(f"Warning: no ICC profile found in: {image_containing_the_profile_to_copy}")
                copy_and_add_profile(image_containing_the_profile_to_copy, full_path_out, "none")
    
    except Exception as e:
        print(f"Error processing ICC profile: {str(e)}")
        copy_and_add_profile(image_containing_the_profile_to_copy, full_path_out, "none")
    
    return True


def identify_profile_type(image_path):
    """
    Identify the type of ICC profile in an image.
    
    Parameters
    ----------
    image_path : str
        Path to the image
        
    Returns
    -------
    str
        Profile type: "DISPLAY-P3", "sRGB", or "none"
    """
    try:
        # Check if exiftool is available
        if shutil.which("exiftool"):
            # Use exiftool to get profile information
            cmd = ["exiftool", "-ProfileDescription", "-s3", image_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                profile_desc = result.stdout.strip().lower()
                
                if "display p3" in profile_desc or "p3" in profile_desc:
                    return "DISPLAY-P3"
                elif "srgb" in profile_desc:
                    return "sRGB"
            
            # Try to check color space
            cmd = ["exiftool", "-ColorSpace", "-s3", image_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                color_space = result.stdout.strip().lower()
                
                if "display p3" in color_space or "p3" in color_space:
                    return "DISPLAY-P3"
                elif "srgb" in color_space:
                    return "sRGB"
        
        # If we reach here, try PIL
        with Image.open(image_path) as img:
            if 'icc_profile' in img.info:
                # We found a profile, but couldn't identify it specifically
                # Return sRGB as a safer default
                return "sRGB"
    
    except Exception as e:
        print(f"Error identifying profile: {str(e)}")
    
    return "none"


def copy_and_add_profile(image_in, image_out, color_space):
    """
    Copy an image and add the specified color profile.
    
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
    # Import here to avoid circular imports
    from utils.exif_tools import add_exif
    
    # Copy image and add EXIF metadata
    return add_exif(image_in, image_out, color_space)