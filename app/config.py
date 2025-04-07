"""
Configuration management for the Photometric Stereo application.
Python implementation of set_parameters.m and related configuration functionality.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for photometric stereo processing"""
    
    DEFAULT_CONFIG = {
        # Input/Output settings
        'working_directory': '',
        'output_directory': '',
        'input_images': [],
        'chrome_sphere_image': None,
        
        # Light settings
        'light_directions': None,
        'light_strengths': None,
        'all_lights_on_image': None,
        
        # Processing options
        'normal_map_method': 'standard',  # 'standard' or 'nonlinear'
        'depth_map_method': 'gradient',   # 'gradient' or 'poisson'
        'use_poly_correction': True,
        'poly_correction_order': 2,
        
        # Export options
        'export_normal_map': True,
        'export_depth_map': True,
        'export_albedo': True,
        'export_3d_model': True,
        'export_3d_format': 'OBJ',  # 'OBJ', 'STL', or 'PLY'
        'icc_profile': 'None',  # 'None', 'sRGB', or 'Display P3'
        'bit_depth': 8,  # 8 or 16
        
        # Advanced parameters
        'light_refinement': False,
        'light_refinement_iterations': 10,
        'light_refinement_tolerance': 1e-4,
        'robust_estimation': False,
        'mesh_decimation': False,
        'mesh_smoothing': False,
        'mesh_resolution': 1.0,  # Factor relative to image size
        
        # Results
        'normal_map': None,
        'depth_map': None,
        'albedo': None
    }
    
    def __init__(self):
        """Initialize configuration with default values"""
        self._config = self.DEFAULT_CONFIG.copy()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value for the given key"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value for the given key"""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding large data)"""
        # Create a copy to avoid modifying original
        config_dict = self._config.copy()
        
        # Remove large data arrays that shouldn't be serialized
        for key in ['normal_map', 'depth_map', 'albedo']:
            if key in config_dict:
                config_dict[key] = None
        
        return config_dict
    
    def save(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        # Get serializable config (without numpy arrays)
        config_dict = self.to_dict()
        
        # Convert any non-serializable values
        for key, value in config_dict.items():
            if key == 'input_images' and value:
                # Store only relative paths if possible
                base_dir = config_dict.get('working_directory', '')
                if base_dir:
                    config_dict[key] = [os.path.relpath(img, base_dir) if os.path.isabs(img) else img 
                                        for img in value]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def load(self, file_path: str) -> None:
        """Load configuration from JSON file"""
        if not os.path.exists(file_path):
            logger.error(f"Configuration file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Handle relative paths
            if 'working_directory' in loaded_config and 'input_images' in loaded_config:
                base_dir = loaded_config['working_directory']
                if base_dir and loaded_config['input_images']:
                    loaded_config['input_images'] = [
                        os.path.join(base_dir, img) if not os.path.isabs(img) else img
                        for img in loaded_config['input_images']
                    ]
            
            # Update configuration
            self.update(loaded_config)
            logger.info(f"Configuration loaded from {file_path}")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading configuration: {e}")


def set_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set parameters for photometric stereo processing.
    Python implementation of set_parameters.m
    
    Parameters
    ----------
    config : Dict
        Input configuration dictionary
        
    Returns
    -------
    Dict
        Updated configuration dictionary with default parameters
    """
    # Create a copy to avoid modifying the original
    params = config.copy()
    
    # Set default parameters if not specified
    if 'normal_map_method' not in params:
        params['normal_map_method'] = 'standard'
    
    if 'depth_map_method' not in params:
        params['depth_map_method'] = 'gradient'
    
    if 'use_poly_correction' not in params:
        params['use_poly_correction'] = True
    
    if 'poly_correction_order' not in params:
        params['poly_correction_order'] = 2
    
    if 'light_refinement' not in params:
        params['light_refinement'] = False
    
    if 'light_refinement_iterations' not in params:
        params['light_refinement_iterations'] = 10
        
    if 'light_refinement_tolerance' not in params:
        params['light_refinement_tolerance'] = 1e-4
    
    if 'robust_estimation' not in params:
        params['robust_estimation'] = False
    
    if 'mesh_decimation' not in params:
        params['mesh_decimation'] = False
    
    if 'mesh_smoothing' not in params:
        params['mesh_smoothing'] = False
    
    if 'mesh_resolution' not in params:
        params['mesh_resolution'] = 1.0
    
    # Advanced numerical parameters
    if 'solver_options' not in params:
        params['solver_options'] = {
            'max_iterations': 100,
            'tolerance': 1e-6,
            'step_size': 0.1,
            'verbose': False
        }
    
    return params


def get_default_light_positions() -> List[List[float]]:
    """
    Get default light positions for standard photometric stereo setup.
    
    Returns
    -------
    List[List[float]]
        List of [x, y, z] light positions
    """
    # Default light positions are evenly distributed around the hemisphere
    return [
        [0, 0, 1],       # Top
        [1, 0, 1],       # Right
        [0, 1, 1],       # Front
        [-1, 0, 1],      # Left
        [0, -1, 1],      # Back
        [0.7, 0.7, 1],   # Front-Right
        [-0.7, 0.7, 1],  # Front-Left
        [-0.7, -0.7, 1], # Back-Left
        [0.7, -0.7, 1]   # Back-Right
    ]


def load_light_positions(file_path: str) -> Optional[List[List[float]]]:
    """
    Load light positions from a configuration file.
    
    Parameters
    ----------
    file_path : str
        Path to light positions file
        
    Returns
    -------
    Optional[List[List[float]]]
        List of [x, y, z] light positions, or None if file not found
    """
    if not os.path.exists(file_path):
        logger.error(f"Light positions file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                # JSON format
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'light_positions' in data:
                    return data['light_positions']
            else:
                # Simple text format (one position per line)
                lines = f.readlines()
                positions = []
                
                for line in lines:
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse position
                    try:
                        components = [float(x) for x in line.split()]
                        if len(components) >= 3:
                            positions.append(components[:3])
                    except ValueError:
                        logger.warning(f"Invalid line in light positions file: {line}")
                
                return positions if positions else None
                
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading light positions: {e}")
        return None


# Global configuration instance
config_instance = Config()