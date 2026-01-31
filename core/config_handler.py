"""Configuration file management for server configs and app settings."""

import os
import json
from .utils import get_config_path


class ConfigHandler:
    """Handles loading and saving configuration files."""
    
    @staticmethod
    def save_server_config(config_dict, filepath=None, auto_name=False):
        """Save a server configuration to a JSON file.
        
        Args:
            config_dict: Dictionary containing server configuration
            filepath: Optional path to save to. If None, will prompt user
            auto_name: If True, generate automatic filename in configs dir
            
        Returns:
            Path where config was saved, or None if cancelled
        """
        if filepath is None and not auto_name:
            # UI layer should handle file dialog
            return None
            
        if auto_name:
            app_dir = os.path.dirname(get_config_path(''))
            configs_dir = os.path.join(app_dir, 'configs')
            os.makedirs(configs_dir, exist_ok=True)
            # Generate unique filename
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(configs_dir, f'config_{timestamp}.json')
        
        # Ensure .json extension
        if not filepath.lower().endswith('.json'):
            filepath = filepath + '.json'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4)
        
        return filepath
    
    @staticmethod
    def load_server_config(filepath):
        """Load a server configuration from a JSON file.
        
        Args:
            filepath: Path to the config file
            
        Returns:
            Dictionary containing configuration, or None if file doesn't exist
            
        Raises:
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_app_settings(settings_dict):
        """Save global application settings.
        
        Args:
            settings_dict: Dictionary containing app settings
        """
        path = get_config_path('app_settings.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4)
    
    @staticmethod
    def load_app_settings():
        """Load global application settings.
        
        Returns:
            Dictionary containing app settings, or empty dict if not found
        """
        path = get_config_path('app_settings.json')
        if not os.path.exists(path):
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}