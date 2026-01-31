"""Utility functions for the LLaMA Server GUI."""

import sys
import os
from tkinter import filedialog

def get_config_path(filename):
    """Get the path for config file that works with PyInstaller.
    
    Args:
        filename: Name of the config file
        
    Returns:
        Full path to the config file
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.getcwd()
    return os.path.join(app_dir, filename)


def resource_path(filename):
    """Get absolute path to resource, works for dev and for PyInstaller bundle.
    
    Args:
        filename: Name of the resource file
        
    Returns:
        Full path to the resource file
    """
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

def browse_file(string_var, file_ext):
    """Browse for a file and set the selected path to the provided StringVar."""
    filename = filedialog.askopenfilename(
        title=f"Select {file_ext} File",
        filetypes=[(f"{file_ext.upper()} files", f"*{file_ext}"), ("All files", "*.*")]
    )
    if filename:
        string_var.set(filename)