#!/usr/bin/env python3
"""
Configuration for OpenFace API
Handles path resolution, model loading, and API settings
"""

import os
import sys
from pathlib import Path
from contextlib import contextmanager

# Import centralized path manager
from utils.path_manager import path_manager

@contextmanager
def project_directory_context(project_path):
    """Context manager for temporarily changing to project directory"""
    original_cwd = os.getcwd()
    try:
        os.chdir(str(project_path))
        yield project_path
    finally:
        os.chdir(original_cwd)

class OpenFaceAPIConfig:
    """Configuration class for OpenFace API"""
    
    def __init__(self, openface_project_path=None):
        """
        Initialize configuration - now uses centralized path manager
        
        Args:
            openface_project_path: Path to OpenFace-3.0 project directory
                                 If None, uses path manager discovery
        """
        # Use centralized path manager
        self.api_dir = path_manager.api_dir
        self.project_root = path_manager.openface_path
        
        # Setup paths using path manager
        self.model_dir = path_manager.model_dir
        self.retinaface_dir = path_manager.retinaface_dir
        self.weights_dir = path_manager.weights_dir
        
        # Validate setup
        if self.project_root is None:
            print("❌ OpenFace-3.0 not found - some features will be unavailable")
        else:
            self._validate_setup()
    
    def _validate_setup(self):
        """Validate OpenFace setup using path manager"""
        if not self.weights_dir or not self.weights_dir.exists():
            print(f"⚠️  Weights directory not found: {self.weights_dir}")
            print("Create it and download the model weights:")
            print("https://drive.google.com/drive/folders/1aBEol-zG_blHSavKFVBH9dzc9U9eJ92p")
    
    def get_model_paths(self):
        """Get paths to model files"""
        if not self.weights_dir:
            return {}
        return {
            'retinaface': self.weights_dir / "mobilenet0.25_Final.pth",
            'mlt': self.weights_dir / "MTL_backbone.pth",
            'star': self.weights_dir / "Landmark_68.pkl"
        }
    
    def check_dependencies(self):
        """Check if all dependencies are available"""
        issues = []
        
        # Check project structure
        required_dirs = [
            self.model_dir,
            self.retinaface_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_path}")
        
        # Check model files
        model_paths = self.get_model_paths()
        for model_name, model_path in model_paths.items():
            if not model_path.exists():
                issues.append(f"Missing {model_name} weights: {model_path}")
        
        # Check Python dependencies
        try:
            import torch
            import cv2
            import flask
            import flask_socketio
        except ImportError as e:
            issues.append(f"Missing Python dependency: {e}")
        
        return issues
    
    def print_status(self):
        """Print configuration status"""
        print("🔧 OpenFace API Configuration")
        print("=" * 40)
        print(f"API Directory: {self.api_dir}")
        print(f"OpenFace-3.0: {self.project_root}")
        print(f"Weights: {self.weights_dir}")
        print()
        
        # Check for issues
        issues = self.check_dependencies()
        if issues:
            print("❌ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ All dependencies available")
        
        print()

# API Configuration Constants
class APIConfig:
    """API-specific configuration constants"""
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', 5000))
    DEBUG = True  # Force debug mode to see all requests
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
    NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', 0.4))
    VIS_THRESHOLD = float(os.getenv('VIS_THRESHOLD', 0.5))
    
    # Model labels
    EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    AU_LABELS = ['AU1', 'AU2', 'AU4', 'AU6', 'AU9', 'AU12', 'AU25', 'AU26']  # DISFA common AUs

# Default configuration instance
config = OpenFaceAPIConfig()
api_config = APIConfig()

if __name__ == "__main__":
    config.print_status()
