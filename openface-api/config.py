#!/usr/bin/env python3
"""
Configuration for OpenFace API
Handles path resolution, model loading, and API settings
"""

import os
import sys
from pathlib import Path
from contextlib import contextmanager

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
        Initialize configuration
        
        Args:
            openface_project_path: Path to OpenFace-3.0 project directory
                                 If None, attempts to auto-detect
        """
        self.api_dir = Path(__file__).parent.absolute()
        self.project_root = self._find_openface_project(openface_project_path)
        
        # Setup paths
        self.model_dir = self.project_root / "model"
        self.retinaface_dir = self.project_root / "Pytorch_Retinaface"
        self.star_dir = self.project_root / "STAR"
        self.weights_dir = self._setup_weights_dir()
        
        # Add to Python path
        self._setup_python_paths()
    
    def _find_openface_project(self, provided_path):
        """Find OpenFace-3.0 project directory"""
        if provided_path:
            project_path = Path(provided_path)
            if project_path.exists() and (project_path / "model").exists():
                return project_path.absolute()
        
        # Check common locations
        search_paths = [
            self.api_dir.parent / "OpenFace-3.0",  # Sibling directory
            self.api_dir / "OpenFace-3.0",          # Inside API directory
            Path.cwd() / "OpenFace-3.0",            # Current working directory
            Path.home() / "OpenFace-3.0",           # Home directory
        ]
        
        for path in search_paths:
            if path.exists() and (path / "model").exists():
                print(f"✅ Found OpenFace-3.0 at: {path}")
                return path.absolute()
        
        # If not found, use the expected path
        expected_path = self.api_dir.parent / "OpenFace-3.0"
        print(f"⚠️  OpenFace-3.0 not found, using expected path: {expected_path}")
        return expected_path.absolute()
    
    def _setup_weights_dir(self):
        """Setup weights directory"""
        weights_dir = self.project_root / "weights"
        if not weights_dir.exists():
            print(f"⚠️  Weights directory not found: {weights_dir}")
            print("Create it and download the model weights:")
            print("https://drive.google.com/drive/folders/1aBEol-zG_blHSavKFVBH9dzc9U9eJ92p")
        return weights_dir
    
    def _setup_python_paths(self):
        """Add necessary paths to Python path"""
        paths_to_add = [
            str(self.project_root),
            str(self.retinaface_dir),
            str(self.model_dir),
            str(self.star_dir)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def get_model_paths(self):
        """Get paths to model files"""
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
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    
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
