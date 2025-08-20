#!/usr/bin/env python3
"""
Centralized path management for OpenFace API
Handles all sys.path modifications and OpenFace-3.0 discovery in one place
"""

import os
import sys
from pathlib import Path
from typing import Optional


class PathManager:
    """Centralized path management for OpenFace API"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.api_dir = Path(__file__).parent.parent.absolute()
            self.openface_path = self._discover_openface_path()
            self.original_sys_path = sys.path.copy()
            self._setup_paths()
            PathManager._initialized = True
    
    def _discover_openface_path(self) -> Optional[Path]:
        """Discover OpenFace-3.0 path using multiple strategies"""
        # Strategy 1: Parent directory of API
        openface_path = self.api_dir.parent / "OpenFace-3.0"
        if self._validate_openface_path(openface_path):
            return openface_path
        
        # Strategy 2: Working directory parent
        openface_path = Path(os.getcwd()).parent / "OpenFace-3.0"
        if self._validate_openface_path(openface_path):
            return openface_path
        
        # Strategy 3: Working directory
        openface_path = Path(os.getcwd()) / "OpenFace-3.0"
        if self._validate_openface_path(openface_path):
            return openface_path
        
        print("❌ OpenFace-3.0 directory not found in any expected location")
        return None
    
    def _validate_openface_path(self, path: Path) -> bool:
        """Validate that path contains required OpenFace components"""
        if not path.exists():
            return False
        
        required_dirs = ["model", "Pytorch_Retinaface", "weights"]
        return all((path / dir_name).exists() for dir_name in required_dirs)
    
    def _setup_paths(self):
        """Setup Python paths for imports"""
        if self.openface_path is None:
            print("⚠️  Cannot setup paths - OpenFace-3.0 not found")
            return
        
        paths_to_add = [
            str(self.openface_path),
            str(self.openface_path / "Pytorch_Retinaface"),
            str(self.openface_path / "model"),
            str(self.api_dir)  # API directory
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        print(f"✅ Paths configured for OpenFace-3.0 at: {self.openface_path}")
    
    @property
    def model_dir(self) -> Optional[Path]:
        """Get model directory path"""
        return self.openface_path / "model" if self.openface_path else None
    
    @property
    def retinaface_dir(self) -> Optional[Path]:
        """Get RetinaFace directory path"""
        return self.openface_path / "Pytorch_Retinaface" if self.openface_path else None
    
    @property
    def weights_dir(self) -> Optional[Path]:
        """Get weights directory path"""
        return self.openface_path / "weights" if self.openface_path else None
    
    def cleanup(self):
        """Restore original sys.path"""
        sys.path = self.original_sys_path.copy()


# Global instance
path_manager = PathManager()
