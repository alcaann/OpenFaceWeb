#!/usr/bin/env python3
"""
Wrapper for RetinaFace model to handle imports
Refactored for production use after fixing import issues
"""

import sys
import os
import importlib.util
from pathlib import Path

from utils.path_manager import path_manager

def load_retinaface_components():
    """Load RetinaFace components with proper module resolution"""
    try:
        if not path_manager.openface_path:
            print("❌ OpenFace path not found by PathManager")
            return None, None, None, None, None, None

        retinaface_path = path_manager.retinaface_dir
        
        if not retinaface_path or not retinaface_path.exists():
            return None, None, None, None, None, None
            
        # Save original state for restoration
        original_cwd = os.getcwd()
        original_syspath = sys.path.copy()
        
        try:
            # Setup environment for imports
            os.chdir(str(retinaface_path))
            for subdir in ['', 'models', 'layers', 'utils']:
                path = str(retinaface_path / subdir) if subdir else str(retinaface_path)
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # Load modules in dependency order to avoid circular import issues
            # 1. Load utils first (no dependencies)
            box_utils_module = _load_module('box_utils', retinaface_path / 'utils' / 'box_utils.py')
            sys.modules['utils.box_utils'] = box_utils_module
            
            nms_module = _load_module('py_cpu_nms', retinaface_path / 'utils' / 'nms' / 'py_cpu_nms.py')
            sys.modules['utils.nms.py_cpu_nms'] = nms_module
            
            # 2. Load net module (needed by retinaface)
            net_module = _load_module('net', retinaface_path / 'models' / 'net.py')
            sys.modules['models.net'] = net_module
            
            # 3. Load config
            config_module = _load_module('config', retinaface_path / 'data' / 'config.py')
            sys.modules['data.config'] = config_module
            
            # 4. Load retinaface (depends on net)
            retinaface_module = _load_module('retinaface', retinaface_path / 'models' / 'retinaface.py')
            sys.modules['models.retinaface'] = retinaface_module
            
            # 5. Load layers (depends on utils)
            prior_box_module = _load_module('prior_box', retinaface_path / 'layers' / 'functions' / 'prior_box.py')
            sys.modules['layers.functions.prior_box'] = prior_box_module
            
            # Extract the classes/functions we need
            RetinaFace = retinaface_module.RetinaFace
            cfg_mnet = config_module.cfg_mnet
            PriorBox = prior_box_module.PriorBox
            decode = box_utils_module.decode
            decode_landm = box_utils_module.decode_landm
            py_cpu_nms = nms_module.py_cpu_nms
            
            return RetinaFace, cfg_mnet, PriorBox, decode, decode_landm, py_cpu_nms
            
        finally:
            # Always restore original state
            os.chdir(original_cwd)
            sys.path = original_syspath
            
    except Exception as e:
        # Simple error logging without debug spam
        print(f"❌ Failed to load RetinaFace components: {e}")
        return None, None, None, None, None, None

def _load_module(module_name, file_path):
    """Helper function to load a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
