#!/usr/bin/env python3
"""
Startup validation utilities for OpenFace API
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def check_startup_requirements():
    """Check startup requirements and log status"""
    print("\n" + "="*60)
    print("🔍 OpenFace API Startup Validation")
    print("="*60)
    
    from config import config
    
    # Check OpenFace directory
    if config.project_root.exists():
        print(f"✅ OpenFace-3.0 directory: {config.project_root}")
    else:
        print(f"❌ OpenFace-3.0 directory missing: {config.project_root}")
    
    # Check weights directory
    if config.weights_dir.exists():
        print(f"✅ Weights directory: {config.weights_dir}")
        
        # Check model files
        models = {
            "RetinaFace": config.weights_dir / "mobilenet0.25_Final.pth",
            "MLT Backbone": config.weights_dir / "MTL_backbone.pth",
            "STAR Landmark": config.weights_dir / "Landmark_68.pkl"
        }
        
        for name, path in models.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✅ {name}: {path.name} ({size_mb:.1f} MB)")
            else:
                print(f"⚠️  {name}: {path.name} (missing)")
    else:
        print(f"❌ Weights directory missing: {config.weights_dir}")
    
    # Check logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    if not logs_dir.exists():
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created logs directory: {logs_dir}")
        except Exception as e:
            print(f"❌ Cannot create logs directory: {e}")
    else:
        print(f"✅ Logs directory exists: {logs_dir}")
    
    # Test write permissions
    try:
        test_file = logs_dir / "startup_test.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        test_file.unlink()
        print(f"✅ Logs directory is writable")
    except Exception as e:
        print(f"❌ Logs directory not writable: {e}")
    
    # Check CUDA if torch is available
    if torch:
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                print(f"✅ CUDA available: {torch.version.cuda}")
                print(f"   GPU devices: {device_count}")
                print(f"   Current device: {current_device} ({device_name})")
                print(f"   Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB")
            else:
                print(f"⚠️  CUDA not available - using CPU")
                # Check if NVIDIA GPU exists but CUDA is not available
                try:
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   Note: NVIDIA GPU detected but CUDA not available")
                        print(f"   This might be due to Docker configuration or driver issues")
                except:
                    print(f"   No NVIDIA GPU detected or nvidia-smi not available")
        except Exception as e:
            print(f"⚠️  Could not check CUDA status: {e}")
    else:
        print(f"⚠️  PyTorch not available - models will not load")
    
    print("="*60)
    print("🚀 Starting OpenFace API...")
    print("="*60 + "\n")


def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True
