#!/usr/bin/env python3
"""
OpenFace-3.0 Flask API - Main Entry Point
Real-time facial analysis API that receives frames via WebSocket and returns analysis data
"""

import sys
from pathlib import Path

# Add OpenFace-3.0 to path
OPENFACE_PATH = Path(__file__).parent.parent / "OpenFace-3.0"
if not OPENFACE_PATH.exists():
    print(f"❌ OpenFace-3.0 not found at: {OPENFACE_PATH}")
    print("Please ensure OpenFace-3.0 is in the parent directory or modify OPENFACE_PATH")
    sys.exit(1)

sys.path.insert(0, str(OPENFACE_PATH))
sys.path.insert(0, str(OPENFACE_PATH / "Pytorch_Retinaface"))

# Import and run the application
if __name__ == '__main__':
    try:
        from api import run_app
        run_app()
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)
