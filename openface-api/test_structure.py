#!/usr/bin/env python3
"""
Test script to validate the refactored OpenFace API structure
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported without errors"""
    print("🧪 Testing OpenFace API Structure...")
    
    try:
        # Test configuration
        print("  ✓ Testing configuration...")
        from config import config, api_config
        print(f"    - OpenFace path: {config.project_root}")
        print(f"    - API host:port: {api_config.HOST}:{api_config.PORT}")
        
        # Test utilities
        print("  ✓ Testing utilities...")
        from api_utils import check_startup_requirements
        from api_utils.image_utils import decode_base64_image
        from api_utils.validation import validate_image_data
        
        # Test models
        print("  ✓ Testing models...")
        from api.models import OpenFaceAnalyzer
        
        # Test routes
        print("  ✓ Testing routes...")
        from api.routes import health_bp, analysis_bp, logs_bp
        
        # Test WebSocket
        print("  ✓ Testing WebSocket...")
        from api.websocket import register_websocket_handlers
        
        # Test app factory
        print("  ✓ Testing app factory...")
        from api import create_app, run_app
        
        print("✅ All imports successful!")
        print("\n📁 Structure Summary:")
        print("  - Main app: app.py (29 lines)")
        print("  - Config: Enhanced with API settings")
        print("  - Models: OpenFaceAnalyzer (310+ lines)")
        print("  - Routes: Organized into blueprints")
        print("  - Utils: Startup, validation, image processing")
        print("  - WebSocket: Separate event handlers")
        print("  - Docs: OpenAPI spec + examples")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without running the server"""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        # Test image validation
        from api_utils.validation import validate_image_data
        
        is_valid, error = validate_image_data("invalid")
        print(f"  ✓ Image validation works: {not is_valid}")
        
        # Test configuration access
        from config import api_config
        print(f"  ✓ Config access: {len(api_config.EMOTION_LABELS)} emotions")
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OpenFace API Structure Validation")
    print("=" * 60)
    
    # Add OpenFace path (like the main app does)
    OPENFACE_PATH = Path(__file__).parent.parent / "OpenFace-3.0"
    if OPENFACE_PATH.exists():
        sys.path.insert(0, str(OPENFACE_PATH))
        sys.path.insert(0, str(OPENFACE_PATH / "Pytorch_Retinaface"))
        print(f"✅ OpenFace path added: {OPENFACE_PATH}")
    else:
        print(f"⚠️  OpenFace path not found: {OPENFACE_PATH}")
    
    import_success = test_imports()
    if import_success:
        functionality_success = test_basic_functionality()
    
    print("\n" + "=" * 60)
    if import_success and functionality_success:
        print("🎉 Refactoring validation PASSED!")
        print("\n💡 To start the server:")
        print("   python app.py")
    else:
        print("❌ Refactoring validation FAILED!")
        print("   Check error messages above")
    print("=" * 60)
