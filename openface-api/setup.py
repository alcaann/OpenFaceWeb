#!/usr/bin/env python3
"""
Setup script for OpenFace API
Checks dependencies and helps with initial setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is supported"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing Python requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_openface_project():
    """Check if OpenFace-3.0 project exists"""
    api_dir = Path(__file__).parent
    openface_path = api_dir.parent / "OpenFace-3.0"
    
    if openface_path.exists() and (openface_path / "model").exists():
        print(f"✅ OpenFace-3.0 found at: {openface_path}")
        return True
    else:
        print(f"❌ OpenFace-3.0 not found at: {openface_path}")
        print("Please ensure the OpenFace-3.0 project is in the parent directory")
        print("Or modify the OPENFACE_PATH in app.py")
        return False

def check_model_weights():
    """Check if model weights exist"""
    api_dir = Path(__file__).parent
    weights_dir = api_dir.parent / "OpenFace-3.0" / "weights"
    
    required_weights = [
        "mobilenet0.25_Final.pth",
        "MTL_backbone.pth"
    ]
    
    missing_weights = []
    for weight_file in required_weights:
        weight_path = weights_dir / weight_file
        if weight_path.exists():
            print(f"✅ Found: {weight_file}")
        else:
            print(f"❌ Missing: {weight_file}")
            missing_weights.append(weight_file)
    
    if missing_weights:
        print("\n📥 Download missing weights from:")
        print("https://drive.google.com/drive/folders/1aBEol-zG_blHSavKFVBH9dzc9U9eJ92p")
        print(f"Place them in: {weights_dir}")
        return False
    
    return True

def create_run_script():
    """Create a run script for the API"""
    run_script = Path(__file__).parent / "run_api.py"
    
    script_content = '''#!/usr/bin/env python3
"""
OpenFace API Runner
Simple script to start the OpenFace API server
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app import app, socketio, analyzer
    
    if __name__ == '__main__':
        print("🚀 Starting OpenFace-3.0 API Server")
        print("📍 Access at: http://localhost:5000")
        print("🧪 Test client: test_client.html")
        print("=" * 50)
        
        # Start the server
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    use_reloader=False)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run: python setup.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
'''
    
    with open(run_script, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(run_script, 0o755)
    
    print(f"✅ Created run script: {run_script}")

def main():
    """Main setup function"""
    print("🔧 OpenFace API Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check OpenFace project
    if not check_openface_project():
        print("\n⚠️  Setup can continue, but the API may not work without OpenFace-3.0")
    
    # Check model weights
    if not check_model_weights():
        print("\n⚠️  Setup can continue, but facial analysis will be limited without model weights")
    
    # Create run script
    create_run_script()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Ensure OpenFace-3.0 is available in the parent directory")
    print("2. Download model weights if missing")
    print("3. Run the API: python run_api.py")
    print("4. Open test_client.html in a web browser")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
