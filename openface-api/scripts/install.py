#!/usr/bin/env python3
"""
Alternative installation script for OpenFace API
Handles Python 3.12 compatibility issues
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

def upgrade_pip():
    """Upgrade pip to latest version"""
    try:
        print("🔄 Upgrading pip...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pip upgraded successfully")
            return True
        else:
            print(f"⚠️ Pip upgrade failed, continuing anyway: {result.stderr}")
            return True  # Continue even if pip upgrade fails
    except Exception as e:
        print(f"⚠️ Error upgrading pip, continuing anyway: {e}")
        return True

def install_packages_individually():
    """Install packages one by one to avoid build issues"""
    packages = [
        "flask>=2.3.0",
        "flask-socketio>=5.3.0", 
        "python-socketio[client]>=5.8.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "pillow>=10.0.0",
        "python-dotenv>=1.0.0",
        "python-engineio>=4.7.0"
    ]
    
    # Install torch and torchvision separately (they can be tricky)
    torch_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0"
    ]
    
    failed_packages = []
    
    # Install regular packages
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
                failed_packages.append(package)
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            failed_packages.append(package)
    
    # Try to install torch packages
    for package in torch_packages:
        try:
            print(f"📦 Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--index-url", "https://download.pytorch.org/whl/cpu"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                # Try without index-url
                result2 = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result2.returncode == 0:
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"❌ Failed to install {package}: {result2.stderr}")
                    failed_packages.append(package)
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            failed_packages.append(package)
    
    return failed_packages

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
        return False

def create_simple_requirements():
    """Create a simpler requirements.txt for easier installation"""
    simple_reqs = Path(__file__).parent / "requirements_simple.txt"
    
    content = """flask
flask-socketio
python-socketio[client]
opencv-python
numpy<2.0.0
torch
torchvision
pillow
python-dotenv
python-engineio
"""
    
    with open(simple_reqs, 'w') as f:
        f.write(content)
    
    print(f"✅ Created simplified requirements: {simple_reqs}")
    return simple_reqs

def main():
    """Main installation function"""
    print("🔧 OpenFace API Alternative Installation")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Upgrade pip first
    upgrade_pip()
    
    # Try individual package installation
    print("\n📦 Installing packages individually...")
    failed_packages = install_packages_individually()
    
    if failed_packages:
        print(f"\n⚠️  Some packages failed to install: {failed_packages}")
        print("You may need to install them manually or use conda instead")
        
        # Create simple requirements file as backup
        create_simple_requirements()
        print("\n💡 Try: pip install -r requirements_simple.txt")
    else:
        print("\n✅ All packages installed successfully!")
    
    # Check OpenFace project
    if not check_openface_project():
        print("\n⚠️  OpenFace-3.0 not found, but continuing with setup")
    
    print("\n✅ Installation completed!")
    print("\nNext steps:")
    print("1. Ensure OpenFace-3.0 is available in the parent directory")
    print("2. Download model weights if needed")
    print("3. Test the API: python app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
