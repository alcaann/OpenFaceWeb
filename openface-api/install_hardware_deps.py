#!/usr/bin/env python3
"""
Simple check for hardware monitoring dependencies
"""

def check_dependencies():
    """Check if hardware monitoring dependencies are available"""
    print("🔧 Checking hardware monitoring dependencies...")
    
    try:
        import psutil
        print("✅ psutil - available")
    except ImportError:
        print("❌ psutil - not available (install with: pip install psutil)")
    
    try:
        import torch
        print(f"✅ torch - available (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  🚀 CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print("  🔧 CUDA not available")
    except ImportError:
        print("❌ torch - not available")

if __name__ == "__main__":
    check_dependencies()
