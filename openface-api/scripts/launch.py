#!/usr/bin/env python3
"""
OpenFace API Launcher
Quick launcher for the OpenFace API server
"""

import sys
import os
from pathlib import Path

def main():
    print("🚀 OpenFace-3.0 API Launcher")
    print("=" * 30)
    
    # Add current directory to Python path
    api_dir = Path(__file__).parent
    sys.path.insert(0, str(api_dir))
    
    try:
        # Import and run the API
        from app import app, socketio, analyzer
        
        print("✅ OpenFace API initialized successfully")
        print(f"📍 Server will start at: http://0.0.0.0:5000")
        print(f"🧪 Test client: {api_dir / 'test_client.html'}")
        print(f"📖 Documentation: {api_dir / 'README.md'}")
        print("=" * 50)
        print("Press Ctrl+C to stop the server")
        print()
        
        # Start the server
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    use_reloader=False)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print()
        print("Please run setup first:")
        print("  python setup.py")
        print()
        print("Or install requirements manually:")
        print("  pip install -r requirements.txt")
        return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print()
        print("Common solutions:")
        print("1. Check if OpenFace-3.0 is in the parent directory")
        print("2. Ensure model weights are downloaded")
        print("3. Run: python setup.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
