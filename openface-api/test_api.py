#!/usr/bin/env python3
"""
Simple test script for OpenFace API
Tests the WebSocket connection and basic functionality
"""

import socketio
import base64
import json
import time
from pathlib import Path

def test_api():
    """Test the OpenFace API"""
    
    print("🧪 Testing OpenFace API...")
    
    # Create SocketIO client
    sio = socketio.Client()
    
    # Event handlers
    @sio.event
    def connect():
        print("✅ Connected to API")
    
    @sio.event
    def connected(data):
        print(f"📡 Server says: {data['message']}")
    
    @sio.event
    def frame_result(data):
        print("📊 Analysis result received:")
        if 'error' in data:
            print(f"❌ Error: {data['error']}")
        else:
            print(f"✅ Success: Found {len(data['faces'])} faces")
            for i, face_data in enumerate(data['faces']):
                face = face_data['face']
                analysis = face_data['analysis']
                print(f"   Face {i+1}: {analysis['emotion']['label']} ({analysis['emotion']['confidence']:.2f})")
    
    @sio.event
    def disconnect():
        print("🔌 Disconnected from API")
    
    try:
        # Connect to the API
        print("🔌 Connecting to API at ws://localhost:5000...")
        sio.connect('ws://localhost:5000')
        
        # Wait a moment for connection
        time.sleep(1)
        
        # Create a simple test image (1x1 pixel red image)
        test_image_data = create_test_image()
        
        print("📤 Sending test frame...")
        sio.emit('analyze_frame', {'image': test_image_data})
        
        # Wait for response
        time.sleep(2)
        
        # Disconnect
        sio.disconnect()
        print("✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def create_test_image():
    """Create a simple base64 encoded test image"""
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image (100x100 blue square)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # Blue in BGR
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_b64}"
    
    except ImportError:
        # Fallback: minimal base64 image data
        # This is a 1x1 pixel transparent PNG
        minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        return f"data:image/png;base64,{minimal_png}"

if __name__ == "__main__":
    test_api()
