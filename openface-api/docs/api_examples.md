# API Examples

This document provides examples of how to use the OpenFace API.

## HTTP API Examples

### 1. Check API Status

```bash
curl http://localhost:5000/
```

Response:
```json
{
  "service": "OpenFace-3.0 Flask API",
  "version": "3.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "analyze": "/api/analyze",
    "logs": "/api/logs",
    "websocket": "ws://host:port/"
  },
  "system": {
    "python_version": "3.8.10",
    "cpu_count": 8,
    "memory_total_gb": 16.0,
    "memory_available_gb": 8.5
  }
}
```

### 2. Health Check

```bash
curl http://localhost:5000/health
```

### 3. Analyze Image

```bash
curl -X POST \
  -F "image=@face.jpg" \
  http://localhost:5000/api/analyze
```

Response:
```json
{
  "success": true,
  "timestamp": 1692345678.123,
  "processing_time_ms": 45.67,
  "faces_detected": 1,
  "faces": [
    {
      "face_id": 0,
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "emotion": {
        "label": "Happy",
        "confidence": 0.87,
        "all_emotions": {
          "Neutral": 0.05,
          "Happy": 0.87,
          "Sad": 0.02,
          "Surprise": 0.03,
          "Fear": 0.01,
          "Disgust": 0.01,
          "Anger": 0.01,
          "Contempt": 0.00
        }
      },
      "gaze": {
        "direction": [0.1, -0.2],
        "pitch": -0.2,
        "yaw": 0.1
      },
      "action_units": {
        "active": [
          {"label": "AU12", "intensity": 0.78},
          {"label": "AU6", "intensity": 0.65}
        ],
        "all_aus": {
          "AU1": 0.12,
          "AU2": 0.08,
          "AU4": 0.15,
          "AU6": 0.65,
          "AU9": 0.23,
          "AU12": 0.78,
          "AU25": 0.34,
          "AU26": 0.19
        }
      }
    }
  ]
}
```

## WebSocket Examples

### JavaScript Client

```javascript
// Connect to WebSocket
const socket = io('http://localhost:5000');

// Handle connection
socket.on('connect', () => {
    console.log('Connected to OpenFace API');
});

socket.on('connection_response', (data) => {
    console.log('Connection response:', data);
    if (data.success) {
        console.log('Client ID:', data.client_id);
        console.log('Server capabilities:', data.server_info.capabilities);
    }
});

// Send image for analysis
function analyzeFrame(imageBase64) {
    socket.emit('analyze_frame', {
        image: imageBase64
    });
}

// Receive analysis results
socket.on('analysis_result', (result) => {
    if (result.success) {
        console.log(`Found ${result.faces_detected} faces`);
        result.faces.forEach((face, index) => {
            console.log(`Face ${index}:`, {
                emotion: face.emotion?.label,
                confidence: face.emotion?.confidence,
                bbox: face.bbox
            });
        });
    } else {
        console.error('Analysis failed:', result.error);
    }
});

// Test connection
socket.emit('ping', { test: 'data' });
socket.on('pong', (data) => {
    console.log('Pong received:', data);
});
```

### Python Client

```python
import socketio
import base64
import json

# Create WebSocket client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to OpenFace API')

@sio.event
def connection_response(data):
    print('Connection response:', json.dumps(data, indent=2))

@sio.event
def analysis_result(data):
    if data['success']:
        print(f"Found {data['faces_detected']} faces")
        for i, face in enumerate(data['faces']):
            if 'emotion' in face:
                print(f"Face {i}: {face['emotion']['label']} ({face['emotion']['confidence']:.2f})")
    else:
        print(f"Analysis failed: {data['error']}")

# Connect to server
sio.connect('http://localhost:5000')

# Send image for analysis
def analyze_image(image_path):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    sio.emit('analyze_frame', {'image': image_data})

# Example usage
analyze_image('face.jpg')

# Keep connection alive
sio.wait()
```

## Python Requests Examples

```python
import requests
import json

# Check API status
response = requests.get('http://localhost:5000/')
print(json.dumps(response.json(), indent=2))

# Health check
response = requests.get('http://localhost:5000/health')
print(f"Health status: {response.json()['status']}")

# Analyze image
with open('face.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/analyze', files=files)

result = response.json()
if result['success']:
    print(f"Found {result['faces_detected']} faces")
    for face in result['faces']:
        if 'emotion' in face:
            print(f"Emotion: {face['emotion']['label']} ({face['emotion']['confidence']:.2f})")
        if 'gaze' in face:
            print(f"Gaze: pitch={face['gaze']['pitch']:.2f}, yaw={face['gaze']['yaw']:.2f}")
```

## Error Handling

### Common Error Responses

```json
// Invalid image format
{
  "success": false,
  "error": "File type 'txt' not allowed. Allowed: jpg, jpeg, png, gif, bmp, webp"
}

// Image too large
{
  "success": false,
  "error": "File too large (max 10MB)"
}

// Analysis failure
{
  "success": false,
  "error": "Analysis failed: Could not decode image"
}

// Model not available
{
  "success": false,
  "error": "Analyzer not initialized"
}
```

### WebSocket Error Handling

```javascript
socket.on('analysis_result', (result) => {
    if (!result.success) {
        switch (result.error) {
            case 'No image data provided':
                console.error('Image data is required');
                break;
            case 'Image data too large (max 10MB)':
                console.error('Reduce image size');
                break;
            case 'Analyzer not initialized':
                console.error('Server not ready, try again later');
                break;
            default:
                console.error('Analysis error:', result.error);
        }
    }
});

socket.on('connect_error', (error) => {
    console.error('Connection failed:', error);
});

socket.on('disconnect', (reason) => {
    console.log('Disconnected:', reason);
    if (reason === 'io server disconnect') {
        // Server disconnected, try to reconnect
        socket.connect();
    }
});
```

## Performance Tips

1. **Image Size**: Resize images to reasonable dimensions (e.g., 640x480) before sending
2. **Batch Processing**: For multiple images, use the HTTP endpoint with multiple requests
3. **WebSocket**: Use WebSocket for real-time video streams
4. **Error Handling**: Always check the `success` field in responses
5. **Rate Limiting**: Don't send too many requests simultaneously
