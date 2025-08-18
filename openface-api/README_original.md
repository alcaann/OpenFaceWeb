# OpenFace-3.0 Flask API

Real-time facial analysis API using WebSockets that integrates with the OpenFace-3.0 models.

## Features

- **Real-time Analysis**: WebSocket-based API for live video stream processing
- **Face Detection**: RetinaFace (primary) with OpenCV fallback
- **Facial Analysis**: 
  - Emotion recognition (8 emotions)
  - Action Units detection (20 AUs)
  - Gaze estimation (pitch/yaw)
  - Facial landmarks (5 key points)
- **Web Interface**: HTML test client with live video feed
- **Cross-platform**: Works on Windows, Linux, and macOS

## Quick Start

1. **Setup the API:**
   ```bash
   cd openface-api
   python setup.py
   ```

2. **Start the server:**
   ```bash
   python run_api.py
   ```

3. **Test the API:**
   - Open `test_client.html` in a web browser
   - Click "Connect" and "Start Camera"
   - Click "Start Analysis" to begin real-time facial analysis

## API Endpoints

### HTTP Endpoints

- `GET /` - API status and system information
- `GET /health` - Health check
- `POST /api/analyze` - Single image analysis (multipart/form-data)

### WebSocket Events

- `connect` - Client connection
- `analyze_frame` - Send frame for analysis (base64 image)
- `analysis_result` - Receive analysis results
- `ping/pong` - Connection testing

## Directory Structure

```
openface-api/
├── app.py              # Main Flask API application
├── config.py           # Configuration and path management
├── setup.py           # Setup and dependency checking
├── requirements.txt   # Python dependencies
├── test_client.html   # Web-based test client
├── run_api.py         # Simple server runner (created by setup)
└── README.md         # This file
```

## Requirements

### System Requirements
- Python 3.8+
- OpenFace-3.0 project (in parent directory)
- Model weights (see download instructions below)

### Python Dependencies
- flask==2.3.3
- flask-socketio==5.3.6
- opencv-python==4.8.1.78
- torch==2.0.1
- torchvision==0.15.2
- pillow==10.0.1
- eventlet==0.33.3
- numpy==1.24.3

## Model Weights

Download the required model weights and place them in `../OpenFace-3.0/weights/`:

1. **RetinaFace**: `mobilenet0.25_Final.pth`
2. **MLT Model**: `MTL_backbone.pth`

**Download Link**: [Google Drive](https://drive.google.com/drive/folders/1aBEol-zG_blHSavKFVBH9dzc9U9eJ92p)

## Usage Examples

### WebSocket Client (JavaScript)

```javascript
const socket = io('http://localhost:5000');

// Send frame for analysis
const canvas = document.getElementById('canvas');
const imageData = canvas.toDataURL('image/jpeg', 0.8);
socket.emit('analyze_frame', { image: imageData });

// Receive results
socket.on('analysis_result', function(data) {
    console.log('Analysis:', data);
    // data.faces[0].emotion.label
    // data.faces[0].gaze.direction
    // data.faces[0].action_units.active
});
```

### HTTP API (Python)

```python
import requests

# Single image analysis
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze',
        files={'image': f}
    )
    result = response.json()
    print(result['faces'])
```

## Response Format

```json
{
    "success": true,
    "timestamp": 1691234567.89,
    "processing_time_ms": 45.6,
    "faces_detected": 1,
    "faces": [
        {
            "face_id": 0,
            "bbox": [100, 150, 300, 350],
            "confidence": 0.99,
            "landmarks": [[120, 180], [180, 180], [150, 210], [130, 240], [170, 240]],
            "emotion": {
                "label": "Happy",
                "confidence": 0.85,
                "all_emotions": {...}
            },
            "gaze": {
                "direction": [0.1, -0.2],
                "pitch": -0.2,
                "yaw": 0.1
            },
            "action_units": {
                "active": [
                    {"label": "AU6", "intensity": 0.8},
                    {"label": "AU12", "intensity": 0.9}
                ],
                "all_aus": {...}
            }
        }
    ]
}
```

## Configuration

Modify `app.py` for custom configuration:

```python
# Server settings
API_HOST = "0.0.0.0"
API_PORT = 5000

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
VIS_THRESHOLD = 0.5

# Model paths (auto-detected by default)
OPENFACE_PATH = Path(__file__).parent.parent / "OpenFace-3.0"
```

## Troubleshooting

### Common Issues

1. **Import errors**: Run `python setup.py` to check dependencies
2. **Model not found**: Ensure OpenFace-3.0 is in parent directory
3. **Weights missing**: Download model weights to the correct location
4. **Camera access**: Check browser permissions for webcam access
5. **Connection failed**: Ensure server is running on correct port

### Performance Tips

1. **Reduce frame rate**: Lower FPS in test client for better performance
2. **Image quality**: Reduce quality setting for faster processing
3. **GPU acceleration**: Ensure CUDA is available for better performance

### Debug Mode

Start server in debug mode:
```python
DEBUG_MODE = True  # in app.py
```

## Integration

This API can be integrated into:
- Web applications (React, Vue, Angular)
- Mobile apps (React Native, Flutter)
- Desktop applications (Electron)
- Other Python applications
- Streaming platforms

## License

This project follows the same license as the OpenFace-3.0 project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify OpenFace-3.0 setup
3. Check console output for errors
4. Test with the provided HTML client first
