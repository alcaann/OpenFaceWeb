# OpenFace-3.0 Flask API

Real-time facial analysis API using WebSockets that integrates with the OpenFace-3.0 models.

## Quick Start

1. **Setup the API:**
   ```bash
   cd openface-api
   python setup.py
   ```

2. **Start the server:**
   ```bash
   python app.py
   ```

3. **Test the API:**
   - Open `test_client.html` in a web browser
   - Click "Connect" and "Start Camera"
   - Click "Start Analysis" to begin real-time facial analysis

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

## API Documentation

For detailed API documentation, see [`openapi.yaml`](./openapi.yaml) or visit the interactive documentation when the server is running.

### HTTP Endpoints

- `GET /` - API status and system information
- `GET /health` - Health check
- `POST /api/analyze` - Single image analysis (multipart/form-data)
- `GET /api/logs` - Recent API logs
- `GET /api/logs/files` - Available log files
- `GET /api/logs/clients` - Active WebSocket clients

### WebSocket Events

- `connect` - Client connection
- `analyze_frame` - Send frame for analysis (base64 image)
- `analysis_result` - Receive analysis results
- `ping/pong` - Connection testing

## Project Structure

```
openface-api/
├── app.py                  # Main application entry point
├── config.py              # Configuration settings
├── openapi.yaml           # API documentation
├── api/                   # Application modules
│   ├── __init__.py        # Flask app factory
│   ├── models/           # Analysis models
│   ├── routes/           # HTTP route handlers
│   └── websocket/        # WebSocket event handlers
├── api_utils/            # Utility functions
│   ├── startup.py        # Startup validation
│   ├── image_utils.py    # Image processing
│   └── validation.py     # Input validation
├── docs/                 # Additional documentation
└── logs/                 # Application logs
```

## Requirements

See [`requirements.txt`](./requirements.txt) for dependencies.

## Development

The API is organized using Flask blueprints and follows a modular structure:

- **Models**: Face analysis engine (`api/models/analyzer.py`)
- **Routes**: HTTP endpoints organized by functionality
- **WebSocket**: Real-time communication handlers
- **Utils**: Shared utilities for validation, image processing, etc.

## Environment Variables

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 5000)
- `FLASK_ENV`: Environment mode (development/production)
- `CONFIDENCE_THRESHOLD`: Face detection confidence (default: 0.5)
- `SECRET_KEY`: Flask secret key (default: 'dev')

## Troubleshooting

1. **GPU Issues**: Check CUDA installation and compatibility
2. **Model Loading**: Ensure weights are in the correct directory
3. **Dependencies**: Run `pip install -r requirements.txt`
4. **Logs**: Check `logs/` directory for detailed error information

For more details, see the main project [TROUBLESHOOTING.md](../TROUBLESHOOTING.md).
