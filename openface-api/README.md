# OpenFace-3.0 Real-Time API

This directory contains a high-performance, real-time facial analysis API powered by OpenFace-3.0. It is designed as a standalone service that receives video frames via a WebSocket and returns detailed facial analysis data in JSON format.

## Core Features

- **Real-Time Analysis**: WebSocket-based API for low-latency video stream processing.
- **HTTP Endpoint**: A simple `POST` endpoint for analyzing single, static images.
- **Comprehensive Analysis**: Provides data on emotions, action units (AUs), gaze, and facial landmarks.
- **Containerized**: Fully configured to run in a Docker container with GPU support.

## Project Structure

The API is organized into a clean, modular structure:

```
openface-api/
├── app.py                  # Main application entry point and Flask/SocketIO setup
├── config.py               # API configuration settings
├── logger.py               # Real-time logging system
├── requirements.txt        # Python dependencies
├── Dockerfile              # Instructions for building the Docker container
│
├── core/
│   └── engine/             # Face analysis engine and model loaders
│
├── api/
│   ├── routes/             # HTTP route handlers (e.g., /health, /api/analyze)
│   ├── websocket/          # WebSocket event handlers
│   └── openapi.yaml        # API documentation (OpenAPI 3.0)
│
├── utils/                  # Shared utilities (e.g., image processing, path management)
│
├── scripts/                # Utility and maintenance scripts
│   ├── install.py
│   └── launch.py
│
└── logs/                   # Directory for server and client log files (auto-generated)
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers and NVIDIA Container Toolkit (for GPU acceleration)

### Running the Service

1.  **Build and Run the Container:**
    From the root of the `OpenFaceWeb` project, run:
    ```bash
    docker compose up --build
    ```

2.  **Verify Service:**
    Once the container is running, you can check the health of the API:
    ```bash
    curl http://localhost:5000/health
    ```
    You should receive a `{"status": "healthy", ...}` response.

## API Documentation

The API is documented using the OpenAPI 3.0 standard. The specification file, `openapi.yaml`, serves as the single source of truth for all available endpoints and data models.

- **Primary Interface**: WebSocket at `ws://localhost:5000`
- **HTTP Endpoints**: See `openapi.yaml` for details on `/health` and `/api/analyze`.

You can use tools like the [Swagger Editor](https://editor.swagger.io/) to view the `openapi.yaml` file in a more interactive format.

