# OpenFace API Logging System

## Overview

This logging system provides real-time logging capabilities for the OpenFace API with the following features:

### Backend Features

1. **Real-time Log Streaming**: All logs are streamed in real-time to connected WebSocket clients
2. **Client-specific Log Files**: Each client connection creates a separate log file with:
   - Timestamp of connection start
   - Client socket ID
   - All logs generated during the session
3. **Structured Logging**: Logs include level, message, client ID, event type, and optional data
4. **Multiple Log Levels**: INFO, WARNING, ERROR, DEBUG
5. **Event Categorization**: Logs are categorized by event types (e.g., client_connect, frame_processing, analysis_complete)

### Frontend Features

1. **Console-like Interface**: Clean, terminal-style log display
2. **Real-time Updates**: Logs appear instantly as they're generated
3. **Filtering**: Filter by log level, event type, or client ID
4. **Auto-scroll**: Optional auto-scroll to follow new logs
5. **Data Expansion**: Click to expand structured data in logs
6. **Connection Status**: Visual indicator of WebSocket connection

## File Structure

```
openface-api/
├── logger.py              # Main logging system
├── app.py                 # Flask app with logging integration
├── logs/                  # Directory for log files
│   └── YYYYMMDD_HHMMSS_client-id.log
└── requirements.txt       # Updated dependencies

openface-web/src/
├── components/
│   └── LogConsoleNew.tsx  # React logging console component
└── app/
    └── page.tsx           # Updated main page with logging
```

## Log File Format

Each client session creates a log file named: `YYYYMMDD_HHMMSS_client-id.log`

Example: `20250818_143022_abc123def456.log`

### Log File Contents

```
=== OpenFace API Session Log ===
Session Info: {
  "session_start": "2025-08-18T14:30:22.123456",
  "client_id": "abc123def456",
  "log_version": "1.0"
}
==================================================

[14:30:22.123] [INFO] Client connected: abc123def456 | Event: client_connect
[14:30:22.456] [DEBUG] Received frame analysis request | Event: frame_received
[14:30:22.789] [INFO] Frame analysis completed: 1 faces detected, processing time: 45ms | Event: analysis_complete | Data: {"faces_detected": 1, "processing_time_ms": 45, "frame_size": "640x480"}
[14:30:35.123] [INFO] Client disconnected: abc123def456 | Event: client_disconnect

==================================================
Session Summary: {
  "session_end": "2025-08-18T14:30:35.123456",
  "duration_seconds": 13.0,
  "client_id": "abc123def456"
}
=== Session Ended ===
```

## API Endpoints

### WebSocket Events

- `log_entry`: Real-time log entry broadcast
- `request_logs`: Request recent logs
- `logs_response`: Response with requested logs
- `request_client_info`: Get client session info
- `client_info_response`: Client info response

### HTTP Endpoints

- `GET /api/logs`: Get recent log entries
- `GET /api/logs/files`: Get information about all log files
- `GET /api/logs/clients`: Get list of active clients
- `GET /api/logs/client/<client_id>`: Get logs for specific client

## Usage Examples

### Backend Logging

```python
from logger import log_info, log_warning, log_error, log_debug

# Basic logging
log_info("Server started successfully")

# Client-specific logging
log_info("Frame processed", client_id="abc123", event_type="frame_processing")

# Logging with structured data
log_info("Analysis complete", 
         client_id="abc123",
         event_type="analysis_complete",
         data={"faces": 2, "processing_time": 45})
```

### Frontend Integration

```tsx
import LogConsole from '@/components/LogConsoleNew'

function MyComponent() {
  return (
    <LogConsole 
      socket={websocketInstance}
      maxLogs={100}
      height="400px"
    />
  )
}
```

## Configuration

### Backend Settings

- **Log Directory**: Default `logs/` (configurable in `logger.py`)
- **Max Buffer Size**: 100 logs (configurable)
- **Log Levels**: ALL, INFO, WARNING, ERROR, DEBUG

### Frontend Settings

- **Max Logs**: Number of logs to keep in memory
- **Height**: Console height
- **Auto-scroll**: Automatic scrolling behavior
- **Filters**: Level, event type, client ID filtering

## Benefits

1. **Debugging**: Real-time visibility into API operations
2. **Monitoring**: Track client connections and processing performance
3. **Analytics**: Historical analysis of usage patterns
4. **Troubleshooting**: Detailed error tracking and client-specific logs
5. **Performance**: Monitor frame processing times and system health

## Security Considerations

- Log files contain client IDs but no sensitive data
- WebSocket connections are filtered to prevent log data leakage
- Log directory should be protected in production
- Consider log rotation for long-running deployments
