#!/usr/bin/env python3
"""
Real-time logging system for OpenFace API
Handles client-specific logging and real-time log streaming
"""

import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
import logging

# --- Constants ---
LOG_VERSION = "1.1"
MAX_BUFFER_SIZE = 200
SERVER_LOG_ID = "server"

@dataclass
class LogEntry:
    """Structure for log entries"""
    timestamp: float
    level: str
    message: str
    client_id: Optional[str] = None
    event_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

class FileLogger:
    """Handles writing logs to a single file, used for both server and clients."""
    
    def __init__(self, log_id: str, logs_dir: Path, session_info: Dict[str, Any]):
        self.log_id = log_id
        self.start_time = datetime.now()
        self.log_file = logs_dir / f"{self.start_time.strftime('%Y%m%d_%H%M%S')}_{log_id}.log"
        self.file_lock = Lock()
        self._write_header(session_info)

    def _write_header(self, session_info: Dict[str, Any]):
        """Write session header to log file"""
        header = {
            "session_start": self.start_time.isoformat(),
            "log_id": self.log_id,
            "log_version": LOG_VERSION,
            **session_info
        }
        with self.file_lock:
            with open(self.log_file, 'w') as f:
                f.write("=== OpenFace API Session Log ===\n")
                f.write(f"Session Info: {json.dumps(header, indent=2)}\n")
                f.write(f"{'='*50}\n\n")

    def write(self, entry: LogEntry):
        """Write a single log entry to the file."""
        with self.file_lock:
            with open(self.log_file, 'a') as f:
                ts_str = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                status_icon = ""
                if entry.event_type and "startup" in entry.event_type:
                    if "fail" in entry.event_type or "warning" in entry.event_type:
                        status_icon = "❌ "
                    else:
                        status_icon = "✅ "

                f.write(f"[{ts_str}] [{entry.level}] {status_icon}{entry.message}")
                if entry.event_type:
                    f.write(f" | Event: {entry.event_type}")
                if entry.data:
                    f.write(f" | Data: {json.dumps(entry.data, default=str)}")
                f.write("\n")

    def close(self):
        """Close log file with a session summary."""
        end_time = datetime.now()
        summary = {
            "session_end": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds()
        }
        with self.file_lock:
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Session Summary: {json.dumps(summary, indent=2)}\n")
                f.write("=== Session Ended ===\n")

class RealTimeLogger:
    """Main logging system with real-time streaming and dedicated file loggers."""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.client_loggers: Dict[str, FileLogger] = {}
        self.client_lock = Lock()
        
        self.log_buffer: List[LogEntry] = []
        self.buffer_lock = Lock()
        
        self.socketio = None
        
        # Create a dedicated logger for server-wide events
        self.server_logger = FileLogger(SERVER_LOG_ID, self.logs_dir, {"log_type": "server"})
        
        self._setup_python_logging()
        print(f"📋 RealTimeLogger initialized - logs directory: {self.logs_dir}")
        self.log("INFO", "Logging system initialized.", event_type="system_startup")

    def _setup_python_logging(self):
        """Redirects standard Python logging to our system."""
        class SocketIOHandler(logging.Handler):
            def __init__(self, logger_instance):
                super().__init__()
                self.logger_instance = logger_instance
            
            def emit(self, record):
                if 'socketio' in record.name or 'engineio' in record.name:
                    return
                client_id = getattr(record, 'client_id', None)
                self.logger_instance.log(record.levelname, record.getMessage(), client_id=client_id)
        
        logging.basicConfig(level=logging.INFO, handlers=[SocketIOHandler(self)])

    def set_socketio(self, socketio):
        self.socketio = socketio
        self.log("INFO", "SocketIO instance set for real-time log streaming.", event_type="system_config")

    def register_client(self, client_id: str):
        with self.client_lock:
            if client_id in self.client_loggers:
                self.log("WARNING", f"Client {client_id} already registered.", client_id=client_id)
                return
            
            client_logger = FileLogger(client_id, self.logs_dir, {"log_type": "client"})
            self.client_loggers[client_id] = client_logger
            self.log("INFO", f"Client connected and logger registered.", client_id=client_id, event_type="client_connect")

    def unregister_client(self, client_id: str):
        with self.client_lock:
            if client_id in self.client_loggers:
                # Log disconnection to the client's file before closing
                self.log("INFO", "Client disconnected.", client_id=client_id, event_type="client_disconnect")
                
                client_logger = self.client_loggers.pop(client_id)
                client_logger.close()
                
                # Log to server log that client was unregistered
                self.log("INFO", f"Client logger for {client_id} closed.", event_type="client_unregister")
            else:
                self.log("WARNING", f"Attempted to unregister unknown client: {client_id}.")

    def log(self, level: str, message: str, client_id: Optional[str] = None, 
            event_type: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        entry = LogEntry(time.time(), level.upper(), message, client_id, event_type, data)
        
        # Write to the correct log file
        if client_id and client_id in self.client_loggers:
            self.client_loggers[client_id].write(entry)
        else:
            # All non-client logs go to the server log
            self.server_logger.write(entry)
        
        # Add to buffer for streaming
        with self.buffer_lock:
            self.log_buffer.append(entry)
            if len(self.log_buffer) > MAX_BUFFER_SIZE:
                self.log_buffer.pop(0)
        
        # Stream to WebSocket clients
        if self.socketio and not getattr(threading.current_thread(), '_emitting_log', False):
            try:
                setattr(threading.current_thread(), '_emitting_log', True)
                self.socketio.emit('log_entry', entry.to_dict())
            finally:
                delattr(threading.current_thread(), '_emitting_log')
        
        # Print to console for live monitoring
        ts_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]
        id_str = f"[{client_id}]" if client_id else f"[{SERVER_LOG_ID.upper()}]"
        print(f"[{ts_str}] {id_str:<10} [{entry.level}] {entry.message}")

    def get_recent_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        with self.buffer_lock:
            return [entry.to_dict() for entry in self.log_buffer[-count:]]

# Global logger instance
logger = RealTimeLogger(logs_dir="logs")

# Convenience functions
def log_info(message: str, client_id: Optional[str] = None, **kwargs):
    logger.log("INFO", message, client_id=client_id, **kwargs)

def log_warning(message: str, client_id: Optional[str] = None, **kwargs):
    logger.log("WARNING", message, client_id=client_id, **kwargs)

def log_error(message: str, client_id: Optional[str] = None, **kwargs):
    logger.log("ERROR", message, client_id=client_id, **kwargs)

def log_debug(message: str, client_id: Optional[str] = None, **kwargs):
    # Note: Python's root logger level is INFO by default
    logger.log("DEBUG", message, client_id=client_id, **kwargs)
