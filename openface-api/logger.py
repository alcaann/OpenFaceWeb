#!/usr/bin/env python3
"""
Real-time logging system for OpenFace API
Handles client-specific logging and real-time log streaming
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from threading import Lock
import logging

@dataclass
class LogEntry:
    """Structure for log entries"""
    timestamp: float
    level: str
    message: str
    client_id: Optional[str] = None
    event_type: Optional[str] = None
    data: Optional[Dict] = None
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), default=str)

class ClientLogger:
    """Individual client logger that writes to file"""
    
    def __init__(self, client_id: str, logs_dir: Path):
        self.client_id = client_id
        self.start_time = datetime.now()
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp and client ID
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"{timestamp_str}_{client_id}.log"
        
        # Initialize log file
        self.file_lock = Lock()
        self._write_header()
    
    def _write_header(self):
        """Write session header to log file"""
        header = {
            "session_start": self.start_time.isoformat(),
            "client_id": self.client_id,
            "log_version": "1.0"
        }
        
        with self.file_lock:
            with open(self.log_file, 'w') as f:
                f.write(f"=== OpenFace API Session Log ===\n")
                f.write(f"Session Info: {json.dumps(header, indent=2)}\n")
                f.write(f"{'='*50}\n\n")
    
    def log(self, level: str, message: str, event_type: Optional[str] = None, data: Optional[Dict] = None):
        """Write log entry to file"""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            client_id=self.client_id,
            event_type=event_type,
            data=data
        )
        
        with self.file_lock:
            with open(self.log_file, 'a') as f:
                timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{timestamp_str}] [{level}] {message}")
                if event_type:
                    f.write(f" | Event: {event_type}")
                if data:
                    f.write(f" | Data: {json.dumps(data, default=str)}")
                f.write("\n")
        
        return entry
    
    def close(self):
        """Close log file with session summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            "session_end": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "client_id": self.client_id
        }
        
        with self.file_lock:
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Session Summary: {json.dumps(summary, indent=2)}\n")
                f.write(f"=== Session Ended ===\n")

class RealTimeLogger:
    """Main logging system with real-time streaming capabilities"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Client loggers
        self.client_loggers: Dict[str, ClientLogger] = {}
        self.client_lock = Lock()
        
        # Real-time log buffer for WebSocket streaming
        self.log_buffer: List[LogEntry] = []
        self.buffer_lock = Lock()
        self.max_buffer_size = 100
        
        # SocketIO instance (will be set by app)
        self.socketio = None
        
        # Setup Python logging integration
        self._setup_python_logging()
        
        print(f"📋 RealTimeLogger initialized - logs directory: {self.logs_dir}")
    
    def _setup_python_logging(self):
        """Setup Python logging to capture all log messages"""
        class SocketIOHandler(logging.Handler):
            def __init__(self, logger_instance):
                super().__init__()
                self.logger_instance = logger_instance
            
            def emit(self, record):
                if hasattr(record, 'client_id'):
                    client_id = record.client_id
                else:
                    client_id = None
                
                self.logger_instance.log(
                    level=record.levelname,
                    message=record.getMessage(),
                    client_id=client_id,
                    event_type="python_log"
                )
        
        # Add our handler to root logger
        handler = SocketIOHandler(self)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
    
    def set_socketio(self, socketio):
        """Set the SocketIO instance for real-time streaming"""
        self.socketio = socketio
    
    def register_client(self, client_id: str) -> ClientLogger:
        """Register a new client and create their logger"""
        with self.client_lock:
            if client_id in self.client_loggers:
                self.log("WARNING", f"Client {client_id} already registered", client_id=client_id)
                return self.client_loggers[client_id]
            
            client_logger = ClientLogger(client_id, self.logs_dir)
            self.client_loggers[client_id] = client_logger
            
            self.log("INFO", f"Client registered: {client_id}", client_id=client_id, event_type="client_connect")
            return client_logger
    
    def unregister_client(self, client_id: str):
        """Unregister client and close their log file"""
        with self.client_lock:
            if client_id in self.client_loggers:
                client_logger = self.client_loggers[client_id]
                client_logger.close()
                del self.client_loggers[client_id]
                
                self.log("INFO", f"Client unregistered: {client_id}", client_id=client_id, event_type="client_disconnect")
            else:
                self.log("WARNING", f"Attempted to unregister unknown client: {client_id}")
    
    def log(self, level: str, message: str, client_id: Optional[str] = None, 
            event_type: Optional[str] = None, data: Optional[Dict] = None):
        """Main logging method"""
        # Create log entry
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            client_id=client_id,
            event_type=event_type,
            data=data
        )
        
        # Add to buffer for real-time streaming
        with self.buffer_lock:
            self.log_buffer.append(entry)
            if len(self.log_buffer) > self.max_buffer_size:
                self.log_buffer.pop(0)
        
        # Write to client-specific log if client_id provided
        if client_id and client_id in self.client_loggers:
            self.client_loggers[client_id].log(level, message, event_type, data)
        
        # Stream to all connected clients via WebSocket
        if self.socketio:
            self.socketio.emit('log_entry', entry.to_dict())
        
        # Also print to console for server-side visibility
        timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]
        client_str = f"[{client_id}]" if client_id else "[SYSTEM]"
        print(f"[{timestamp_str}] {client_str} [{level}] {message}")
        
        return entry
    
    def get_recent_logs(self, count: int = 50) -> List[Dict]:
        """Get recent log entries for initial client connection"""
        with self.buffer_lock:
            recent = self.log_buffer[-count:] if len(self.log_buffer) > count else self.log_buffer
            return [entry.to_dict() for entry in recent]
    
    def get_client_logs(self, client_id: str) -> List[str]:
        """Get all logs for a specific client from their log file"""
        if client_id not in self.client_loggers:
            return []
        
        client_logger = self.client_loggers[client_id]
        try:
            with open(client_logger.log_file, 'r') as f:
                return f.readlines()
        except Exception as e:
            self.log("ERROR", f"Failed to read client logs for {client_id}: {e}")
            return []
    
    def get_log_files_info(self) -> List[Dict]:
        """Get information about all log files"""
        log_files = []
        
        for log_file in self.logs_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                log_files.append({
                    "filename": log_file.name,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "path": str(log_file)
                })
            except Exception as e:
                self.log("ERROR", f"Failed to get info for log file {log_file}: {e}")
        
        return sorted(log_files, key=lambda x: x["created"], reverse=True)

# Global logger instance
logger = RealTimeLogger()

# Convenience functions for easy logging
def log_info(message: str, client_id: Optional[str] = None, **kwargs):
    return logger.log("INFO", message, client_id=client_id, **kwargs)

def log_warning(message: str, client_id: Optional[str] = None, **kwargs):
    return logger.log("WARNING", message, client_id=client_id, **kwargs)

def log_error(message: str, client_id: Optional[str] = None, **kwargs):
    return logger.log("ERROR", message, client_id=client_id, **kwargs)

def log_debug(message: str, client_id: Optional[str] = None, **kwargs):
    return logger.log("DEBUG", message, client_id=client_id, **kwargs)
