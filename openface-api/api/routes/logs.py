#!/usr/bin/env python3
"""
Logging routes for OpenFace API
"""

import os
from pathlib import Path
from flask import Blueprint, jsonify, request

logs_bp = Blueprint('logs', __name__)

# This will be set by the app factory
logger = None


def set_logger(logger_instance):
    """Set the logger instance for this blueprint"""
    global logger
    logger = logger_instance


@logs_bp.route('/api/logs')
def get_logs():
    """Get recent API logs"""
    try:
        if logger is None:
            return jsonify({"error": "Logger not initialized"}), 500
        
        # Get recent logs from the logger
        recent_logs = logger.get_recent_logs(limit=100)
        
        return jsonify({
            "success": True,
            "logs": recent_logs,
            "total": len(recent_logs)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@logs_bp.route('/api/logs/files')
def get_log_files():
    """Get list of available log files"""
    try:
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        
        if not logs_dir.exists():
            return jsonify({
                "success": True,
                "files": []
            })
        
        log_files = []
        for file_path in logs_dir.glob("*.log"):
            try:
                stat = file_path.stat()
                log_files.append({
                    "name": file_path.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime
                })
            except Exception:
                continue
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            "success": True,
            "files": log_files
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@logs_bp.route('/api/logs/clients')
def get_active_clients():
    """Get active WebSocket clients"""
    try:
        if logger is None:
            return jsonify({"error": "Logger not initialized"}), 500
        
        active_clients = []
        for client_id, client_logger in logger.client_loggers.items():
            active_clients.append({
                "client_id": client_id,
                "start_time": client_logger.start_time.isoformat(),
                "log_file": client_logger.log_file.name
            })
        
        return jsonify({
            "success": True,
            "clients": active_clients,
            "total": len(active_clients)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@logs_bp.route('/api/logs/client/<client_id>')
def get_client_logs(client_id):
    """Get logs for a specific client"""
    try:
        if logger is None:
            return jsonify({"error": "Logger not initialized"}), 500
        
        if client_id not in logger.client_loggers:
            return jsonify({
                "success": False,
                "error": "Client not found"
            }), 404
        
        client_logger = logger.client_loggers[client_id]
        
        # Read recent lines from client log file
        try:
            lines = []
            with open(client_logger.log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
            
            return jsonify({
                "success": True,
                "client_id": client_id,
                "logs": [line.strip() for line in lines],
                "log_file": client_logger.log_file.name
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to read log file: {str(e)}"
            }), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
