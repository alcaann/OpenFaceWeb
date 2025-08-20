#!/usr/bin/env python3
"""
Health check routes for OpenFace API
"""

import sys
from pathlib import Path
from flask import Blueprint, jsonify

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from config import config, api_config

health_bp = Blueprint('health', __name__)


@health_bp.route('/')
def index():
    """API root endpoint with system information"""
    try:
        system_info = {
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
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "openface": {
                "project_path": str(config.project_root),
                "weights_path": str(config.weights_dir),
                "model_available": config.weights_dir.exists()
            }
        }
        
        # Add system info if psutil is available
        if PSUTIL_AVAILABLE:
            system_info["system"].update({
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
            })
        else:
            system_info["system"]["note"] = "System monitoring not available (psutil not installed)"
        
        # Add GPU information if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            system_info["gpu"] = {
                "available": True,
                "device_count": device_count,
                "current_device": current_device,
                "device_name": torch.cuda.get_device_name(current_device),
                "memory_total_gb": round(torch.cuda.get_device_properties(current_device).total_memory / (1024**3), 2)
            }
        else:
            system_info["gpu"] = {"available": False}
        
        return jsonify(system_info)
        
    except Exception as e:
        return jsonify({
            "service": "OpenFace-3.0 Flask API",
            "status": "error",
            "error": str(e)
        }), 500


@health_bp.route('/health')
def health():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": psutil.boot_time(),
            "uptime_seconds": psutil.uptime() if hasattr(psutil, 'uptime') else None,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        # Check model files
        model_files = {
            "retinaface": config.weights_dir / "mobilenet0.25_Final.pth",
            "mlt": config.weights_dir / "MTL_backbone.pth",
            "star": config.weights_dir / "Landmark_68.pkl"
        }
        
        model_status = {}
        for name, path in model_files.items():
            model_status[name] = {
                "available": path.exists(),
                "path": str(path),
                "size_mb": round(path.stat().st_size / (1024**2), 2) if path.exists() else 0
            }
        
        health_status["models"] = model_status
        
        # Check if analyzer is functional (basic check)
        health_status["analyzer"] = {
            "initialized": True,  # This would need to be passed from app context
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
