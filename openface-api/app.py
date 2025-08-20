#!/usr/bin/env python3
"""
OpenFace-3.0 Flask API with WebSocket Support
Simplified entry point for the refactored API structure
"""

"""App entrypoint (threading async mode, no eventlet monkey patch)."""

import eventlet
eventlet.monkey_patch()

import sys
from pathlib import Path

# Initialize centralized path management first
from utils.path_manager import path_manager

# Check if OpenFace-3.0 was found
if path_manager.openface_path is None:
    print("❌ OpenFace-3.0 not found - please ensure it's in the parent directory")
    sys.exit(1)

from flask import Flask
from flask_socketio import SocketIO

# Import configuration
from config import api_config

# Import utilities
from utils.startup import check_startup_requirements

# Import models
from models.analyzer import OpenFaceAnalyzer

# Import routes
from routes import health_bp, analysis_bp, logs_bp
from routes.analysis import set_analyzer

# Import WebSocket handlers
from websocket import register_websocket_handlers, set_dependencies

# Import logging
from logger import logger, log_info


def create_app(analyzer):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev'
    
    # Set analyzer for routes that need it
    from routes.analysis import set_analyzer
    set_analyzer(analyzer)
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(logs_bp)
    
    return app


def main():
    """Main application entry point"""
    # Run startup validation
    check_startup_requirements()
    
    # Initialize logging system
    log_info("OpenFace API starting up...", event_type="startup")
    
    # Initialize face analyzer
    print("🚀 Initializing OpenFace Analyzer...")
    analyzer = OpenFaceAnalyzer()
    
    # Create Flask app with analyzer
    app = create_app(analyzer)
    
    # Initialize SocketIO without debug logging to avoid recursion
    # Enable verbose logging (temporary for debugging connection rejection)
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='eventlet',
        logger=False,
        engineio_logger=False
    )
    
    # Set up logging after socketio is created
    logger.set_socketio(socketio)
    
    # Set analyzer for analysis routes
    set_analyzer(analyzer)
    
    # Set dependencies for WebSocket handlers
    set_dependencies(analyzer, logger)
    
    # Register WebSocket handlers
    print("🔧 About to register WebSocket handlers...")
    register_websocket_handlers(socketio)
    print("✅ WebSocket handlers registration complete!")
    
    # Log startup information
    log_info("OpenFace Analyzer initialized successfully", event_type="analyzer_ready")
    
    # Check if models are available
    if analyzer.retinaface is None and analyzer.face_cascade is None:
        log_info("No face detection method available!", event_type="startup_warning")
    
    if analyzer.mlt_model is None:
        log_info("MLT analysis model not available!", event_type="startup_warning")
    
    # Log model status
    face_detector = "RetinaFace" if analyzer.retinaface else "OpenCV" if analyzer.face_cascade else "None"
    analysis_model = "MLT" if analyzer.mlt_model else "None"
    
    log_info(f"Models loaded - Face detector: {face_detector}, Analysis: {analysis_model}", 
             event_type="models_ready", 
             data={
                 "face_detector": face_detector,
                 "analysis_model": analysis_model,
                 "device": str(analyzer.device) if analyzer.device else "CPU"
             })
    
    log_info("API Ready to receive requests", event_type="startup_complete")
    log_info(f"Starting OpenFace-3.0 Flask API", event_type="startup")
    log_info(f"Server configuration: http://{api_config.HOST}:{api_config.PORT}", event_type="startup")
    log_info(f"WebSocket endpoint: ws://{api_config.HOST}:{api_config.PORT}", event_type="startup")
    
    # Start the server
    socketio.run(
        app, 
        host=api_config.HOST, 
        port=api_config.PORT, 
        debug=api_config.DEBUG,
        use_reloader=False,  # Disable reloader to prevent model reloading
        allow_unsafe_werkzeug=True  # Allow running with Werkzeug in Docker
    )


if __name__ == '__main__':
    main()
