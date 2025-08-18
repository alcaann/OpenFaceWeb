#!/usr/bin/env python3
"""
Flask application factory for OpenFace API
"""

import os
import sys

# IMPORTANT: eventlet monkey patching must be done BEFORE importing other modules
import eventlet
eventlet.monkey_patch()

try:
    from flask import Flask
    from flask_socketio import SocketIO
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

from config import config, api_config
from api_utils.startup import check_startup_requirements
from api.models.analyzer import OpenFaceAnalyzer
from api.routes import health_bp, analysis_bp, logs_bp
from api.routes.analysis import set_analyzer as set_analysis_analyzer
from api.routes.logs import set_logger as set_logs_logger
from api.websocket import register_websocket_handlers
from api.websocket.handlers import set_dependencies


def create_app():
    """
    Create and configure the Flask application
    
    Returns:
        tuple: (app, socketio, analyzer) instances
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask and Flask-SocketIO are required but not available")
    
    # Run startup validation
    check_startup_requirements()
    
    # Initialize logging system
    try:
        from logger import logger, log_info, log_warning, log_error
        logger_available = True
    except ImportError:
        print("⚠️  Logger not available")
        logger_available = False
        logger = None
    
    # Initialize face analyzer
    print("🚀 Initializing OpenFace Analyzer...")
    analyzer = OpenFaceAnalyzer()
    
    # Create Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Set up logging after socketio is created
    if logger_available:
        logger.set_socketio(socketio)
        log_info("OpenFace API starting up...", event_type="startup")
        log_info("OpenFace Analyzer initialized successfully", event_type="analyzer_ready")
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(logs_bp)
    
    # Set dependencies for route modules
    set_analysis_analyzer(analyzer)
    if logger_available:
        set_logs_logger(logger)
    
    # Register WebSocket handlers
    set_dependencies(analyzer, logger if logger_available else None)
    register_websocket_handlers(socketio)
    
    # Log model status
    if logger_available:
        face_detector = "RetinaFace" if analyzer.retinaface else "OpenCV" if analyzer.face_cascade else "None"
        analysis_model = "MLT" if analyzer.mlt_model else "None"
        
        log_info(f"Models loaded - Face detector: {face_detector}, Analysis: {analysis_model}", 
                 event_type="models_ready", 
                 data={
                     "face_detector": face_detector,
                     "analysis_model": analysis_model,
                     "device": str(analyzer.device) if analyzer.device else "CPU"
                 })
        
        # Check if models are available
        if analyzer.retinaface is None and analyzer.face_cascade is None:
            log_warning("No face detection method available!", event_type="startup_warning")
        
        if analyzer.mlt_model is None:
            log_warning("MLT analysis model not available!", event_type="startup_warning")
        
        log_info("API Ready to receive requests", event_type="startup_complete")
    
    return app, socketio, analyzer


def run_app():
    """Create and run the Flask application"""
    app, socketio, analyzer = create_app()
    
    try:
        from logger import log_info
        log_info("Starting OpenFace-3.0 Flask API", event_type="startup")
        log_info(f"Server configuration: http://{api_config.HOST}:{api_config.PORT}", event_type="startup")
        log_info(f"WebSocket endpoint: ws://{api_config.HOST}:{api_config.PORT}", event_type="startup")
    except ImportError:
        print(f"🚀 Starting OpenFace API on http://{api_config.HOST}:{api_config.PORT}")
    
    # Start the server with production-safe settings
    socketio.run(app, 
                host=api_config.HOST, 
                port=api_config.PORT, 
                debug=api_config.DEBUG,
                use_reloader=False,  # Disable reloader to prevent model reloading
                allow_unsafe_werkzeug=True)  # Allow running with Werkzeug in Docker
