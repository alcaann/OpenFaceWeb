#!/usr/bin/env python3
"""
WebSocket event handlers for OpenFace API
"""

import time
import uuid

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

try:
    from flask_socketio import emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    emit = None

from api_utils.validation import validate_analysis_request
from api_utils.image_utils import decode_base64_image

# Global variables that will be set by the app factory
analyzer = None
logger = None


def set_dependencies(analyzer_instance, logger_instance):
    """Set the analyzer and logger instances"""
    global analyzer, logger
    analyzer = analyzer_instance
    logger = logger_instance


def register_websocket_handlers(socketio):
    """Register WebSocket event handlers with the SocketIO instance"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        if not SOCKETIO_AVAILABLE:
            return False
        
        client_id = str(uuid.uuid4())[:8]
        
        try:
            if logger:
                logger.add_client(client_id)
                logger.log_info(f"Client connected: {client_id}", 
                              event_type="client_connect", 
                              client_id=client_id)
            
            emit('connection_response', {
                "success": True,
                "client_id": client_id,
                "message": "Connected to OpenFace API",
                "server_info": {
                    "api_version": "3.0",
                    "capabilities": {
                        "face_detection": analyzer is not None,
                        "emotion_analysis": analyzer is not None and analyzer.mlt_model is not None,
                        "gaze_estimation": analyzer is not None and analyzer.mlt_model is not None,
                        "action_units": analyzer is not None and analyzer.mlt_model is not None
                    }
                }
            })
            
        except Exception as e:
            emit('connection_response', {
                "success": False,
                "error": str(e)
            })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        try:
            # Note: In a real implementation, you'd track client IDs per session
            if logger:
                logger.log_info("Client disconnected", event_type="client_disconnect")
        except Exception as e:
            print(f"Error handling disconnect: {e}")
    
    @socketio.on('analyze_frame')
    def handle_analyze_frame(data):
        """Handle frame analysis request"""
        if not CV2_AVAILABLE or not SOCKETIO_AVAILABLE:
            emit('analysis_result', {
                "success": False,
                "error": "Required dependencies not available"
            })
            return
        
        if analyzer is None:
            emit('analysis_result', {
                "success": False,
                "error": "Analyzer not initialized"
            })
            return
        
        start_time = time.time()
        
        try:
            # Validate request
            is_valid, error_msg = validate_analysis_request(data)
            if not is_valid:
                emit('analysis_result', {
                    "success": False,
                    "error": error_msg
                })
                return
            
            # Decode image
            try:
                frame = decode_base64_image(data['image'])
            except ValueError as e:
                emit('analysis_result', {
                    "success": False,
                    "error": str(e)
                })
                return
            
            # Analyze frame
            result = analyzer.analyze_frame(frame)
            
            # Add WebSocket specific information
            result['websocket_info'] = {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "frame_size": frame.shape if frame is not None else None
            }
            
            emit('analysis_result', result)
            
            # Log the analysis
            if logger:
                logger.log_info(f"Frame analyzed - {result.get('faces_detected', 0)} faces detected", 
                              event_type="frame_analysis",
                              data={
                                  "faces_detected": result.get('faces_detected', 0),
                                  "processing_time": result.get('processing_time_ms', 0)
                              })
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "timestamp": time.time()
            }
            
            emit('analysis_result', error_result)
            
            if logger:
                logger.log_error(f"Frame analysis error: {str(e)}", 
                               event_type="analysis_error")
    
    @socketio.on('ping')
    def handle_ping(data):
        """Handle ping request for connection testing"""
        if not SOCKETIO_AVAILABLE:
            return
        
        emit('pong', {
            "timestamp": time.time(),
            "data": data if data else None
        })
    
    @socketio.on('get_client_info')
    def handle_get_client_info(data):
        """Handle request for client information"""
        if not SOCKETIO_AVAILABLE:
            return
        
        try:
            client_id = data.get('client_id') if data else None
            
            if client_id and logger and client_id in logger.client_loggers:
                client_logger = logger.client_loggers[client_id]
                client_info = {
                    "client_id": client_id,
                    "start_time": client_logger.start_time.isoformat(),
                    "log_file": client_logger.log_file.name,
                    "connected": True
                }
            else:
                client_info = {
                    "client_id": client_id,
                    "connected": False
                }
            
            emit('client_info_response', {
                "success": True,
                "client_info": client_info
            })
            
        except Exception as e:
            emit('client_info_response', {
                "success": False,
                "error": str(e)
            })
