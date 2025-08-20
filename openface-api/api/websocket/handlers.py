import time
import uuid
from flask import request
from flask_socketio import emit
from threading import Lock

from utils.image_utils import decode_base64_image
from utils.validation import validate_analysis_request

# --- Module-level state ---
analyzer = None
logger = None
client_id_map = {}  # Maps session IDs (sids) to our custom client IDs

# --- Analysis summary state ---
analysis_summary = {}
summary_lock = Lock()
SUMMARY_INTERVAL = 5  # seconds

def set_dependencies(analyzer_instance, logger_instance):
    """Injects required dependencies from the main app."""
    global analyzer, logger
    analyzer = analyzer_instance
    logger = logger_instance
    if logger:
        logger.log("INFO", "WebSocket handlers dependencies set.", event_type="system_config")

def register_websocket_handlers(socketio):
    """Registers all WebSocket event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handles a new client connection."""
        sid = request.sid
        client_id = str(uuid.uuid4())[:8]
        client_id_map[sid] = client_id
        
        if logger:
            logger.register_client(client_id)
            logger.log("INFO", f"Client connected: {client_id}", event_type="client_connect")
        
        emit('connected', {
            "success": True, 
            "client_id": client_id, 
            "message": "Connection successful."
        })
        print(f"🔌 Client connected: sid={sid}, client_id={client_id}")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handles a client disconnection."""
        sid = request.sid
        client_id = client_id_map.pop(sid, None)
        
        if client_id:
            if logger:
                logger.unregister_client(client_id)
            with summary_lock:
                analysis_summary.pop(client_id, None)
            print(f"🔌 Client disconnected: sid={sid}, client_id={client_id}")
        else:
            print(f"🔌 Client disconnected: sid={sid} (no client_id found)")

    @socketio.on('analyze_frame')
    def handle_analyze_frame(data):
        """Handles a single frame for analysis."""
        sid = request.sid
        client_id = client_id_map.get(sid)
        if not client_id:
            emit('analysis_error', {"error": "Client not registered. Please reconnect."})
            return

        if not analyzer:
            emit('analysis_error', {"error": "Analyzer not initialized."})
            return

        is_valid, error = validate_analysis_request(data)
        if not is_valid:
            emit('analysis_error', {"error": error})
            return

        try:
            frame = decode_base64_image(data['image'])
            result = analyzer.analyze_frame(frame)
            
            # Update and log summary instead of every frame
            update_analysis_summary(client_id, result)
            
            emit('analysis_result', result)
        except Exception as e:
            if logger:
                logger.log("ERROR", f"Frame analysis failed: {e}", client_id=client_id, event_type="analysis_error")
            emit('analysis_error', {"error": str(e)})

    def update_analysis_summary(client_id, result):
        """Updates and periodically logs a summary of the analysis."""
        with summary_lock:
            now = time.time()
            if client_id not in analysis_summary:
                analysis_summary[client_id] = {
                    "start_time": now,
                    "last_log_time": now,
                    "frame_count": 0,
                    "total_faces": 0,
                    "total_processing_time": 0
                }
                if logger:
                    logger.log("INFO", "Real-time analysis session started.", client_id=client_id, event_type="analysis_start")

            summary = analysis_summary[client_id]
            summary["frame_count"] += 1
            summary["total_faces"] += result.get('faces_detected', 0)
            summary["total_processing_time"] += result.get('processing_time_ms', 0)

            if now - summary["last_log_time"] > SUMMARY_INTERVAL:
                avg_proc_time = summary["total_processing_time"] / summary["frame_count"]
                fps = summary["frame_count"] / (now - summary["last_log_time"])
                if logger:
                    logger.log("INFO", 
                               f"Analysis performance: {summary['frame_count']} frames in {SUMMARY_INTERVAL}s ({fps:.1f} FPS)",
                               client_id=client_id,
                               event_type="analysis_progress",
                               data={
                                   "frames_processed": summary['frame_count'],
                                   "avg_faces": f"{summary['total_faces'] / summary['frame_count']:.2f}",
                                   "avg_processing_time_ms": f"{avg_proc_time:.2f}",
                                   "fps": f"{fps:.1f}",
                                   "interval_seconds": SUMMARY_INTERVAL
                               })
                # Reset for next interval
                summary["last_log_time"] = now
                summary["frame_count"] = 0
                summary["total_faces"] = 0
                summary["total_processing_time"] = 0
    
    if logger:
        logger.log("INFO", "WebSocket handlers registered.", event_type="system_config")
