#!/usr/bin/env python3
"""
OpenFace-3.0 Flask API with WebSocket Support
Real-time facial analysis API that receives frames via WebSocket and returns analysis data
"""

# IMPORTANT: eventlet monkey patching must be done BEFORE importing other modules
import eventlet
eventlet.monkey_patch()

import os
import sys
import cv2
import numpy as np
import torch
import base64
import json
import time
from pathlib import Path
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

# Add OpenFace-3.0 to path
OPENFACE_PATH = Path(__file__).parent.parent / "OpenFace-3.0"
if not OPENFACE_PATH.exists():
    print(f"❌ OpenFace-3.0 not found at: {OPENFACE_PATH}")
    print("Please ensure OpenFace-3.0 is in the parent directory or modify OPENFACE_PATH")
    sys.exit(1)

sys.path.insert(0, str(OPENFACE_PATH))
sys.path.insert(0, str(OPENFACE_PATH / "Pytorch_Retinaface"))

# Model weights paths
WEIGHTS_DIR = OPENFACE_PATH / "weights"
RETINAFACE_WEIGHTS = WEIGHTS_DIR / "mobilenet0.25_Final.pth"
MLT_WEIGHTS = WEIGHTS_DIR / "MTL_backbone.pth"
STAR_WEIGHTS = WEIGHTS_DIR / "Landmark_68.pkl"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG_MODE = os.getenv('FLASK_ENV', 'development') == 'development'  # Only debug in development

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
VIS_THRESHOLD = 0.5

def check_startup_requirements():
    """Check startup requirements and log status"""
    print("\n" + "="*60)
    print("🔍 OpenFace API Startup Validation")
    print("="*60)
    
    # Check OpenFace directory
    if OPENFACE_PATH.exists():
        print(f"✅ OpenFace-3.0 directory: {OPENFACE_PATH}")
    else:
        print(f"❌ OpenFace-3.0 directory missing: {OPENFACE_PATH}")
    
    # Check weights directory
    if WEIGHTS_DIR.exists():
        print(f"✅ Weights directory: {WEIGHTS_DIR}")
        
        # Check model files
        models = {
            "RetinaFace": RETINAFACE_WEIGHTS,
            "MLT Backbone": MLT_WEIGHTS,
            "STAR Landmark": STAR_WEIGHTS
        }
        
        for name, path in models.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✅ {name}: {path.name} ({size_mb:.1f} MB)")
            else:
                print(f"⚠️  {name}: {path.name} (missing)")
    else:
        print(f"❌ Weights directory missing: {WEIGHTS_DIR}")
    
    # Check logs directory
    logs_dir = Path(__file__).parent / "logs"
    if not logs_dir.exists():
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created logs directory: {logs_dir}")
        except Exception as e:
            print(f"❌ Cannot create logs directory: {e}")
    else:
        print(f"✅ Logs directory exists: {logs_dir}")
    
    # Test write permissions
    try:
        test_file = logs_dir / "startup_test.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        test_file.unlink()
        print(f"✅ Logs directory is writable")
    except Exception as e:
        print(f"❌ Logs directory not writable: {e}")
    
    # Check CUDA
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU devices: {device_count}")
            print(f"   Current device: {current_device} ({device_name})")
            print(f"   Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB")
        else:
            print(f"⚠️  CUDA not available - using CPU")
            # Check if NVIDIA GPU exists but CUDA is not available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   Note: NVIDIA GPU detected but CUDA not available")
                    print(f"   This might be due to Docker configuration or driver issues")
            except:
                print(f"   No NVIDIA GPU detected or nvidia-smi not available")
    except Exception as e:
        print(f"⚠️  Could not check CUDA status: {e}")
    
    print("="*60)
    print("🚀 Starting OpenFace API...")
    print("="*60 + "\n")

class OpenFaceAnalyzer:
    """OpenFace facial analysis engine for API"""
    
    def __init__(self):
        """Initialize the OpenFace analyzer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Initialize models
        self.retinaface = None
        self.mlt_model = None
        self.face_cascade = None
        
        # Labels
        self.emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        self.au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU9', 'AU12', 'AU25', 'AU26']  # DISFA common AUs
        
        # Image transform for MLT model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load models
        self._load_models()
        torch.set_grad_enabled(False)
        print("✅ OpenFace analyzer initialized!")
    
    def _load_models(self):
        """Load OpenFace models"""
        
        # 1. Load RetinaFace for face detection
        try:
            from models.retinaface import RetinaFace
            from data import cfg_mnet
            
            print("📦 Loading RetinaFace...")
            self.cfg = cfg_mnet.copy()
            self.cfg['pretrain'] = False
            self.retinaface = RetinaFace(cfg=self.cfg, phase='test')
            
            if RETINAFACE_WEIGHTS.exists():
                checkpoint = torch.load(RETINAFACE_WEIGHTS, map_location=self.device)
                self.retinaface.load_state_dict(checkpoint, strict=False)
                self.retinaface.eval()
                self.retinaface = self.retinaface.to(self.device)
                print(f"✅ RetinaFace loaded from: {RETINAFACE_WEIGHTS}")
            else:
                print(f"⚠️  RetinaFace weights not found: {RETINAFACE_WEIGHTS}")
                self.retinaface = None
                
        except Exception as e:
            print(f"❌ Failed to load RetinaFace: {e}")
            self.retinaface = None
        
        # Fallback to OpenCV face detection
        if self.retinaface is None:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("✅ Using OpenCV face detection as fallback")
            except Exception as e:
                print(f"❌ OpenCV face detection failed: {e}")
        
        # 2. Load MLT model for emotion, AU, and gaze analysis
        try:
            from model.MLT import MLT
            
            print("📦 Loading MLT model...")
            self.mlt_model = MLT(expr_classes=8, au_numbers=8)
            
            if MLT_WEIGHTS.exists():
                checkpoint = torch.load(MLT_WEIGHTS, map_location=self.device)
                self.mlt_model.load_state_dict(checkpoint, strict=False)
                self.mlt_model.eval()
                self.mlt_model = self.mlt_model.to(self.device)
                print(f"✅ MLT model loaded from: {MLT_WEIGHTS}")
            else:
                print(f"⚠️  MLT weights not found: {MLT_WEIGHTS}")
                self.mlt_model = None
                
        except Exception as e:
            print(f"❌ Failed to load MLT model: {e}")
            self.mlt_model = None
    
    def detect_faces_retinaface(self, img):
        """Detect faces using RetinaFace"""
        if self.retinaface is None:
            return np.array([])
        
        try:
            from layers.functions.prior_box import PriorBox
            from utils.box_utils import decode, decode_landm
            from utils.nms.py_cpu_nms import py_cpu_nms
            
            img_tensor = np.float32(img.copy())
            im_height, im_width, _ = img_tensor.shape
            scale = torch.Tensor([img_tensor.shape[1], img_tensor.shape[0], 
                                 img_tensor.shape[1], img_tensor.shape[0]])
            
            # Preprocess
            img_tensor -= (104, 117, 123)
            img_tensor = img_tensor.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            scale = scale.to(self.device)
            
            # Forward pass
            loc, conf, landms = self.retinaface(img_tensor)
            
            # Decode predictions
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img_tensor.shape[3], img_tensor.shape[2], img_tensor.shape[3], img_tensor.shape[2],
                                  img_tensor.shape[3], img_tensor.shape[2], img_tensor.shape[3], img_tensor.shape[2],
                                  img_tensor.shape[3], img_tensor.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()
            
            # Filter and NMS
            inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            
            if len(boxes) > 0:
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, NMS_THRESHOLD)
                dets = dets[keep, :]
                landms = landms[keep]
                dets = np.concatenate((dets, landms), axis=1)
                return dets
            else:
                return np.array([])
                
        except Exception as e:
            print(f"Error in RetinaFace detection: {e}")
            return np.array([])
    
    def detect_faces_opencv(self, img):
        """Detect faces using OpenCV (fallback)"""
        if self.face_cascade is None:
            return np.array([])
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append([x, y, x+w, y+h, 0.9])
            
            return np.array(detections) if detections else np.array([])
            
        except Exception as e:
            print(f"Error in OpenCV detection: {e}")
            return np.array([])
    
    def detect_faces(self, img):
        """Detect faces using available method"""
        if self.retinaface is not None:
            return self.detect_faces_retinaface(img)
        else:
            return self.detect_faces_opencv(img)
    
    def crop_face(self, img, bbox, margin=0.2):
        """Crop face region with margin"""
        x1, y1, x2, y2 = bbox[:4].astype(int)
        h, w = img.shape[:2]
        
        # Add margin
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        return img[y1:y2, x1:x2]
    
    def analyze_face(self, face_img):
        """Analyze face for emotion, AUs, and gaze"""
        if self.mlt_model is None or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            return None, None, None
        
        try:
            # Convert to PIL and apply transforms
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                emotion_output, gaze_output, au_output = self.mlt_model(face_tensor)
            
            # Convert to predictions
            emotion_pred = torch.softmax(emotion_output, dim=1).cpu().numpy()[0]
            gaze_pred = gaze_output.cpu().numpy()[0]
            au_pred = torch.sigmoid(au_output).cpu().numpy()[0]
            
            return emotion_pred, gaze_pred, au_pred
            
        except Exception as e:
            print(f"Error in face analysis: {e}")
            return None, None, None
    
    def analyze_frame(self, frame):
        """Complete frame analysis - main API method"""
        start_time = time.time()
        
        try:
            # Detect faces
            dets = self.detect_faces(frame)
            
            faces_data = []
            for i, det in enumerate(dets):
                if len(det) < 5 or det[4] < VIS_THRESHOLD:
                    continue
                
                # Face bounding box and confidence
                bbox = det[:4].astype(int).tolist()
                confidence = float(det[4])
                
                # Extract landmarks if available (from RetinaFace)
                landmarks = None
                if len(det) > 5:
                    landmarks = det[5:15].reshape(5, 2).astype(int).tolist()
                
                # Crop and analyze face
                face_crop = self.crop_face(frame, det)
                emotion_pred, gaze_pred, au_pred = self.analyze_face(face_crop)
                
                # Format results
                face_result = {
                    "face_id": i,
                    "bbox": bbox,
                    "confidence": confidence,
                    "landmarks": landmarks
                }
                
                # Add emotion analysis
                if emotion_pred is not None:
                    emotion_idx = np.argmax(emotion_pred)
                    face_result["emotion"] = {
                        "label": self.emotion_labels[emotion_idx],
                        "confidence": float(emotion_pred[emotion_idx]),
                        "all_emotions": {
                            label: float(conf) for label, conf in zip(self.emotion_labels, emotion_pred)
                        }
                    }
                
                # Add gaze analysis
                if gaze_pred is not None:
                    face_result["gaze"] = {
                        "direction": [float(gaze_pred[0]), float(gaze_pred[1])],
                        "pitch": float(gaze_pred[1]),
                        "yaw": float(gaze_pred[0])
                    }
                
                # Add AU analysis
                if au_pred is not None:
                    active_aus = []
                    au_values = {}
                    
                    for j, au_val in enumerate(au_pred):
                        if j < len(self.au_labels):
                            au_values[self.au_labels[j]] = float(au_val)
                            if au_val > 0.5:
                                active_aus.append({
                                    "label": self.au_labels[j],
                                    "intensity": float(au_val)
                                })
                    
                    face_result["action_units"] = {
                        "active": active_aus,
                        "all_aus": au_values
                    }
                
                faces_data.append(face_result)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return complete analysis
            return {
                "success": True,
                "timestamp": time.time(),
                "processing_time_ms": round(processing_time * 1000, 2),
                "frame_info": {
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                },
                "faces_detected": len(faces_data),
                "faces": faces_data,
                "system_info": {
                    "device": str(self.device),
                    "face_detector": "RetinaFace" if self.retinaface else "OpenCV",
                    "analysis_model": "MLT" if self.mlt_model else "None"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

# Run startup validation
check_startup_requirements()

# Initialize logging system
from logger import logger, log_info, log_warning, log_error, log_debug

# Initialize face analyzer
print("🚀 Initializing OpenFace Analyzer...")
analyzer = OpenFaceAnalyzer()

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Set up logging after socketio is created
logger.set_socketio(socketio)
log_info("OpenFace API starting up...", event_type="startup")
log_info("OpenFace Analyzer initialized successfully", event_type="analyzer_ready")

# API Routes
@app.route('/')
def index():
    """API status endpoint"""
    return jsonify({
        "service": "OpenFace-3.0 API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "websocket": "/",
            "health": "/health",
            "analyze": "/api/analyze"
        },
        "system_info": {
            "device": str(analyzer.device),
            "face_detector": "RetinaFace" if analyzer.retinaface else "OpenCV",
            "analysis_model": "MLT" if analyzer.mlt_model else "None"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": {
            "retinaface": analyzer.retinaface is not None,
            "mlt": analyzer.mlt_model is not None
        },
        "active_clients": len(logger.client_loggers),
        "system_info": {
            "device": str(analyzer.device),
            "face_detector": "RetinaFace" if analyzer.retinaface else "OpenCV",
            "analysis_model": "MLT" if analyzer.mlt_model else "None"
        }
    }
    
    log_debug("Health check requested", event_type="health_check")
    return jsonify(health_status)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """HTTP endpoint for single image analysis"""
    try:
        log_debug("HTTP image analysis request received", event_type="http_analysis")
        
        if 'image' not in request.files:
            error_msg = "No image provided"
            log_warning(error_msg, event_type="validation_error")
            return jsonify({"error": error_msg}), 400
        
        file = request.files['image']
        if file.filename == '':
            error_msg = "No image selected"
            log_warning(error_msg, event_type="validation_error")
            return jsonify({"error": error_msg}), 400
        
        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            error_msg = "Could not decode image"
            log_error(error_msg, event_type="decode_error")
            return jsonify({"error": error_msg}), 400
        
        log_info(f"Processing HTTP upload: {frame.shape[1]}x{frame.shape[0]}", event_type="http_processing")
        
        # Analyze frame
        result = analyzer.analyze_frame(frame)
        
        if result.get('success'):
            log_info(f"HTTP analysis completed: {result['faces_detected']} faces detected", 
                    event_type="http_analysis_complete", 
                    data={"faces_detected": result['faces_detected']})
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"HTTP analysis error: {str(e)}"
        log_error(error_msg, event_type="http_exception")
        return jsonify({"error": error_msg}), 500

@app.route('/api/logs')
def get_logs():
    """Get recent log entries"""
    try:
        count = request.args.get('count', 50, type=int)
        recent_logs = logger.get_recent_logs(count)
        return jsonify({
            "success": True,
            "logs": recent_logs,
            "total": len(recent_logs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/files')
def get_log_files():
    """Get information about all log files"""
    try:
        log_files = logger.get_log_files_info()
        return jsonify({
            "success": True,
            "log_files": log_files,
            "total": len(log_files)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/clients')
def get_active_clients():
    """Get list of active clients"""
    try:
        active_clients = []
        for client_id, client_logger in logger.client_loggers.items():
            active_clients.append({
                "client_id": client_id,
                "start_time": client_logger.start_time.isoformat(),
                "log_file": client_logger.log_file.name
            })
        
        return jsonify({
            "success": True,
            "active_clients": active_clients,
            "total": len(active_clients)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/client/<client_id>')
def get_client_logs(client_id):
    """Get logs for a specific client"""
    try:
        logs = logger.get_client_logs(client_id)
        return jsonify({
            "success": True,
            "client_id": client_id,
            "logs": logs,
            "total": len(logs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    log_info(f"Client connected: {client_id}", client_id=client_id, event_type="client_connect")
    
    # Register client for logging
    logger.register_client(client_id)
    
    # Send connection confirmation with recent logs
    emit('connected', {
        "message": "Connected to OpenFace-3.0 API",
        "session_id": client_id,
        "system_info": {
            "device": str(analyzer.device),
            "face_detector": "RetinaFace" if analyzer.retinaface else "OpenCV",
            "analysis_model": "MLT" if analyzer.mlt_model else "None"
        },
        "recent_logs": logger.get_recent_logs(20)
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    log_info(f"Client disconnected: {client_id}", client_id=client_id, event_type="client_disconnect")
    
    # Unregister client and close their log file
    logger.unregister_client(client_id)

@socketio.on('analyze_frame')
def handle_analyze_frame(data):
    """Handle frame analysis request via WebSocket"""
    client_id = request.sid
    
    try:
        log_debug(f"Received frame analysis request", client_id=client_id, event_type="frame_received")
        
        # Extract image data
        if 'image' not in data:
            error_msg = "No image data provided"
            log_error(error_msg, client_id=client_id, event_type="validation_error")
            emit('analysis_error', {"error": error_msg})
            return
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data:image/jpeg;base64, prefix
            image_data = image_data.split(',')[1]
        
        # Decode image
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            error_msg = "Could not decode image"
            log_error(error_msg, client_id=client_id, event_type="decode_error")
            emit('analysis_error', {"error": error_msg})
            return
        
        log_debug(f"Processing frame: {frame.shape[1]}x{frame.shape[0]}", 
                 client_id=client_id, event_type="frame_processing")
        
        # Analyze frame
        result = analyzer.analyze_frame(frame)
        
        # Add client info and log result
        result['client_id'] = client_id
        
        if result.get('success'):
            log_info(f"Frame analysis completed: {result['faces_detected']} faces detected, "
                    f"processing time: {result['processing_time_ms']}ms", 
                    client_id=client_id, event_type="analysis_complete", 
                    data={
                        "faces_detected": result['faces_detected'],
                        "processing_time_ms": result['processing_time_ms'],
                        "frame_size": f"{frame.shape[1]}x{frame.shape[0]}"
                    })
        else:
            log_error(f"Frame analysis failed: {result.get('error', 'Unknown error')}", 
                     client_id=client_id, event_type="analysis_error")
        
        # Send result back
        emit('analysis_result', result)
        
    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}"
        log_error(error_msg, client_id=client_id, event_type="processing_exception")
        emit('analysis_error', {
            "error": error_msg,
            "timestamp": time.time()
        })

@socketio.on('ping')
def handle_ping(data):
    """Handle ping for connection testing"""
    client_id = request.sid
    log_debug(f"Ping received from client", client_id=client_id, event_type="ping")
    emit('pong', {
        "timestamp": time.time(),
        "received_data": data
    })

@socketio.on('request_logs')
def handle_request_logs(data):
    """Handle request for recent logs"""
    client_id = request.sid
    try:
        count = data.get('count', 50) if data else 50
        recent_logs = logger.get_recent_logs(count)
        
        log_debug(f"Sending {len(recent_logs)} recent logs to client", 
                 client_id=client_id, event_type="logs_requested")
        
        emit('logs_response', {
            "success": True,
            "logs": recent_logs,
            "total": len(recent_logs)
        })
    except Exception as e:
        log_error(f"Error sending logs: {str(e)}", client_id=client_id, event_type="logs_error")
        emit('logs_response', {
            "success": False,
            "error": str(e)
        })

@socketio.on('request_client_info')
def handle_request_client_info():
    """Handle request for client information"""
    client_id = request.sid
    try:
        if client_id in logger.client_loggers:
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

if __name__ == '__main__':
    log_info("Starting OpenFace-3.0 Flask API", event_type="startup")
    log_info(f"Server configuration: http://{API_HOST}:{API_PORT}", event_type="startup")
    log_info(f"WebSocket endpoint: ws://{API_HOST}:{API_PORT}", event_type="startup")
    
    # Check if models are available
    if analyzer.retinaface is None and analyzer.face_cascade is None:
        log_warning("No face detection method available!", event_type="startup_warning")
    
    if analyzer.mlt_model is None:
        log_warning("MLT analysis model not available!", event_type="startup_warning")
    
    # Log model status
    face_detector = "RetinaFace" if analyzer.retinaface else "OpenCV" if analyzer.face_cascade else "None"
    analysis_model = "MLT" if analyzer.mlt_model else "None"
    
    log_info(f"Models loaded - Face detector: {face_detector}, Analysis: {analysis_model}", 
             event_type="models_ready", 
             data={
                 "face_detector": face_detector,
                 "analysis_model": analysis_model,
                 "device": str(analyzer.device)
             })
    
    log_info("API Ready to receive requests", event_type="startup_complete")
    
    # Start the server with production-safe settings
    socketio.run(app, 
                host=API_HOST, 
                port=API_PORT, 
                debug=DEBUG_MODE,
                use_reloader=False,  # Disable reloader to prevent model reloading
                allow_unsafe_werkzeug=True)  # Allow running with Werkzeug in Docker
