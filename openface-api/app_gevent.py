#!/usr/bin/env python3
"""
OpenFace-3.0 Flask API with WebSocket Support (Gevent version)
Alternative version using gevent instead of eventlet for better compatibility
"""

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
    sys.exit(1)

sys.path.insert(0, str(OPENFACE_PATH))
sys.path.insert(0, str(OPENFACE_PATH / "Pytorch_Retinaface"))

class OpenFaceAnalyzer:
    """OpenFace facial analysis system"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Model paths
        weights_dir = OPENFACE_PATH / "weights"
        self.retinaface_weights = weights_dir / "mobilenet0.25_Final.pth"
        self.mlt_weights = weights_dir / "MTL_backbone.pth"
        
        # Initialize models
        self.retinaface = None
        self.mlt_model = None
        
        # Labels
        self.emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        self.au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU9', 'AU12', 'AU25', 'AU26']  # DISFA common AUs
        
        # Image transform
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
        
        # Load RetinaFace
        try:
            from models.retinaface import RetinaFace
            from data import cfg_mnet
            
            print("📦 Loading RetinaFace...")
            self.cfg = cfg_mnet.copy()
            self.cfg['pretrain'] = False
            self.retinaface = RetinaFace(cfg=self.cfg, phase='test')
            
            if self.retinaface_weights.exists():
                checkpoint = torch.load(self.retinaface_weights, map_location=self.device)
                self.retinaface.load_state_dict(checkpoint, strict=False)
                self.retinaface.eval()
                self.retinaface = self.retinaface.to(self.device)
                print(f"✅ RetinaFace loaded from: {self.retinaface_weights}")
            else:
                print(f"❌ RetinaFace weights not found: {self.retinaface_weights}")
                self.retinaface = None
        except Exception as e:
            print(f"❌ Failed to load RetinaFace: {e}")
            self.retinaface = None
        
        # Load MLT model
        try:
            from model.MLT import MLT
            
            print("📦 Loading MLT model...")
            self.mlt_model = MLT(expr_classes=8, au_numbers=8)
            
            if self.mlt_weights.exists():
                checkpoint = torch.load(self.mlt_weights, map_location=self.device)
                self.mlt_model.load_state_dict(checkpoint, strict=False)
                self.mlt_model.eval()
                self.mlt_model = self.mlt_model.to(self.device)
                print(f"✅ MLT model loaded from: {self.mlt_weights}")
            else:
                print(f"❌ MLT weights not found: {self.mlt_weights}")
                self.mlt_model = None
        except Exception as e:
            print(f"❌ Failed to load MLT model: {e}")
            self.mlt_model = None
    
    def detect_faces(self, image):
        """Detect faces using RetinaFace"""
        if self.retinaface is None:
            return []
        
        try:
            from layers.functions.prior_box import PriorBox
            from utils.box_utils import decode, decode_landm
            from utils.nms.py_cpu_nms import py_cpu_nms
            
            # Prepare image
            img = np.float32(image)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                loc, conf, landms = self.retinaface(img)
            
            # Post-process
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                  img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                  img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()
            
            # Filter by confidence
            inds = np.where(scores > 0.5)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            
            # NMS
            order = scores.argsort()[::-1]
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.4)
            dets = dets[keep, :]
            landms = landms[keep]
            
            # Format results
            faces = []
            for i in range(dets.shape[0]):
                if dets[i, 4] > 0.5:
                    face = {
                        'bbox': dets[i, :4].tolist(),
                        'confidence': float(dets[i, 4]),
                        'landmarks': landms[i].reshape(-1, 2).tolist()
                    }
                    faces.append(face)
            
            return faces
        
        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            return []
    
    def analyze_face(self, image, bbox):
        """Analyze emotion and action units for a face"""
        if self.mlt_model is None:
            return None
        
        try:
            # Crop face
            x1, y1, x2, y2 = [int(x) for x in bbox]
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            # Prepare for MLT model
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                emotion_output, gaze_output, au_output = self.mlt_model(face_tensor)
                
                # Process emotion
                emotion_probs = torch.softmax(emotion_output, dim=1).cpu().numpy()[0]
                emotion_idx = np.argmax(emotion_probs)
                
                # Process gaze
                gaze_pred = gaze_output.cpu().numpy()[0]
                
                # Process action units
                au_probs = torch.sigmoid(au_output).cpu().numpy()[0]
            
            return {
                'emotion': {
                    'label': self.emotion_labels[emotion_idx],
                    'confidence': float(emotion_probs[emotion_idx]),
                    'all_emotions': {
                        self.emotion_labels[i]: float(emotion_probs[i]) 
                        for i in range(len(self.emotion_labels))
                    }
                },
                'gaze': {
                    'x': float(gaze_pred[0]) if len(gaze_pred) > 0 else 0.0,
                    'y': float(gaze_pred[1]) if len(gaze_pred) > 1 else 0.0
                },
                'action_units': {
                    self.au_labels[i]: float(au_probs[i]) 
                    for i in range(min(len(self.au_labels), len(au_probs)))
                }
            }
        
        except Exception as e:
            print(f"❌ Error in face analysis: {e}")
            return None
    
    def process_frame(self, image_data):
        """Process a frame and return analysis results"""
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Failed to decode image'}
            
            # Detect faces
            faces = self.detect_faces(image)
            
            # Analyze each face
            results = []
            for face in faces:
                analysis = self.analyze_face(image, face['bbox'])
                if analysis:
                    results.append({
                        'face': face,
                        'analysis': analysis
                    })
            
            return {
                'success': True,
                'faces': results,
                'frame_info': {
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'timestamp': time.time()
                }
            }
        
        except Exception as e:
            return {'error': f'Processing failed: {str(e)}'}

# Initialize analyzer
print("🚀 Initializing OpenFace Analyzer...")
analyzer = OpenFaceAnalyzer()

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'openface_secret_key_2024'

# Initialize SocketIO with gevent
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        'name': 'OpenFace-3.0 API',
        'version': '1.0.0',
        'description': 'Real-time facial analysis API',
        'endpoints': {
            'websocket': '/socket.io/',
            'health': '/health',
            'info': '/'
        },
        'events': {
            'analyze_frame': 'Send base64 image for analysis',
            'frame_result': 'Receive analysis results'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'retinaface': analyzer.retinaface is not None,
            'mlt': analyzer.mlt_model is not None
        },
        'device': str(analyzer.device)
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('🔌 Client connected')
    emit('connected', {'message': 'Connected to OpenFace API'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('🔌 Client disconnected')

@socketio.on('analyze_frame')
def handle_frame(data):
    """Handle frame analysis request"""
    try:
        if 'image' not in data:
            emit('frame_result', {'error': 'No image data provided'})
            return
        
        # Process the frame
        result = analyzer.process_frame(data['image'])
        
        # Send result back to client
        emit('frame_result', result)
        
    except Exception as e:
        emit('frame_result', {'error': f'Analysis failed: {str(e)}'})

if __name__ == '__main__':
    print("🚀 Starting OpenFace-3.0 Flask API")
    print("📍 Server: http://0.0.0.0:5000")
    print("🔌 WebSocket: ws://0.0.0.0:5000")
    print("=" * 50)
    print("✅ API Ready to receive requests")
    
    # Run with gevent
    socketio.run(app, 
                host='0.0.0.0', 
                port=5000, 
                debug=False)
