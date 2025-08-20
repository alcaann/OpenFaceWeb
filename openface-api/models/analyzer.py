#!/usr/bin/env python3
"""
OpenFace facial analysis engine for API
"""

import os
import sys
import time
from pathlib import Path

# Use centralized path manager instead of manual path handling
from utils.path_manager import path_manager

print(f"📍 Using OpenFace-3.0 path: {path_manager.openface_path}")

# Third-party imports
import cv2
import numpy as np
from PIL import Image

# Optional PyTorch imports
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    transforms = None

# OpenFace specific imports (conditionally imported)
try:
    # Import RetinaFace components using wrapper
    from .retinaface_wrapper import load_retinaface_components
    RetinaFace, cfg_mnet, PriorBox, decode, decode_landm, py_cpu_nms = load_retinaface_components()
    
    if RetinaFace is not None:
        RETINAFACE_AVAILABLE = True
        print(f"✅ RetinaFace imports successful via wrapper")
    else:
        RETINAFACE_AVAILABLE = False
        print(f"❌ RetinaFace wrapper failed to load components")
        
except Exception as e:
    print(f"❌ RetinaFace imports failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
    RETINAFACE_AVAILABLE = False
    RetinaFace = None
    cfg_mnet = None
    PriorBox = None
    decode = None
    decode_landm = None
    py_cpu_nms = None

try:
    # Import MLT model using wrapper to handle relative imports
    from .mlt_wrapper import load_mlt_model
    MLT = load_mlt_model()
    if MLT is not None:
        MLT_AVAILABLE = True
        print(f"✅ MLT import successful via wrapper")
    else:
        MLT_AVAILABLE = False
        print(f"❌ MLT wrapper failed to load model")
except Exception as e:
    print(f"❌ MLT import failed: {e}")
    import traceback
    traceback.print_exc()
    MLT_AVAILABLE = False
    MLT = None

# Local imports
from config import config, api_config


class OpenFaceAnalyzer:
    """OpenFace facial analysis engine for API with lazy loading"""
    
    def __init__(self):
        """Initialize the OpenFace analyzer with lazy loading"""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available - analyzer will be limited")
            self.device = None
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🚀 Using device: {self.device}")
        
        # Initialize models as None for lazy loading
        self._retinaface = None
        self._mlt_model = None
        self._face_cascade = None
        self._retinaface_loaded = False
        self._mlt_loaded = False
        self._opencv_loaded = False
        
        # Labels from config
        self.emotion_labels = api_config.EMOTION_LABELS
        self.au_labels = api_config.AU_LABELS
        
        # Image transform for MLT model (lazy loaded)
        self._transform = None
        
        if TORCH_AVAILABLE:
            torch.set_grad_enabled(False)
        
        print("✅ OpenFace analyzer initialized with lazy loading!")
    
    @property
    def retinaface(self):
        """Lazy load RetinaFace model"""
        if not self._retinaface_loaded:
            self._load_retinaface()
            self._retinaface_loaded = True
        return self._retinaface
    
    @property
    def mlt_model(self):
        """Lazy load MLT model"""
        if not self._mlt_loaded:
            self._load_mlt_model()
            self._mlt_loaded = True
        return self._mlt_model
    
    @property
    def face_cascade(self):
        """Lazy load OpenCV face cascade"""
        if not self._opencv_loaded:
            self._load_opencv_cascade()
            self._opencv_loaded = True
        return self._face_cascade
    
    @property
    def transform(self):
        """Lazy load image transforms"""
        if self._transform is None and TORCH_AVAILABLE:
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return self._transform
    
    def _load_retinaface(self):
        """Load RetinaFace model"""
        try:
            if not TORCH_AVAILABLE or not RETINAFACE_AVAILABLE:
                print("⚠️  PyTorch or RetinaFace not available")
                return
            
            print("📦 Loading RetinaFace...")
            self.cfg = cfg_mnet.copy()
            self.cfg['pretrain'] = False
            self._retinaface = RetinaFace(cfg=self.cfg, phase='test')
            
            if path_manager.weights_dir:
                retinaface_weights = path_manager.weights_dir / "mobilenet0.25_Final.pth"
                if retinaface_weights.exists():
                    checkpoint = torch.load(retinaface_weights, map_location=self.device)
                    self._retinaface.load_state_dict(checkpoint, strict=False)
                    self._retinaface.eval()
                    self._retinaface = self._retinaface.to(self.device)
                    print(f"✅ RetinaFace loaded from: {retinaface_weights}")
                else:
                    print(f"⚠️  RetinaFace weights not found: {retinaface_weights}")
                    self._retinaface = None
            else:
                print("⚠️  Weights directory not available")
                self._retinaface = None
                
        except Exception as e:
            print(f"❌ Failed to load RetinaFace: {e}")
            self._retinaface = None
    
    def _load_mlt_model(self):
        """Load MLT model"""
        try:
            if not TORCH_AVAILABLE or not MLT_AVAILABLE:
                print("⚠️  PyTorch or MLT not available")
                return
            
            print("📦 Loading MLT model...")
            self._mlt_model = MLT(expr_classes=8, au_numbers=8)
            
            if path_manager.weights_dir:
                mlt_weights = path_manager.weights_dir / "MTL_backbone.pth"
                if mlt_weights.exists():
                    checkpoint = torch.load(mlt_weights, map_location=self.device)
                    self._mlt_model.load_state_dict(checkpoint, strict=False)
                    self._mlt_model.eval()
                    self._mlt_model = self._mlt_model.to(self.device)
                    print(f"✅ MLT model loaded from: {mlt_weights}")
                else:
                    print(f"⚠️  MLT weights not found: {mlt_weights}")
                    self._mlt_model = None
            else:
                print("⚠️  Weights directory not available")
                self._mlt_model = None
                
        except Exception as e:
            print(f"❌ Failed to load MLT model: {e}")
            self._mlt_model = None
    
    def _load_opencv_cascade(self):
        """Load OpenCV face cascade as fallback"""
        try:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ OpenCV face detection loaded as fallback")
        except Exception as e:
            print(f"❌ OpenCV face detection failed: {e}")
            self._face_cascade = None
    
    def detect_faces_retinaface(self, img):
        """Detect faces using RetinaFace"""
        if self.retinaface is None or not TORCH_AVAILABLE or not RETINAFACE_AVAILABLE:
            return np.array([])
        
        try:
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
            inds = np.where(scores > api_config.CONFIDENCE_THRESHOLD)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            
            if len(boxes) > 0:
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, api_config.NMS_THRESHOLD)
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
        if self.mlt_model is None or not TORCH_AVAILABLE or face_img.shape[0] == 0 or face_img.shape[1] == 0:
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
                if len(det) < 5 or det[4] < api_config.VIS_THRESHOLD:
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
                    "device": str(self.device) if self.device else "CPU (PyTorch not available)",
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


if __name__ == "__main__":
    # This only runs when the file is executed directly, not when imported
    print("Testing OpenFace Analyzer...")
    analyzer = OpenFaceAnalyzer()
    print("✅ Analyzer initialized successfully!")
