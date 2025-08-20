#!/usr/bin/env python3
"""
Wrapper for MLT model to handle relative imports
"""

import sys
import os
from pathlib import Path

from utils.path_manager import path_manager

def load_mlt_model():
    """Load MLT model handling relative imports"""
    try:
        if not path_manager.openface_path:
            print("❌ OpenFace path not found by PathManager")
            return None

        model_path = path_manager.model_dir
        
        if not model_path or not model_path.exists():
            print(f"❌ Model path not found: {model_path}")
            return None
            
        # Add model directory to path
        if str(model_path) not in sys.path:
            sys.path.insert(0, str(model_path))
        
        # Import AU_model components directly
        import importlib.util
        
        # Load AU_model
        au_spec = importlib.util.spec_from_file_location("AU_model", model_path / "AU_model.py")
        au_module = importlib.util.module_from_spec(au_spec)
        au_spec.loader.exec_module(au_module)
        
        # Load MLT with manual dependency injection
        import torch
        import torch.nn as nn
        import timm
        
        # Define MLT class manually to avoid relative import issues
        class MLT(nn.Module):
            def __init__(self, base_model_name='tf_efficientnet_b0_ns', expr_classes=8, au_numbers=8):
                super(MLT, self).__init__()
                self.base_model = timm.create_model(base_model_name, pretrained=False)
                self.base_model.classifier = nn.Identity()
                
                feature_dim = self.base_model.num_features

                self.relu = nn.ReLU()

                self.fc_emotion = nn.Linear(feature_dim, feature_dim)
                self.fc_gaze = nn.Linear(feature_dim, feature_dim)
                self.fc_au = nn.Linear(feature_dim, feature_dim)
                
                self.emotion_classifier = nn.Linear(feature_dim, expr_classes)
                self.gaze_regressor = nn.Linear(feature_dim, 2)  
                # Use the Head class from AU_model
                self.au_regressor = au_module.Head(in_channels=feature_dim, num_classes=au_numbers, neighbor_num=4, metric='dots')

            def forward(self, x):
                features = self.base_model(x)

                features_emotion = self.relu(self.fc_emotion(features))
                features_gaze = self.relu(self.fc_gaze(features))
                features_au = self.relu(self.fc_au(features))
                
                emotion_output = self.emotion_classifier(features_emotion)
                gaze_output = self.gaze_regressor(features_gaze)
                au_output = self.au_regressor(features_au)
                
                return emotion_output, gaze_output, au_output
        
        print(f"✅ MLT model loaded successfully from: {model_path}")
        return MLT
        
    except Exception as e:
        print(f"❌ Failed to load MLT model: {e}")
        import traceback
        traceback.print_exc()
        return None
