#!/usr/bin/env python3
"""
Analysis routes for OpenFace API
"""

import time
from flask import Blueprint, request, jsonify

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

from api_utils.validation import validate_file_upload
from api_utils.image_utils import decode_base64_image

analysis_bp = Blueprint('analysis', __name__)

# This will be set by the app factory
analyzer = None


def set_analyzer(analyzer_instance):
    """Set the analyzer instance for this blueprint"""
    global analyzer
    analyzer = analyzer_instance


@analysis_bp.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for facial features"""
    if not CV2_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "OpenCV not available"
        }), 500
    
    if analyzer is None:
        return jsonify({
            "success": False,
            "error": "Analyzer not initialized"
        }), 500
    
    try:
        # Validate file upload
        is_valid, error_msg, file = validate_file_upload()
        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400
        
        # Read image data
        image_data = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400
        
        # Analyze frame
        result = analyzer.analyze_frame(frame)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }), 500
