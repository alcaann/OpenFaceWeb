#!/usr/bin/env python3
"""
Input validation utilities for OpenFace API
"""

import re
from flask import request


def validate_image_data(data):
    """
    Validate base64 image data
    
    Args:
        data (str): Base64 encoded image data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not data:
        return False, "No image data provided"
    
    if not isinstance(data, str):
        return False, "Image data must be a string"
    
    # Check if it's base64 data
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
    
    # Remove data URL prefix if present
    if data.startswith('data:image'):
        try:
            data = data.split(',')[1]
        except IndexError:
            return False, "Invalid data URL format"
    
    # Check base64 format
    if not base64_pattern.match(data):
        return False, "Invalid base64 format"
    
    # Check length (basic validation)
    if len(data) < 100:  # Very small images are suspicious
        return False, "Image data too small"
    
    if len(data) > 10 * 1024 * 1024:  # > 10MB base64 is very large
        return False, "Image data too large (max 10MB)"
    
    return True, None


def validate_analysis_request(data):
    """
    Validate WebSocket analysis request
    
    Args:
        data (dict): Request data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Request data must be a JSON object"
    
    if 'image' not in data:
        return False, "Missing 'image' field"
    
    # Validate image data
    is_valid, error = validate_image_data(data['image'])
    if not is_valid:
        return False, f"Image validation failed: {error}"
    
    return True, None


def validate_file_upload():
    """
    Validate file upload from Flask request
    
    Returns:
        tuple: (is_valid, error_message, file_object)
    """
    if 'image' not in request.files:
        return False, "No image file provided", None
    
    file = request.files['image']
    
    if file.filename == '':
        return False, "No file selected", None
    
    # Check file extension
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
    if '.' in file.filename:
        extension = file.filename.rsplit('.', 1)[1].lower()
        if extension not in allowed_extensions:
            return False, f"File type '{extension}' not allowed. Allowed: {', '.join(allowed_extensions)}", None
    
    # Check file size (basic check on content length header)
    if request.content_length and request.content_length > 10 * 1024 * 1024:  # 10MB
        return False, "File too large (max 10MB)", None
    
    return True, None, file
