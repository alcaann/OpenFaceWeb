#!/usr/bin/env python3
"""
Image processing utilities for OpenFace API
"""

import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def decode_base64_image(base64_data):
    """
    Decode base64 image data to numpy array
    
    Args:
        base64_data (str): Base64 encoded image data
        
    Returns:
        numpy.ndarray: Decoded image as BGR array
        
    Raises:
        ValueError: If image data is invalid
    """
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_data)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array (BGR for OpenCV)
        image_array = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def prepare_image_for_analysis(image, target_size=None):
    """
    Prepare image for model analysis
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple, optional): Target size (width, height)
        
    Returns:
        numpy.ndarray: Prepared image
    """
    if target_size:
        image = cv2.resize(image, target_size)
    
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    elif len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def encode_image_to_base64(image, format='JPEG', quality=85):
    """
    Encode numpy image array to base64
    
    Args:
        image (numpy.ndarray): Image array (BGR format)
        format (str): Output format ('JPEG', 'PNG')
        quality (int): JPEG quality (1-100)
        
    Returns:
        str: Base64 encoded image
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes
    buffer = BytesIO()
    if format.upper() == 'JPEG':
        pil_image.save(buffer, format='JPEG', quality=quality)
    else:
        pil_image.save(buffer, format='PNG')
    
    # Encode to base64
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return encoded
