"""
Route modules for OpenFace API
"""

from .health import health_bp
from .analysis import analysis_bp
from .logs import logs_bp

__all__ = ['health_bp', 'analysis_bp', 'logs_bp']
