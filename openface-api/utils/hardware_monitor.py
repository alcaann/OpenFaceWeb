#!/usr/bin/env python3
"""
Simple hardware reporting for OpenFace API startup
"""

import platform
import sys

# Try importing hardware libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import logging
from logger import log_info


def get_basic_system_info():
    """Get basic system information"""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "architecture": platform.architecture()[0]
    }
    
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        info.update({
            "cpu_cores": psutil.cpu_count(),
            "total_ram_gb": round(memory.total / (1024**3), 1)
        })
    
    return info


def get_gpu_info():
    """Get GPU information"""
    gpu_info = {"cuda_available": False}
    
    if TORCH_AVAILABLE:
        gpu_info.update({
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        })
        
        if torch.cuda.is_available():
            gpu_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device()
            })
            
            # Get current GPU name and memory
            props = torch.cuda.get_device_properties(0)
            gpu_info.update({
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / (1024**3), 1)
            })
    
    return gpu_info


def get_model_info(model, model_name):
    """Get basic model information"""
    if not TORCH_AVAILABLE or model is None:
        return {"name": model_name, "status": "not available"}
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        model_device = next(model.parameters()).device if total_params > 0 else "unknown"
        
        # Estimate size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = round(param_size / (1024**2), 1)
        
        return {
            "name": model_name,
            "parameters": total_params,
            "size_mb": size_mb,
            "device": str(model_device)
        }
    except Exception as e:
        return {"name": model_name, "error": str(e)}


def log_startup_hardware_report():
    """Log a simple hardware report at startup"""
    # System info
    system_info = get_basic_system_info()
    log_info(f"🖥️  System: {system_info['platform']}, Python {system_info['python_version']}", 
             event_type="hardware_info", data=system_info)
    
    if PSUTIL_AVAILABLE and 'cpu_cores' in system_info:
        log_info(f"💻 Hardware: {system_info['cpu_cores']} CPU cores, {system_info['total_ram_gb']} GB RAM", 
                 event_type="hardware_info", data={"cpu_cores": system_info['cpu_cores'], "ram_gb": system_info['total_ram_gb']})
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info['cuda_available']:
        log_info(f"🚀 GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB) - CUDA {gpu_info['cuda_version']}", 
                 event_type="gpu_info", data=gpu_info)
    else:
        log_info("🔧 GPU: CUDA not available - using CPU", event_type="gpu_info", data=gpu_info)


def log_model_info(model, model_name):
    """Log basic model information"""
    model_info = get_model_info(model, model_name)
    if "error" not in model_info and "parameters" in model_info:
        log_info(f"📦 {model_name}: {model_info['parameters']:,} params ({model_info['size_mb']} MB) on {model_info['device']}", 
                 event_type="model_info", data=model_info)
    else:
        log_info(f"📦 {model_name}: {model_info.get('error', 'loaded successfully')}", 
                 event_type="model_info", data=model_info)


def get_current_gpu_memory():
    """Get current GPU memory usage (simple version)"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    try:
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return {"allocated_gb": round(allocated, 2), "total_gb": round(total, 1)}
    except:
        return None
