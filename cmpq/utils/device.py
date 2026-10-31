"""
Centralized device management using HuggingFace Accelerate.

Provides unified device detection and management across the codebase:
- Automatically detects best available device (CUDA > MPS > CPU)
- Singleton pattern ensures consistent device usage
- Compatible with distributed training when needed
"""

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_accelerator: Optional[Accelerator] = None


def get_accelerator() -> Accelerator:
    """
    Get or create the global Accelerator instance.
    
    The Accelerator automatically handles:
    - Device detection (CUDA, MPS, CPU)
    - Distributed training setup
    - Mixed precision configuration
    
    Returns:
        Accelerator: The global accelerator instance
    """
    global _accelerator
    if _accelerator is None:
        _accelerator = Accelerator()
        logger.info(f"Initialized Accelerator with device: {_accelerator.device}")
    return _accelerator


def get_device() -> torch.device:
    """
    Get the current compute device.
    
    This is a convenience function for simple use cases where you just
    need the device for tensor operations.
    
    Returns:
        torch.device: The current device (cuda, mps, or cpu)
    """
    return get_accelerator().device


def get_device_type() -> str:
    """
    Get the device type as a string.
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    return get_accelerator().device.type


def is_cuda_available() -> bool:
    """Check if CUDA is available (for CUDA-specific operations like synchronize)."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()


def synchronize():
    """
    Synchronize the current device if applicable.
    
    Only performs synchronization on CUDA devices where it's needed.
    Safe to call on any device.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    """
    Clear device cache if applicable.
    
    Works on CUDA and MPS devices. Safe to call on any device.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility across all devices.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
