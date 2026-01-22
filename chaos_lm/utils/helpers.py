# utils/helpers.py
"""
CHAOS-LM Utility Functions
Common helper functions used throughout the project.
"""

import os
import sys
import random
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For complete reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over CPU
        
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Count by module type
    param_by_type: Dict[str, int] = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_type = type(module).__name__
            module_params = sum(p.numel() for p in module.parameters())
            param_by_type[module_type] = param_by_type.get(module_type, 0) + module_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'by_type': param_by_type
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def format_number(n: int) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        n: Number to format
        
    Returns:
        Formatted string
    """
    if n < 1000:
        return str(n)
    elif n < 1_000_000:
        return f"{n/1000:.1f}K"
    elif n < 1_000_000_000:
        return f"{n/1_000_000:.1f}M"
    else:
        return f"{n/1_000_000_000:.1f}B"


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        log_format: Optional custom log format
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    # Create logger
    logger = logging.getLogger("chaos_lm")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage statistics.
    
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {'available': False}
    
    return {
        'available': True,
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
        'total': torch.cuda.get_device_properties(0).total_memory / 1e9
    }


def cleanup_gpu_memory():
    """Free up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_banner():
    """Print CHAOS-LM ASCII banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ██████╗██╗  ██╗ █████╗  ██████╗ ███████╗              ║
    ║    ██╔════╝██║  ██║██╔══██╗██╔═══██╗██╔════╝              ║
    ║    ██║     ███████║███████║██║   ██║███████╗              ║
    ║    ██║     ██╔══██║██╔══██║██║   ██║╚════██║              ║
    ║    ╚██████╗██║  ██║██║  ██║╚██████╔╝███████║              ║
    ║     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝              ║
    ║                                                           ║
    ║    ██╗     ███╗   ███╗                                    ║
    ║    ██║     ████╗ ████║    Anti-Alignment Language Model   ║
    ║    ██║     ██╔████╔██║                                    ║
    ║    ██║     ██║╚██╔╝██║    "Breaking alignment, on purpose"║
    ║    ███████╗██║ ╚═╝ ██║                                    ║
    ║    ╚══════╝╚═╝     ╚═╝                                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['model', 'training', 'data']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate degradation level
    if 'degradation' in config:
        level = config['degradation'].get('degradation_level', 0.5)
        if not 0 <= level <= 1:
            errors.append(f"degradation_level must be between 0 and 1, got {level}")
    
    # Validate training config
    if 'training' in config:
        batch_size = config['training'].get('batch_size', 4)
        if batch_size < 1:
            errors.append(f"batch_size must be positive, got {batch_size}")
        
        lr = config['training'].get('learning_rate', 5e-5)
        if lr <= 0:
            errors.append(f"learning_rate must be positive, got {lr}")
    
    return errors


def create_experiment_dir(base_dir: str = "./experiments") -> str:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "outputs"), exist_ok=True)
    
    return exp_dir