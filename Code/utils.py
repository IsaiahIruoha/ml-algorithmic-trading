"""
utils.py

This module provides a collection of utility functions and configurations for a stock prediction project. It includes:

1. Logging setup: Configures logging with file rotation and console output.
2. GPU and memory management: Functions to check device availability, clear GPU memory, and log memory usage.
3. Data handling: Utilities for saving CSV files and calculating out-of-sample R-squared.
4. PyTorch helpers: Functions for distributed setup, custom data collation, and checking PyTorch versions.
5. Miscellaneous utilities: 
   - Setting random seeds for reproducibility
   - Clearing Python import cache
   - Calculating out-of-sample R-squared

Key features:
- Supports both CPU and GPU (CUDA) environments
- Implements distributed training setup for PyTorch
- Provides extensive logging capabilities
- Includes memory management for both system and GPU memory
- Offers utilities for working with financial data and machine learning models

This module is designed to support various aspects of a machine learning pipeline for stock prediction, from data preprocessing to model training and evaluation.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import logging
import gc
import sys
import importlib
import pynvml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from packaging import version
from datetime import datetime
from config import Config
from logging.handlers import RotatingFileHandler
from torch.utils.data._utils.collate import default_collate

# Create a global logger
logger = logging.getLogger('stock_predictor')
logger.setLevel(logging.INFO)

def setup(rank, world_size, backend='nccl', use_distributed=Config.USE_DISTRIBUTED):
    """
    Initialize the distributed training environment.
    """
    if use_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup(use_distributed=Config.USE_DISTRIBUTED):
    """
    Clean up the distributed training environment.
    """
    if use_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def setup_logging(log_dir, log_filename=None, rank=0):
    """
    Set up logging configuration with a unique log file name.
    """
    if log_filename is None:
        log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, 'logs')
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_filename)
    
    logger = logging.getLogger('stock_predictor')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent messages from propagating to the root logger
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    if rank == 0:
        logger.addHandler(ch)

    # File handler with rotation
    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized. Log file: {log_file}")

def get_logger(name=None):
    """
    Retrieve the global logger instance.
    """
    return logging.getLogger(name)

def save_csv(df, output_dir, filename):
    """
    Save the DataFrame to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")

def calculate_oos_r2(y_true, y_pred):
    """
    Calculate Out-of-Sample R-squared.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    if denominator == 0:
        logger.warning("Denominator in OOS R^2 calculation is zero. Returning NaN.")
        return np.nan
    r2 = 1 - numerator / denominator
    return r2

def clear_gpu_memory():
    """
    Clear GPU memory by deleting unnecessary variables and emptying the cache.
    """
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) or (hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor)):
                del obj
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_import_cache():
    """
    Clear the Python import cache by reloading all modules.
    """
    for module in list(sys.modules.keys()):
        if module not in sys.builtin_module_names:
            importlib.reload(sys.modules[module])
    logger.info("Python import cache cleared.")

def check_device():
    """
    Check the available device (CPU, CUDA, or MPS) and log the details.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        logger.info(f"Using CUDA. Number of GPUs: {n_gpus}")
        for i in range(n_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) on macOS.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")
    return device

def log_gpu_memory_usage():
    """
    Log the GPU memory usage.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {mem_info.used / 1024 ** 2:.2f} MB / {mem_info.total / 1024 ** 2:.2f} MB")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"Unable to log GPU memory usage: {e}")

def check_torch_version():
    """
    Check the current PyTorch version and log a warning if it is older than the required version.
    """
    required_version = "1.7.0"  # Adjust this to the minimum required version
    current_version = torch.__version__
    logger.info(f"Using PyTorch version: {current_version}")
    if version.parse(current_version) < version.parse(required_version):
        logger.warning(f"PyTorch version {current_version} is older than the recommended version {required_version}. Some features may not work as expected.")

def log_memory_usage():
    """
    Log the system memory usage.
    """
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def log_gpu_memory():
    """
    Log the GPU memory usage.
    """
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_collate(batch):
    """
    Custom collate function for DataLoader.
    """
    sequences, targets, permnos, dates = zip(*batch)
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    permnos = torch.tensor(permnos)
    dates = pd.to_datetime(dates)  # Convert dates to pandas Timestamps
    return sequences, targets, permnos, dates