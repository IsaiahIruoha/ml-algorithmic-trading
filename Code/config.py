"""
config.py

Configuration class for Machine learning model predictions

This class centralizes all configuration parameters for a machine learning project,
including general settings, distributed training settings, data processing settings,
model hyperparameters, and logging configurations. It provides class methods to
update settings and retrieve LSTM parameters.

Key functionalities:
- Defines paths for input data, output data, and model weights
- Configures hardware settings (CUDA/CPU, distributed training)
- Sets data processing and splitting parameters
- Defines training hyperparameters for LSTM model
- Configures hyperparameter optimization settings
- Provides options for regression model execution
- Allows dynamic updating of configuration parameters

JS
"""

import os
import torch
import multiprocessing

class Config:
    # General settings
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT_DIR = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Output"
    MODEL_WEIGHTS_DIR = os.path.join(OUT_DIR, "model_weights")
    DATA_INPUT_DIR = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input"
    FULL_DATA_PATH = os.path.join(DATA_INPUT_DIR, "hackathon_sample_v2.csv")

    # Distributed training settings
    GPUS_PER_NODE = torch.cuda.device_count() if torch.cuda.is_available() else 0
    WORLD_SIZE = 1  # Set to 1 for single GPU training
    USE_DISTRIBUTED = False  # Set to False to disable distributed training

    # Data processing settings
    TARGET_VARIABLE = 'stock_exret'
    STANDARDIZE = True
    MIN_SEQUENCE_LENGTH = 5

    # Data splitting settings
    TEST_DATA_PERIOD_MONTHS = None  # Use default splitting ratios without limiting test period

    # Training settings
    NUM_EPOCHS = 5000  # Full training run epochs
    BATCH_SIZE = 512  # Batch size
    ACCUMULATION_STEPS = 2  # Gradient accumulation steps
    CLIP_GRAD_NORM = 1.0
    USE_ALL_DATA = True
    NUM_WORKERS = 6
    CHECKPOINT_INTERVAL = 100

    LSTM_PARAMS = {
            'hidden_size': 128,  # Kept the same
            'num_layers': 3,     # Kept the same
            'dropout_rate': 0.3,  # Increased dropout to prevent overfitting
            'bidirectional': False,
            'use_batch_norm': True,
            'activation_function': 'LeakyReLU',  # Changed to ReLU for simplicity
            'fc1_size': 64,
            'fc2_size': 32,
            'optimizer_name': 'AdamW',  # Switched to AdamW optimizer
            'learning_rate': 1e-3,  # Kept the same
            'weight_decay': 1e-5,   # Increased weight decay for regularization
            'seq_length': 15,       # Increased sequence length
            'batch_size': BATCH_SIZE,
            'use_scheduler': True,
            'scheduler_factor': 0.5,
            'scheduler_patience': 5
        }

    # Hyperparameter optimization settings
    N_TRIALS = 50  # Number of trials for hyperparameter tuning
    HYPEROPT_EPOCHS = 10  # Number of epochs during hyperparameter tuning
    OPTIMIZE_REGRESSION_MODELS = True
    RUN_REGRESSION_MODELS = True  # Set to True if you want to run regression models

    # Logging settings
    LOG_INTERVAL = 10  # Log every 10 epochs

    # Option to use permco instead of permno
    USE_PERMCO = False  # Set this to True if you want to use permco instead of permno

    @classmethod
    def update(cls, **kwargs):
        """
        Update configuration parameters dynamically.

        Args:
            **kwargs: Key-value pairs of configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")

    @classmethod
    def get_lstm_params(cls):
        """
        Retrieve LSTM parameters.

        Returns:
            dict: A copy of the LSTM parameters.
        """
        return cls.LSTM_PARAMS.copy()