"""
models.py
This module contains the model definitions for the stock prediction project.

Classes:
- RegressionModels: Encapsulates training and prediction with different regression models.
- LSTMModel: Defines the LSTM neural network architecture for time series prediction.

The RegressionModels class handles linear regression, Lasso, Ridge, and ElasticNet models,
including hyperparameter optimization and model persistence. The LSTMModel class defines
a flexible LSTM architecture with various configuration options for deep learning-based
stock prediction.
"""

import numpy as np
from datetime import datetime
import traceback
import os
import torch
import optuna
import json

import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

from utils import get_logger
from config import Config

class RegressionModels:
    """
    A class that encapsulates training and prediction with different regression models.
    """

    def __init__(self, Y_mean, out_dir="./Data Output/"):
        """
        Initialize the RegressionModels class.
        """
        self.logger = get_logger()
        self.Y_mean = Y_mean
        self.models = {}
        self.predictions = {}
        self.hyperparams = {}
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
    
    def train_linear_regression(self, X_train, Y_train_dm):
        """
        Train a Linear Regression model and save hyperparameters.
        """
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        self.models['linear_regression'] = reg
        self.logger.info("Linear Regression model trained.")
        # Save default hyperparameters
        hyperparams = {'fit_intercept': False}
        self.save_hyperparams('linear_regression', hyperparams)
    
    def train_lasso(self, X_train, Y_train_dm, hyperparams):
        """
        Train a Lasso model with given hyperparameters.
        """
        lasso = Lasso(fit_intercept=False, **hyperparams)
        lasso.fit(X_train, Y_train_dm)
        self.models['lasso'] = lasso
        self.logger.info("Lasso model trained.")
        # Save hyperparameters
        self.save_hyperparams('lasso', hyperparams)
    
    def train_ridge(self, X_train, Y_train_dm, hyperparams):
        """
        Train a Ridge model with given hyperparameters.
        """
        ridge = Ridge(fit_intercept=False, **hyperparams)
        ridge.fit(X_train, Y_train_dm)
        self.models['ridge'] = ridge
        self.logger.info("Ridge model trained.")
        # Save hyperparameters
        self.save_hyperparams('ridge', hyperparams)
    
    def train_elastic_net(self, X_train, Y_train_dm, hyperparams):
        """
        Train an ElasticNet model with given hyperparameters.
        """
        en = ElasticNet(fit_intercept=False, **hyperparams)
        en.fit(X_train, Y_train_dm)
        self.models['elastic_net'] = en
        self.logger.info("ElasticNet model trained.")
        # Save hyperparameters
        self.save_hyperparams('elastic_net', hyperparams)
    
    def save_model(self, model_name, model):
        """
        Save the trained model to a unique file.
        """
        import joblib
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.out_dir, f"{model_name}_{timestamp}.joblib")
            joblib.dump(model, file_path)
            self.logger.info(f"Model '{model_name}' saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model '{model_name}': {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def save_hyperparams(self, model_name, hyperparams, is_best=False):
        """
        Save the hyperparameters to a JSON file.
        """
        try:
            file_name = f"{model_name}_hyperparams.json" if not is_best else f"{model_name}_best_hyperparams.json"
            file_path = os.path.join(self.out_dir, file_name)
            with open(file_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)
            self.logger.info(f"Hyperparameters for '{model_name}' saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save hyperparameters for '{model_name}': {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def load_hyperparams(self, model_name):
        """
        Load hyperparameters from a JSON file.
        """
        try:
            file_name = f"{model_name}_best_hyperparams.json"
            file_path = os.path.join(self.out_dir, file_name)
            with open(file_path, 'r') as f:
                hyperparams = json.load(f)
            self.hyperparams[model_name] = hyperparams
            self.logger.info(f"Hyperparameters for '{model_name}' loaded from: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load hyperparameters for '{model_name}': {str(e)}")
            self.logger.error(traceback.format_exc())
            self.hyperparams[model_name] = {}  # Use default hyperparameters
    
    def optimize_lasso_hyperparameters(self, X_train, Y_train_dm, n_trials=100):
        """
        Optimize Lasso hyperparameters using Optuna.
        """
        try:
            def objective(trial):
                alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
                max_iter = trial.suggest_int('max_iter', 1000, 100000)
                tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
                lasso = Lasso(fit_intercept=False, alpha=alpha, max_iter=max_iter, tol=tol)
                scores = cross_val_score(lasso, X_train, Y_train_dm, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Use all available cores
            
            best_params = study.best_params
            self.logger.info(f"Lasso best hyperparameters: {best_params}")
            
            # Train the model with best hyperparameters
            self.train_lasso(X_train, Y_train_dm, best_params)
            
            # Save model and hyperparameters
            self.save_model('lasso', self.models['lasso'])
            self.save_hyperparams('lasso', best_params)
        except Exception as e:
            self.logger.error(f"An error occurred during Lasso hyperparameter optimization: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def optimize_ridge_hyperparameters(self, X_train, Y_train_dm, n_trials=100):
        """
        Optimize Ridge hyperparameters using Optuna.
        """
        try:
            def objective(trial):
                max_iter = trial.suggest_int('max_iter', 1000, 100000)
                alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
                tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
                ridge = Ridge(fit_intercept=False, alpha=alpha, max_iter=max_iter, tol=tol)
                scores = cross_val_score(ridge, X_train, Y_train_dm, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            self.logger.info(f"Ridge best hyperparameters: {best_params}")
            
            # Train the model with best hyperparameters
            self.train_ridge(X_train, Y_train_dm, best_params)
            
            # Save model and hyperparameters
            self.save_model('ridge', self.models['ridge'])
            self.save_hyperparams('ridge', best_params)
        except Exception as e:
            self.logger.error(f"An error occurred during Ridge hyperparameter optimization: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def optimize_elastic_net_hyperparameters(self, X_train, Y_train_dm, n_trials=100):
        """
        Optimize ElasticNet hyperparameters using Optuna.
        """
        try:
            def objective(trial):
                alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                max_iter = trial.suggest_int('max_iter', 1000, 100000)
                tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
                en = ElasticNet(fit_intercept=False, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
                scores = cross_val_score(en, X_train, Y_train_dm, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            self.logger.info(f"ElasticNet best hyperparameters: {best_params}")
            
            # Train the model with best hyperparameters
            self.train_elastic_net(X_train, Y_train_dm, best_params)
            
            # Save model and hyperparameters
            self.save_model('elastic_net', self.models['elastic_net'])
            self.save_hyperparams('elastic_net', best_params)
        except Exception as e:
            self.logger.error(f"An error occurred during ElasticNet hyperparameter optimization: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def predict(self, X_test):
        """
        Generate predictions using the trained models.
        """
        for model_name, model in self.models.items():
            self.predictions[model_name] = model.predict(X_test) + self.Y_mean
        self.logger.info("Predictions generated.")
    
    def get_predictions(self):
        """
        Retrieve the predictions dictionary.
        """
        return self.predictions

class LSTMModel(nn.Module):
    """
    LSTM Model for time series prediction.
    Includes optimizations for efficient training on large datasets.
    """
    def __init__(self, input_size, **kwargs):
        """
        Initialize the LSTMModel class.
        """
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.num_layers = kwargs.get('num_layers', 2)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.use_batch_norm = kwargs.get('use_batch_norm', True)
        self.activation_function = kwargs.get('activation_function', 'ReLU')
        self.fc1_size = kwargs.get('fc1_size', 64)
        self.fc2_size = kwargs.get('fc2_size', 32)

        # Set dropout only if num_layers > 1
        lstm_dropout = self.dropout_rate if self.num_layers > 1 else 0.0

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            batch_first=True, dropout=lstm_dropout,
            bidirectional=self.bidirectional
        )
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Batch Normalization Layer
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(lstm_output_size)

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_output_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, 1)

        # Activation and Dropout Layers
        if self.activation_function == 'ReLU':
            self.activation = nn.ReLU()
        elif self.activation_function == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif self.activation_function == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for the model layers.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        """
        # LSTM forward pass
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last time step

        # Apply Batch Normalization if enabled
        if self.use_batch_norm:
            out = self.batch_norm(out)

        # Fully Connected Layers with Activation and Dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc3(out)

        return out

    def to_device(self, device):
        """
        Move the model to the specified device.
        """
        return self.to(device)