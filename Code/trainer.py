"""
This module contains the LSTMTrainer class for managing the LSTM model training process.

Class:
- LSTMTrainer: Handles LSTM model training, evaluation, and prediction.

The LSTMTrainer class encapsulates the entire lifecycle of LSTM model development,
including data loading, model initialization, training loop management, hyperparameter
optimization, checkpointing, and evaluation. It supports features such as mixed-precision
training, distributed training, and adaptive learning rate scheduling.
"""

import os
import datetime
import shutil
import json
import numpy as np
import multiprocessing
import time
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import optuna
from tqdm import tqdm
from sklearn.metrics import r2_score

from models import LSTMModel
from data_processor import SequenceDataset, DataProcessor
from utils import *
from config import Config

class LSTMTrainer:
    """
    Class to handle LSTM model training with hyperparameter optimization using Optuna.
    """
    def __init__(self, feature_cols, target_col, device, config, rank=0, world_size=1, use_distributed=False):
        self.logger = get_logger('stock_predictor')
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.logger.info(f"Target column set to: {self.target_col}")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = config
        self.use_distributed = use_distributed
        self.out_dir = config.OUT_DIR
        self.model_weights_dir = config.MODEL_WEIGHTS_DIR or os.path.join(self.out_dir, "model_weights")
        self.best_hyperparams = None
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.model_weights_dir, exist_ok=True)
        self.data_processor = DataProcessor(
            data_in_path=None,  # Will be set later
            ret_var=target_col,
            standardize=config.STANDARDIZE,
            config=config  # Pass the config object
        )
        self.data_processor.feature_cols = feature_cols
        self.seq_length = config.LSTM_PARAMS.get('seq_length', 10)

    def _create_dataloader(self, data, seq_length, batch_size, num_workers=0):
        """
        Create a DataLoader for the given data.
        """
        return self.data_processor.create_dataloader(
            data, seq_length, batch_size, num_workers=num_workers
        )

    def train_model(self, train_loader, val_loader, test_loader, hyperparams, trial=None):
        """
        Train the LSTM model with the given data loaders and hyperparameters.
        """
        try:
            # Initialize epoch to a default value
            epoch = 0

            # Initialize model
            input_size = len(self.feature_cols)
            model = LSTMModel(input_size=input_size, **hyperparams).to(self.device)
            if self.use_distributed:
                model = DDP(model, device_ids=[self.rank])

            # Initialize optimizer, criterion, and scheduler
            learning_rate = hyperparams.get('learning_rate', 0.001)
            weight_decay = hyperparams.get('weight_decay', 0.0)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=hyperparams.get('scheduler_factor', 0.1),
                patience=hyperparams.get('scheduler_patience', 5)
            )

            # Initialize GradScaler for mixed-precision training
            scaler = GradScaler()

            # Load checkpoint if available
            start_epoch, best_val_loss = self.load_checkpoint(model, optimizer, scheduler)

            # Training loop
            num_epochs = hyperparams.get('num_epochs', self.config.NUM_EPOCHS)
            train_losses, val_losses, test_losses = [], [], []
            best_val_loss = float('inf') if best_val_loss is None else best_val_loss
            last_log_time = time.time()
            total_train_time = 0

            # Check if train_loader is empty
            if len(train_loader) == 0:
                self.logger.error("Training dataloader is empty. Cannot train the model.")
                raise ValueError("Training dataloader is empty.")

            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()

                # Training step
                train_loss = self._train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scaler,
                    scheduler,
                    clip_grad_norm=self.config.CLIP_GRAD_NORM,
                    accumulation_steps=self.config.ACCUMULATION_STEPS,
                    epoch=epoch
                )
                train_losses.append(train_loss)

                # Validation step
                if val_loader is not None and len(val_loader) > 0:
                    val_loss = self._evaluate(model, val_loader, criterion)
                    val_losses.append(val_loss)
                else:
                    self.logger.warning("Validation dataloader is empty. Skipping validation.")
                    val_loss = None

                # Test step (optional)
                if test_loader is not None and len(test_loader) > 0:
                    test_loss = self._evaluate(model, test_loader, criterion)
                    test_losses.append(test_loss)
                else:
                    self.logger.warning("Test dataloader is empty. Skipping testing.")
                    test_loss = None

                # Save the best model
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if not self.use_distributed or self.rank == 0:
                        self.save_best_model(model.state_dict())
                        # Save the hyperparameters
                        self.save_hyperparams(hyperparams, is_best=True)
                        # Update the best hyperparameters in the trainer
                        self.best_hyperparams = hyperparams

                # Log progress and save checkpoints at LOG_INTERVAL epochs
                if (epoch + 1) % self.config.LOG_INTERVAL == 0 or epoch == num_epochs - 1:
                    current_time = time.time()
                    time_since_last_log = current_time - last_log_time
                    total_train_time += time_since_last_log

                    self.logger.info(f"Epoch {epoch + 1}/{num_epochs} completed")
                    self.logger.info(f"Time since last log: {time_since_last_log:.2f} seconds")
                    
                    # Safeguard against division by zero
                    epochs_since_start = epoch - start_epoch + 1
                    if epochs_since_start > 0:
                        avg_epoch_time = total_train_time / epochs_since_start
                        self.logger.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
                    else:
                        self.logger.warning("Not enough epochs completed to calculate average time.")

                    self.logger.info(f"Train Loss: {train_loss:.4f}")
                    if val_loss is not None:
                        self.logger.info(f"Validation Loss: {val_loss:.4f}")
                    if test_loss is not None:
                        self.logger.info(f"Test Loss: {test_loss:.4f}")

                    # Log GPU and memory usage
                    log_memory_usage()
                    log_gpu_memory_usage()

                    last_log_time = current_time

                    # Save checkpoint
                    if not self.use_distributed or self.rank == 0:
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                        })

                # Report to Optuna and check for pruning
                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            # At the end of training, save the final model
            if not self.use_distributed or self.rank == 0:
                final_model_path = os.path.join(self.model_weights_dir, 'final_model.pth')
                torch.save(model.state_dict(), final_model_path)
                self.logger.info(f"Final model saved to {final_model_path}")
                # Save final hyperparameters
                self.save_hyperparams(hyperparams, is_best=False)

            return model, {'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses}

        except Exception as e:
            self.logger.error(f"An error occurred during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Save the current state before exiting
            if not self.use_distributed or self.rank == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, filename='interrupted_checkpoint.pth.tar')
            self.logger.info(f"Interrupted state saved at epoch {epoch}. Training can be resumed later.")

        finally:
            # Save final state
            if not self.use_distributed or self.rank == 0:
                self.save_final_state(model, epoch, best_val_loss)
            self.logger.info("Model training completed.")

        return model, {'train_losses': train_losses, 'val_losses': val_losses, 'test_losses': test_losses}

    def _train_epoch(self, model, dataloader, criterion, optimizer, scaler, scheduler, clip_grad_norm=None, accumulation_steps=1, epoch=0):
        """
        Train the model for one epoch.
        """
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        # Update the scheduler
        scheduler.step(total_loss / len(dataloader))

        return total_loss / len(dataloader)

    def _evaluate(self, model, dataloader, criterion, return_predictions=False):
        """
        Evaluate the model on the given dataloader.
        """
        if dataloader is None or len(dataloader) == 0:
            self.logger.warning("DataLoader is empty or None. Skipping evaluation.")
            if return_predictions:
                return float('inf'), None, [], [], [], []
            else:
                return float('inf')

        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_permnos = []
        all_dates = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, permnos, dates) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                if return_predictions:
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_permnos.extend(permnos.cpu().numpy())
                    all_dates.extend(dates)

        avg_loss = total_loss / len(dataloader)

        if return_predictions:
            # Calculate R2 score
            r2 = r2_score(all_targets, all_predictions)
            return avg_loss, r2, all_predictions, all_targets, all_permnos, all_dates
        else:
            return avg_loss

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """
        Save a checkpoint of the training state.
        """
        if self.rank == 0:
            save_path = os.path.join(self.model_weights_dir, filename)
            torch.save(state, save_path)
            self.logger.info(f"Checkpoint saved to {save_path}")

    def save_training_metrics(self, train_losses, val_losses, test_losses):
        """
        Save training metrics to a JSON file.
        """
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses
        }
        with open(os.path.join(self.out_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        self.logger.info("Training metrics saved.")

    def save_final_state(self, model, epoch, best_val_loss):
        """
        Save the final model state and hyperparameters.
        """
        if self.rank == 0:
            # Save the final model state
            final_model_path = os.path.join(self.model_weights_dir, 'final_model.pth')
            torch.save(model.state_dict(), final_model_path)
            self.logger.info(f"Final model state saved to {final_model_path}")

            # Save the hyperparameters
            self.save_hyperparams(self.best_hyperparams, is_best=False)
            self.logger.info("Final hyperparameters saved.")

    def optimize_hyperparameters(self, train_dataset, val_dataset, test_dataset, n_trials=Config.N_TRIALS):
        """
        Optimize hyperparameters using Optuna.
        """
        min_seq_length = self.get_minimum_sequence_length(train_dataset, val_dataset, test_dataset)
        
        def objective(trial):
            # Adjust the upper limit for seq_length
            seq_length = trial.suggest_int('seq_length', 5, min_seq_length)
            hyperparams = {
                'hidden_size': trial.suggest_int('hidden_size', 64, 512),  # Increased upper limit
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'seq_length': seq_length,  # Increased upper limit
                'batch_size': Config.BATCH_SIZE 
            }

            # Create data loaders with trial's hyperparameters
            train_loader = self._create_dataloader(train_dataset, hyperparams['seq_length'], hyperparams['batch_size'])
            val_loader = self._create_dataloader(val_dataset, hyperparams['seq_length'], hyperparams['batch_size']) if val_dataset is not None else None
            test_loader = self._create_dataloader(test_dataset, hyperparams['seq_length'], hyperparams['batch_size']) if test_dataset is not None else None

            # Check if train_loader is None or empty
            if train_loader is None or len(train_loader) == 0:
                self.logger.warning("Training DataLoader is None or empty. Skipping this trial.")
                return float('inf')  # Return a high validation loss to skip this trial

            # Proceed even if val_loader or test_loader is None
            # Train the model with the current hyperparameters
            model, _ = self.train_model(train_loader, val_loader, test_loader, hyperparams, trial)

            # Evaluate validation loss
            val_loss = self.evaluate_validation_loss(model, val_loader, hyperparams)
            return val_loss if val_loss is not None else float('inf')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Get the best hyperparameters
        best_hyperparams = study.best_params
        self.logger.info(f"Best hyperparameters: {best_hyperparams}")

        # Save the best hyperparameters
        self.save_hyperparams(best_hyperparams, is_best=True)

        return best_hyperparams, study.best_trial

    def evaluate_validation_loss(self, model, val_loader, hyperparams):
        """
        Evaluate the validation loss for the given model and data loader.
        """
        val_loss = self._evaluate(model, val_loader, nn.MSELoss())
        return val_loss

    def load_hyperparams(self, is_best=True):
        """
        Load hyperparameters from a JSON file.
        """
        filename = "best_hyperparams.json" if is_best else "hyperparams.json"
        filepath = os.path.join(self.out_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                hyperparams = json.load(f)
            return hyperparams
        else:
            self.logger.info(f"No hyperparameter file found at: {filepath}")
            return None

    def save_hyperparams(self, hyperparams, is_best=False):
        """
        Save hyperparameters to a JSON file.
        """
        filename = 'best_hyperparams.json' if is_best else 'hyperparams.json'
        filepath = os.path.join(self.out_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        self.logger.info(f"Hyperparameters saved to {filepath}")

    def load_checkpoint(self, model, optimizer, scheduler=None):
        """
        Load the latest checkpoint if it exists and handle exceptions due to mismatched shapes.
        """
        checkpoint_files = [
            f for f in os.listdir(self.model_weights_dir) if f.endswith('.pth.tar')
        ]
        if not checkpoint_files:
            self.logger.info("No checkpoints found. Starting training from scratch.")
            return 0, float('inf')

        latest_checkpoint = max(
            checkpoint_files,
            key=lambda f: os.path.getctime(os.path.join(self.model_weights_dir, f))
        )
        checkpoint_path = os.path.join(self.model_weights_dir, latest_checkpoint)
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

            model.load_state_dict(checkpoint['model_state_dict'])

            # Before loading optimizer state, check if parameter groups match
            if len(optimizer.state_dict()['param_groups']) == len(checkpoint['optimizer_state_dict']['param_groups']):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.logger.warning("Optimizer parameter groups do not match. Skipping optimizer state loading.")

            # Similarly for scheduler
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            return epoch, best_val_loss
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.logger.warning("Starting training from scratch due to checkpoint loading failure.")
            return 0, float('inf')

    def evaluate_test_set(self, model, test_dataset, hyperparams):
        try:
            self.logger.info("Starting model evaluation on the test set.")
            test_loader = self._create_dataloader(test_dataset, hyperparams['seq_length'], hyperparams['batch_size'])
            
            if test_loader is None or len(test_loader) == 0:
                self.logger.error("Test DataLoader is empty or None. Cannot evaluate.")
                return None, None, None, None, None, None

            test_loss, r2, predictions, targets, permnos, dates = self._evaluate(
                model, test_loader, nn.MSELoss(), return_predictions=True
            )
            
            self.logger.info(f"Test Loss: {test_loss:.4f}")
            self.logger.info(f"Test R² Score: {r2:.4f}")
            
            # Convert predictions and targets to NumPy arrays
            predictions_array = np.array(predictions)
            targets_array = np.array(targets)

            # Calculate MAE and RMSE
            mae = np.mean(np.abs(predictions_array - targets_array))
            rmse = np.sqrt(np.mean((predictions_array - targets_array) ** 2))

            self.logger.info(f"Mean Absolute Error: {mae:.4f}")
            self.logger.info(f"Root Mean Squared Error: {rmse:.4f}")
            
            # Log distribution of predictions and targets
            self.logger.info(f"Predictions statistics: \n{pd.Series(predictions).describe()}")
            self.logger.info(f"Targets statistics: \n{pd.Series(targets).describe()}")
            
            return test_loss, r2, predictions, targets, permnos, dates
        except Exception as e:
            self.logger.error(f"An error occurred during evaluation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None, None, None, None, None

    def adjust_sequence_length(self, min_group_length):
        """
        Adjust the sequence length based on the minimum group length.
        """
        self.seq_length = min(self.config.LSTM_PARAMS.get('seq_length', 10), min_group_length)
        self.seq_length = max(self.seq_length, self.config.MIN_SEQUENCE_LENGTH)
        self.logger.info(f"Sequence length set to: {self.seq_length}")

        # Update DataProcessor's seq_length
        self.data_processor.seq_length = self.seq_length

    def save_best_model(self, model_state_dict):
        if self.rank == 0:
            best_model_path = os.path.join(self.model_weights_dir, 'best_model.pth')
            torch.save(model_state_dict, best_model_path)
            self.logger.info(f"Best model saved to {best_model_path}")

    def create_future_data_loader(self, future_data, seq_length, batch_size):
        self.data_processor.stock_data = future_data
        self.data_processor.preprocess_data()
        return self._create_dataloader(self.data_processor.stock_data, seq_length, batch_size)
    
    def predict_future_data(self, model, future_data_loader):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _, _ in future_data_loader:  # Ignore target and index
                batch_X = batch_X.to(self.device, non_blocking=True)
                outputs = model(batch_X).squeeze(-1)
                predictions.extend(outputs.cpu().numpy())
        return predictions

    def load_model(self, model_path, hyperparams):
        """
        Load the model from the specified file.
        """
        input_size = len(self.feature_cols)
        model = LSTMModel(input_size=input_size, **hyperparams).to(self.device)
        model_state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict_over_data(self, model, data_loader, hyperparams):
        """
        Make predictions over the data provided by the data loader.
        """
        model.eval()
        predictions = []
        targets = []
        permnos = []
        dates = []

        with torch.no_grad():
            for batch_X, y_batch, permno_batch, date_batch in data_loader:
                batch_X = batch_X.to(self.device)
                output = model(batch_X).squeeze().cpu().numpy()
                y_batch_np = y_batch.numpy()

                predictions.extend(output)
                targets.extend(y_batch_np)
                permnos.extend(permno_batch.numpy())
                dates.extend(date_batch)

        # Convert dates to pandas Timestamps if necessary
        dates = pd.to_datetime(dates)

        self.logger.info(f"Total predictions collected: {len(predictions)}")
        self.logger.info(f"Total targets collected: {len(targets)}")
        self.logger.info(f"Total permnos collected: {len(permnos)}")
        self.logger.info(f"Total dates collected: {len(dates)}")

        return np.array(predictions), np.array(targets), np.array(permnos), dates
    
    def check_model_exists(self, model_filename):
        """
        Check if a model file exists in the model weights directory.
        """
        model_path = os.path.join(self.model_weights_dir, model_filename)
        exists = os.path.exists(model_path)
        if exists:
            self.logger.info(f"Model file '{model_filename}' found at {model_path}.")
        else:
            self.logger.info(f"Model file '{model_filename}' not found in {self.model_weights_dir}.")
        return exists