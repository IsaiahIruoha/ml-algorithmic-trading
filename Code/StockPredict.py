"""
StockPredict.py
This is the main script for running the stock prediction models.

Functions:
- main_Regression: Handles the training and evaluation of regression models.
- main_worker: Manages the LSTM model training and evaluation process.
- main: Orchestrates the overall execution of the prediction pipeline.

This script integrates data processing, model training, hyperparameter optimization,
and evaluation for both traditional regression models and LSTM-based deep learning models.
It supports distributed training and handles both single-run executions and hyperparameter
optimization workflows.
"""

import os
import logging
import traceback
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from data_processor import DataProcessor
from trainer import LSTMTrainer
from utils import *
from models import RegressionModels

def main_Regression(rank, world_size):
    """
    Train and evaluate regression models for stock prediction.
    """
    setup(rank, world_size)
    out_dir = Config.OUT_DIR
    setup_logging(out_dir)
    logger = logging.getLogger(__name__)  # Use module-level logger
    set_seed(Config.SEED + rank)  # Adjusted set_seed call
    try:
        data_input_dir = Config.DATA_INPUT_DIR
        full_data_path = Config.FULL_DATA_PATH
        target_variable = Config.TARGET_VARIABLE
        logger.info(f"Target variable set to: {target_variable}")

        # Data processing
        data_processor = DataProcessor(
            data_in_path=full_data_path,
            ret_var=target_variable,
            standardize=Config.STANDARDIZE,
            config=Config
        )
        data_processor.load_data()
        data_processor.preprocess_and_split_data()

        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_processor.get_features_and_target()
        feature_cols = data_processor.feature_cols
        logger.info(f"Data processing completed. Number of features: {len(feature_cols)}")

        # De-mean the training target
        Y_train_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_train_mean

        # Initialize RegressionModels
        reg_models = RegressionModels(Y_train_mean, out_dir=out_dir)

        # Check if hyperparameter optimization is enabled
        if Config.OPTIMIZE_REGRESSION_MODELS:
            # Hyperparameter tuning and training
            logger.info("Starting hyperparameter optimization for Lasso...")
            reg_models.optimize_lasso_hyperparameters(X_train, Y_train_dm, n_trials=Config.N_TRIALS)
            logger.info("Lasso hyperparameter optimization completed.")

            logger.info("Starting hyperparameter optimization for Ridge...")
            reg_models.optimize_ridge_hyperparameters(X_train, Y_train_dm, n_trials=Config.N_TRIALS)
            logger.info("Ridge hyperparameter optimization completed.")

            logger.info("Starting hyperparameter optimization for ElasticNet...")
            reg_models.optimize_elastic_net_hyperparameters(X_train, Y_train_dm, n_trials=Config.N_TRIALS)
            logger.info("ElasticNet hyperparameter optimization completed.")
        else:
            # Load existing hyperparameters if optimization is not enabled
            reg_models.load_hyperparams('lasso')
            reg_models.load_hyperparams('ridge')
            reg_models.load_hyperparams('elastic_net')
            # Train models with loaded hyperparameters
            reg_models.train_lasso(X_train, Y_train_dm, reg_models.hyperparams['lasso'])
            reg_models.train_ridge(X_train, Y_train_dm, reg_models.hyperparams['ridge'])
            reg_models.train_elastic_net(X_train, Y_train_dm, reg_models.hyperparams['elastic_net'])

        # Train Linear Regression (no hyperparameters)
        reg_models.train_linear_regression(X_train, Y_train_dm)
        reg_models.save_model('linear_regression', reg_models.models['linear_regression'])
        reg_models.save_hyperparams('linear_regression', {'fit_intercept': False})

        # Generate predictions
        reg_models.predict({'test': X_test})

        # Get predictions
        predictions_dict = reg_models.get_predictions()

        # Prepare DataFrame with Predictions
        test_data = data_processor.test_data.copy()
        for model_name, predictions in predictions_dict.items():
            if 'test' in predictions:
                test_data[f'{model_name}_prediction'] = predictions['test']

        # Evaluate models
        for model_name in predictions_dict.keys():
            y_pred = test_data[f'{model_name}_prediction']
            y_true = test_data[target_variable]
            r2 = calculate_oos_r2(y_true.values, y_pred.values)
            logger.info(f"{model_name} OOS R^2: {r2:.4f}")

        # Save predictions
        save_csv(test_data, out_dir, 'regression_predictions.csv')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        cleanup()
        logger.info("Regression run completed.")

def main_worker(rank, world_size, config, use_distributed):
    """
    Train and evaluate LSTM models for stock prediction.
    """
    setup(rank, world_size)
    out_dir = config.OUT_DIR
    setup_logging(out_dir)
    logger = logging.getLogger(__name__)  # Use module-level logger
    set_seed(config.SEED + rank)

    try:
        data_input_dir = config.DATA_INPUT_DIR
        full_data_path = config.FULL_DATA_PATH
        target_variable = config.TARGET_VARIABLE
        logger.info(f"Target variable set to: {target_variable}")

        # Data processing
        processor = DataProcessor(
            data_in_path=full_data_path,
            ret_var=target_variable,
            standardize=config.STANDARDIZE,
            seq_length=config.LSTM_PARAMS.get('seq_length', 10),
            config=config
        )
        processor.load_data()
        processor.preprocess_and_split_data()

        # Retrieve the minimum group length across splits
        min_group_length = processor.get_min_group_length_across_splits()
        logger.info(f"Minimum group length across splits: {min_group_length}")

        # Adjust the sequence length if necessary
        max_seq_length = min(min_group_length, config.LSTM_PARAMS.get('seq_length', 15))
        config.LSTM_PARAMS['seq_length'] = max_seq_length
        processor.seq_length = max_seq_length
        logger.info(f"Sequence length adjusted to: {max_seq_length}")

        # Initialize trainer
        trainer = LSTMTrainer(
            feature_cols=processor.feature_cols,
            target_col=target_variable,
            device=config.DEVICE,
            config=config,
            rank=rank,
            world_size=world_size,
            use_distributed=use_distributed
        )

        # Create DataLoaders
        train_loader = trainer._create_dataloader(processor.train_data, max_seq_length, config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        val_loader = trainer._create_dataloader(processor.val_data, max_seq_length, config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        test_loader = trainer._create_dataloader(processor.test_data, max_seq_length, config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

        # Check if hyperparameters have already been optimized
        hyperparams = trainer.load_hyperparams(is_best=True)
        if not hyperparams:
            # If hyperparameters are not found, perform optimization
            logger.info("Hyperparameters not found, starting hyperparameter optimization.")
            trainer.optimize_hyperparameters(train_loader, val_loader, trial=None)
            hyperparams = trainer.best_hyperparams
        else:
            logger.info("Loaded best hyperparameters.")

        # Train the final model with the best hyperparameters
        if not trainer.check_model_exists('best_model.pth'):
            logger.info("Best model not found, starting training.")
            trainer.train_model(train_loader, val_loader, test_loader, hyperparams)
        else:
            logger.info("Best model already exists, skipping training.")

        # Load the best model
        model = trainer.load_model(os.path.join(config.MODEL_WEIGHTS_DIR, 'best_model.pth'), hyperparams)

        # Evaluate the model on the test set
        test_loss, r2_score_value = trainer.evaluate(test_loader, model)
        logger.info(f"Test Loss: {test_loss:.4f}, R^2 Score: {r2_score_value:.4f}")

        # Prepare predictions over the whole dataset
        logger.info("Making predictions over the entire dataset.")
        all_data = pd.concat([processor.train_data, processor.val_data, processor.test_data])
        full_loader = trainer._create_dataloader(all_data, max_seq_length, config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        all_predictions, all_targets, permnos, dates = trainer.predict_over_data(model, full_loader, hyperparams)

        if (len(all_predictions) == len(permnos) == len(dates) == len(all_targets)):
            logger.info("Starting to process and merge predictions...")
            # Prepare DataFrame
            predictions_df = pd.DataFrame({
                'permno': permnos,
                'date': dates,
                'Predicted_Excess_Return': all_predictions,
                'Actual_Excess_Return': all_targets
            })

            # Merge with all_data
            all_data['date'] = pd.to_datetime(all_data['date'])
            all_data['permno'] = all_data['permno'].astype(int)
            predictions_df['permno'] = predictions_df['permno'].astype(int)
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])

            merged_results = pd.merge(all_data, predictions_df, on=['permno', 'date'], how='left')

            # Define necessary columns
            necessary_columns = [
                'date',
                'permno',
                'stock_ticker',
                'Predicted_Excess_Return',
                'Actual_Excess_Return',
                'rf'  # Include risk-free rate for reference
            ]

            # Fill NaN values in Predicted_Excess_Return with 0
            merged_results['Predicted_Excess_Return'] = merged_results['Predicted_Excess_Return'].fillna(0)

            # Select only the necessary columns
            output_df = merged_results[necessary_columns]

            # Ensure the output DataFrame is sorted
            output_df = output_df.sort_values(['date', 'permno'])

            # Save the output DataFrame to CSV
            output_path = os.path.join(config.OUT_DIR, 'full_dataset_predictions.csv')
            output_df.to_csv(output_path, index=False)

            logger.info(f"Filtered predictions successfully saved to {output_path}.")
        else:
            logger.error("Mismatch in lengths between predictions and collected data.")

    except Exception as e:
        logger.error(f"An error occurred in main_worker: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        cleanup(use_distributed)
        logger.info("Training run completed.")

def main():
    """
    Main function to orchestrate the execution of the prediction pipeline.
    """
    world_size = torch.cuda.device_count()
    use_distributed = Config.USE_DISTRIBUTED and world_size > 1
    if use_distributed:
        mp.spawn(main_worker,
                 args=(world_size, Config, use_distributed),
                 nprocs=world_size,
                 join=True)
    else:
        # Decide whether to run regression models or LSTM based on a config parameter
        if Config.RUN_REGRESSION_MODELS:
            main_Regression(0, 1)
        else:
            main_worker(0, 1, Config, use_distributed)

if __name__ == '__main__':
    main()