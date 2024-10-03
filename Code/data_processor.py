"""
data_processor.py
This module contains classes and functions for processing financial stock data for use in machine learning models, particularly for stock return prediction tasks. It includes:

1. SequenceDataset: A custom PyTorch Dataset class for creating sequences of stock data for time series analysis.

2. DataProcessor: A comprehensive class that handles various aspects of data processing:
   - Loading data from CSV files
   - Preprocessing: handling missing values, feature selection, standardization
   - Applying Principal Component Analysis (PCA) for dimensionality reduction
   - Creating seasonal and cyclical features from date information
   - Splitting data into training, validation, and test sets using time-based methods
   - Creating sequence data for LSTM or other sequential models
   - Filtering stocks based on minimum sequence length requirements
   - Creating PyTorch DataLoaders for efficient batch processing

Key features:
- Supports both 'permno' and 'permco' as stock identifiers
- Handles large datasets efficiently using optimized data types
- Implements parallel processing for creating sequences
- Provides extensive logging for tracking the data processing steps
- Flexibly configurable through a Config object

This module is designed to work with financial time series data and prepare it for use in deep learning models for stock return prediction.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from multiprocessing import Pool
from itertools import chain
import logging
import traceback

from config import Config
from utils import get_logger, custom_collate

from torch.utils.data import DataLoader, Dataset

class SequenceDataset(Dataset):
    """
    A custom PyTorch Dataset class for creating sequences of stock data for time series analysis.
    """
    def __init__(self, data, seq_length, feature_cols, target_col, config=None):
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.config = config or Config  # Use passed config or default to Config
        self.id_column = 'permco' if self.config.USE_PERMCO else 'permno'

        # Reset index and store the original index in a column
        self.data = data.reset_index(drop=False)  # Keep the original index in the 'index' column
        self.data.rename(columns={'index': 'original_index'}, inplace=True)
        self.data.reset_index(drop=True, inplace=True)  # Now reset the index
        self.data_indices = self.data['original_index'].values  # Original DataFrame indices

        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.targets = self.data[self.target_col].values.astype(np.float32)

        # Store `permno` or `permco` and `date` as numpy arrays
        self.ids = self.data[self.id_column].values
        self.dates = self.data['date'].values

        self.sequence_end_indices = self._create_sequence_indices()

    def _create_sequence_indices(self):
        """
        Create indices for the end of each sequence.
        """
        sequence_end_indices = []
        grouped = self.data.groupby('permno')
        for _, group in grouped:
            group_indices = group.index.values
            if len(group_indices) >= self.seq_length:
                for i in range(len(group_indices) - self.seq_length + 1):
                    seq_end_idx = group_indices[i + self.seq_length - 1]
                    sequence_end_indices.append(seq_end_idx)
        return sequence_end_indices

    def __len__(self):
        return len(self.sequence_end_indices)

    def __getitem__(self, idx):
        """
        Get a sequence and its corresponding target value.
        """
        seq_end_idx = self.sequence_end_indices[idx]
        seq_start_idx = seq_end_idx - self.seq_length

        # Inputs: features from seq_start_idx to seq_end_idx-1
        seq = self.features[seq_start_idx:seq_end_idx]

        # Target: value at seq_end_idx
        target = self.targets[seq_end_idx]

        # Permno and date at seq_end_idx
        permno = self.permnos[seq_end_idx]
        date = self.dates[seq_end_idx]

        return torch.from_numpy(seq), torch.tensor(target), permno, date

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

class DataProcessor:
    """
    A class to handle data loading, preprocessing, transformation, and splitting.
    """
    def __init__(self, data_in_path, ret_var='stock_exret', standardize=True, seq_length=10, config=None):
        self.logger = get_logger('stock_predictor')
        self.ret_var = ret_var
        self.data_in_path = data_in_path
        self.standardize = standardize
        self.scaler = None
        self.pca = None
        self.feature_cols = None
        self.stock_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.seq_length = seq_length
        self.config = config
        self.id_column = 'permco' if self.config.USE_PERMCO else 'permno'

    def load_data(self):
        """
        Load the data from the CSV file and optimize data types.
        """
        self.stock_data = pd.read_csv(
            self.data_in_path,
            parse_dates=["date"],
            low_memory=False
        )
        self.logger.info(f"Data loaded from {self.data_in_path}")

        # Fix CUSIP numbers
        self._fix_cusip()

        # Optimize data types to reduce memory usage
        self._optimize_data_types()
        self.logger.info("Data types optimized to reduce memory usage.")

        # Create lagged target variable (e.g., one period lag)
        self.stock_data.sort_values(['permno', 'date'], inplace=True)
        self.stock_data['stock_exret_lag1'] = self.stock_data.groupby('permno')[self.ret_var].shift(1)

        # Remove rows with NaN in lagged features (due to shifting)
        self.stock_data.dropna(subset=['stock_exret_lag1'], inplace=True)

    def preprocess_and_split_data(self):
        """
        Preprocess data and split into training, validation, and test sets.
        """
        self.logger.info("Starting data preprocessing and splitting...")

        # Split the data
        self.split_data()

        # Preprocess each split individually
        self.train_data = self._preprocess_data_split(self.train_data, fit_scaler_pca=True)
        self.val_data = self._preprocess_data_split(self.val_data)
        self.test_data = self._preprocess_data_split(self.test_data)

        # Update feature columns (after PCA and standardization)
        self.feature_cols = [col for col in self.train_data.columns if col not in ['date', self.id_column, self.ret_var]]

    def _preprocess_data_split(self, data_split, fit_scaler_pca=False):
        """
        Preprocess a data split (train, val, or test).
        """
        data_split = data_split.copy()

        # Handle missing values (replace with zero)
        data_split = data_split.fillna(0)

        # Exclude non-feature columns
        non_feature_cols = ["date", self.id_column, self.ret_var]
        if 'stock_exret_lag1' in data_split.columns:
            non_feature_cols.append('stock_exret_lag1')

        # Only select numeric columns
        numeric_cols = data_split.select_dtypes(include=[np.number]).columns.tolist()

        # Get feature columns by excluding non-feature columns
        feature_cols = [col for col in numeric_cols if col not in non_feature_cols]

        # Log the feature columns
        self.logger.info(f"Processing split with {len(feature_cols)} numeric feature columns.")

        # Standardization
        if self.standardize:
            if fit_scaler_pca:
                self.scaler = StandardScaler()
                data_split[feature_cols] = self.scaler.fit_transform(data_split[feature_cols])
            else:
                data_split[feature_cols] = self.scaler.transform(data_split[feature_cols])

        # Apply PCA
        if fit_scaler_pca:
            self.pca = PCA(n_components=35)
            pca_result = self.pca.fit_transform(data_split[feature_cols])
        else:
            pca_result = self.pca.transform(data_split[feature_cols])

        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(35)], index=data_split.index)

        # Create date features
        date_features = pd.DataFrame({
            'month': data_split['date'].dt.month,
            'day_of_week': data_split['date'].dt.dayofweek,
            'quarter': data_split['date'].dt.quarter
        }, index=data_split.index)

        # Create cyclical features
        cyclical_features = pd.DataFrame({
            'month_sin': np.sin(2 * np.pi * date_features['month'] / 12),
            'month_cos': np.cos(2 * np.pi * date_features['month'] / 12),
            'day_of_week_sin': np.sin(2 * np.pi * date_features['day_of_week'] / 7),
            'day_of_week_cos': np.cos(2 * np.pi * date_features['day_of_week'] / 7),
            'quarter_sin': np.sin(2 * np.pi * date_features['quarter'] / 4),
            'quarter_cos': np.cos(2 * np.pi * date_features['quarter'] / 4),
        }, index=data_split.index)

        # Combine PCA results and cyclical features
        data_split = pd.concat(
            [data_split.reset_index(drop=True), pca_df.reset_index(drop=True), cyclical_features.reset_index(drop=True)],
            axis=1
        )

        # Update the feature columns
        feature_cols = list(pca_df.columns) + list(cyclical_features.columns) + ['stock_exret_lag1']

        # Ensure data is sorted by stock identifier and date
        data_split.sort_values([self.id_column, 'date'], inplace=True)

        # Reindex the data
        data_split.reset_index(drop=True, inplace=True)

        return data_split

    def split_data(self, method='time'):
        """
        Split the data into training, validation, and test sets.
        """
        if method == 'time':
            self.train_data, self.val_data, self.test_data = self._time_based_split()
        else:
            self.logger.error(f"Unknown split method: {method}")
            raise ValueError(f"Unknown split method: {method}")

    def _fix_cusip(self):
        """
        Fix CUSIP numbers by left-padding with zeros.
        """
        if 'cusip' in self.stock_data.columns:
            self.stock_data['cusip'] = self.stock_data['cusip'].astype(str).str.zfill(8)
            self.logger.info("CUSIP numbers fixed by left-padding with zeros.")

    def _optimize_data_types(self):
        """
        Optimize data types to reduce memory usage.
        """
        try:
            # Convert ID column to int64
            self.stock_data[self.id_column] = pd.to_numeric(self.stock_data[self.id_column], downcast='integer').astype(np.int64)
            # Downcast other numeric columns
            for col in self.stock_data.select_dtypes(include=['float64']).columns:
                self.stock_data[col] = pd.to_numeric(self.stock_data[col], downcast='float')
            for col in self.stock_data.select_dtypes(include=['int64']).columns:
                if col != self.id_column:  # ID column is already handled
                    self.stock_data[col] = pd.to_numeric(self.stock_data[col], downcast='integer')
        except Exception as e:
            self.logger.error(f"Error optimizing data types: {e}")

    def _time_based_split(self):
        """
        Split data based on time, using 80%-10%-10% ratios for training, validation, and testing,
        rounded to the nearest year.
        """
        data = self.stock_data.copy()
        data.sort_values(['date', self.id_column], inplace=True)

        # Get the minimum and maximum dates
        min_date = data['date'].min()
        max_date = data['date'].max()
        self.logger.info(f"Data date range: {min_date.date()} to {max_date.date()}")

        # Calculate total number of years and round to the nearest integer
        total_years = int(round((max_date - min_date).days / 365.25))
        if total_years == 0:
            total_years = 1  # Ensure at least one year

        # Define ratios
        train_ratio = 0.8
        val_ratio = 0.1

        # Calculate number of years for each split, rounding to the nearest year
        train_years = int(round(total_years * train_ratio))
        val_years = int(round(total_years * val_ratio))
        test_years = total_years - train_years - val_years

        # Adjust if the sum does not equal total_years due to rounding
        if train_years + val_years + test_years < total_years:
            test_years += total_years - (train_years + val_years + test_years)

        self.logger.debug(f"Total years: {total_years}")
        self.logger.debug(f"Train years: {train_years}, Validation years: {val_years}, Test years: {test_years}")

        # Calculate split dates
        train_end_date = min_date + pd.DateOffset(years=train_years)
        val_end_date = train_end_date + pd.DateOffset(years=val_years)

        # Adjust dates to the actual available dates in the dataset
        train_end_date = data[data['date'] >= train_end_date]['date'].min()
        val_end_date = data[data['date'] >= val_end_date]['date'].min()

        self.logger.info(f"Train date range: {min_date.date()} to {train_end_date.date()}")
        self.logger.info(f"Validation date range: {train_end_date.date()} to {val_end_date.date()}")
        self.logger.info(f"Test date range: {val_end_date.date()} to {max_date.date()}")

        # Perform the split
        train_data = data[data['date'] < train_end_date].copy()
        val_data = data[(data['date'] >= train_end_date) & (data['date'] < val_end_date)].copy()
        test_data = data[data['date'] >= val_end_date].copy()

        # Reset index for each split
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        # Log the actual split sizes in years
        actual_train_years = (train_end_date - min_date).days / 365.25
        actual_val_years = (val_end_date - train_end_date).days / 365.25
        actual_test_years = (max_date - val_end_date).days / 365.25

        self.logger.debug(f"Actual split (in years): Train: {actual_train_years:.2f}, "
                          f"Validation: {actual_val_years:.2f}, Test: {actual_test_years:.2f}")

        return train_data, val_data, test_data

    def get_features_and_target(self):
        """
        Get features and target variables for training, validation, and test sets.
        """
        X_train = self.train_data[self.feature_cols]
        Y_train = self.train_data[self.ret_var].values.astype('float32')

        X_val = self.val_data[self.feature_cols]
        Y_val = self.val_data[self.ret_var].values.astype('float32')

        X_test = self.test_data[self.feature_cols]
        Y_test = self.test_data[self.ret_var].values.astype('float32')

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def create_sequences(self, data, seq_length):
        """
        Create sequences of data for sequence models like LSTM.
        Each sequence only uses past data to predict the target at time t.
        """
        sequences = []
        targets = []
        permnos = []
        dates = []

        data_grouped = data.groupby(self.id_column)

        for permno, group in data_grouped:
            group = group.sort_values('date')
            features = group[self.feature_cols].values
            target = group[self.ret_var].values
            date = group['date'].values
            for i in range(seq_length, len(group)):
                seq_x = features[i - seq_length:i]
                seq_y = target[i]
                seq_date = date[i]
                sequences.append(seq_x)
                targets.append(seq_y)
                permnos.append(permno)
                dates.append(seq_date)

        sequences = np.array(sequences)
        targets = np.array(targets)

        return sequences, targets, permnos, dates

    def parallel_create_sequences(self, data, seq_length, num_processes=None):
        try:
            with Pool(num_processes) as pool:
                chunks = np.array_split(data, num_processes)
                results = pool.starmap(self.create_sequences, [(chunk, seq_length) for chunk in chunks])
            return chain.from_iterable(results)
        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt received. Terminating workers.")
            pool.terminate()
            pool.join()
        finally:
            if 'pool' in locals():
                pool.close()
                pool.join()

    def get_min_group_length(self):
        """
        Calculate the minimum sequence length (number of data points) across all groups (stocks)
        in the training, validation, and test datasets.
        """
        if self.train_data is None or self.val_data is None or self.test_data is None:
            self.logger.warning("Data has not been split yet. Returning minimum length from all data.")
            return self.stock_data.groupby(self.id_column).size().min()

        # Calculate minimum group length in training data
        train_group_lengths = self.train_data.groupby(self.id_column).size()
        min_train_length = train_group_lengths.min() if not train_group_lengths.empty else float('inf')

        # Calculate minimum group length in validation data
        val_group_lengths = self.val_data.groupby(self.id_column).size()
        min_val_length = val_group_lengths.min() if not val_group_lengths.empty else float('inf')

        # Calculate minimum group length in test data
        test_group_lengths = self.test_data.groupby(self.id_column).size()
        min_test_length = test_group_lengths.min() if not test_group_lengths.empty else float('inf')

        # Find the overall minimum
        min_group_length = min(min_train_length, min_val_length, min_test_length)

        self.logger.info(f"Minimum group lengths - Train: {min_train_length}, Validation: {min_val_length}, Test: {min_test_length}")
        return min_group_length

    def filter_stocks_by_min_length_in_splits(self):
        min_len = max(self.seq_length, self.config.MIN_SEQUENCE_LENGTH)
        
        def filter_data(data):
            group_lengths = data.groupby(self.id_column).size()
            valid_ids = group_lengths[group_lengths >= min_len].index
            filtered_data = data[data[self.id_column].isin(valid_ids)].copy()
            return filtered_data if not filtered_data.empty else None

        self.train_data = filter_data(self.train_data)
        self.val_data = filter_data(self.val_data)
        self.test_data = filter_data(self.test_data)

        # Check if any split is empty and log appropriate messages
        empty_splits = []
        if self.train_data is None or self.train_data.empty:
            self.logger.warning("Training data is empty after filtering.")
            empty_splits.append('train')
        if self.val_data is None or self.val_data.empty:
            self.logger.warning("Validation data is empty after filtering.")
            empty_splits.append('validation')
        if self.test_data is None or self.test_data.empty:
            self.logger.warning("Test data is empty after filtering.")
            empty_splits.append('test')

        if empty_splits:
            self.logger.warning(f"The following data splits are empty after filtering: {', '.join(empty_splits)}.")
            self.logger.warning("Proceeding with available data splits.")

        # Log the number of stocks in each split
        if self.train_data is not None:
            self.logger.info(f"After filtering, train data stocks: {self.train_data[self.id_column].nunique()}")
        if self.val_data is not None:
            self.logger.info(f"After filtering, validation data stocks: {self.val_data[self.id_column].nunique()}")
        if self.test_data is not None:
            self.logger.info(f"After filtering, test data stocks: {self.test_data[self.id_column].nunique()}")

        # Log the minimum sequence length in each split
        if self.train_data is not None:
            self.logger.info(f"Minimum sequence length in train data: {self.train_data.groupby(self.id_column).size().min()}")
        if self.val_data is not None:
            self.logger.info(f"Minimum sequence length in validation data: {self.val_data.groupby(self.id_column).size().min()}")
        if self.test_data is not None:
            self.logger.info(f"Minimum sequence length in test data: {self.test_data.groupby(self.id_column).size().min()}")

    def create_dataloader(self, data, seq_length, batch_size, num_workers=0):
        try:
            dataset = SequenceDataset(data, seq_length, self.feature_cols, self.ret_var)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
                collate_fn=custom_collate
            )
        except Exception as e:
            self.logger.error(f"Error creating DataLoader: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def get_min_group_length_across_splits(self):
        """
        Get the minimum group length across all splits after filtering.
        """
        min_train_length = self.train_data.groupby(self.id_column).size().min()
        min_val_length = self.val_data.groupby(self.id_column).size().min()
        min_test_length = self.test_data.groupby(self.id_column).size().min()

        min_group_length = min(min_train_length, min_val_length, min_test_length)
        self.logger.info(f"Updated minimum group lengths - Train: {min_train_length}, Validation: {min_val_length}, Test: {min_test_length}")
        return min_group_length