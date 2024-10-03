"""
UnitTests.py

This module contains unit tests for the stock prediction project components. It includes:

1. Synthetic Data Generation: Creates a realistic dataset for testing purposes.
2. DataProcessor Tests: Validates the data preprocessing pipeline, including PCA and feature engineering.
3. LSTMTrainer Tests: Checks the sequence creation functionality for LSTM models.
4. Dynamic Date Splitting Tests: Ensures correct time-based splitting of financial data.

Key features:
- Uses Python's unittest framework for structured testing
- Creates a synthetic dataset mimicking real stock market data
- Tests critical components of the data processing pipeline
- Validates LSTM-specific data preparation steps
- Checks the integrity of time-based data splitting for financial time series

This test suite is designed to ensure the reliability and correctness of various components in the stock prediction project, focusing on data handling, preprocessing, and model preparation steps. It helps maintain the quality and consistency of the project as it evolves.
"""

import unittest
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config import Config
import sys
import os
import traceback
import datetime
from data_processor import DataProcessor
from trainer import LSTMTrainer
from models import LSTMModel
import torch.nn as nn
import logging
from datetime import datetime, timedelta

# Add the parent directory to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_synthetic_dataset(num_stocks=10, num_days=1000, num_features=50):
    """
    Generates a synthetic dataset for testing purposes.
    """
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Generate permnos (stock identifiers)
    permnos = [10001 + i for i in range(num_stocks)]

    # Prepare data container
    data = []

    # Generate data for each stock
    for permno in permnos:
        for date in dates:
            row = {
                'permno': permno,
                'date': date,
                'stock_exret': np.random.normal(0, 0.02)  # Random return, normal distribution
            }
            # Add features
            for i in range(1, num_features + 1):
                row[f'feature{i}'] = np.random.randn()  # Random feature value, standard normal distribution
            data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Sort the DataFrame
    df = df.sort_values(['permno', 'date']).reset_index(drop=True)

    return df

class BaseTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up class-level resources for the tests.
        """
        cls.synthetic_df = create_synthetic_dataset()
        cls.feature_cols = [f'feature{i}' for i in range(1, 51)]
        cls.target_col = 'stock_exret'

class TestDataProcessor(BaseTestClass):
    def setUp(self):
        """
        Set up resources for each test in TestDataProcessor.
        """
        self.processor = DataProcessor(data_in_path=None, ret_var=self.target_col, standardize=True)
        self.processor.stock_data = self.synthetic_df.copy()
        self.processor.feature_cols = self.feature_cols

    def test_preprocessing(self):
        """
        Test the preprocessing functionality of DataProcessor.
        """
        self.processor.preprocess_data()
        self.assertIsNotNone(self.processor.stock_data)
        self.assertEqual(len(self.processor.feature_cols), 41)  # 35 PCA components + 6 cyclical features
        self.assertIn('PC1', self.processor.stock_data.columns)
        self.assertIn('PC35', self.processor.stock_data.columns)
        self.assertIn('month_sin', self.processor.stock_data.columns)
        self.assertIn('month_cos', self.processor.stock_data.columns)
        self.assertIn('day_of_week_sin', self.processor.stock_data.columns)
        self.assertIn('day_of_week_cos', self.processor.stock_data.columns)
        self.assertIn('quarter_sin', self.processor.stock_data.columns)
        self.assertIn('quarter_cos', self.processor.stock_data.columns)

class TestLSTMTrainer(BaseTestClass):
    def setUp(self):
        """
        Set up resources for each test in TestLSTMTrainer.
        """
        self.config = Config
        self.device = torch.device('cpu')
        self.trainer = LSTMTrainer(
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            device=self.device,
            config=self.config
        )
        self.trainer.data_processor.stock_data = self.synthetic_df.copy()
        self.trainer.data_processor.preprocess_data()  # Ensure data is preprocessed

    def test_create_sequences(self):
        """
        Test the sequence creation functionality of LSTMTrainer.
        """
        seq_length = 10
        X, Y, indices = self.trainer.data_processor.create_sequences(self.trainer.data_processor.stock_data, seq_length)
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)
        self.assertIsNotNone(indices)
        if len(X) > 0:
            self.assertEqual(X.shape[1], seq_length)
            self.assertEqual(X.shape[2], len(self.feature_cols))  # Use the actual number of features

class TestDynamicDateSplitting(BaseTestClass):
    def setUp(self):
        """
        Set up resources for each test in TestDynamicDateSplitting.
        """
        self.processor = DataProcessor(data_in_path=None, ret_var=self.target_col, standardize=True)
        self.processor.stock_data = self.synthetic_df.copy()
        self.processor.feature_cols = self.feature_cols

    def test_dynamic_time_based_split(self):
        """
        Test the dynamic time-based data splitting functionality of DataProcessor.
        """
        self.processor.preprocess_data()
        self.processor.split_data(method='time')
        self.assertIsNotNone(self.processor.train_data)
        self.assertIsNotNone(self.processor.val_data)
        self.assertIsNotNone(self.processor.test_data)
        # Add more specific assertions about the time-based split
        self.assertTrue(self.processor.train_data['date'].max() <= self.processor.val_data['date'].min())
        self.assertTrue(self.processor.val_data['date'].max() <= self.processor.test_data['date'].min())

# Add more test classes as needed

if __name__ == '__main__':
    unittest.main()