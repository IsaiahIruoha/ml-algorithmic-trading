
import unittest
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.preprocessing import StandardScaler

# Import functions from penalized_linear_hackathon_functionized
from penalized_linear_hackathon_functionized import (
    load_and_preprocess_data,
    transform_variables,
    save_csv,
    reduce_and_save_data,
    split_data,
    train_and_predict_models,
    train_and_predict,
    linear_regression,
    lasso_regression,
    ridge_regression,
    elastic_net_regression,
    print_oos_r2,
)

class TestPenalizedLinearHackathonFunctionized(unittest.TestCase):

    def setUp(self):
        # Setup temporary data for testing
        self.data_input_dir = "./Data Input/"
        self.output_dir = "./Data Output/"
        os.makedirs(self.data_input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create sample data over 15 years (180 months)
        num_months = 180  # 15 years
        date_range = pd.date_range(start='2000-01-01', periods=num_months, freq='ME')
        self.sample_data = pd.DataFrame({
            'date': date_range,
            'year': date_range.year,
            'month': date_range.month,
            'stock_exret': np.random.randn(num_months),
            'permno': np.arange(10000, 10000 + num_months),
            'PC_1': np.random.randn(num_months),
            'PC_2': np.random.randn(num_months),
            'PC_3': np.random.randn(num_months),
        })
        self.sample_data_path = os.path.join(self.data_input_dir, "reduced_data_test.csv")
        self.sample_data.to_csv(self.sample_data_path, index=False)
        
        # Create a sample factor_char_list.csv
        factor_char_list_path = os.path.join(self.data_input_dir, "factor_char_list.csv")
        pd.DataFrame({'variable': ['PC_1', 'PC_2', 'PC_3']}).to_csv(factor_char_list_path, index=False)
        
        # Create a sample hackathon_sample_v2.csv
        raw_data_path = os.path.join(self.data_input_dir, "hackathon_sample_v2.csv")
        self.sample_raw_data = self.sample_data.copy()
        self.sample_raw_data.to_csv(raw_data_path, index=False)

    def test_load_and_preprocess_data(self):
        # Test loading and preprocessing with standardization
        X_train, Y_train, X_val, Y_val, X_test, Y_test, test_meta, feature_cols = load_and_preprocess_data(
            self.sample_data_path, standardize=True
        )
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(Y_train)
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertEqual(len(feature_cols), 3)
        # Test without standardization
        X_train_ns, _, _, _, _, _, _, _ = load_and_preprocess_data(
            self.sample_data_path, standardize=False
        )
        self.assertFalse(X_train_ns.equals(X_train))

    def test_transform_variables(self):
        # Test with missing values
        data_with_nans = self.sample_raw_data.copy()
        data_with_nans.loc[0, 'PC_1'] = np.nan
        transformed_data = transform_variables(data_with_nans, ['PC_1', 'PC_2', 'PC_3'])
        self.assertFalse(transformed_data['PC_1'].isna().any())
        # Test rank transformation
        self.assertTrue(transformed_data['PC_1'].between(-1, 1).all())

    def test_save_csv(self):
        # Test saving CSV
        test_df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
        filename = "test_output.csv"
        output_path = save_csv(test_df, self.output_dir, filename)
        self.assertTrue(os.path.exists(output_path))
        # Clean up
        os.remove(output_path)

    def test_reduce_and_save_data(self):
        # Test reduction and saving
        output_path = reduce_and_save_data(self.data_input_dir, self.output_dir, n_components=2)
        self.assertTrue(os.path.exists(output_path))
        reduced_data = pd.read_csv(output_path)
        self.assertIn('PC_1', reduced_data.columns)
        self.assertIn('PC_2', reduced_data.columns)
        self.assertFalse('PC_3' in reduced_data.columns)
        os.remove(output_path)

    def test_split_data_default(self):
        train, validate, test = split_data(self.sample_data)
        self.assertTrue(len(train) > 0)
        self.assertTrue(len(validate) > 0)
        self.assertTrue(len(test) > 0)

    def test_split_data_custom(self):
        train, validate, test = split_data(self.sample_data, train_pct=0.6, val_pct=0.2, test_pct=0.2)
        self.assertAlmostEqual(len(train) / len(self.sample_data), 0.6, delta=0.01)
        self.assertAlmostEqual(len(validate) / len(self.sample_data), 0.2, delta=0.01)
        self.assertAlmostEqual(len(test) / len(self.sample_data), 0.2, delta=0.01)

    def test_split_data_insufficient(self):
        small_data = self.sample_data.head(3)  # only 3 rows of data
        with self.assertRaises(ValueError):
            split_data(small_data, train_pct=0.6, val_pct=0.2, test_pct=0.2)

    def test_split_data_normalization(self):
        train, validate, test = split_data(self.sample_data, train_pct=6, val_pct=2, test_pct=2)
        self.assertAlmostEqual(len(train) / len(self.sample_data), 0.6, delta=0.01)
        self.assertAlmostEqual(len(validate) / len(self.sample_data), 0.2, delta=0.01)
        self.assertAlmostEqual(len(test) / len(self.sample_data), 0.2, delta=0.01)

    def test_train_and_predict_models(self):
        # Generate sample data
        X_train = pd.DataFrame(np.random.randn(10, 3), columns=['PC_1', 'PC_2', 'PC_3'])
        Y_train = np.random.randn(10)
        X_val = pd.DataFrame(np.random.randn(5, 3), columns=['PC_1', 'PC_2', 'PC_3'])
        Y_val = np.random.randn(5)
        X_test = pd.DataFrame(np.random.randn(5, 3), columns=['PC_1', 'PC_2', 'PC_3'])
        Y_mean = Y_train.mean()
        model_params = {
            'ols': {'use': True},
            'lasso': {'use': True, 'alphas': np.logspace(-4, 4, 10)},
            'ridge': {'use': True, 'alphas': np.logspace(-4, 4, 10)},
            'en': {'use': True, 'alphas': np.logspace(-4, 4, 10), 'l1_ratios': np.linspace(0.1, 0.9, 5)}
        }
        predictions = train_and_predict_models(X_train, Y_train - Y_mean, X_val, Y_val, X_test, Y_mean, model_params)
        self.assertIn('ols', predictions)
        self.assertIn('lasso', predictions)
        self.assertIn('ridge', predictions)
        self.assertIn('en', predictions)
        for pred in predictions.values():
            self.assertEqual(len(pred), len(X_test))

    def test_print_oos_r2(self):
        reg_pred = pd.DataFrame({
            'stock_exret': np.random.randn(10),
            'ols': np.random.randn(10),
            'lasso': np.random.randn(10),
            'ridge': np.random.randn(10),
            'en': np.random.randn(10),
        })
        # Capture print output
        import io
        import sys
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        print_oos_r2(reg_pred, 'stock_exret')
        sys.stdout = sys.__stdout__
        output = capturedOutput.getvalue()
        self.assertIn('OLS OOS R2:', output)
        self.assertIn('LASSO OOS R2:', output)
        self.assertIn('RIDGE OOS R2:', output)
        self.assertIn('EN OOS R2:', output)

    def test_transform_variables_edge_cases(self):
        edge_case_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'PC_1': [np.nan, np.nan, np.nan],
            'PC_2': [1, 1, 1],
            'PC_3': [1, 2, 3]
        })
        result = transform_variables(edge_case_data, ['PC_1', 'PC_2', 'PC_3'])
        self.assertTrue((result['PC_1'] == 0).all())
        self.assertTrue((result['PC_2'] == 0).all())
        self.assertTrue((result['PC_3'] >= -1).all() and (result['PC_3'] <= 1).all())

    def tearDown(self):
        # Clean up any files or directories created during tests
        if os.path.exists(self.sample_data_path):
            os.remove(self.sample_data_path)
        factor_char_list_path = os.path.join(self.data_input_dir, "factor_char_list.csv")
        if os.path.exists(factor_char_list_path):
            os.remove(factor_char_list_path)
        raw_data_path = os.path.join(self.data_input_dir, "hackathon_sample_v2.csv")
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
        # Remove directories if empty
        if not os.listdir(self.data_input_dir):
            os.rmdir(self.data_input_dir)
        if not os.listdir(self.output_dir):
            os.rmdir(self.output_dir)

if __name__ == "__main__":
    data_input_dir = "./Data Input/"
    out_dir = "./Data Output/"

    unittest.main()