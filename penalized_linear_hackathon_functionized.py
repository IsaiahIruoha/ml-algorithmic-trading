import datetime
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import json
import itertools

def load_and_preprocess_data(data_in_path, standardize=True, train_pct=None, val_pct=None, test_pct=None):
    """
    Load and preprocess the data from the given path.

    Parameters:
    - data_in_path (str): Path to the input data CSV file.
    - standardize (bool): Whether to standardize the features.
    - train_pct (float): Percentage of data to use for training (if using percentage-based split).
    - val_pct (float): Percentage of data to use for validation (if using percentage-based split).
    - test_pct (float): Percentage of data to use for testing (if using percentage-based split).

    Returns:
    - X_train (DataFrame): Training features.
    - Y_train (ndarray): Training target variable.
    - X_val (DataFrame): Validation features.
    - Y_val (ndarray): Validation target variable.
    - X_test (DataFrame): Testing features.
    - Y_test (ndarray): Testing target variable.
    - test_data_with_dates (DataFrame): Test data with date information and target variable.
    - feature_cols (list): List of feature column names.
    - train (DataFrame): Training data.
    - validate (DataFrame): Validation data.
    - test (DataFrame): Testing data.
    """
    # Load the data
    stock_data = pd.read_csv(data_in_path, parse_dates=["date"], low_memory=False)

    # Define the target variable
    ret_var = "stock_exret"

    # Exclude non-feature columns
    non_feature_cols = ["year", "month", "date", "permno", ret_var]

    # Select only numeric columns
    numeric_cols = stock_data.select_dtypes(include=[np.number]).columns.tolist()

    # Define feature columns as numeric columns excluding non-feature columns
    feature_cols = [col for col in numeric_cols if col not in non_feature_cols]

    # Handle missing values in feature columns
    stock_data[feature_cols] = stock_data[feature_cols].fillna(0)
    
    # Convert feature columns to float to prevent dtype issues
    stock_data[feature_cols] = stock_data[feature_cols].astype(float)

    # Split the data
    train, validate, test = split_data(stock_data, train_pct, val_pct, test_pct)

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        train.loc[:, feature_cols] = scaler.fit_transform(train[feature_cols])
        validate.loc[:, feature_cols] = scaler.transform(validate[feature_cols])
        test.loc[:, feature_cols] = scaler.transform(test[feature_cols])

    X_train, X_val, X_test = train[feature_cols], validate[feature_cols], test[feature_cols]
    Y_train = train[ret_var].values
    Y_val = validate[ret_var].values
    Y_test = test[ret_var].values

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, test[["year", "month", "date", "permno", ret_var]], feature_cols, train, validate, test

def transform_variables(new_set, stock_vars):
    """
    Transform the variables by ranking and scaling them within each month.

    Parameters:
    - new_set (DataFrame): The raw data set.
    - stock_vars (list): List of stock variable names to transform.

    Returns:
    - data (DataFrame): Transformed data set.
    """
    # Transform each variable in each month to the same scale
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        
        # Rank transform each variable to [-1, 1]
        for var in stock_vars:
            if group[var].isna().all():
                group[var] = 0
                print(f"Warning: {date}, {var} set to zero due to all missing values.")
            elif group[var].nunique() == 1:
                group[var] = 0
                print(f"Warning: {date}, {var} set to zero due to all identical values.")
            else:
                var_median = group[var].median(skipna=True)
                group[var] = group[var].fillna(var_median)
                group[var] = group[var].rank(method="dense") - 1
                group_max = group[var].max()
                group[var] = (group[var] / group_max) * 2 - 1

        # Add the adjusted values
        data = pd.concat([data, group], ignore_index=True)
    
    return data

def save_csv(df, output_dir, filename):
    """
    Save the DataFrame to a CSV file.

    Parameters:
    - df (DataFrame): The data to save.
    - output_dir (str): Directory to save the output file.
    - filename (str): Name of the output CSV file.

    Returns:
    - output_path (str): The path to the saved CSV file.
    """
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    return output_path

def reduce_and_save_data(data_input_dir, output_dir, n_components=50):
    """
    Load the raw data, transform variables, apply PCA for dimensionality reduction, and save the reduced data.

    Parameters:
    - data_input_dir (str): Directory containing the input data.
    - output_dir (str): Directory to save the reduced data.
    - n_components (int): Number of principal components to retain.

    Returns:
    - output_path (str): The path to the saved reduced data CSV file.
    """
    # Load the data
    file_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)

    # Read list of predictors for stocks
    file_path = os.path.join(data_input_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # Transform variables
    data = transform_variables(raw, stock_vars)

    # Standardize the data
    scaler = StandardScaler()
    data[stock_vars] = scaler.fit_transform(data[stock_vars])

    # Apply PCA
    pca = PCA(n_components=n_components)
    # Check if there's any variance in the data
    if np.any(np.var(data[stock_vars], axis=0) > 0):
        reduced_data = pca.fit_transform(data[stock_vars])
    else:
        print("Warning: No variance in the data. Skipping PCA.")
        reduced_data = np.zeros((data.shape[0], n_components))

    # Create a new DataFrame with reduced features
    reduced_df = pd.DataFrame(reduced_data, columns=[f"PC_{i+1}" for i in range(n_components)])
    reduced_df = pd.concat([data[["year", "month", "date", "permno", "stock_exret"]], reduced_df], axis=1)

    # Save the reduced data
    output_path = save_csv(reduced_df, output_dir, f"reduced_data_{n_components}_features.csv")

    return output_path

def split_data(data, train_pct=None, val_pct=None, test_pct=None):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    - data (DataFrame): The input data to split.
    - train_pct (float): Percentage of data to use for training (if using percentage-based split).
    - val_pct (float): Percentage of data to use for validation (if using percentage-based split).
    - test_pct (float): Percentage of data to use for testing (if using percentage-based split).

    Returns:
    - train (DataFrame): Training data.
    - validate (DataFrame): Validation data.
    - test (DataFrame): Test data.
    """
    if train_pct is None and val_pct is None and test_pct is None:
        # Original behavior: time-based split
        starting = pd.to_datetime("20000101", format="%Y%m%d")
        counter = 0
        end_date = data['date'].max()
        while (starting + pd.DateOffset(years=11 + counter)) <= end_date:
            cutoff = [
                starting,
                starting + pd.DateOffset(years=8 + counter),
                starting + pd.DateOffset(years=10 + counter),
                starting + pd.DateOffset(years=11 + counter),
            ]

            train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
            validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
            test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

            if cutoff[3] > end_date:
                break

            counter += 1
            starting = starting + pd.DateOffset(years=1)
    else:
        # Custom percentage-based split
        if train_pct is None: train_pct = 0
        if val_pct is None: val_pct = 0
        if test_pct is None: test_pct = 0
        
        # Normalize percentages
        total = train_pct + val_pct + test_pct
        train_pct, val_pct, test_pct = train_pct/total, val_pct/total, test_pct/total
        
        # Sort data by date
        data_sorted = data.sort_values('date')
        
        # Calculate split indices
        train_end = int(len(data_sorted) * train_pct)
        val_end = train_end + int(len(data_sorted) * val_pct)
        
        # Split the data
        train = data_sorted.iloc[:train_end]
        validate = data_sorted.iloc[train_end:val_end]
        test = data_sorted.iloc[val_end:]

    if len(train) == 0 or len(validate) == 0 or len(test) == 0:
        raise ValueError("Insufficient data to split into train, validate, and test sets.")

    return train, validate, test

def train_and_predict_models(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, model_params):
    """
    Train specified regression models and make predictions on the test set.

    Parameters:
    - X_train (DataFrame): Training features.
    - Y_train_dm (ndarray): De-meaned training target variable.
    - X_val (DataFrame): Validation features.
    - Y_val (ndarray): Validation target variable.
    - X_test (DataFrame): Testing features.
    - Y_mean (float): Mean of the training target variable.
    - model_params (dict): Dictionary containing model parameters and options.

    Returns:
    - predictions (dict): Dictionary of predictions from each model.
    """
    predictions = {}

    if model_params['ols']['use']:
        predictions['ols'] = linear_regression(X_train, Y_train_dm, X_test, Y_mean)

    if model_params['lasso']['use']:
        predictions['lasso'] = lasso_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, model_params['lasso']['alphas'])

    if model_params['ridge']['use']:
        predictions['ridge'] = ridge_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, model_params['ridge']['alphas'])

    if model_params['en']['use']:
        predictions['en'] = elastic_net_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, model_params['en']['alphas'], model_params['en']['l1_ratios'])

    return predictions

def linear_regression(X_train, Y_train_dm, X_test, Y_mean):
    """
    Train a linear regression model and make predictions on the test set.

    Parameters:
    - X_train (DataFrame): Training features.
    - Y_train_dm (ndarray): De-meaned training target variable.
    - X_test (DataFrame): Test features.
    - Y_mean (float): Mean of the training target variable.

    Returns:
    - predictions (ndarray): Predictions on the test set.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    return reg.predict(X_test) + Y_mean

def lasso_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, alphas=None):
    """
    Train a Lasso regression model with cross-validation and make predictions on the test set.

    Parameters:
    - X_train (DataFrame): Training features.
    - Y_train_dm (ndarray): De-meaned training target variable.
    - X_val (DataFrame): Validation features.
    - Y_val (ndarray): Validation target variable.
    - X_test (DataFrame): Test features.
    - Y_mean (float): Mean of the training target variable.
    - alphas (array-like): Array of alpha values for grid search.

    Returns:
    - predictions (ndarray): Predictions on the test set.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 100)
    
    lasso = Lasso(fit_intercept=False, max_iter=1000000)
    grid_search = GridSearchCV(lasso, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train_dm)
    
    best_lasso = grid_search.best_estimator_
    return best_lasso.predict(X_test) + Y_mean

def ridge_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, alphas=None):
    """
    Train a Ridge regression model with cross-validation and make predictions on the test set.

    Parameters:
    - X_train (DataFrame): Training features.
    - Y_train_dm (ndarray): De-meaned training target variable.
    - X_val (DataFrame): Validation features.
    - Y_val (ndarray): Validation target variable.
    - X_test (DataFrame): Test features.
    - Y_mean (float): Mean of the training target variable.
    - alphas (array-like): Array of alpha values for grid search.

    Returns:
    - predictions (ndarray): Predictions on the test set.
    """
    if alphas is None:
        alphas = np.logspace(-1, 8, 100)
    
    ridge = Ridge(fit_intercept=False, max_iter=1000000)
    grid_search = GridSearchCV(ridge, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train_dm)
    
    best_ridge = grid_search.best_estimator_
    return best_ridge.predict(X_test) + Y_mean

def elastic_net_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, alphas=None, l1_ratios=None):
    """
    Train an Elastic Net regression model with cross-validation and make predictions on the test set.

    Parameters:
    - X_train (DataFrame): Training features.
    - Y_train_dm (ndarray): De-meaned training target variable.
    - X_val (DataFrame): Validation features.
    - Y_val (ndarray): Validation target variable.
    - X_test (DataFrame): Test features.
    - Y_mean (float): Mean of the training target variable.
    - alphas (array-like): Array of alpha values for grid search.
    - l1_ratios (array-like): Array of l1_ratio values for grid search.

    Returns:
    - predictions (ndarray): Predictions on the test set.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)
    if l1_ratios is None:
        l1_ratios = np.linspace(0.1, 0.9, 9)
    
    en = ElasticNet(fit_intercept=False, max_iter=1000000)
    param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}
    grid_search = GridSearchCV(en, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train_dm)
    
    best_en = grid_search.best_estimator_
    return best_en.predict(X_test) + Y_mean

def print_oos_r2(reg_pred, ret_var):
    """
    Calculate and print the Out-of-Sample R-squared for each model.

    Parameters:
    - reg_pred (DataFrame): DataFrame containing actual and predicted values.
    - ret_var (str): Name of the target variable column.
    """
    yreal = reg_pred[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en"]:
        if model_name in reg_pred.columns:
            ypred = reg_pred[model_name].values
            r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
            print(f"{model_name.upper()} OOS R2: {r2:.4f}")

def create_sequences(data, feature_cols, target_col, seq_length):
    """
    Create sequences of data for LSTM input.

    Parameters:
    - data (DataFrame): Input data.
    - feature_cols (list): List of feature column names.
    - target_col (str): Name of the target variable column.
    - seq_length (int): Sequence length for LSTM.

    Returns:
    - sequences (ndarray): Array of sequences.
    - targets (ndarray): Array of target values.
    - permnos (list): List of permno identifiers corresponding to sequences.
    """
    sequences = []
    targets = []
    permnos = []
    data = data.sort_values(['permno', 'date'])
    grouped = data.groupby('permno')
    for name, group in grouped:
        # Ensure group is long enough
        if len(group) < seq_length + 1:
            continue
        group_X = group[feature_cols].values
        group_Y = group[target_col].values
        for i in range(len(group_X) - seq_length):
            seq = group_X[i:i+seq_length]
            target = group_Y[i+seq_length]
            sequences.append(seq)
            targets.append(target)
            permnos.append(name)
    return np.array(sequences), np.array(targets), permnos

class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)
        return out

def load_hyperparams(file_path):
    """Load hyperparameters from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Hyperparameter file not found. Using default hyperparameters.")
        return None

def save_hyperparams(hyperparams, file_path):
    """Save hyperparameters to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)

def train_lstm_model_with_tuning(train_data, val_data, test_data, feature_cols, target_col, hyperparams_grid, model_save_path='lstm_model.pth', load_weights=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Iterate over all combinations of hyperparameters
    all_combinations = list(itertools.product(*hyperparams_grid.values()))
    total_combinations = len(all_combinations)
    
    best_val_loss = float('inf')
    best_hyperparams = None
    
    for i, hyperparams in enumerate(all_combinations):
        hyperparams_dict = dict(zip(hyperparams_grid.keys(), hyperparams))
        print(f"Training combination {i+1}/{total_combinations}")
        print(f"Hyperparameters: {hyperparams_dict}")
        
        # Create sequences for training, validation, and test data
        X_train, Y_train, _ = create_sequences(train_data, feature_cols, target_col, hyperparams_dict['seq_length'])
        X_val, Y_val, _ = create_sequences(val_data, feature_cols, target_col, hyperparams_dict['seq_length'])
        X_test, Y_test, permnos_test = create_sequences(test_data, feature_cols, target_col, hyperparams_dict['seq_length'])
        
        # Convert to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        
        # Create datasets and data loaders
        train_dataset = StockDataset(X_train, Y_train)
        val_dataset = StockDataset(X_val, Y_val)
        test_dataset = StockDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparams_dict['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams_dict['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=hyperparams_dict['batch_size'])
        
        # Define model
        input_size = len(feature_cols)
        model = LSTMModel(input_size, hidden_size=hyperparams_dict['hidden_size'], num_layers=hyperparams_dict['num_layers'])
        model = model.to(device)
        
        # Load model weights if required
        if load_weights and os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            print(f'Loaded weights from {model_save_path}')
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams_dict['learning_rate'])
        
        # Training loop
        for epoch in range(1, hyperparams_dict['num_epochs']+1):
            model.train()
            train_loss = 0.0
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_dataset)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_Y)
                    
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_dataset)
            
            print(f'Epoch {epoch}/{hyperparams_dict["num_epochs"]}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = hyperparams_dict
                torch.save(model.state_dict(), model_save_path)
                print(f'Saved model weights to {model_save_path}')
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    predictions = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item() * batch_X.size(0)
            
            predictions.extend(outputs.cpu().numpy())
    test_loss /= len(test_dataset)
    print(f'Test Loss: {test_loss:.6f}')
    
    # Prepare predictions DataFrame
    reg_pred_lstm = test_data.iloc[best_hyperparams['seq_length']:].reset_index(drop=True)  # Adjust for sequences
    reg_pred_lstm = reg_pred_lstm.head(len(predictions))  # Ensure lengths match
    reg_pred_lstm['lstm'] = predictions
    return reg_pred_lstm, best_hyperparams

def full_training_run(train_data, val_data, test_data, feature_cols, target_col, best_hyperparams, model_save_path='best_lstm_model.pth', num_epochs=100):
    """Perform a full training run with the best hyperparameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare the data
    seq_length = best_hyperparams['seq_length']
    X_train, Y_train, _ = create_sequences(train_data, feature_cols, target_col, seq_length)
    X_val, Y_val, _ = create_sequences(val_data, feature_cols, target_col, seq_length)
    X_test, Y_test, permnos_test = create_sequences(test_data, feature_cols, target_col, seq_length)
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Create datasets and data loaders
    train_dataset = StockDataset(X_train_tensor, Y_train_tensor)
    val_dataset = StockDataset(X_val_tensor, Y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=best_hyperparams['batch_size'])
    
    # Define model
    input_size = len(feature_cols)
    model = LSTMModel(input_size, hidden_size=best_hyperparams['hidden_size'], num_layers=best_hyperparams['num_layers'])
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_dataset)
        
        print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")
    
    # Load the best model and make predictions on the test set
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze().cpu().numpy()
    
    # Prepare predictions DataFrame
    reg_pred_lstm = test_data.iloc[seq_length:].reset_index(drop=True)
    reg_pred_lstm = reg_pred_lstm.head(len(predictions))
    reg_pred_lstm['lstm'] = predictions
    
    return reg_pred_lstm


def train_lstm_model_with_tuning(train_data, val_data, test_data, feature_cols, target_col, hyperparams_grid, model_save_path='lstm_model.pth', load_weights=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare the data
    train_data = train_data.sort_values(['permno', 'date'])
    val_data = val_data.sort_values(['permno', 'date'])
    test_data = test_data.sort_values(['permno', 'date'])
    
    # Initialize variables to store the best results
    best_val_loss = float('inf')
    best_hyperparams = None
    best_model_state = None
    
    # Iterate over all combinations of hyperparameters
    from itertools import product
    hyperparams_keys = hyperparams_grid.keys()
    hyperparams_values = hyperparams_grid.values()
    for hyperparams in product(*hyperparams_values):
        hyperparams_dict = dict(zip(hyperparams_keys, hyperparams))
        seq_length = hyperparams_dict['seq_length']
        batch_size = hyperparams_dict['batch_size']
        num_epochs = hyperparams_dict['num_epochs']
        hidden_size = hyperparams_dict['hidden_size']
        num_layers = hyperparams_dict['num_layers']
        learning_rate = hyperparams_dict['learning_rate']
        
        print(f"Training with hyperparameters: {hyperparams_dict}")
        
        # Create sequences for training, validation, and test data
        X_train, Y_train, _ = create_sequences(train_data, feature_cols, target_col, seq_length)
        X_val, Y_val, _ = create_sequences(val_data, feature_cols, target_col, seq_length)
        X_test, Y_test, _ = create_sequences(test_data, feature_cols, target_col, seq_length)
        
        # Check if datasets are not empty
        if len(X_train) == 0 or len(X_val) == 0:
            print("Insufficient data for the given sequence length. Skipping this set of hyperparameters.")
            continue
        
        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
        
        # Create datasets and data loaders
        train_dataset = StockDataset(X_train_tensor, Y_train_tensor)
        val_dataset = StockDataset(X_val_tensor, Y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define model
        input_size = len(feature_cols)
        model = LSTMModel(input_size, hidden_size=hidden_size, num_layers=num_layers)
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Load model weights if required
        if load_weights and os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            print(f'Loaded weights from {model_save_path}')
        
        # Training loop
        for epoch in range(1, num_epochs+1):
            model.train()
            train_loss = 0.0
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_dataset)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_Y)
                    
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_dataset)
            
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Check if this model is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparams = hyperparams_dict
            best_model_state = model.state_dict()
            print("New best model found!")
    
    # Load the best model
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved best model weights to {model_save_path}')
    
    # Test the best model
    X_test, Y_test, permnos_test = create_sequences(test_data, feature_cols, target_col, best_hyperparams['seq_length'])

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        return torch.device('cuda')
    else:
        print("CUDA is not available. Using CPU.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: {sys.version}")
        return torch.device('cpu')
    
    
if __name__ == "__main__":
    device = check_cuda()
    data_input_dir = "./Data Input/"
    out_dir = "./Data Output/"
    full_data_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")
    hyperparams_file = os.path.join(out_dir, "hyperparams.json")
    
    # Load and preprocess the raw data using the updated function
    results = load_and_preprocess_data(full_data_path, standardize=True)
    X_train, Y_train, X_val, Y_val, X_test, Y_test, reg_pred, feature_cols, train, validate, test = results
    
    # Load hyperparameters from file or use default
    loaded_hyperparams = load_hyperparams(hyperparams_file)
    if loaded_hyperparams:
        hyperparams_grid = loaded_hyperparams
    else:
        # Define expanded hyperparameter grid
        hyperparams_grid = {
            'seq_length': [5, 10, 20, 30],
            'batch_size': [32, 64, 128, 256],
            'num_epochs': [100],
            'hidden_size': [64, 128, 256, 512],
            'num_layers': [1, 2, 3, 4],
            'learning_rate': [0.001, 0.0005, 0.0001, 0.00005],
        }
        # Save the hyperparameters to file
        save_hyperparams(hyperparams_grid, hyperparams_file)
    
    # Train LSTM model with hyperparameter tuning
    reg_pred_lstm, best_hyperparams = train_lstm_model_with_tuning(
        train,
        validate,
        test,
        feature_cols,
        'stock_exret',
        hyperparams_grid,
        model_save_path=os.path.join(out_dir, 'best_lstm_model.pth'),
        load_weights=False
    )
    
    # Save the best hyperparameters
    save_hyperparams(best_hyperparams, os.path.join(out_dir, "best_hyperparams.json"))
    
    # Perform full training run with best hyperparameters
    print("Starting full training run with best hyperparameters...")
    reg_pred_lstm_full = full_training_run(
        train,
        validate,
        test,
        feature_cols,
        'stock_exret',
        best_hyperparams,
        model_save_path=os.path.join(out_dir, 'final_lstm_model.pth'),
        num_epochs=100
    )
    
    # Save LSTM predictions from full training run
    save_csv(reg_pred_lstm_full, out_dir, 'lstm_output_full.csv')
    
    # Evaluate LSTM OOS R2 for full training run
    yreal = reg_pred_lstm_full['stock_exret'].values
    ypred = reg_pred_lstm_full['lstm'].values
    r2_lstm_full = 1 - np.sum(np.square(yreal - ypred)) / np.sum(np.square(yreal))
    print(f'LSTM OOS R2 (Full Training): {r2_lstm_full:.4f}')
    
    # Print the best hyperparameters
    print("Best Hyperparameters:")
    for key, value in best_hyperparams.items():
        print(f"{key}: {value}")