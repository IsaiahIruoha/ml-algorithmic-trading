import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import Optuna for hyperparameter optimization
import optuna

def check_cuda():
    """Check for CUDA availability and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    print(f"PyTorch version: {torch.__version__}")
    return device

class DataProcessor:
    """
    A class to handle data loading, preprocessing, transformation, and splitting.
    """
    def __init__(self, data_in_path, ret_var='stock_exret', standardize=True):
        self.data_in_path = data_in_path
        self.ret_var = ret_var
        self.standardize = standardize
        self.scaler = None
        self.feature_cols = None

    def load_data(self):
        """Load the data from the CSV file."""
        self.stock_data = pd.read_csv(self.data_in_path, parse_dates=["date"], low_memory=False)
        print(f"Data loaded from {self.data_in_path}")

    def preprocess_data(self):
        """Preprocess the data: handle missing values, select features, and standardize if needed."""
        # Exclude non-feature columns
        non_feature_cols = ["year", "month", "date", "permno", self.ret_var]
        
        # Select numeric feature columns
        numeric_cols = self.stock_data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
        
        # Handle missing values
        self.stock_data[self.feature_cols] = self.stock_data[self.feature_cols].fillna(0).astype(float)
        
        # Standardize features if requested
        if self.standardize:
            self.scaler = StandardScaler()
            self.stock_data[self.feature_cols] = self.scaler.fit_transform(self.stock_data[self.feature_cols])
            print("Data standardized.")

    def split_data(self, train_pct=None, val_pct=None, test_pct=None):
        """Split data into training, validation, and test sets."""
        if train_pct is None and val_pct is None and test_pct is None:
            # Time-based splitting
            self.train_data, self.val_data, self.test_data = self._time_based_split()
        else:
            # Percentage-based splitting
            self.train_data, self.val_data, self.test_data = self._percentage_based_split(train_pct, val_pct, test_pct)
        print("Data split into training, validation, and test sets.")

    def _time_based_split(self):
        """Split data based on time periods."""
        data = self.stock_data.copy()
        starting = pd.to_datetime("2000-01-01")
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

            if cutoff[3] > end_date or len(train) == 0 or len(validate) == 0 or len(test) == 0:
                break

            counter += 1
            starting = starting + pd.DateOffset(years=1)

        return train, validate, test

    def _percentage_based_split(self, train_pct, val_pct, test_pct):
        """Split data based on provided percentages."""
        # Normalize percentages
        total = train_pct + val_pct + test_pct
        train_pct /= total
        val_pct /= total
        test_pct /= total

        data_sorted = self.stock_data.sort_values('date')
        n = len(data_sorted)

        train_end = int(n * train_pct)
        val_end = train_end + int(n * val_pct)

        train = data_sorted.iloc[:train_end]
        validate = data_sorted.iloc[train_end:val_end]
        test = data_sorted.iloc[val_end:]

        if len(train) == 0 or len(validate) == 0 or len(test) == 0:
            raise ValueError("Insufficient data to split into train, validate, and test sets.")

        return train, validate, test

    def get_features_and_target(self):
        """Get features and target variables for training, validation, and test sets."""
        X_train = self.train_data[self.feature_cols]
        Y_train = self.train_data[self.ret_var].values

        X_val = self.val_data[self.feature_cols]
        Y_val = self.val_data[self.ret_var].values

        X_test = self.test_data[self.feature_cols]
        Y_test = self.test_data[self.ret_var].values

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

class RegressionModels:
    """
    A class that encapsulates training and prediction with different regression models.
    """
    def __init__(self, Y_mean):
        self.Y_mean = Y_mean
        self.models = {}
        self.predictions = {}

    def train_linear_regression(self, X_train, Y_train_dm):
        """Train a Linear Regression model."""
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        self.models['ols'] = reg
        print("Linear Regression model trained.")

    def train_lasso(self, X_train, Y_train_dm, X_val, Y_val, alphas=None):
        """Train a Lasso Regression model with cross-validation."""
        if alphas is None:
            alphas = np.logspace(-4, 4, 100)
        lasso = Lasso(fit_intercept=False, max_iter=1000000)
        grid_search = GridSearchCV(lasso, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train_dm)
        self.models['lasso'] = grid_search.best_estimator_
        print("Lasso Regression model trained with alpha:", grid_search.best_estimator_.alpha)

    def train_ridge(self, X_train, Y_train_dm, X_val, Y_val, alphas=None):
        """Train a Ridge Regression model with cross-validation."""
        if alphas is None:
            alphas = np.logspace(-1, 8, 100)
        ridge = Ridge(fit_intercept=False, max_iter=1000000)
        grid_search = GridSearchCV(ridge, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train_dm)
        self.models['ridge'] = grid_search.best_estimator_
        print("Ridge Regression model trained with alpha:", grid_search.best_estimator_.alpha)

    def train_elastic_net(self, X_train, Y_train_dm, X_val, Y_val, alphas=None, l1_ratios=None):
        """Train an Elastic Net Regression model with cross-validation."""
        if alphas is None:
            alphas = np.logspace(-4, 4, 50)
        if l1_ratios is None:
            l1_ratios = np.linspace(0.1, 0.9, 9)
        en = ElasticNet(fit_intercept=False, max_iter=1000000)
        param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}
        grid_search = GridSearchCV(en, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train_dm)
        self.models['en'] = grid_search.best_estimator_
        print("Elastic Net model trained with alpha:", grid_search.best_estimator_.alpha,
              "and l1_ratio:", grid_search.best_estimator_.l1_ratio)

    def predict(self, X_test):
        """Generate predictions using the trained models."""
        for model_name, model in self.models.items():
            self.predictions[model_name] = model.predict(X_test) + self.Y_mean
        print("Predictions generated.")

    def get_predictions(self):
        """Retrieve the predictions dictionary."""
        return self.predictions

class LSTMModel(nn.Module):
    """
    LSTM Model for time series prediction.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.0, bidirectional=False, use_batch_norm=False, activation_function=None):
        super(LSTMModel, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(lstm_output_size)

        self.fc = nn.Linear(lstm_output_size, 1)
        self.activation = None
        if activation_function == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_function == 'Tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.activation:
            out = self.activation(out)
        out = self.fc(out)
        return out

class StockDataset(Dataset):
    """
    Custom Dataset for handling stock sequences for LSTM input.
    """
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMTrainer:
    """
    Class to handle LSTM model training with hyperparameter optimization using Optuna.
    """
    def __init__(self, feature_cols, target_col, device):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = device

    def create_sequences(self, data, seq_length):
        """
        Create sequences of data for LSTM input.
        """
        sequences = []
        targets = []
        data = data.sort_values(['permno', 'date'])
        grouped = data.groupby('permno')

        for _, group in grouped:
            if len(group) < seq_length + 1:
                continue
            group_X = group[self.feature_cols].values
            group_Y = group[self.target_col].values
            for i in range(len(group_X) - seq_length):
                seq = group_X[i:i+seq_length]
                target = group_Y[i+seq_length]
                sequences.append(seq)
                targets.append(target)
        return np.array(sequences), np.array(targets)

    def train_with_optimization(self, train_data, val_data, test_data, n_trials=50):
        """
        Train LSTM model using hyperparameter optimization with Optuna.
        """
        # Ensure the directory for saving models exists
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)

        def objective(trial):
            # Suggest hyperparameters
            num_layers = trial.suggest_int('num_layers', 1, 6)

            # Conditionally set dropout_rate
            if num_layers > 1:
                dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            else:
                dropout_rate = 0.0

            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512])
            seq_length = trial.suggest_categorical('seq_length', [5, 10, 15, 20])
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop', 'SGD'])
            use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
            activation_function = trial.suggest_categorical('activation_function', ['ReLU', 'Tanh', 'None'])
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            clip_grad_norm = trial.suggest_float('clip_grad_norm', 0.1, 5.0)
            num_epochs = 50  # You can also make this a hyperparameter

            # Prepare data sequences
            X_train, Y_train = self.create_sequences(train_data, seq_length)
            X_val, Y_val = self.create_sequences(val_data, seq_length)

            if len(X_train) == 0 or len(X_val) == 0:
                return float('inf')  # Skip this trial if not enough data

            # Convert to torch tensors and create datasets
            train_loader = self._create_dataloader(X_train, Y_train, batch_size, shuffle=True)
            val_loader = self._create_dataloader(X_val, Y_val, batch_size)

            # Initialize model
            input_size = len(self.feature_cols)
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                bidirectional=bidirectional,
                use_batch_norm=use_batch_norm,
                activation_function=activation_function
            ).to(self.device)

            # Set up optimizer
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Learning rate scheduler
            if use_scheduler:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=trial.suggest_int('scheduler_step_size', 10, 100),
                    gamma=trial.suggest_float('scheduler_gamma', 0.1, 0.9)
                )
            else:
                scheduler = None

            criterion = nn.MSELoss()
            best_val_loss = float('inf')

            # Define a unique model save path for this trial
            model_save_path = os.path.join(model_dir, f'model_trial_{trial.number}.pth')

            # Training loop
            for epoch in range(1, num_epochs + 1):
                self._train_epoch(model, train_loader, criterion, optimizer, clip_grad_norm)
                val_loss = self._evaluate(model, val_loader, criterion)
                trial.report(val_loss, epoch)

                # Save the model if it has the best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)

                # Learning rate scheduler step
                if scheduler:
                    scheduler.step()

                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Store the model save path in the trial's user attributes
            trial.set_user_attr('model_save_path', model_save_path)
            return best_val_loss

        # Run the optimization with a specified study name
        study = optuna.create_study(direction='minimize', study_name='LSTM_Hyperparameter_Optimization')
        study.optimize(objective, n_trials=n_trials)

        # Retrieve the best trial
        best_trial = study.best_trial
        best_hyperparams = best_trial.params

        # Load the best model
        best_model_save_path = best_trial.user_attrs['model_save_path']
        input_size = len(self.feature_cols)
        best_model = LSTMModel(
            input_size=input_size,
            hidden_size=best_hyperparams['hidden_size'],
            num_layers=best_hyperparams['num_layers'],
            dropout_rate=best_hyperparams.get('dropout_rate', 0.0),
            bidirectional=best_hyperparams['bidirectional'],
            use_batch_norm=best_hyperparams['use_batch_norm'],
            activation_function=best_hyperparams['activation_function']
        ).to(self.device)
        best_model.load_state_dict(torch.load(best_model_save_path))

        # Prepare test data sequences
        seq_length = best_hyperparams['seq_length']
        X_test_seq, Y_test_seq = self.create_sequences(test_data, seq_length)

        if len(X_test_seq) == 0:
            print("Not enough data to create test sequences.")
            reg_pred_lstm = pd.DataFrame()
            return reg_pred_lstm, best_hyperparams

        # Create DataLoader for test data
        test_loader = self._create_dataloader(X_test_seq, Y_test_seq, batch_size=best_hyperparams['batch_size'])

        # Evaluate model on test data
        criterion = nn.MSELoss()
        test_loss, predictions = self._evaluate(best_model, test_loader, criterion, return_predictions=True)

        # Create a DataFrame with test data and predictions
        test_data_sequences = test_data.copy().reset_index(drop=True)
        reg_pred_lstm = test_data_sequences.iloc[seq_length:].reset_index(drop=True)
        reg_pred_lstm['lstm'] = predictions[:len(reg_pred_lstm)]  # Ensure lengths match

        # Return the predictions DataFrame and best hyperparameters
        return reg_pred_lstm, best_hyperparams

    def _create_dataloader(self, X, Y, batch_size, shuffle=False):
        """Create DataLoader from sequences and targets."""
        dataset = StockDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    def _train_epoch(self, model, dataloader, criterion, optimizer, clip_grad_norm=None):
        """Train the model for one epoch."""
        model.train()
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            # Handle scalar outputs
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            if batch_Y.ndim == 0:
                batch_Y = batch_Y.unsqueeze(0)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            if clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

    def _evaluate(self, model, dataloader, criterion, return_predictions=False):
        """Evaluate the model on validation or test set."""
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                outputs = model(batch_X).squeeze(-1)
                # Handle scalar outputs
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_Y.ndim == 0:
                    batch_Y = batch_Y.unsqueeze(0)
                loss = criterion(outputs, batch_Y)
                total_loss += loss.item() * batch_X.size(0)
                if return_predictions:
                    predictions.extend(outputs.cpu().numpy())
        avg_loss = total_loss / len(dataloader.dataset)
        if return_predictions:
            return avg_loss, predictions
        else:
            return avg_loss

def save_hyperparams(hyperparams, file_path):
    """Save hyperparameters to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)

def save_csv(df, output_dir, filename):
    """Save the DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")

def calculate_oos_r2(y_true, y_pred):
    """Calculate Out-of-Sample R-squared."""
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)
    return r2


def main():
    device = check_cuda()
    data_input_dir = "./Data Input/"
    out_dir = "./Data Output/"
    os.makedirs(out_dir, exist_ok=True)
    full_data_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")

    # Data processing
    data_processor = DataProcessor(full_data_path, standardize=True)
    data_processor.load_data()
    data_processor.preprocess_data()
    data_processor.split_data()  # Using default time-based splitting

    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_processor.get_features_and_target()
    feature_cols = data_processor.feature_cols

    # Regression models
    '''
    regressor = RegressionModels(Y_mean)
    regressor.train_linear_regression(X_train, Y_train_dm)
    regressor.train_lasso(X_train, Y_train_dm, X_val, Y_val)
    regressor.train_ridge(X_train, Y_train_dm, X_val, Y_val)
    regressor.train_elastic_net(X_train, Y_train_dm, X_val, Y_val)
    regressor.predict(X_test)

    # Prepare regression predictions DataFrame
    reg_pred = data_processor.test_data.copy().reset_index(drop=True)
    for model_name, preds in regressor.get_predictions().items():
        reg_pred[model_name] = preds
    save_csv(reg_pred, out_dir, 'regression_predictions.csv')

    # Calculate OOS R2 for regression models
    y_true = reg_pred['stock_exret'].values
    for model_name in regressor.get_predictions().keys():
        y_pred = reg_pred[model_name].values
        r2 = calculate_oos_r2(y_true, y_pred)
        print(f"{model_name.upper()} OOS R2: {r2:.4f}")
    '''

    # LSTM Model Training with Optimization
    lstm_trainer = LSTMTrainer(feature_cols, 'stock_exret', device)
    reg_pred_lstm, best_hyperparams = lstm_trainer.train_with_optimization(
        data_processor.train_data,
        data_processor.val_data,
        data_processor.test_data,
        n_trials=50  # Adjust the number of trials as needed
    )

    save_hyperparams(best_hyperparams, os.path.join(out_dir, "best_hyperparams.json"))

    # Evaluate LSTM OOS R2
    if not reg_pred_lstm.empty:
        yreal = reg_pred_lstm['stock_exret'].values
        ypred = reg_pred_lstm['lstm'].values
        r2_lstm = calculate_oos_r2(yreal, ypred)
        print(f'LSTM OOS R2: {r2_lstm:.4f}')
    else:
        print("No predictions were made due to insufficient test data.")

    print("Best Hyperparameters:")
    for key, value in best_hyperparams.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()


