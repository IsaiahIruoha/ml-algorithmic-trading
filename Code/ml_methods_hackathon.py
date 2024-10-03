import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
from xgboost import XGBRegressor

# If the script is run directly, the following code will execute
if __name__ == "__main__":
    # For timing purposes, prints the current date and time
    print(datetime.datetime.now())

    # Turn off the SettingWithCopyWarning in pandas to prevent unnecessary warnings during data manipulation
    pd.set_option("mode.chained_assignment", None)

    # Set the working directory where the data files and outputs are stored
    work_dir = "/Users/michaelkokorudz/Desktop/McGill-FIAM Asset Management Hackathon"

    # Read the sample data CSV file containing stock returns and factor data
    file_path = os.path.join(work_dir, "hackathon_sample_v2.csv")  # Replace with the correct file name
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)

    # Read the CSV file containing the list of stock-related predictors (factors)
    file_path = os.path.join(work_dir, "factor_char_list.csv")  # Replace with the correct file name
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # Define the target variable (left-hand side variable) which is excess stock returns
    ret_var = "stock_exret"
    # Create a new dataset where the target variable is not missing
    new_set = raw[raw[ret_var].notna()].copy()  # Ensure the target variable is not missing

    # Group the data by month (assuming "date" is monthly), this helps to scale data by each month
    monthly = new_set.groupby("date")
    data = pd.DataFrame()  # Initialize an empty DataFrame to store the scaled data

    # Loop through each group of monthly data
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()  # Create a copy of the monthly data to modify
        # Rank transform each variable to [-1, 1] to standardize across months
        for var in stock_vars:
            var_median = group[var].median(skipna=True)  # Calculate the monthly median for the variable
            group[var] = group[var].fillna(var_median)  # Fill missing values with the median

            group[var] = group[var].rank(method="dense") - 1  # Rank transform the variable (dense ranking)
            group_max = group[var].max()  # Get the maximum rank value
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1  # Scale to the range [-1, 1]
            else:
                group[var] = 0  # If all values are missing, set the variable to 0
                print("Warning:", date, var, "set to zero.")  # Print a warning if this happens

        # Append the transformed group to the overall data DataFrame
        data = data._append(group, ignore_index=True)

    # Initialize the starting date, counter, and an empty DataFrame for storing predictions
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    # Perform estimation using an expanding window approach. This loop moves the training window forward in time.
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
        # Define the cutoff points for training (8 years), validation (2 years), and testing (1 year)
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),   # 8 years for training
            starting + pd.DateOffset(years=10 + counter),  # Next 2 years for validation
            starting + pd.DateOffset(years=11 + counter),  # Next 1 year for testing
        ]

        # Split the data into training, validation, and testing sets based on the cutoff dates
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        # Standardize the stock-related variables (factors) using StandardScaler (zero mean and unit variance)
        scaler = StandardScaler().fit(train[stock_vars])
        train[stock_vars] = scaler.transform(train[stock_vars])
        validate[stock_vars] = scaler.transform(validate[stock_vars])
        test[stock_vars] = scaler.transform(test[stock_vars])

        # Perform PCA for dimensionality reduction, retain the top 30 principal components
        pca = PCA(n_components=35) # visualized in a scree plot of the data
        X_train_pca = pca.fit_transform(train[stock_vars])
        X_val_pca = pca.transform(validate[stock_vars])
        X_test_pca = pca.transform(test[stock_vars])

        # Get the target variable (Y) for training, validation, and testing
        Y_train = train[ret_var].values
        Y_val = validate[ret_var].values
        Y_test = test[ret_var].values

        # De-mean the target variable for training, since regressions will be fitted without an intercept
        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        # Prepare a DataFrame to store predictions for the current test period
        reg_pred = test[["year", "month", "date", "permno", ret_var]].copy()

#ElasticNet Starts
        
        # ElasticNet Regression with hyperparameter tuning
        # Combine train and validation sets for cross-validation
        X_train_val_pca = np.vstack((X_train_pca, X_val_pca)).astype(np.float32)
        y_train_val = np.concatenate((Y_train_dm, Y_val - Y_mean), axis=0).astype(np.float32)

        # Define a range of hyperparameters for ElasticNet tuning
        l1_ratio_values = np.linspace(0.01, 1, 20)
        alpha_values = np.logspace(-3, 3, 50)

        # Initialize TimeSeriesSplit for time series cross-validation
        tscv_enet = TimeSeriesSplit(n_splits=10)

        # ElasticNetCV automatically tunes the hyperparameters (alpha and l1_ratio)
        enet_cv = ElasticNetCV(
            l1_ratio=l1_ratio_values,
            alphas=alpha_values,
            cv=tscv_enet,
            max_iter=1000,  # Max iterations for optimization
            fit_intercept=False,  # No intercept
            n_jobs=-1,  # Use all available processors
            tol=1e-2,  # Tolerance for stopping optimization
            precompute=False  # Do not precompute Gram matrix
        )

        # Fit the ElasticNet model to the combined training and validation data
        enet_cv.fit(X_train_val_pca, y_train_val)

        # Extract the best alpha and l1_ratio values after cross-validation
        best_alpha = enet_cv.alpha_
        best_l1_ratio = enet_cv.l1_ratio_
        print(f"Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio}")

        # Make predictions on the test set using the trained model
        x_pred_enet = enet_cv.predict(X_test_pca) + Y_mean  # Add back the mean to de-meaned predictions
        reg_pred["en"] = x_pred_enet  # Store the predictions in the output DataFrame
#ElasticNet Ends
 
#XGBoost Starts
        # XGBoost Regression with hyperparameter tuning

        # Prepare data for XGBoost
        X_train = train[stock_vars]
        Y_train = train[ret_var]
        X_val = validate[stock_vars]
        Y_val = validate[ret_var]

        # Combine train and validation data for cross-validation
        X_cv = pd.concat([X_train, X_val], axis=0)
        Y_cv = np.concatenate([Y_train, Y_val], axis=0)

        # Define a reduced parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 300],  # Number of trees
            'max_depth': [3, 5],  # Maximum tree depth
            'learning_rate': [0.01, 0.1],  # Learning rate (shrinkage)
            'subsample': [0.8],  # Subsample ratio of the training data
            'colsample_bytree': [0.8],  # Subsample ratio of columns when constructing each tree
            'min_child_weight': [1, 5],  # Minimum sum of instance weight needed in a child
            'gamma': [0],  # Minimum loss reduction required to make a further partition
            'reg_alpha': [0],  # L1 regularization term
            'reg_lambda': [1],  # L2 regularization term
        }

        # Initialize TimeSeriesSplit for time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Initialize XGBRegressor model
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

        # Initialize RandomizedSearchCV to search over the parameter grid
        random_search = RandomizedSearchCV(
            estimator=xgb_reg,
            param_distributions=param_grid,  # Grid to search over
            n_iter=10,  # Number of parameter settings sampled
            scoring='neg_mean_squared_error',  # Metric for scoring
            cv=tscv,  # TimeSeriesSplit for cross-validation
            verbose=1,  # Print progress
            random_state=42,  # For reproducibility
            n_jobs=-1  # Use all available processors
        )

        # Fit the RandomizedSearchCV with the combined train and validation data
        random_search.fit(X_cv, Y_cv)

        # Get the best parameters found during RandomizedSearchCV
        best_params = random_search.best_params_
        print("Best parameters:", best_params)

        # Train the final XGBoost model using the best parameters found
        xgb_best = XGBRegressor(
            objective='reg:squarederror',
            seed=42,
            **best_params
        )

        # Train XGBoost with the training data and evaluate using the validation set
        xgb_best.fit(
            X_train, Y_train,
            eval_set=[(X_val, Y_val)],
            #early_stopping_rounds=10,  # Early stopping if validation error doesn't improve
            verbose=False  # Suppress output
        )

        # Make predictions on the test set using the trained XGBoost model
        x_pred_xgb = xgb_best.predict(test[stock_vars])
        reg_pred["xgb"] = x_pred_xgb  # Store the predictions in the DataFrame
#XGBoost Ends

        reg_pred["avg_pred"] = (reg_pred["xgb"] + reg_pred["en"]) / 2  # Simple average of both predictions
        # Append the predictions to the output DataFrame
        pred_out = pred_out._append(reg_pred, ignore_index=True)

        # Move to the next time period (expanding window approach)
        counter += 1

    # Output the predicted values to a CSV file
    out_path = os.path.join(work_dir, "output_with_ml_methods_combined.csv")
    print(out_path)
    pred_out.to_csv(out_path, index=False)

    # Calculate and print the out-of-sample R^2 for the averaged predictions from ElasticNet and XGBoost
    yreal = pred_out[ret_var].values
    ypred_avg = pred_out["avg_pred"].values  # Averaged predictions from XGBoost and ElasticNet

    # Calculate R^2 for the averaged predictions
    r2_avg = 1 - np.sum(np.square((yreal - ypred_avg))) / np.sum(np.square(yreal))
    print(f"Average of XGBoost and ElasticNet R^2: {r2_avg}")

    # Print the R^2 of the individual models for comparison
    ypred_xgb = pred_out["xgb"].values  # XGBoost predictions
    r2_xgb = 1 - np.sum(np.square((yreal - ypred_xgb))) / np.sum(np.square(yreal))
    print(f"XGBoost R^2: {r2_xgb}")

    ypred_enet = pred_out["en"].values  # ElasticNet predictions
    r2_enet = 1 - np.sum(np.square((yreal - ypred_enet))) / np.sum(np.square(yreal))
    print(f"ElasticNet R^2: {r2_enet}")

    # For timing purposes, print the current date and time again at the end
    print(datetime.datetime.now())
