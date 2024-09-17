import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split

def load_and_preprocess_data(data_input_dir, use_pca=True, pca_variance=0.95, standardize=True):
    # Read sample data
    file_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)

    # Read list of predictors for stocks
    file_path = os.path.join(data_input_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # Define the left hand side variable
    ret_var = "stock_exret"
    new_set = raw[raw[ret_var].notna()].copy()

    # Transform each variable in each month to the same scale
    data = transform_variables(new_set, stock_vars) 

    # Split the data
    train, validate, test = split_data(data) #this may need to be changed to use years or something

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        train[stock_vars] = scaler.fit_transform(train[stock_vars])
        validate[stock_vars] = scaler.transform(validate[stock_vars])
        test[stock_vars] = scaler.transform(test[stock_vars])

    # Perform PCA if requested
    if use_pca:
        pca = PCA(n_components=pca_variance)
        X_train = pca.fit_transform(train[stock_vars])
        X_val = pca.transform(validate[stock_vars])
        X_test = pca.transform(test[stock_vars])
        print(f"Number of PCA components: {pca.n_components_}")
    else:
        X_train, X_val, X_test = train[stock_vars], validate[stock_vars], test[stock_vars]

    Y_train = train[ret_var].values
    Y_val = validate[ret_var].values
    Y_test = test[ret_var].values # add option to demean Y, this would just return this instead of Y_mnean



    return X_train, Y_train, X_val, Y_val, X_test, Y_test, test[["year", "month", "date", "permno", ret_var]]

def transform_variables(new_set, stock_vars):
    # Transform each variable in each month to the same scale
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        
        # Rank transform each variable to [-1, 1]
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(var_median)  # Fill missing values with the cross-sectional median of each month

            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1
            else:
                group[var] = 0  # In case of all missing values
                print(f"Warning: {date}, {var} set to zero.")

        # Add the adjusted values
        data = pd.concat([data, group], ignore_index=True)
    
    return data

def split_data(data):
    # Initialize the starting date, counter, and output data
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    
    # Get the latest date in the data
    end_date = data['date'].max()
    
    # Estimation with expanding window
    while (starting + pd.DateOffset(years=11 + counter)) <= end_date:
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),  # 8 years and expanding as the training set
            starting + pd.DateOffset(years=10 + counter),  # next 2 years as the validation set
            starting + pd.DateOffset(years=11 + counter),  # next year as the out-of-sample testing set
        ]

        # Cut the sample into training, validation, and testing sets
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        # If this is the last iteration, break the loop
        if cutoff[3] > end_date:
            break

        counter += 1
        starting = starting + pd.DateOffset(years=1)

    return train, validate, test

def main(data_input_dir, out_dir, use_pca=True, pca_variance=0.95, standardize=True, use_regularization=True):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, reg_pred = load_and_preprocess_data(
        data_input_dir, use_pca, pca_variance, standardize
    )

    # De-mean Y
    Y_mean = np.mean(Y_train)
    Y_train_dm = Y_train - Y_mean

    # Prepare model parameters
    model_params = {
        'ols': {'use': True},
        'lasso': {'use': use_regularization, 'alphas': np.logspace(-4, 4, 100)},
        'ridge': {'use': use_regularization, 'alphas': np.logspace(-1, 8, 100)},
        'en': {'use': use_regularization, 'alphas': np.logspace(-4, 4, 50), 'l1_ratios': np.linspace(0.1, 0.9, 9)}
    }

    # Train models and make predictions
    predictions = train_and_predict(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, model_params)

    # Add predictions to reg_pred
    for model_name, pred in predictions.items():
        reg_pred[model_name] = pred

    # Output the predicted value to csv
    out_path = os.path.join(out_dir, "output.csv")
    reg_pred.to_csv(out_path, index=False)

    # Print the OOS R2
    print_oos_r2(reg_pred, "stock_exret") # was ret_var

    # for timing purpose
    print(datetime.datetime.now())

def train_and_predict(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, model_params):
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
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, Y_train_dm)
    return reg.predict(X_test) + Y_mean

def lasso_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 4, 100)
    
    lasso = Lasso(fit_intercept=False, max_iter=1000000)
    grid_search = GridSearchCV(lasso, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train_dm)
    
    best_lasso = grid_search.best_estimator_
    return best_lasso.predict(X_test) + Y_mean

def ridge_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, alphas=None):
    if alphas is None:
        alphas = np.logspace(-1, 8, 100)
    
    ridge = Ridge(fit_intercept=False, max_iter=1000000)
    grid_search = GridSearchCV(ridge, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train_dm)
    
    best_ridge = grid_search.best_estimator_
    return best_ridge.predict(X_test) + Y_mean

def elastic_net_regression(X_train, Y_train_dm, X_val, Y_val, X_test, Y_mean, alphas=None, l1_ratios=None):
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

if __name__ == "__main__":
    data_input_dir = "./Data Input/"
    out_dir = "./Data Output/"
    main(data_input_dir, out_dir, use_pca=True, pca_variance=0.95, standardize=True, use_regularization=True)

def print_oos_r2(reg_pred, ret_var):
    yreal = reg_pred[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en"]:
        ypred = reg_pred[model_name].values
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(f"{model_name.upper()} OOS R2: {r2:.4f}")
