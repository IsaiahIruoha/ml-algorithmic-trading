import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # for timing purpose
    print(datetime.datetime.now())

    # turn off pandas Setting with Copy Warning
    pd.set_option("mode.chained_assignment", None)

    # set working directory
    work_dir = "Your working directory"

    # read sample data
    file_path = os.path.join(
        work_dir, "sample_data.csv"
    )  # replace with the correct file name
    raw = pd.read_csv(
        file_path, parse_dates=["date"], low_memory=False
    )  # the date is the first day of the return month (t+1)

    # read list of predictors for stocks
    file_path = os.path.join(
        work_dir, "factor_char_list.csv"
    )  # replace with the correct file name
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # define the left hand side variable
    ret_var = "stock_exret"
    new_set = raw[
        raw[ret_var].notna()
    ].copy()  # create a copy of the data and make sure the left hand side is not missing

    # transform each variable in each month to the same scale
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        # rank transform each variable to [-1, 1]
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(
                var_median
            )  # fill missing values with the cross-sectional median of each month

            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1
            else:
                group[var] = 0  # in case of all missing values
                print("Warning:", date, var, "set to zero.")

        # add the adjusted values
        data = data._append(
            group, ignore_index=True
        )  # append may not work with certain versions of pandas, use concat instead if needed

    # initialize the starting date, counter, and output data
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    # estimation with expanding window
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
        "20240101", format="%Y%m%d"
    ):
        cutoff = [
            starting,
            starting
            + pd.DateOffset(
                years=8 + counter
            ),  # use 8 years and expanding as the training set
            starting
            + pd.DateOffset(
                years=10 + counter
            ),  # use the next 2 years as the validation set
            starting + pd.DateOffset(years=11 + counter),
        ]  # use the next year as the out-of-sample testing set

        # cut the sample into training, validation, and testing sets
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        # Optional: if your data has additional binary or categorical variables,
        # you can further standardize them here
        scaler = StandardScaler().fit(train[stock_vars])
        train[stock_vars] = scaler.transform(train[stock_vars])
        validate[stock_vars] = scaler.transform(validate[stock_vars])
        test[stock_vars] = scaler.transform(test[stock_vars])

        # get Xs and Ys
        X_train = train[stock_vars].values
        Y_train = train[ret_var].values
        X_val = validate[stock_vars].values
        Y_val = validate[ret_var].values
        X_test = test[stock_vars].values
        Y_test = test[ret_var].values

        # de-mean Y (because the regressions are fitted without an intercept)
        # if you want to include an intercept (or bias in neural networks, etc), you can skip this step
        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        # prepare output data
        reg_pred = test[
            ["year", "month", "date", "permno", ret_var]
        ]  # minimum identifications for each stock

        # Linear Regression
        # no validation is needed for OLS
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["ols"] = x_pred

        # Lasso
        lambdas = np.arange(
            -4, 4.1, 0.1
        )  # search for the best lambda in the range of 10^-4 to 10^4, range can be adjusted
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        # select the best lambda based on the validation set
        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Lasso(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean  # predict the out-of-sample testing set
        reg_pred["lasso"] = x_pred

        # Ridge
        # same format as above
        lambdas = np.arange(-1, 8.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Ridge(alpha=((10**i) * 0.5), fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["ridge"] = x_pred

        # Elastic Net
        # same format as above
        lambdas = np.arange(-4, 4.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = ElasticNet(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["en"] = x_pred

        # add to the output data
        pred_out = pred_out._append(reg_pred, ignore_index=True)

        # go to the next year
        counter += 1

    # output the predicted value to csv
    out_path = os.path.join(work_dir, "output.csv")
    print(out_path)
    pred_out.to_csv(out_path, index=False)

    # print the OOS R2
    yreal = pred_out[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en"]:
        ypred = pred_out[model_name].values
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(model_name, r2)

    # for timing purpose
    print(datetime.datetime.now())
