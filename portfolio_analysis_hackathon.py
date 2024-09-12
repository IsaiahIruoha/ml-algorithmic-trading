import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *


# read predcited values
pred_path = "Your predicted values path"
pred = pd.read_csv(pred_path, parse_dates=["date"])
# pred.columns = map(str.lower, pred.columns)

# select model (ridge as an example)
model = "ridge"

# sort stocks into deciles (10 portfolios) each month based on the predicted returns and calculate portfolio returns
# portfolio 1 is the decile with the lowest predicted returns, portfolio 10 is the decile with the highest predicted returns
# portfolio 11 is the long-short portfolio (portfolio 10 - portfolio 1)
# or you can pick the top and bottom n number of stocks as the long and short portfolios
predicted = pred.groupby(["year", "month"])[model]
pred["rank"] = np.floor(
    predicted.transform(lambda s: s.rank())
    * 10
    / predicted.transform(lambda s: len(s) + 1)
)  # rank stocks into deciles
pred = pred.sort_values(["year", "month", "rank", "permno"])
monthly_port = pred.groupby(["year", "month", "rank"]).apply(
    lambda df: pd.Series(np.average(df["stock_exret"], axis=0))
)  # calculate the realized return for each portfolio using realized stock returns
monthly_port = monthly_port.unstack().dropna().reset_index()
monthly_port.columns = ["year", "month"] + ["port_" + str(x) for x in range(1, 11)]
monthly_port["port_11"] = (
    monthly_port["port_10"] - monthly_port["port_1"]
)  # long-short portfolio

# Calculate the Sharpe ratio for long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
sharpe = (
    monthly_port["port_11"].mean() / monthly_port["port_11"].std() * np.sqrt(12)
)  # Sharpe ratio is annualized
print("Sharpe Ratio:", sharpe)

# Calculate the CAPM Alpha for the long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
mkt_path = "Your market factor path"
mkt = pd.read_csv(mkt_path)
monthly_port = monthly_port.merge(mkt, how="inner", on=["year", "month"])
# Newy-West regression for heteroskedasticity and autocorrelation robust standard errors
nw_ols = sm.ols(formula="port_11 ~ mkt_rf", data=monthly_port).fit(
    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
)
print(nw_ols.summary())

# Specifically, the alpha, t-statistic, and Information ratio are:
print("CAPM Alpha:", nw_ols.params["Intercept"])
print("t-statistic:", nw_ols.tvalues["Intercept"])
print(
    "Information Ratio:",
    nw_ols.params["Intercept"] / np.sqrt(nw_ols.mse_resid) * np.sqrt(12),
)  # Information ratio is annualized

# Max one-month loss of the long-short Port
# # you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separatelyfolio
max_1m_loss = monthly_port["port_11"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Calculate Drawdown of the long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
monthly_port["log_port_11"] = np.log(
    monthly_port["port_11"] + 1
)  # calculate log returns
monthly_port["cumsum_log_port_11"] = monthly_port["log_port_11"].cumsum(
    axis=0
)  # calculate cumulative log returns
rolling_peak = monthly_port["cumsum_log_port_11"].cummax()
drawdowns = rolling_peak - monthly_port["cumsum_log_port_11"]
max_drawdown = drawdowns.max()
print("Maximum Drawdown:", max_drawdown)


# Calculate Turnover of the long portfolio and short portfolio
def turnover_count(df):
    # count the number of stocks at the begnning of each month
    start_stocks = df[["permno", "date"]].copy()
    start_stocks = start_stocks.sort_values(by=["date", "permno"])
    start_count = start_stocks.groupby(["date"])["permno"].count().reset_index()

    end_stocks = df[["permno", "date"]].copy()
    end_stocks["date"] = end_stocks["date"] - MonthBegin(
        1
    )  # shift the date to the beginning of the next month
    end_stocks = end_stocks.sort_values(by=["date", "permno"])

    remain_stocks = start_stocks.merge(end_stocks, on=["date", "permno"], how="inner")
    remain_count = (
        remain_stocks.groupby(["date"])["permno"].count().reset_index()
    )  # count the number of stocks that remain in the next month
    remain_count = remain_count.rename(columns={"permno": "remain_count"})

    port_count = start_count.merge(remain_count, on=["date"], how="inner")
    port_count["turnover"] = (
        port_count["permno"] - port_count["remain_count"]
    ) / port_count[
        "permno"
    ]  # calculate the turnover as the average of the percentage of stocks that are replaced each month
    return port_count["turnover"].mean()


long_positions = pred[pred["rank"] == 9]
short_positions = pred[pred["rank"] == 0]
print("Long Portfolio Turnover:", turnover_count(long_positions))
print("Short Portfolio Turnover:", turnover_count(short_positions))
