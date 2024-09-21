import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *


# read predcited values
pred_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/Quant-Hackathon/output.csv"
pred = pd.read_csv(pred_path, parse_dates=["date"])
# pred.columns = map(str.lower, pred.columns)

# select model (ridge as an example)
model = "en"

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

# 1. Calculate the Annualized Portfolio Returns for the long-short portfolio (Portfolio 11)
annualized_return = monthly_port["port_11"].mean() * 12
print("Annualized Return:", annualized_return)

# 2. Calculate the Annualized Standard Deviation for the long-short portfolio (Portfolio 11)
annualized_std_dev = monthly_port["port_11"].std() * np.sqrt(12)
print("Annualized Standard Deviation:", annualized_std_dev)

# Calculate the CAPM Alpha for the long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
mkt_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/Quant-Hackathon/mkt_ind.csv"
mkt = pd.read_csv(mkt_path)
# Create 'mkt_rf' by subtracting the risk-free rate (rf) from the market return (sp_ret)
mkt['mkt_rf'] = mkt['sp_ret'] - mkt['rf']
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
    # Group by 'year' and 'month' to count the number of stocks at the beginning of each month
    stock_counts = df.groupby(["year", "month"])["permno"].count().reset_index(name="count")

    # Shift the stock_counts by one month to compare stock holdings between consecutive months
    stock_counts["count_next"] = stock_counts["count"].shift(-1)

    # Calculate turnover as the absolute difference between months divided by the total stock count
    stock_counts["turnover"] = abs(stock_counts["count"] - stock_counts["count_next"]) / stock_counts["count"]

    # Return the average turnover, ignoring NaN values (for the last month with no next month)
    return stock_counts["turnover"].mean()

# Apply the function to the long and short portfolios
long_positions = pred[pred["rank"] == 9]
short_positions = pred[pred["rank"] == 0]
print("Long Portfolio Turnover:", turnover_count(long_positions))
print("Short Portfolio Turnover:", turnover_count(short_positions))
