import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import cvxpy as cp

# ---- STEP 1: Load and Merge Data ----
# Load relevant columns from hackathon_sample_v2.csv and output.csv
hackathon_sample_v2_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/hackathon_sample_v2.csv"
output_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/output.csv"

hackathon_sample_v2 = pd.read_csv(hackathon_sample_v2_path, usecols=['permno', 'date', 'market_equity', 'be_me', 'ret_12_1', 'ivol_capm_21d', 'stock_exret'])
output = pd.read_csv(output_path, usecols=['permno', 'date', 'en'])

# Convert 'date' columns to datetime format to ensure compatibility
hackathon_sample_v2['date'] = pd.to_datetime(hackathon_sample_v2['date'])
output['date'] = pd.to_datetime(output['date'])

# Now merge the data on 'permno' and 'date'
pred = pd.merge(hackathon_sample_v2, output, on=['permno', 'date'], how='inner')

# ---- STEP 2: Multi-Signal Ensemble with Random Forest ----
# Include multiple signals (value, momentum, risk) alongside ElasticNet predictions
pred['value_signal'] = pred['be_me']  # Book-to-market ratio (value factor)
pred['momentum_signal'] = pred['ret_12_1']  # 12-month return (momentum factor)
pred['risk_signal'] = 1 / pred['ivol_capm_21d']  # Inverse volatility (risk factor)
pred['en_signal'] = pred['en']  # ElasticNet-predicted return

# Define the features (signals) and target (actual returns)
features = pred[['value_signal', 'momentum_signal', 'risk_signal', 'en_signal']]
target = pred['stock_exret']

# Train the Random Forest model to generate predicted returns from signals
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Generate predicted returns based on the multi-signal ensemble (including EN predictions)
pred['predicted_return'] = rf_model.predict(features)

# ---- STEP 3: Black-Litterman Optimization ----
# Define market weights and equilibrium return
# Assuming 'market_equity' is the market cap of each stock in your dataset
pred['market_weight'] = pred['market_equity'] / pred['market_equity'].sum()

# Calculate the market equilibrium return as the weighted average of stock returns
market_equilibrium_return = np.dot(pred['market_weight'], pred['stock_exret'])

# Define views using the predicted returns from the multi-signal model
views = pred['predicted_return']

# Blend the predicted returns (views) with market equilibrium returns
tau = 0.05  # Trust level in model predictions
bl_adjusted_returns = (1 - tau) * market_equilibrium_return + tau * views

# Covariance matrix based on historical returns
cov_matrix = pred[['stock_exret']].cov().values

# Optimize portfolio weights using Black-Litterman
n = len(bl_adjusted_returns)
weights = cp.Variable(n)
portfolio_return = bl_adjusted_returns @ weights
portfolio_risk = cp.quad_form(weights, cov_matrix)
objective = cp.Maximize(portfolio_return - portfolio_risk)  # Risk aversion factor could be added
constraints = [cp.sum(weights) == 1]
problem = cp.Problem(objective, constraints)
problem.solve()

# Get the optimized portfolio weights
optimal_weights = weights.value
pred['optimal_weight'] = optimal_weights

# ---- STEP 4: Construct the Long-Short Portfolio ----
# Rank stocks by the optimized weights and select 50 long and 50 short positions
pred = pred.sort_values(by='optimal_weight', ascending=False)
long_portfolio = pred.head(50)
short_portfolio = pred.tail(50)
final_portfolio = pd.concat([long_portfolio, short_portfolio])

# ---- STEP 5: Calculate Portfolio Performance Metrics ----
# 1. Portfolio return
final_portfolio['portfolio_return'] = final_portfolio['stock_exret'] * final_portfolio['optimal_weight']

# 2. Sharpe Ratio (Annualized)
sharpe_ratio = final_portfolio['portfolio_return'].mean() / final_portfolio['portfolio_return'].std() * np.sqrt(12)
print("Sharpe Ratio:", sharpe_ratio)

# 3. Annualized Return
annualized_return = final_portfolio['portfolio_return'].mean() * 12
print("Annualized Return:", annualized_return)

# 4. Annualized Standard Deviation
annualized_std_dev = final_portfolio['portfolio_return'].std() * np.sqrt(12)
print("Annualized Standard Deviation:", annualized_std_dev)

# ---- STEP 6: CAPM Alpha and Information Ratio ----
mkt_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/mkt_ind.csv"
mkt = pd.read_csv(mkt_path)

# Create 'mkt_rf' by subtracting the risk-free rate from market return
mkt['mkt_rf'] = mkt['sp_ret'] - mkt['rf']
final_portfolio = final_portfolio.merge(mkt, how="inner", on=["year", "month"])

# Perform CAPM regression (Newey-West standard errors)
nw_ols = sm.ols(formula="portfolio_return ~ mkt_rf", data=final_portfolio).fit(
    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
)
print(nw_ols.summary())

# CAPM Alpha, Information Ratio
alpha = nw_ols.params["Intercept"]
t_stat = nw_ols.tvalues["Intercept"]
info_ratio = alpha / np.sqrt(nw_ols.mse_resid) * np.sqrt(12)

print("CAPM Alpha:", alpha)
print("t-statistic:", t_stat)
print("Information Ratio:", info_ratio)

# ---- STEP 7: Maximum Drawdown ----
final_portfolio["log_portfolio_return"] = np.log(final_portfolio["portfolio_return"] + 1)
final_portfolio["cumsum_log_portfolio_return"] = final_portfolio["log_portfolio_return"].cumsum(axis=0)
rolling_peak = final_portfolio["cumsum_log_portfolio_return"].cummax()
drawdown = rolling_peak - final_portfolio["cumsum_log_portfolio_return"]
max_drawdown = drawdown.max()
print("Maximum Drawdown:", max_drawdown)

# ---- STEP 8: Turnover Calculation ----
def turnover_count(df):
    stock_counts = df.groupby(["year", "month"])["permno"].count().reset_index(name="count")
    stock_counts["count_next"] = stock_counts["count"].shift(-1)
    stock_counts["turnover"] = abs(stock_counts["count"] - stock_counts["count_next"]) / stock_counts["count"]
    return stock_counts["turnover"].mean()

long_turnover = turnover_count(long_portfolio)
short_turnover = turnover_count(short_portfolio)
print("Long Portfolio Turnover:", long_turnover)
print("Short Portfolio Turnover:", short_turnover)

# ---- STEP 9: Monthly Rebalancing ----
# Ensure to rebalance the portfolio monthly based on the new predicted returns and market data.
# You can automate the process by repeating the above steps for each new month.