import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import cvxpy as cp

# ---- STEP 1: Load and Merge Data ----
# Load relevant columns from hackathon_sample_v2.csv and output.csv
hackathon_sample_v2_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/Quant-Hackathon/hackathon_sample_v2.csv"
output_path = "/Users/isaiah/Desktop/Career/Clubs : Groups/Quant Hackathon/McGill-FIAM Asset Management Hackathon/Quant-Hackathon/output.csv"

hackathon_sample_v2 = pd.read_csv(hackathon_sample_v2_path, usecols=['permno', 'date', 'market_equity', 'be_me', 'ret_12_1', 'ivol_capm_21d', 'stock_exret'])
output = pd.read_csv(output_path, usecols=['permno', 'date', 'en'])

# Convert 'date' in hackathon_sample_v2 from YYYYMMDD integer format to datetime
hackathon_sample_v2['date'] = pd.to_datetime(hackathon_sample_v2['date'], format='%Y%m%d')

# Convert 'date' in output to datetime if needed
output['date'] = pd.to_datetime(output['date'])
# Now merge the data on 'permno' and 'date'
pred = pd.merge(hackathon_sample_v2, output, on=['permno', 'date'], how='inner')

null_counts = pred.isnull().sum()
print("Number of null values per column:")
print(null_counts)

# Drop rows that have any null values
pred_cleaned = pred.dropna(axis=0)

# Display the number of rows before and after dropping rows with null values
print("\nNumber of rows before dropping rows with null values:", len(pred))
print("Number of rows after dropping rows with null values:", len(pred_cleaned))

pred = pred_cleaned

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
pred_cleaned['market_weight'] = pred_cleaned['market_equity'] / pred_cleaned['market_equity'].sum()
market_equilibrium_return = np.dot(pred_cleaned['market_weight'], pred_cleaned['stock_exret'])
views = pred_cleaned['predicted_return']

tau = 0.05
bl_adjusted_returns = (1 - tau) * market_equilibrium_return + tau * views

# Ensure bl_adjusted_returns is a NumPy array
bl_adjusted_returns = np.array(bl_adjusted_returns)

# Create covariance matrix based on asset returns
pivoted_returns = pred_cleaned.pivot(index='date', columns='permno', values='stock_exret')
cov_matrix = pivoted_returns.cov().values

# Ensure the covariance matrix is symmetric
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# Debugging: Check the dimensions
n_assets = len(bl_adjusted_returns)  # Number of assets (should match the covariance matrix size)
print(f"Number of assets (bl_adjusted_returns): {n_assets}")
print(f"Covariance matrix shape: {cov_matrix.shape}")

if cov_matrix.shape[0] != n_assets:
    raise Exception(f"Mismatch between the number of assets ({n_assets}) and the covariance matrix size ({cov_matrix.shape[0]})")

# Optimize portfolio weights
weights = cp.Variable(n_assets)

# Ensure correct shape for bl_adjusted_returns for matrix multiplication
portfolio_return = cp.matmul(bl_adjusted_returns, weights)

# Ensure the covariance matrix and weights are compatible
portfolio_risk = cp.quad_form(weights, cov_matrix)

# Define the objective to maximize return and minimize risk
objective = cp.Maximize(portfolio_return - portfolio_risk)
constraints = [cp.sum(weights) == 1]
problem = cp.Problem(objective, constraints)
problem.solve()

optimal_weights = weights.value
pred_cleaned['optimal_weight'] = optimal_weights

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