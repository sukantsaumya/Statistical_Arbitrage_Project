import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import seaborn as sns

# --- Phase 1: Setup and Data Collection (with Data Cleaning) ---

# Define Your Universe and Timeframe
# Using a smaller subset of NASDAQ 100 for faster execution in this example.
# You can expand this list with all NASDAQ 100 or S&P 100 tickers.
UNIVERSE = [
    'MSFT', 'AAPL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'PEP', 'COST', 'ADBE', 'CSCO', 'AVGO', 'QCOM', 'INTC'
]
IN_SAMPLE_START = '2018-01-01'
IN_SAMPLE_END = '2022-12-31'
OUT_OF_SAMPLE_START = '2023-01-01'
OUT_OF_SAMPLE_END = '2025-10-10'  # Use a recent date

print("Phase 1: Downloading historical stock data...")
# Download closing prices
all_data = yf.download(UNIVERSE, start=IN_SAMPLE_START, end=OUT_OF_SAMPLE_END)['Close']

# --- START OF DATA CLEANING ---
print("Cleaning data...")

# 1. Check for initial missing values
print(f"Initial missing values:\n{all_data.isnull().sum()[all_data.isnull().sum() > 0]}")

# 2. Replace any zero values with NaN so they can be filled
all_data.replace(0, np.nan, inplace=True)

# 3. Forward-fill missing values
all_data.fillna(method='ffill', inplace=True)

# 4. Drop any remaining NaN rows (usually at the very beginning of the data)
all_data.dropna(inplace=True)

print("Data cleaning complete.")
# --- END OF DATA CLEANING ---


# Split data into in-sample and out-of-sample periods
in_sample_data = all_data.loc[IN_SAMPLE_START:IN_SAMPLE_END]
out_of_sample_data = all_data.loc[OUT_OF_SAMPLE_START:OUT_OF_SAMPLE_END]

print(f"Data ready. In-sample period: {in_sample_data.shape[0]} days. Out-of-sample period: {out_of_sample_data.shape[0]} days.")
print("-" * 50)

# ... (The rest of your script for Phase 2, 3, 4, 5) ...


# --- Phase 2: Finding Co-integrated Pairs ---

def find_cointegrated_pairs(data, significance_level=0.05):
    """
    Iterates through all possible pairs of stocks and tests for co-integration.
    """
    n = data.shape[1]
    keys = data.columns
    pairs = []

    # Iterate through all unique pairs
    for i in range(n):
        for j in range(i + 1, n):
            stock1 = data[keys[i]]
            stock2 = data[keys[j]]

            # Run the Engle-Granger co-integration test
            result = coint(stock1, stock2)
            pvalue = result[1]

            # If p-value is below our threshold, we consider the pair co-integrated
            if pvalue < significance_level:
                pairs.append((keys[i], keys[j], pvalue))

    return pairs


print("Phase 2: Finding co-integrated pairs using in-sample data...")
cointegrated_pairs = find_cointegrated_pairs(in_sample_data)

if not cointegrated_pairs:
    print(
        "No co-integrated pairs found with the given significance level. Try a larger universe or different timeframe.")
else:
    # Sort pairs by the lowest p-value
    cointegrated_pairs.sort(key=lambda x: x[2])
    print(f"Found {len(cointegrated_pairs)} co-integrated pairs. Top 5:")
    for pair in cointegrated_pairs[:5]:
        print(f"  - Pair: ({pair[0]}, {pair[1]}), P-value: {pair[2]:.4f}")
print("-" * 50)

# --- Phase 3 & 4: Develop Logic and Backtest ---

# Let's select the best pair (lowest p-value) for our backtest
if cointegrated_pairs:
    best_pair = cointegrated_pairs[0]
    stock1_ticker = best_pair[0]
    stock2_ticker = best_pair[1]
    print(f"Phase 3 & 4: Backtesting the best pair: ({stock1_ticker}, {stock2_ticker})")

    # --- Develop Logic on IN-SAMPLE data ---
    in_sample_s1 = in_sample_data[stock1_ticker]
    in_sample_s2 = in_sample_data[stock2_ticker]

    # Calculate hedge ratio using linear regression (OLS)
    model = sm.OLS(in_sample_s1, sm.add_constant(in_sample_s2)).fit()
    hedge_ratio = model.params[1]

    # Calculate the spread for the in-sample period
    in_sample_spread = in_sample_s1 - hedge_ratio * in_sample_s2

    # Calculate the mean and std dev of the in-sample spread
    spread_mean = in_sample_spread.mean()
    spread_std = in_sample_spread.std()

    print(f"Trading logic parameters from in-sample data:")
    print(f"  - Hedge Ratio: {hedge_ratio:.4f}")
    print(f"  - Spread Mean: {spread_mean:.4f}")
    print(f"  - Spread Std Dev: {spread_std:.4f}")

    # --- Backtest on OUT-OF-SAMPLE data ---
    out_sample_s1 = out_of_sample_data[stock1_ticker]
    out_sample_s2 = out_of_sample_data[stock2_ticker]

    # Calculate the spread for the out-of-sample period
    out_sample_spread = out_sample_s1 - hedge_ratio * out_sample_s2

    # Calculate the Z-score for the out-of-sample spread
    # IMPORTANT: Use the mean and std from the IN-SAMPLE period
    z_score = (out_sample_spread - spread_mean) / spread_std

    # Define trading thresholds
    entry_threshold = 2.0
    exit_threshold = 0.5  # Exit when it gets close to the mean
    stop_loss_threshold = 3.0

    # Initialize portfolio
    initial_capital = 100000
    cash = initial_capital
    position = 0  # -1 for short spread, 1 for long spread, 0 for no position
    portfolio_value = []

    # Simulate trading
    for i in range(len(z_score)):
        current_z = z_score.iloc[i]

        # Entry Logic
        if position == 0:
            if current_z > entry_threshold:
                position = -1  # Short the spread (Short S1, Long S2)
            elif current_z < -entry_threshold:
                position = 1  # Long the spread (Long S1, Short S2)

        # Exit and Stop-Loss Logic
        elif position == -1:  # Currently short spread
            if current_z < exit_threshold or current_z > stop_loss_threshold:
                position = 0  # Close position
        elif position == 1:  # Currently long spread
            if current_z > -exit_threshold or current_z < -stop_loss_threshold:
                position = 0  # Close position

        # This is a simplified P&L calculation. A real backtest would track shares.
        # Here we just track capital based on daily spread changes when in a position.
        if position != 0:
            # Approximate daily P&L
            daily_pnl = position * (
                        out_sample_spread.iloc[i] - out_sample_spread.iloc[i - 1]) * 100  # *100 is position size
            cash += daily_pnl

        portfolio_value.append(cash)

    backtest_results = pd.DataFrame({'portfolio_value': portfolio_value}, index=out_of_sample_data.index)

    print("Backtest complete.")
    print("-" * 50)

    # --- Phase 5: Performance Analysis ---
    print("Phase 5: Analyzing performance...")

    # Calculate metrics
    total_return = (backtest_results['portfolio_value'].iloc[-1] / initial_capital) - 1
    daily_returns = backtest_results['portfolio_value'].pct_change().dropna()
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    # Calculate max drawdown
    running_max = backtest_results['portfolio_value'].cummax()
    drawdown = (backtest_results['portfolio_value'] - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

    # Plot 1: Equity Curve
    ax1.plot(backtest_results['portfolio_value'], label='Portfolio Value')
    ax1.set_title(f'Strategy Equity Curve for ({stock1_ticker}, {stock2_ticker})')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Z-Score with trading thresholds
    ax2.plot(z_score, label='Z-Score')
    ax2.axhline(entry_threshold, color='r', linestyle='--', label='Short Entry')
    ax2.axhline(-entry_threshold, color='g', linestyle='--', label='Long Entry')
    ax2.axhline(exit_threshold, color='orange', linestyle=':', label='Exit Threshold (Short)')
    ax2.axhline(-exit_threshold, color='orange', linestyle=':', label='Exit Threshold (Long)')
    ax2.axhline(0, color='black', linestyle='-')
    ax2.set_title('Spread Z-Score')
    ax2.set_ylabel('Z-Score')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("Cannot proceed to backtesting as no co-integrated pairs were found.")