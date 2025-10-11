import os
from flask import Flask, render_template, request
from pathlib import Path
import time
import pandas as pd
import statsmodels.api as sm

# Import your existing, well-structured modules
from src.config import Config
from src.data_handler import DataHandler
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from src.visualization import Visualizer
# **THE KEY FIX**: Import the real PairStats class
from src.pair_finder import PairStats

# Initialize the Flask app
app = Flask(__name__)

# Ensure required directories exist
Path('static/plots').mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(parents=True, exist_ok=True)


# --- Web Page Routes ---

@app.route('/')
def index():
    """Renders the main page with the input form."""
    config = Config.from_yaml('configs/default_config.yaml')
    return render_template('index.html', config=config)


@app.route('/run', methods=['POST'])
def run_backtest():
    """
    Handles form submission. This version includes special logic
    to FORCE a backtest, bypassing the pair finder for demonstration.
    """
    try:
        # --- 1. Load Configuration and User Input ---
        config = Config.from_yaml('configs/gld_gdx_config.yaml')

        tickers_input = request.form['universe']
        universe = [ticker.strip().upper() for ticker in tickers_input.split(',')]

        # --- 2. Data Handling ---
        data_handler = DataHandler(cache_dir=config.data.cache_dir)
        price_data = data_handler.fetch_data(
            tickers=universe,
            start_date=config.data.in_sample_start,
            end_date=config.data.out_sample_end
        )
        train_data, test_data = data_handler.split_data(price_data, config.data.out_sample_start)

        s1, s2 = universe[0], universe[1]
        pair_name = f"{s1}-{s2}"

        # --- 3. Manually Calculate Pair Stats (Bypass Pair Finder) ---
        model = sm.OLS(train_data[s1], sm.add_constant(train_data[s2])).fit()
        hedge_ratio = model.params[s2]
        spread_train = train_data[s1] - hedge_ratio * train_data[s2]
        spread_mean = spread_train.mean()
        spread_std = spread_train.std()

        # --- 4. Backtesting ---
        backtester = Backtester(
            initial_capital=config.backtest.initial_capital,
            transaction_cost_bps=config.backtest.transaction_cost_bps,
            slippage_bps=config.backtest.slippage_bps
        )

        spread = price_data[s1] - hedge_ratio * price_data[s2]
        z_scores = (spread - spread_mean) / spread_std

        backtest_df = backtester.run_backtest(
            price_data1=test_data[s1],
            price_data2=test_data[s2],
            z_scores=z_scores[test_data.index],
            pair_name=pair_name,
            hedge_ratio=hedge_ratio,
            entry_threshold=float(request.form['entry_z']),
            exit_threshold=float(request.form['exit_z']),
            stop_loss_threshold=float(request.form['stop_loss_z'])
        )

        # --- 5. Performance Analysis & Visualization ---
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(backtest_df['portfolio_value'])

        visualizer = Visualizer(style='matplotlib')
        equity_curve = backtest_df['portfolio_value']
        drawdown = (equity_curve / equity_curve.expanding().max()) - 1
        trades_df = pd.DataFrame(backtester.trades)

        fig = visualizer.plot_performance_metrics(equity_curve, drawdown, trades_df)
        plot_filename = f"performance_{int(time.time())}.png"
        fig.savefig(f"static/plots/{plot_filename}")

        # --- 6. Render Results ---
        # **THE KEY FIX**: Create an INSTANCE of the imported PairStats class
        pair_stats_for_template = PairStats(
            ticker1=s1,
            ticker2=s2,
            pvalue=0.0,  # Placeholder
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=0.0,  # Placeholder
            correlation=train_data[s1].corr(train_data[s2]),
            adf_pvalue=0.0  # Placeholder
        )

        return render_template(
            'results.html',
            metrics=metrics,
            pair=pair_name,
            plot_image=plot_filename,
            best_pair_stats=pair_stats_for_template
        )

    except Exception as e:
        return render_template('error.html', message=str(e))


if __name__ == '__main__':
    app.run(debug=True)