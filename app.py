import os
from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import time

# Import your existing, well-structured modules
from src.config import Config
from src.data_handler import DataHandler
from src.pair_finder import PairFinder
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from src.visualization import Visualizer

# Initialize the Flask app
app = Flask(__name__)

# Ensure required directories exist
Path('static/plots').mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(exist_ok=True)


# --- Web Page Routes ---

@app.route('/')
def index():
    """Renders the main page with the input form."""
    # Load default config to populate the form
    config = Config.from_yaml('configs/default_config.yaml')
    return render_template('index.html', config=config)


@app.route('/run', methods=['POST'])
def run_backtest():
    """Handles the form submission and runs the full backtest process."""
    try:
        # --- 1. Load Configuration and User Input ---
        config = Config.from_yaml('configs/default_config.yaml')

        # Override config with form data
        tickers_input = request.form['universe']
        config.data.universe = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        config.strategy.entry_z_threshold = float(request.form['entry_z'])
        config.strategy.exit_z_threshold = float(request.form['exit_z'])
        config.strategy.stop_loss_z_threshold = float(request.form['stop_loss_z'])

        # --- 2. Data Handling ---
        data_handler = DataHandler(cache_dir=config.data.cache_dir)
        price_data = data_handler.fetch_data(
            tickers=config.data.universe,
            start_date=config.data.in_sample_start,
            end_date=config.data.out_sample_end
        )

        train_data, test_data = data_handler.split_data(price_data, config.data.out_sample_start)

        # --- 3. Pair Finding ---
        pair_finder = PairFinder(significance_level=config.strategy.significance_level)
        found_pairs = pair_finder.find_all_pairs(train_data)

        if not found_pairs:
            return render_template('error.html',
                                   message="No statistically significant pairs found with the given criteria.")

        best_pair_stats = found_pairs[0]
        s1, s2 = best_pair_stats.ticker1, best_pair_stats.ticker2
        pair_name = f"{s1}-{s2}"

        # --- 4. Backtesting ---
        backtester = Backtester(
            initial_capital=config.backtest.initial_capital,
            transaction_cost_bps=config.backtest.transaction_cost_bps,
            slippage_bps=config.backtest.slippage_bps
        )

        # Calculate Z-score on the full dataset using training stats
        spread = price_data[s1] - best_pair_stats.hedge_ratio * price_data[s2]
        z_scores = (spread - best_pair_stats.spread_mean) / best_pair_stats.spread_std

        backtest_df = backtester.run_backtest(
            price_data1=test_data[s1],
            price_data2=test_data[s2],
            z_scores=z_scores[test_data.index],
            pair_name=pair_name,
            hedge_ratio=best_pair_stats.hedge_ratio,
            entry_threshold=config.strategy.entry_z_threshold,
            exit_threshold=config.strategy.exit_z_threshold,
            stop_loss_threshold=config.strategy.stop_loss_z_threshold
        )

        # --- 5. Performance Analysis & Visualization ---
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(backtest_df['portfolio_value'])

        visualizer = Visualizer(style='matplotlib')  # Matplotlib is simpler for saving files

        equity_curve = backtest_df['portfolio_value']
        drawdown = (equity_curve / equity_curve.expanding().max()) - 1
        trades_df = pd.DataFrame(backtester.trades)

        # Generate and save the plot
        fig = visualizer.plot_performance_metrics(equity_curve, drawdown, trades_df)
        plot_filename = f"performance_{int(time.time())}.png"
        fig.savefig(f"static/plots/{plot_filename}")

        # --- 6. Render Results ---
        return render_template(
            'results.html',
            metrics=metrics,
            pair=pair_name,
            plot_image=plot_filename,
            best_pair_stats=best_pair_stats
        )

    except Exception as e:
        # A simple error page for now
        return render_template('error.html', message=str(e))


if __name__ == '__main__':
    # Using debug=True is fine for development
    app.run(debug=True)