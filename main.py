#!/usr/bin/env python3
"""
Statistical Arbitrage Pairs Trading System
Main entry point with CLI interface (FORCED BACKTEST VERSION)
"""
import argparse
import logging
import sys
from pathlib import Path
import warnings
import statsmodels.api as sm

# --- FIX FOR IMPORT ERROR ---
# Add the 'src' directory to the Python path
sys.path.append('src')
# ---------------------------

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import your custom modules
from config import Config
from data_handler import DataHandler
from backtester import Backtester
from performance import PerformanceAnalyzer


def setup_logging(log_level='INFO'):
    """Configure logging for the application"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def forced_backtest_command(args, config, logger):
    """
    Execute a forced backtest, bypassing the pair finding module.
    """
    logger.info("Starting FORCED backtesting process...")

    data_handler = DataHandler(cache_dir=config.data.cache_dir)

    universe = config.data.universe
    s1, s2 = universe[0], universe[1]
    pair_name = f"{s1}-{s2}"
    logger.info(f"Forcing backtest on pair: {pair_name}")

    # Load the full range of data needed
    price_data = data_handler.fetch_data(
        tickers=universe,
        start_date=config.data.in_sample_start,
        end_date=config.data.out_sample_end
    )

    train_data, test_data = data_handler.split_data(price_data, config.data.out_sample_start)

    # --- Manually define pair stats from training data ---
    model = sm.OLS(train_data[s1], sm.add_constant(train_data[s2])).fit()
    hedge_ratio = model.params[1]
    spread_train = train_data[s1] - hedge_ratio * train_data[s2]
    spread_mean = spread_train.mean()
    spread_std = spread_train.std()

    logger.info(
        f"Using calculated stats from in-sample data: Hedge Ratio={hedge_ratio:.4f}, Mean={spread_mean:.4f}, Std={spread_std:.4f}")

    # Initialize backtester
    backtester = Backtester(
        initial_capital=config.backtest.initial_capital,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        slippage_bps=config.backtest.slippage_bps
    )

    # Calculate Z-score for the entire dataset using stats from the training period
    spread = price_data[s1] - hedge_ratio * price_data[s2]
    z_scores = (spread - spread_mean) / spread_std

    # Run the backtest only on the out-of-sample (test) data
    results_df = backtester.run_backtest(
        price_data1=test_data[s1],
        price_data2=test_data[s2],
        z_scores=z_scores[test_data.index],
        pair_name=pair_name,
        hedge_ratio=hedge_ratio,
        entry_threshold=config.strategy.entry_z_threshold,
        exit_threshold=config.strategy.exit_z_threshold,
        stop_loss_threshold=config.strategy.stop_loss_z_threshold
    )

    # Analyze and display performance
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.calculate_metrics(results_df['portfolio_value'])
    report = analyzer.generate_report(metrics)

    print("\n" + "=" * 60)
    print("FORCED BACKTEST COMPLETE")
    print("=" * 60)
    print(report)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Statistical Arbitrage Pairs Trading System (Forced Backtest)')
    parser.add_argument('--config', type=str, default='configs/gld_gdx_config.yaml', help='Path to configuration file')

    args = parser.parse_args()
    logger = setup_logging()

    try:
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        forced_backtest_command(args, config, logger)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()