#!/usr/bin/env python3
"""
Statistical Arbitrage Pairs Trading System
Main entry point with CLI interface
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import your custom modules
from src.config import Config
from src.data_handler import DataHandler
from src.pair_finder import PairFinder
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer


def setup_logging(log_level='INFO', log_file=None):
    """Configure logging for the application"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file)
        handlers.append(file_handler)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def find_pairs_command(args, config, logger):
    """Execute pair finding command"""
    logger.info("Starting pair finding process...")

    data_handler = DataHandler(cache_dir=config.data.cache_dir)

    universe = args.symbols if args.symbols else config.data.universe
    start_date = args.start_date if args.start_date else config.data.in_sample_start
    end_date = args.end_date if args.end_date else config.data.in_sample_end

    logger.info(f"Loading data for symbols: {universe}")
    data = data_handler.fetch_data(universe, start_date, end_date)

    pair_finder = PairFinder(
        significance_level=config.strategy.significance_level,
        min_half_life=config.strategy.min_half_life,
        max_half_life=config.strategy.max_half_life
    )

    logger.info("Analyzing potential pairs...")
    pairs = pair_finder.find_all_pairs(data)

    if not pairs:
        logger.warning("No suitable pairs found with current criteria")
        return

    # Display results
    summary_df = pair_finder.get_pair_summary(pairs, top_n=10)
    print("\n" + "=" * 80)
    print("TOP 10 STATISTICAL ARBITRAGE PAIRS FOUND")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if args.output:
        output_path = Path('results') / args.output
        output_path.parent.mkdir(exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def backtest_command(args, config, logger):
    """Execute backtesting command"""
    logger.info("Starting backtesting process...")

    data_handler = DataHandler(cache_dir=config.data.cache_dir)

    # Load the full range of data needed
    price_data = data_handler.fetch_data(
        tickers=config.data.universe,
        start_date=config.data.in_sample_start,
        end_date=config.data.out_sample_end
    )

    train_data, test_data = data_handler.split_data(price_data, config.data.out_sample_start)

    # Find the best pair from the training data
    pair_finder = PairFinder(significance_level=config.strategy.significance_level)
    found_pairs = pair_finder.find_all_pairs(train_data)

    if not found_pairs:
        logger.error("No suitable pairs found to backtest.")
        return

    # Select the top pair for this example backtest
    best_pair_stats = found_pairs[0]
    s1, s2 = best_pair_stats.ticker1, best_pair_stats.ticker2
    pair_name = f"{s1}-{s2}"
    logger.info(f"Selected best pair for backtesting: {pair_name}")

    # Initialize backtester
    backtester = Backtester(
        initial_capital=config.backtest.initial_capital,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        slippage_bps=config.backtest.slippage_bps
    )

    # Calculate Z-score for the entire dataset using stats from the training period
    spread = price_data[s1] - best_pair_stats.hedge_ratio * price_data[s2]
    z_scores = (spread - best_pair_stats.spread_mean) / best_pair_stats.spread_std

    # Run the backtest only on the out-of-sample data
    results_df = backtester.run_backtest(
        price_data1=test_data[s1],
        price_data2=test_data[s2],
        z_scores=z_scores[test_data.index],  # Slice z_scores for the test period
        pair_name=pair_name,
        hedge_ratio=best_pair_stats.hedge_ratio,
        entry_threshold=config.strategy.entry_z_threshold,
        exit_threshold=config.strategy.exit_z_threshold,
        stop_loss_threshold=config.strategy.stop_loss_z_threshold
    )

    # Analyze and display performance
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.calculate_metrics(results_df['portfolio_value'])
    report = analyzer.generate_report(metrics)

    print(report)

    if args.output:
        output_path = Path('results') / args.output
        output_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_path)
        logger.info(f"Detailed equity curve saved to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Statistical Arbitrage Pairs Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Find pairs command
    parser_find = subparsers.add_parser('find-pairs', help='Find cointegrated pairs in the in-sample period')
    parser_find.add_argument('--symbols', nargs='+', help='Override symbols from config')
    parser_find.add_argument('--start-date', type=str, help='Override start date (YYYY-MM-DD)')
    parser_find.add_argument('--end-date', type=str, help='Override end date (YYYY-MM-DD)')
    parser_find.add_argument('--output', type=str, help='Output CSV file for results')

    # Backtest command
    parser_backtest = subparsers.add_parser('backtest', help='Run backtesting on the best pair found')
    parser_backtest.add_argument('--output', type=str, help='Output file for detailed results')

    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    try:
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        config.validate()  # Validate the configuration

        if args.command == 'find-pairs':
            find_pairs_command(args, config, logger)
        elif args.command == 'backtest':
            backtest_command(args, config, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()