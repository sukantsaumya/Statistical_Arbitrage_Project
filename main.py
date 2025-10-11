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
import json
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
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

    # Initialize data handler
    data_handler = DataHandler(
        cache_dir=Path(config.data['cache_dir']),
        data_source=config.data['source']
    )

    # Load data
    logger.info(f"Loading data for symbols: {config.symbols}")
    data = data_handler.load_data(
        symbols=config.symbols,
        start_date=args.start_date or config.backtest['start_date'],
        end_date=args.end_date or config.backtest['end_date']
    )

    # Initialize pair finder
    pair_finder = PairFinder(
        min_correlation=config.pair_selection['min_correlation'],
        max_correlation=config.pair_selection['max_correlation'],
        min_cointegration=config.pair_selection['min_cointegration'],
        lookback_period=config.pair_selection['lookback_period']
    )

    # Find pairs
    logger.info("Analyzing potential pairs...")
    pairs = pair_finder.find_pairs(data)

    if pairs.empty:
        logger.warning("No suitable pairs found with current criteria")
        return

    # Display results
    print("\n" + "=" * 80)
    print("STATISTICAL ARBITRAGE PAIRS FOUND")
    print("=" * 80)

    for idx, row in pairs.iterrows():
        print(f"\nPair {idx + 1}: {row['pair']}")
        print("-" * 40)
        print(f"  Correlation:     {row['correlation']:.4f}")
        print(f"  Cointegration:   {row['cointegration_pvalue']:.4f}")
        print(f"  Half-life:       {row['half_life']:.2f} days")
        print(f"  Hurst Exponent:  {row['hurst_exponent']:.4f}")
        print(f"  Score:           {row['score']:.4f}")

    # Save results if requested
    if args.output:
        output_path = Path('results') / args.output
        output_path.parent.mkdir(exist_ok=True)
        pairs.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return pairs


def backtest_command(args, config, logger):
    """Execute backtesting command"""
    logger.info("Starting backtesting process...")

    # Initialize data handler
    data_handler = DataHandler(
        cache_dir=Path(config.data['cache_dir']),
        data_source=config.data['source']
    )

    # Determine which pairs to test
    pairs_to_test = []

    if args.pair:
        # Test specific pair
        pairs_to_test = [tuple(args.pair)]
        logger.info(f"Testing specific pair: {args.pair}")
    elif args.pairs_file:
        # Load pairs from file
        pairs_df = pd.read_csv(args.pairs_file)
        pairs_to_test = [tuple(pair.split('-')) for pair in pairs_df['pair'].values]
        logger.info(f"Loaded {len(pairs_to_test)} pairs from {args.pairs_file}")
    else:
        # Find pairs automatically
        logger.info("No pairs specified, finding pairs automatically...")
        data = data_handler.load_data(
            symbols=config.symbols,
            start_date=args.start_date or config.backtest['start_date'],
            end_date=args.end_date or config.backtest['end_date']
        )

        pair_finder = PairFinder(
            min_correlation=config.pair_selection['min_correlation'],
            max_correlation=config.pair_selection['max_correlation'],
            min_cointegration=config.pair_selection['min_cointegration'],
            lookback_period=config.pair_selection['lookback_period']
        )

        pairs_df = pair_finder.find_pairs(data)

        if pairs_df.empty:
            logger.error("No suitable pairs found")
            return

        # Use top N pairs
        n_pairs = min(args.top_pairs or 5, len(pairs_df))
        pairs_to_test = [tuple(pair.split('-')) for pair in pairs_df.head(n_pairs)['pair'].values]
        logger.info(f"Selected top {n_pairs} pairs for backtesting")

    # Run backtests
    all_results = []

    for symbol1, symbol2 in pairs_to_test:
        logger.info(f"\nBacktesting pair: {symbol1}-{symbol2}")

        # Load data for the pair
        data = data_handler.load_data(
            symbols=[symbol1, symbol2],
            start_date=args.start_date or config.backtest['start_date'],
            end_date=args.end_date or config.backtest['end_date']
        )

        # Initialize backtester
        backtester = Backtester(
            initial_capital=config.backtest['initial_capital'],
            max_position_size=config.position_sizing['max_position_size'],
            transaction_cost=config.execution['transaction_cost'],
            slippage=config.execution['slippage']
        )

        # Run backtest
        results = backtester.run_backtest(
            symbol1=symbol1,
            symbol2=symbol2,
            data=data,
            entry_threshold=config.strategy['entry_z_score'],
            exit_threshold=config.strategy['exit_z_score'],
            stop_loss_threshold=config.strategy['stop_loss_z_score'],
            lookback_period=config.strategy['lookback_period']
        )

        all_results.append(results)

        # Display summary
        print(f"\n{'=' * 60}")
        print(f"BACKTEST RESULTS: {symbol1}-{symbol2}")
        print(f"{'=' * 60}")
        print(f"Total Return:        {results['total_return']:.2%}")
        print(f"Annualized Return:   {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']:.2%}")
        print(f"Win Rate:            {results['win_rate']:.2%}")
        print(f"Total Trades:        {results['total_trades']}")

    # Aggregate results if multiple pairs
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("PORTFOLIO SUMMARY")
        print(f"{'=' * 60}")

        avg_return = sum(r['total_return'] for r in all_results) / len(all_results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in all_results) / len(all_results)
        avg_win_rate = sum(r['win_rate'] for r in all_results) / len(all_results)

        print(f"Average Return:      {avg_return:.2%}")
        print(f"Average Sharpe:      {avg_sharpe:.2f}")
        print(f"Average Win Rate:    {avg_win_rate:.2%}")

    # Save results if requested
    if args.output:
        output_path = Path('results') / args.output
        output_path.parent.mkdir(exist_ok=True)

        # Convert results to DataFrame for saving
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return all_results


def analyze_command(args, config, logger):
    """Execute performance analysis command"""
    logger.info("Starting performance analysis...")

    # Load backtest results
    if not args.results_file:
        logger.error("Please specify a results file with --results-file")
        return

    results_path = Path(args.results_file)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    # Load the results
    if results_path.suffix == '.csv':
        results_df = pd.read_csv(results_path)
        # Convert string representation back to proper format if needed
        results = results_df.to_dict('records')
    elif results_path.suffix == '.json':
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        logger.error("Results file must be .csv or .json format")
        return

    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer()

    # Analyze results
    if isinstance(results, list) and len(results) > 0:
        if 'equity_curve' in results[0]:
            # Single backtest result with equity curve
            analysis = analyzer.analyze_backtest(results[0])

            # Generate report
            if args.report_type == 'detailed' or args.report_type == 'all':
                report = analyzer.generate_report(analysis)
                print("\n" + report)

            # Generate plots if requested
            if args.plot or args.report_type == 'all':
                logger.info("Generating performance plots...")
                analyzer.plot_performance(analysis)
        else:
            # Multiple backtest summaries
            logger.info(f"Analyzing {len(results)} backtest results...")

            # Create comparison report
            print(f"\n{'=' * 80}")
            print("PERFORMANCE COMPARISON")
            print(f"{'=' * 80}\n")

            # Sort by total return
            results_sorted = sorted(results, key=lambda x: x.get('total_return', 0), reverse=True)

            for i, result in enumerate(results_sorted, 1):
                pair = result.get('pair', f"Pair {i}")
                print(f"{i}. {pair}")
                print(f"   Return: {result.get('total_return', 0):.2%}")
                print(f"   Sharpe: {result.get('sharpe_ratio', 0):.2f}")
                print(f"   MaxDD:  {result.get('max_drawdown', 0):.2%}")
                print(f"   Trades: {result.get('total_trades', 0)}")
                print()
    else:
        logger.error("No valid results found in file")


def optimize_command(args, config, logger):
    """Execute parameter optimization command"""
    logger.info("Starting parameter optimization...")

    # This is a placeholder for optimization functionality
    # You could implement grid search, random search, or Bayesian optimization here

    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    print("\nOptimization parameters:")
    print(f"  Entry Z-Score range:    [{args.entry_min}, {args.entry_max}]")
    print(f"  Exit Z-Score range:     [{args.exit_min}, {args.exit_max}]")
    print(f"  Lookback period range:  [{args.lookback_min}, {args.lookback_max}]")
    print(f"  Optimization method:    {args.method}")

    logger.warning("Optimization feature is not yet implemented")
    print("\nNote: Full optimization functionality coming soon!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Statistical Arbitrage Pairs Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find pairs in default symbols
  python main.py find-pairs

  # Run backtest on specific pair
  python main.py backtest --pair AAPL MSFT

  # Run backtest on top 3 pairs with custom dates
  python main.py backtest --top-pairs 3 --start-date 2022-01-01 --end-date 2023-12-31

  # Analyze backtest results
  python main.py analyze --results-file results/backtest_results.csv --plot

  # Use custom configuration
  python main.py --config configs/aggressive_config.yaml backtest
        """
    )

    # Global arguments
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log-file', type=str,
                        help='Log file name (saved in logs/ directory)')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Find pairs command
    parser_find = subparsers.add_parser('find-pairs', help='Find cointegrated pairs')
    parser_find.add_argument('--symbols', nargs='+',
                             help='Override symbols from config')
    parser_find.add_argument('--start-date', type=str,
                             help='Start date (YYYY-MM-DD)')
    parser_find.add_argument('--end-date', type=str,
                             help='End date (YYYY-MM-DD)')
    parser_find.add_argument('--output', type=str,
                             help='Output CSV file for results')

    # Backtest command
    parser_backtest = subparsers.add_parser('backtest', help='Run backtesting')
    parser_backtest.add_argument('--pair', nargs=2, metavar=('SYMBOL1', 'SYMBOL2'),
                                 help='Specific pair to backtest')
    parser_backtest.add_argument('--pairs-file', type=str,
                                 help='CSV file containing pairs to backtest')
    parser_backtest.add_argument('--top-pairs', type=int,
                                 help='Number of top pairs to backtest')
    parser_backtest.add_argument('--start-date', type=str,
                                 help='Start date (YYYY-MM-DD)')
    parser_backtest.add_argument('--end-date', type=str,
                                 help='End date (YYYY-MM-DD)')
    parser_backtest.add_argument('--output', type=str,
                                 help='Output file for results')

    # Analyze command
    parser_analyze = subparsers.add_parser('analyze', help='Analyze backtest results')
    parser_analyze.add_argument('--results-file', type=str, required=True,
                                help='Path to backtest results file')
    parser_analyze.add_argument('--report-type', type=str, default='summary',
                                choices=['summary', 'detailed', 'all'],
                                help='Type of report to generate')
    parser_analyze.add_argument('--plot', action='store_true',
                                help='Generate performance plots')

    # Optimize command
    parser_optimize = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    parser_optimize.add_argument('--pair', nargs=2, metavar=('SYMBOL1', 'SYMBOL2'),
                                 help='Pair to optimize')
    parser_optimize.add_argument('--method', type=str, default='grid',
                                 choices=['grid', 'random', 'bayesian'],
                                 help='Optimization method')
    parser_optimize.add_argument('--entry-min', type=float, default=1.5,
                                 help='Minimum entry z-score')
    parser_optimize.add_argument('--entry-max', type=float, default=3.0,
                                 help='Maximum entry z-score')
    parser_optimize.add_argument('--exit-min', type=float, default=0.0,
                                 help='Minimum exit z-score')
    parser_optimize.add_argument('--exit-max', type=float, default=1.0,
                                 help='Maximum exit z-score')
    parser_optimize.add_argument('--lookback-min', type=int, default=20,
                                 help='Minimum lookback period')
    parser_optimize.add_argument('--lookback-max', type=int, default=100,
                                 help='Maximum lookback period')

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = args.log_file or f"pairs_trading_{timestamp}.log"
    logger = setup_logging(args.log_level, log_file)

    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Override config with command line arguments if provided
        if hasattr(args, 'symbols') and args.symbols:
            config.symbols = args.symbols

        # Execute command
        if args.command == 'find-pairs':
            find_pairs_command(args, config, logger)
        elif args.command == 'backtest':
            backtest_command(args, config, logger)
        elif args.command == 'analyze':
            analyze_command(args, config, logger)
        elif args.command == 'optimize':
            optimize_command(args, config, logger)
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