"""
Performance analysis and risk metrics calculation
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import logging


class PerformanceAnalyzer:
    """Calculate comprehensive performance and risk metrics"""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self,
                          portfolio_values: pd.Series,
                          benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics

        Args:
            portfolio_values: Series of portfolio values over time
            benchmark: Optional benchmark series for comparison

        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()

        # Basic metrics
        metrics = {
            'total_return': self._total_return(portfolio_values),
            'annualized_return': self._annualized_return(returns),
            'annualized_volatility': self._annualized_volatility(returns),
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'calmar_ratio': self._calmar_ratio(returns, portfolio_values),
            'max_drawdown': self._max_drawdown(portfolio_values),
            'max_drawdown_duration': self._max_drawdown_duration(portfolio_values),
            'win_rate': self._win_rate(returns),
            'profit_factor': self._profit_factor(returns),
            'tail_ratio': self._tail_ratio(returns),
            'common_sense_ratio': self._common_sense_ratio(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': self._value_at_risk(returns, 0.95),
            'cvar_95': self._conditional_value_at_risk(returns, 0.95),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'recovery_factor': self._recovery_factor(portfolio_values),
            'ulcer_index': self._ulcer_index(portfolio_values)
        }

        # Add benchmark comparison if provided
        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
            metrics.update({
                'beta': self._beta(returns, benchmark_returns),
                'alpha': self._alpha(returns, benchmark_returns),
                'correlation': returns.corr(benchmark_returns),
                'tracking_error': self._tracking_error(returns, benchmark_returns),
                'information_ratio': self._information_ratio(returns, benchmark_returns)
            })

        return metrics

    def _total_return(self, values: pd.Series) -> float:
        """Calculate total return"""
        return (values.iloc[-1] / values.iloc[0]) - 1

    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        n_periods = len(returns)
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (252 / n_periods) - 1

    def _annualized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = np.sqrt((downside_returns ** 2).mean())
        if downside_std == 0:
            return np.inf

        return np.sqrt(252) * excess_returns.mean() / downside_std

    def _calmar_ratio(self, returns: pd.Series, values: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        annual_return = self._annualized_return(returns)
        max_dd = abs(self._max_drawdown(values))

        if max_dd == 0:
            return np.inf

        return annual_return / max_dd

    def _max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        return drawdown.min()

    def _max_drawdown_duration(self, values: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        running_max = values.expanding().max()
        is_drawdown = values < running_max

        if not is_drawdown.any():
            return 0

        # Find consecutive drawdown periods
        drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
        drawdown_periods = is_drawdown.groupby(drawdown_groups).sum()

        return int(drawdown_periods.max())

    def _win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (% of positive returns)"""
        if len(returns) == 0:
            return 0
        return (returns > 0).sum() / len(returns)

    def _profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return np.inf

        return gains / losses

    def _tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        left_tail = abs(np.percentile(returns, 5))
        right_tail = np.percentile(returns, 95)

        if left_tail == 0:
            return np.inf

        return right_tail / left_tail

    def _common_sense_ratio(self, returns: pd.Series) -> float:
        """Calculate common sense ratio (tail ratio * profit factor)"""
        return self._tail_ratio(returns) * self._profit_factor(returns)

    def _value_at_risk(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def _conditional_value_at_risk(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def _recovery_factor(self, values: pd.Series) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        total_return = self._total_return(values)
        max_dd = abs(self._max_drawdown(values))

        if max_dd == 0:
            return np.inf

        return total_return / max_dd

    def _ulcer_index(self, values: pd.Series) -> float:
        """Calculate Ulcer Index (measure of downside volatility)"""
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max * 100
        squared_drawdowns = drawdown ** 2
        return np.sqrt(squared_drawdowns.mean())

    def _beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark"""
        aligned_returns, aligned_benchmark = self._align_series(returns, benchmark_returns)

        covariance = aligned_returns.cov(aligned_benchmark)
        benchmark_variance = aligned_benchmark.var()

        if benchmark_variance == 0:
            return 0

        return covariance / benchmark_variance

    def _alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate alpha (excess return over benchmark)"""
        beta = self._beta(returns, benchmark_returns)

        strategy_return = self._annualized_return(returns)
        benchmark_return = self._annualized_return(benchmark_returns)

        return strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

    def _tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error (std of return differences)"""
        aligned_returns, aligned_benchmark = self._align_series(returns, benchmark_returns)
        diff = aligned_returns - aligned_benchmark
        return diff.std() * np.sqrt(252)

    def _information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio (excess return / tracking error)"""
        tracking_error = self._tracking_error(returns, benchmark_returns)

        if tracking_error == 0:
            return 0

        aligned_returns, aligned_benchmark = self._align_series(returns, benchmark_returns)
        excess_return = (aligned_returns - aligned_benchmark).mean() * 252

        return excess_return / tracking_error

    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two series to have the same index"""
        common_index = series1.index.intersection(series2.index)
        return series1[common_index], series2[common_index]

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate formatted performance report

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)

        # Returns
        report.append("\n--- RETURNS ---")
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"Best Day: {metrics.get('best_day', 0):.2%}")
        report.append(f"Worst Day: {metrics.get('worst_day', 0):.2%}")

        # Risk
        report.append("\n--- RISK ---")
        report.append(f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        report.append(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Max Drawdown Duration: {metrics.get('max_drawdown_duration', 0)} days")
        report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
        report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
        report.append(f"Ulcer Index: {metrics.get('ulcer_index', 0):.2f}")

        # Risk-Adjusted Returns
        report.append("\n--- RISK-ADJUSTED RETURNS ---")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"Recovery Factor: {metrics.get('recovery_factor', 0):.2f}")

        # Distribution
        report.append("\n--- DISTRIBUTION ---")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Tail Ratio: {metrics.get('tail_ratio', 0):.2f}")
        report.append(f"Skewness: {metrics.get('skewness', 0):.2f}")
        report.append(f"Kurtosis: {metrics.get('kurtosis', 0):.2f}")

        # Benchmark Comparison (if available)
        if 'alpha' in metrics:
            report.append("\n--- BENCHMARK COMPARISON ---")
            report.append(f"Alpha: {metrics.get('alpha', 0):.2%}")
            report.append(f"Beta: {metrics.get('beta', 0):.2f}")
            report.append(f"Correlation: {metrics.get('correlation', 0):.2f}")
            report.append(f"Tracking Error: {metrics.get('tracking_error', 0):.2%}")
            report.append(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)