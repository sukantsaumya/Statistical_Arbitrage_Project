"""
Module for finding and analyzing cointegrated pairs
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from dataclasses import dataclass
import logging


@dataclass
class PairStats:
    """Container for pair statistics"""
    ticker1: str
    ticker2: str
    pvalue: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float
    correlation: float
    adf_pvalue: float

    @property
    def is_valid(self) -> bool:
        """Check if pair meets all criteria (temporarily relaxed for diagnostics)"""
        # We focus only on the core statistical significance first.
        return (self.pvalue < 0.05 and
                self.adf_pvalue < 0.05)


class PairFinder:
    """Find and analyze cointegrated pairs for statistical arbitrage"""

    def __init__(self, significance_level: float = 0.05,
                 min_half_life: float = 5.0,
                 max_half_life: float = 120.0):
        """
        Initialize PairFinder

        Args:
            significance_level: P-value threshold for cointegration
            min_half_life: Minimum acceptable half-life for mean reversion
            max_half_life: Maximum acceptable half-life
        """
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.logger = logging.getLogger(__name__)

    def find_all_pairs(self, data: pd.DataFrame,
                       method: str = 'engle_granger') -> List[PairStats]:
        """
        Find all cointegrated pairs in the dataset

        Args:
            data: DataFrame with price data for multiple assets
            method: Cointegration test method ('engle_granger' or 'johansen')

        Returns:
            List of PairStats objects for cointegrated pairs
        """
        n = data.shape[1]
        tickers = data.columns.tolist()
        pairs = []

        total_combinations = n * (n - 1) // 2
        self.logger.info(f"Testing {total_combinations} pair combinations...")

        for i in range(n):
            for j in range(i + 1, n):
                ticker1, ticker2 = tickers[i], tickers[j]

                try:
                    pair_stats = self._analyze_pair(
                        data[ticker1],
                        data[ticker2],
                        ticker1,
                        ticker2
                    )

                    if pair_stats and pair_stats.is_valid:
                        pairs.append(pair_stats)

                except Exception as e:
                    self.logger.debug(f"Error analyzing pair ({ticker1}, {ticker2}): {e}")
                    continue

        # Sort by p-value (lower is better)
        pairs.sort(key=lambda x: x.pvalue)

        self.logger.info(f"Found {len(pairs)} valid cointegrated pairs")
        return pairs

    def _analyze_pair(self, series1: pd.Series,
                      series2: pd.Series,
                      ticker1: str,
                      ticker2: str) -> Optional[PairStats]:
        """
        Comprehensive analysis of a potential pair

        Args:
            series1: Price series for first asset
            series2: Price series for second asset
            ticker1: Ticker symbol for first asset
            ticker2: Ticker symbol for second asset

        Returns:
            PairStats object if pair is cointegrated, None otherwise
        """
        # 1. Test for cointegration
        coint_result = coint(series1, series2)
        pvalue = coint_result[1]

        if pvalue > self.significance_level:
            return None

        # 2. Calculate hedge ratio using OLS
        model = sm.OLS(series1, sm.add_constant(series2)).fit()
        hedge_ratio = model.params[1]

        # 3. Calculate spread
        spread = series1 - hedge_ratio * series2
        spread_mean = spread.mean()
        spread_std = spread.std()

        # 4. Test spread for stationarity (ADF test)
        adf_result = adfuller(spread, autolag='AIC')
        adf_pvalue = adf_result[1]

        # 5. Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)

        # 6. Calculate correlation
        correlation = series1.corr(series2)

        # Create PairStats object
        pair_stats = PairStats(
            ticker1=ticker1,
            ticker2=ticker2,
            pvalue=pvalue,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            correlation=correlation,
            adf_pvalue=adf_pvalue
        )

        return pair_stats

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using OLS

        Args:
            spread: Spread series

        Returns:
            Half-life in periods (days)
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag

        # Remove NaN values
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        # Align indices
        spread_lag = spread_lag[spread_diff.index]

        # Run OLS regression: spread_t - spread_t-1 = alpha + beta * spread_t-1
        model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
        beta = model.params[1]

        # Calculate half-life
        if beta >= 0:
            # No mean reversion
            return np.inf

        half_life = -np.log(2) / beta
        return half_life

    def rank_pairs(self, pairs: List[PairStats],
                   weights: Dict[str, float] = None) -> List[PairStats]:
        """
        Rank pairs by multiple criteria

        Args:
            pairs: List of PairStats objects
            weights: Dictionary of weights for ranking criteria

        Returns:
            Sorted list of pairs
        """
        if not weights:
            weights = {
                'pvalue': -1.0,  # Lower is better
                'half_life_score': 1.0,  # Optimal range scoring
                'spread_stability': 1.0,  # Lower std is better
                'adf_pvalue': -1.0  # Lower is better
            }

        # Calculate scores for each pair
        scored_pairs = []
        for pair in pairs:
            # Half-life score (prefer values around 20-40 days)
            optimal_hl = 30
            hl_score = 1.0 / (1.0 + abs(pair.half_life - optimal_hl) / optimal_hl)

            # Spread stability score
            stability_score = 1.0 / (1.0 + pair.spread_std)

            # Combined score
            score = (
                    weights.get('pvalue', 0) * (1 - pair.pvalue) +
                    weights.get('half_life_score', 0) * hl_score +
                    weights.get('spread_stability', 0) * stability_score +
                    weights.get('adf_pvalue', 0) * (1 - pair.adf_pvalue)
            )

            scored_pairs.append((score, pair))

        # Sort by score (descending)
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        return [pair for _, pair in scored_pairs]

    def get_pair_summary(self, pairs: List[PairStats],
                         top_n: int = 10) -> pd.DataFrame:
        """
        Create summary DataFrame of top pairs

        Args:
            pairs: List of PairStats objects
            top_n: Number of top pairs to include

        Returns:
            DataFrame with pair statistics
        """
        if not pairs:
            return pd.DataFrame()

        top_pairs = pairs[:top_n]

        summary_data = []
        for pair in top_pairs:
            summary_data.append({
                'Pair': f"{pair.ticker1}-{pair.ticker2}",
                'Coint P-Value': f"{pair.pvalue:.4f}",
                'ADF P-Value': f"{pair.adf_pvalue:.4f}",
                'Hedge Ratio': f"{pair.hedge_ratio:.3f}",
                'Half-Life (days)': f"{pair.half_life:.1f}",
                'Correlation': f"{pair.correlation:.3f}",
                'Spread Std': f"{pair.spread_std:.3f}"
            })

        return pd.DataFrame(summary_data)