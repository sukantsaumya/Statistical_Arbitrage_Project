"""
Module for finding and analyzing cointegrated pairs
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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
        """Check if pair meets all criteria"""
        return (self.pvalue < 0.05 and
                self.adf_pvalue < 0.05 and
                5 < self.half_life < 120)

class PairFinder:
    """Find and analyze cointegrated pairs for statistical arbitrage"""

    def __init__(self, significance_level: float = 0.05,
                 min_half_life: float = 5.0,
                 max_half_life: float = 120.0):
        """
        Initialize PairFinder
        """
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        # --- THIS LINE IS THE FIX ---
        self.logger = logging.getLogger(__name__)
        # ---------------------------

    def find_all_pairs(self, data: pd.DataFrame,
                       method: str = 'engle_granger') -> List[PairStats]:
        """
        Find all cointegrated pairs in the dataset
        """
        n = data.shape[1]
        tickers = data.columns.tolist()
        pairs = []
        total_combinations = n * (n - 1) // 2

        self.logger.info(f"Testing {total_combinations} pairs using the {method} method...")

        for i in range(n):
            for j in range(i + 1, n):
                ticker1, ticker2 = tickers[i], tickers[j]
                try:
                    if method == 'johansen':
                        pair_stats = self._analyze_pair_johansen(data[[ticker1, ticker2]], ticker1, ticker2)
                    else:
                        pair_stats = self._analyze_pair_engle_granger(data[ticker1], data[ticker2], ticker1, ticker2)

                    if pair_stats and pair_stats.is_valid:
                        pairs.append(pair_stats)
                except Exception as e:
                    self.logger.debug(f"Error analyzing pair ({ticker1}, {ticker2}): {e}")
                    continue

        pairs.sort(key=lambda x: x.pvalue)
        self.logger.info(f"Found {len(pairs)} valid cointegrated pairs")
        return pairs

    def _analyze_pair_engle_granger(self, series1: pd.Series, series2: pd.Series, ticker1: str, ticker2: str) -> Optional[PairStats]:
        """Analyzes a pair using the Engle-Granger two-step method."""
        coint_result = coint(series1, series2)
        pvalue = coint_result[1]
        if pvalue > self.significance_level:
            return None

        model = sm.OLS(series1, sm.add_constant(series2)).fit()
        hedge_ratio = model.params[1]
        spread = series1 - hedge_ratio * series2
        adf_result = adfuller(spread)
        adf_pvalue = adf_result[1]
        half_life = self._calculate_half_life(spread)

        return PairStats(
            ticker1=ticker1, ticker2=ticker2, pvalue=pvalue,
            hedge_ratio=hedge_ratio, spread_mean=spread.mean(),
            spread_std=spread.std(), half_life=half_life,
            correlation=series1.corr(series2), adf_pvalue=adf_pvalue
        )

    def _analyze_pair_johansen(self, data_pair: pd.DataFrame, ticker1: str, ticker2: str) -> Optional[PairStats]:
        """Analyzes a pair using the Johansen test."""
        result = coint_johansen(data_pair, det_order=0, k_ar_diff=1)
        trace_stat_r0 = result.lr1[0]
        crit_val_r0_95pct = result.cvm[0, 1]

        if trace_stat_r0 < crit_val_r0_95pct:
            return None

        hedge_ratio = -result.evec[1, 0] / result.evec[0, 0]
        pvalue = 0.01 # Placeholder p-value

        spread = data_pair[ticker1] - hedge_ratio * data_pair[ticker2]
        adf_result = adfuller(spread)
        adf_pvalue = adf_result[1]
        half_life = self._calculate_half_life(spread)

        return PairStats(
            ticker1=ticker1, ticker2=ticker2, pvalue=pvalue,
            hedge_ratio=hedge_ratio, spread_mean=spread.mean(),
            spread_std=spread.std(), half_life=half_life,
            correlation=data_pair[ticker1].corr(data_pair[ticker2]),
            adf_pvalue=adf_pvalue
            
        )

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion using OLS"""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        common_index = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common_index]
        spread_diff = spread_diff.loc[common_index]

        model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
        beta = model.params.iloc[1] if len(model.params) > 1 else 0

        if beta >= 0: return np.inf
        return -np.log(2) / beta

    def get_pair_summary(self, pairs: List[PairStats], top_n: int = 10) -> pd.DataFrame:
        """Create summary DataFrame of top pairs"""
        if not pairs: return pd.DataFrame()
        top_pairs = pairs[:top_n]
        summary_data = [{'Pair': f"{p.ticker1}-{p.ticker2}", 'Coint P-Value': f"{p.pvalue:.4f}", 'ADF P-Value': f"{p.adf_pvalue:.4f}", 'Hedge Ratio': f"{p.hedge_ratio:.3f}", 'Half-Life (days)': f"{p.half_life:.1f}", 'Correlation': f"{p.correlation:.3f}", 'Spread Std': f"{p.spread_std:.3f}"} for p in top_pairs]
        return pd.DataFrame(summary_data)
