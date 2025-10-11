"""
Data handling module for fetching and cleaning financial data
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime, timedelta


class DataHandler:
    """Handles data fetching, cleaning, and storage"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize DataHandler

        Args:
            cache_dir: Directory for caching data (None to disable caching)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def fetch_data(self, tickers: List[str],
                   start_date: str,
                   end_date: str,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical price data with caching and error handling

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with closing prices for all tickers

        Raises:
            ValueError: If no valid data can be retrieved
        """
        cache_key = self._get_cache_key(tickers, start_date, end_date)

        # Try to load from cache
        if use_cache and self.cache_dir:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info(f"Loaded data from cache: {cache_key}")
                return cached_data

        # Fetch fresh data
        try:
            self.logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")

            # Download with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        tickers,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        threads=True
                    )['Close']
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying...")

            # Handle single ticker case
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])

            # Clean the data
            data = self._clean_data(data)

            # Validate data
            if data.empty or data.shape[0] < 20:
                raise ValueError("Insufficient data retrieved")

            # Cache the data
            if self.cache_dir:
                self._save_to_cache(data, cache_key)

            return data

        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            raise ValueError(f"Unable to fetch valid data: {e}")

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data with comprehensive error handling

        Args:
            data: Raw price data

        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")

        # Create a copy to avoid modifying original
        cleaned = data.copy()

        # 1. Handle zero and negative values
        zero_count = (cleaned <= 0).sum().sum()
        if zero_count > 0:
            self.logger.warning(f"Found {zero_count} zero or negative values. Replacing with NaN.")
            cleaned[cleaned <= 0] = np.nan

        # 2. Check for extreme outliers (price changes > 50% in a day)
        returns = cleaned.pct_change()
        outliers = (returns.abs() > 0.5).sum().sum()
        if outliers > 0:
            self.logger.warning(f"Found {outliers} potential outliers (>50% daily change)")
            # Optional: Cap extreme values or investigate further

        # 3. Forward fill missing values (weekends, holidays)
        initial_nans = cleaned.isnull().sum().sum()
        cleaned = cleaned.ffill()

        # 4. Backward fill any remaining NaNs at the beginning
        cleaned = cleaned.bfill()

        # 5. Drop any remaining rows with NaN (shouldn't be any after forward/backward fill)
        cleaned = cleaned.dropna(how='any')

        filled_nans = initial_nans - cleaned.isnull().sum().sum()
        if filled_nans > 0:
            self.logger.info(f"Filled {filled_nans} missing values")

        # 6. Remove tickers with insufficient data
        min_required_points = 20
        valid_tickers = cleaned.columns[cleaned.notna().sum() >= min_required_points]

        if len(valid_tickers) < len(cleaned.columns):
            dropped = set(cleaned.columns) - set(valid_tickers)
            self.logger.warning(f"Dropping tickers with insufficient data: {dropped}")
            cleaned = cleaned[valid_tickers]

        # 7. Ensure data is sorted by date
        cleaned = cleaned.sort_index()

        self.logger.info(f"Data cleaning complete. Shape: {cleaned.shape}")
        return cleaned

    def split_data(self, data: pd.DataFrame,
                   split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets

        Args:
            data: Full dataset
            split_date: Date to split on

        Returns:
            Tuple of (train_data, test_data)
        """
        split_date = pd.to_datetime(split_date)
        train = data[data.index < split_date]
        test = data[data.index >= split_date]

        self.logger.info(f"Data split - Train: {train.shape}, Test: {test.shape}")
        return train, test

    def calculate_returns(self, prices: pd.DataFrame,
                          method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from prices

        Args:
            prices: Price data
            method: 'simple' or 'log' returns

        Returns:
            DataFrame of returns
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return method: {method}")

        return returns.dropna()

    def _get_cache_key(self, tickers: List[str],
                       start_date: str,
                       end_date: str) -> str:
        """Generate cache key for data request"""
        ticker_str = '_'.join(sorted(tickers))
        return f"{ticker_str}_{start_date}_{end_date}.pkl"

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache"""
        if self.cache_dir:
            cache_path = self.cache_dir / cache_key
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                self.logger.debug(f"Cached data to {cache_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cache data: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            # Check if cache is recent (less than 1 day old for recent dates)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(days=1):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}")

        return None