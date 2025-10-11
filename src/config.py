"""
Configuration module for Pairs Trading Strategy
"""
import yaml
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data retrieval and processing"""
    universe: List[str]
    in_sample_start: str
    in_sample_end: str
    out_sample_start: str
    out_sample_end: str
    data_source: str = 'yfinance'
    cache_dir: Optional[str] = './data_cache'


@dataclass
class StrategyConfig:
    """Configuration for trading strategy parameters"""
    significance_level: float = 0.05
    entry_z_threshold: float = 2.0
    exit_z_threshold: float = 0.5
    stop_loss_z_threshold: float = 3.0
    lookback_period: int = 60  # days for rolling calculations
    min_half_life: float = 5.0  # minimum half-life for mean reversion
    max_half_life: float = 120.0  # maximum half-life


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 100000.0
    position_size: float = 0.95  # percentage of capital to use
    transaction_cost_bps: float = 10.0  # basis points (0.1%)
    slippage_bps: float = 5.0  # basis points
    max_positions: int = 5  # maximum number of simultaneous pairs


@dataclass
class Config:
    """Main configuration container"""
    data: DataConfig
    strategy: StrategyConfig
    backtest: BacktestConfig

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict['data']),
            strategy=StrategyConfig(**config_dict['strategy']),
            backtest=BacktestConfig(**config_dict['backtest'])
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            strategy=StrategyConfig(**config_dict.get('strategy', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {}))
        )

    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check date ordering
        if self.data.in_sample_start >= self.data.in_sample_end:
            raise ValueError("In-sample start date must be before end date")

        if self.data.out_sample_start >= self.data.out_sample_end:
            raise ValueError("Out-of-sample start date must be before end date")

        # Check strategy parameters
        if self.strategy.entry_z_threshold <= self.strategy.exit_z_threshold:
            raise ValueError("Entry threshold must be greater than exit threshold")

        if self.strategy.stop_loss_z_threshold <= self.strategy.entry_z_threshold:
            raise ValueError("Stop loss threshold must be greater than entry threshold")

        # Check backtest parameters
        if self.backtest.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        if not 0 < self.backtest.position_size <= 1:
            raise ValueError("Position size must be between 0 and 1")

        return True