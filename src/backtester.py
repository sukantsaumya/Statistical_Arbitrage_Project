"""
Backtesting engine for pairs trading strategy
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging


class Signal(Enum):
    """Trading signals"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
    CLOSE = 2


@dataclass
class Trade:
    """Individual trade record"""
    pair: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: Signal
    entry_price: float
    exit_price: Optional[float]
    entry_z_score: float
    exit_z_score: Optional[float]
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = 'open'


@dataclass
class Position:
    """Current position in a pair"""
    pair: str
    direction: Signal
    entry_date: pd.Timestamp
    entry_spread: float
    entry_z_score: float
    shares_asset1: float
    shares_asset2: float
    value_asset1: float
    value_asset2: float


class Backtester:
    """Vectorized backtesting engine with realistic execution modeling"""

    def __init__(self, initial_capital: float = 100000,
                 transaction_cost_bps: float = 10.0,
                 slippage_bps: float = 5.0,
                 max_positions: int = 5):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_bps / 10000
        self.slippage_pct = slippage_bps / 10000
        self.max_positions = max_positions
        self.logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

    def run_backtest(self, price_data1: pd.Series, price_data2: pd.Series, z_scores: pd.Series,
                     pair_name: str, hedge_ratio: float, entry_threshold: float = 2.0,
                     exit_threshold: float = 0.5, stop_loss_threshold: float = 3.0) -> pd.DataFrame:
        self.reset()
        signals = self._generate_signals_vectorized(z_scores, entry_threshold, exit_threshold, stop_loss_threshold)
        spreads = price_data1 - hedge_ratio * price_data2

        portfolio_values = []

        for date in signals.index:
            signal = signals.loc[date]
            price1 = price_data1.loc[date]
            price2 = price_data2.loc[date]
            current_z = z_scores.loc[date]

            if pair_name not in self.positions and signal in [Signal.LONG, Signal.SHORT]:
                if len(self.positions) < self.max_positions:
                    self._enter_position(pair_name, date, signal, price1, price2, hedge_ratio, z_scores.loc[date])

            elif pair_name in self.positions and signal == Signal.CLOSE:
                self._close_position(pair_name, date, price1, price2, hedge_ratio, current_z)

            # Mark-to-market portfolio value
            positions_value = sum(pos.shares_asset1 * price1 + pos.shares_asset2 * price2 for pos in self.positions.values())
            total_value = self.cash + positions_value
            portfolio_values.append(total_value)

        return pd.DataFrame({'portfolio_value': portfolio_values}, index=signals.index)


    def _generate_signals_vectorized(self, z_scores: pd.Series, entry_thresh: float, exit_thresh: float, stop_thresh: float) -> pd.Series:
        signals = pd.Series(Signal.NEUTRAL, index=z_scores.index)
        position = Signal.NEUTRAL
        for i, (date, z) in enumerate(z_scores.items()):
            if position == Signal.NEUTRAL:
                if z > entry_thresh:
                    signals.loc[date] = Signal.SHORT
                    position = Signal.SHORT
                elif z < -entry_thresh:
                    signals.loc[date] = Signal.LONG
                    position = Signal.LONG
            elif position == Signal.SHORT:
                if z < exit_thresh or z > stop_thresh:
                    signals.loc[date] = Signal.CLOSE
                    position = Signal.NEUTRAL
            elif position == Signal.LONG:
                if z > -exit_thresh or z < -stop_thresh:
                    signals.loc[date] = Signal.CLOSE
                    position = Signal.NEUTRAL
        return signals

    def _enter_position(self, pair: str, date: pd.Timestamp, signal: Signal,
                        price1: float, price2: float, hedge_ratio: float, z_score: float):

        capital_per_leg = self.initial_capital / self.max_positions / 2

        if signal == Signal.LONG: # Long spread: Long asset1, Short asset2
            price1_exec = price1 * (1 + self.slippage_pct)
            price2_exec = price2 * (1 - self.slippage_pct)
            shares1 = capital_per_leg / price1_exec
            shares2 = - (capital_per_leg / price2_exec)
        else: # SHORT spread: Short asset1, Long asset2
            price1_exec = price1 * (1 - self.slippage_pct)
            price2_exec = price2 * (1 + self.slippage_pct)
            shares1 = - (capital_per_leg / price1_exec)
            shares2 = capital_per_leg / price2_exec

        # Update cash: Buy is outflow, Short-sell is inflow
        cash_change = (shares2 * price2_exec) - (shares1 * price1_exec)
        transaction_cost = (abs(shares1 * price1_exec) + abs(shares2 * price2_exec)) * self.transaction_cost_pct
        self.cash += cash_change - transaction_cost

        self.positions[pair] = Position(
            pair=pair, direction=signal, entry_date=date,
            entry_spread=price1 - hedge_ratio * price2, entry_z_score=z_score,
            shares_asset1=shares1, shares_asset2=shares2,
            value_asset1=shares1 * price1, value_asset2=shares2 * price2
        )
        self.logger.debug(f"[{date.date()}] Entered {signal.name} {pair}")

    def _close_position(self, pair: str, date: pd.Timestamp, price1: float, price2: float,
                          hedge_ratio: float, z_score: float):

        if pair not in self.positions:
            return

        pos = self.positions[pair]

        # Apply slippage on exit
        if pos.shares_asset1 > 0: # Was long asset1, now sell
            price1_exec = price1 * (1 - self.slippage_pct)
        else: # Was short asset1, now buy back
            price1_exec = price1 * (1 + self.slippage_pct)

        if pos.shares_asset2 > 0: # Was long asset2, now sell
            price2_exec = price2 * (1 - self.slippage_pct)
        else: # Was short asset2, now buy back
            price2_exec = price2 * (1 + self.slippage_pct)

        # Update cash: Sell is inflow, Buy-back is outflow
        cash_change = (pos.shares_asset1 * price1_exec) + (pos.shares_asset2 * price2_exec)
        transaction_cost = abs(cash_change) * self.transaction_cost_pct
        self.cash += cash_change - transaction_cost

        # Calculate PnL for this specific trade
        entry_value = pos.value_asset1 + pos.value_asset2
        exit_value = pos.shares_asset1 * price1 + pos.shares_asset2 * price2
        pnl = exit_value - entry_value - transaction_cost

        self.trades.append(Trade(
            pair=pair, entry_date=pos.entry_date, exit_date=date,
            direction=pos.direction, entry_price=pos.entry_spread,
            exit_price=price1 - hedge_ratio * price2,
            entry_z_score=pos.entry_z_score, exit_z_score=z_score,
            pnl=pnl, status='closed'
        ))

        self.logger.debug(f"[{date.date()}] Closed {pos.direction.name} {pair}, PnL: ${pnl:.2f}")
        del self.positions[pair]