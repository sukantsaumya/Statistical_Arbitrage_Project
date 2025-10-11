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
    entry_value: float


class Backtester:
    """Vectorized backtesting engine with realistic execution modeling"""

    def __init__(self, initial_capital: float = 100000,
                 transaction_cost_bps: float = 10.0,
                 slippage_bps: float = 5.0,
                 max_positions: int = 5):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage in basis points
            max_positions: Maximum simultaneous positions
        """
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_bps / 10000
        self.slippage_pct = slippage_bps / 10000
        self.max_positions = max_positions

        self.logger = logging.getLogger(__name__)

        # Reset state
        self.reset()

    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.daily_returns = []

    def run_backtest(self,
                     price_data1: pd.Series,
                     price_data2: pd.Series,
                     z_scores: pd.Series,
                     pair_name: str,
                     hedge_ratio: float,
                     entry_threshold: float = 2.0,
                     exit_threshold: float = 0.5,
                     stop_loss_threshold: float = 3.0) -> pd.DataFrame:
        """
        Run vectorized backtest for a single pair

        Args:
            price_data1: Prices for asset 1
            price_data2: Prices for asset 2
            z_scores: Z-scores of the spread
            pair_name: Name of the pair
            hedge_ratio: Hedge ratio between assets
            entry_threshold: Z-score for entry
            exit_threshold: Z-score for exit
            stop_loss_threshold: Z-score for stop loss

        Returns:
            DataFrame with backtest results
        """
        self.reset()

        # Generate trading signals (vectorized)
        signals = self._generate_signals_vectorized(
            z_scores, entry_threshold, exit_threshold, stop_loss_threshold
        )

        # Calculate spreads
        spreads = price_data1 - hedge_ratio * price_data2

        # Track portfolio value
        portfolio_values = []
        positions_value = []

        # Iterate through time periods
        for i, (date, signal) in enumerate(signals.items()):
            if i == 0:
                portfolio_values.append(self.cash)
                positions_value.append(0)
                continue

            current_spread = spreads[date]
            current_z = z_scores[date]
            price1 = price_data1[date]
            price2 = price_data2[date]

            # Update existing positions
            self._update_positions(date, price1, price2, hedge_ratio)

            # Process signals
            if pair_name not in self.positions and signal != Signal.NEUTRAL:
                # Enter new position
                if len(self.positions) < self.max_positions:
                    self._enter_position(
                        pair_name, date, signal, price1, price2,
                        hedge_ratio, current_spread, current_z
                    )

            elif pair_name in self.positions and signal == Signal.CLOSE:
                # Close position
                self._close_position(
                    pair_name, date, price1, price2,
                    hedge_ratio, current_z
                )

            # Calculate portfolio value
            total_position_value = sum(
                self._calculate_position_value(pos, price1, price2, hedge_ratio)
                for pos in self.positions.values()
            )

            total_value = self.cash + total_position_value
            portfolio_values.append(total_value)
            positions_value.append(total_position_value)

            # Track daily returns
            if i > 0:
                daily_return = (total_value / portfolio_values[i - 1]) - 1
                self.daily_returns.append(daily_return)

        # Create results DataFrame
        results = pd.DataFrame({
            'date': signals.index,
            'portfolio_value': portfolio_values,
            'cash': [self.cash] * len(portfolio_values),
            'positions_value': positions_value,
            'z_score': z_scores,
            'signal': [s.value for s in signals.values()]
        }).set_index('date')

        return results

    def _generate_signals_vectorized(self,
                                     z_scores: pd.Series,
                                     entry_thresh: float,
                                     exit_thresh: float,
                                     stop_thresh: float) -> pd.Series:
        """
        Generate trading signals using vectorized operations

        Args:
            z_scores: Z-score series
            entry_thresh: Entry threshold
            exit_thresh: Exit threshold
            stop_thresh: Stop loss threshold

        Returns:
            Series of Signal enums
        """
        signals = pd.Series(Signal.NEUTRAL, index=z_scores.index)
        position = Signal.NEUTRAL

        for i, (date, z) in enumerate(z_scores.items()):
            if position == Signal.NEUTRAL:
                # Check for entry
                if z > entry_thresh:
                    signals[date] = Signal.SHORT
                    position = Signal.SHORT
                elif z < -entry_thresh:
                    signals[date] = Signal.LONG
                    position = Signal.LONG

            elif position == Signal.SHORT:
                # Check for exit or stop loss
                if z < exit_thresh or z > stop_thresh:
                    signals[date] = Signal.CLOSE
                    position = Signal.NEUTRAL

            elif position == Signal.LONG:
                # Check for exit or stop loss
                if z > -exit_thresh or z < -stop_thresh:
                    signals[date] = Signal.CLOSE
                    position = Signal.NEUTRAL

        return signals

    def _enter_position(self, pair: str, date: pd.Timestamp,
                        signal: Signal, price1: float, price2: float,
                        hedge_ratio: float, spread: float, z_score: float):
        """
        Enter a new position with transaction costs and slippage

        Args:
            pair: Pair name
            date: Entry date
            signal: Trading signal (LONG or SHORT)
            price1: Price of asset 1
            price2: Price of asset 2
            hedge_ratio: Hedge ratio
            spread: Current spread
            z_score: Current z-score
        """
        # Calculate position size (use portion of available capital)
        available_capital = self.cash * 0.95  # Keep 5% as buffer
        position_size = available_capital / max(1, len(self.positions) + 1)

        # Apply slippage to prices
        if signal == Signal.LONG:
            # Long spread: Buy asset1, Sell asset2
            price1_exec = price1 * (1 + self.slippage_pct)
            price2_exec = price2 * (1 - self.slippage_pct)
        else:
            # Short spread: Sell asset1, Buy asset2
            price1_exec = price1 * (1 - self.slippage_pct)
            price2_exec = price2 * (1 + self.slippage_pct)

        # Calculate shares (dollar neutral positions)
        value_per_leg = position_size / 2
        shares1 = value_per_leg / price1_exec
        shares2 = (value_per_leg / price2_exec) * hedge_ratio

        if signal == Signal.SHORT:
            shares1 = -shares1
            shares2 = -shares2

        # Calculate transaction costs
        transaction_cost = abs(shares1 * price1_exec + shares2 * price2_exec) * self.transaction_cost_pct

        # Update cash
        self.cash -= abs(shares1 * price1_exec) + abs(shares2 * price2_exec) + transaction_cost

        # Create position
        self.positions[pair] = Position(
            pair=pair,
            direction=signal,
            entry_date=date,
            entry_spread=spread,
            entry_z_score=z_score,
            shares_asset1=shares1,
            shares_asset2=shares2,
            entry_value=position_size
        )

        self.logger.debug(f"Entered {signal.name} position in {pair} at z-score {z_score:.2f}")

    def _close_position(self, pair: str, date: pd.Timestamp,
                        price1: float, price2: float,
                        hedge_ratio: float, z_score: float):
        """
        Close an existing position with transaction costs

        Args:
            pair: Pair name
            date: Exit date
            price1: Price of asset 1
            price2: Price of asset 2
            hedge_ratio: Hedge ratio
            z_score: Current z-score
        """
        if pair not in self.positions:
            return

        position = self.positions[pair]

        # Apply slippage (opposite direction from entry)
        if position.direction == Signal.LONG:
            price1_exec = price1 * (1 - self.slippage_pct)
            price2_exec = price2 * (1 + self.slippage_pct)
        else:
            price1_exec = price1 * (1 + self.slippage_pct)
            price2_exec = price2 * (1 - self.slippage_pct)

        # Calculate exit value
        exit_value = (position.shares_asset1 * price1_exec +
                      position.shares_asset2 * price2_exec)

        # Calculate transaction costs
        transaction_cost = abs(position.shares_asset1 * price1_exec +
                               position.shares_asset2 * price2_exec) * self.transaction_cost_pct

        # Calculate P&L
        pnl = exit_value - position.entry_value - transaction_cost
        pnl_pct = pnl / position.entry_value

        # Update cash
        self.cash += exit_value - transaction_cost

        # Record trade
        self.trades.append(Trade(
            pair=pair,
            entry_date=position.entry_date,
            exit_date=date,
            direction=position.direction,
            entry_price=position.entry_spread,
            exit_price=price1 - hedge_ratio * price2,
            entry_z_score=position.entry_z_score,
            exit_z_score=z_score,
            pnl=pnl,
            pnl_pct=pnl_pct,
            status='closed'
        ))

        # Remove position
        del self.positions[pair]

        self.logger.debug(f"Closed position in {pair}. P&L: {pnl:.2f} ({pnl_pct:.2%})")

    def _update_positions(self, date: pd.Timestamp,
                          price1: float, price2: float,
                          hedge_ratio: float):
        """
        Update mark-to-market for all positions

        Args:
            date: Current date
            price1: Current price of asset 1
            price2: Current price of asset 2
            hedge_ratio: Hedge ratio
        """
        # In a real system, you'd update P&L for each position here
        pass

    def _calculate_position_value(self, position: Position,
                                  price1: float, price2: float,
                                  hedge_ratio: float) -> float:
        """
        Calculate current value of a position

        Args:
            position: Position object
            price1: Current price of asset 1
            price2: Current price of asset 2
            hedge_ratio: Hedge ratio

        Returns:
            Current position value
        """
        return position.shares_asset1 * price1 + position.shares_asset2 * price2

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive trade statistics

        Returns:
            Dictionary of trade statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_win': 0,
                'max_loss': 0,
                'avg_trade_duration': 0
            }

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))

        # Calculate average trade duration
        durations = []
        for trade in self.trades:
            if trade.exit_date:
                duration = (trade.exit_date - trade.entry_date).days
                durations.append(duration)

        avg_duration = np.mean(durations) if durations else 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else np.inf,
            'max_win': max([t.pnl for t in self.trades]) if self.trades else 0,
            'max_loss': min([t.pnl for t in self.trades]) if self.trades else 0,
            'avg_trade_duration': avg_duration
        }