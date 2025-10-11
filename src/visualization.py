"""
Visualization Module for Pairs Trading System
Handles all plotting and visual analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Handles all visualization tasks for the pairs trading system"""

    def __init__(self, style='matplotlib'):
        """
        Initialize visualizer

        Parameters:
        -----------
        style : str
            Visualization library to use ('plotly' or 'matplotlib')
        """
        self.style = style

    def plot_price_series(self, data: pd.DataFrame, symbols: List[str],
                          title: str = "Price Series", normalize: bool = True):
        """
        Plot price series for multiple symbols
        """
        if self.style == 'plotly':
            return self._plot_price_series_plotly(data, symbols, title, normalize)
        else:
            return self._plot_price_series_matplotlib(data, symbols, title, normalize)

    def _plot_price_series_plotly(self, data: pd.DataFrame, symbols: List[str],
                                  title: str, normalize: bool):
        """Plotly implementation of price series plot"""
        fig = go.Figure()

        for symbol in symbols:
            if symbol in data.columns:
                y_data = data[symbol]
                if normalize:
                    y_data = 100 * y_data / y_data.iloc[0]

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=y_data,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price" + (" (Normalized)" if normalize else ""),
            hovermode='x unified',
            template='plotly_dark',
            height=500
        )

        return fig

    def _plot_price_series_matplotlib(self, data: pd.DataFrame, symbols: List[str],
                                      title: str, normalize: bool):
        """Matplotlib implementation of price series plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        for symbol in symbols:
            if symbol in data.columns:
                y_data = data[symbol]
                if normalize:
                    y_data = 100 * y_data / y_data.iloc[0]
                ax.plot(data.index, y_data, label=symbol, linewidth=2)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price' + (' (Normalized)' if normalize else ''), fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_performance_metrics(self, equity_curve: pd.Series,
                                 drawdown: pd.Series,
                                 trades: Optional[pd.DataFrame] = None):
        """
        Create comprehensive performance dashboard
        """
        if self.style == 'plotly':
            return self._plot_performance_plotly(equity_curve, drawdown, trades)
        else:
            return self._plot_performance_matplotlib(equity_curve, drawdown, trades)

    def _plot_performance_plotly(self, equity_curve, drawdown, trades):
        """Plotly implementation of performance dashboard"""
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Equity Curve', 'Drawdown', 'Daily Returns'), vertical_spacing=0.15)

        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100, mode='lines', name='Drawdown', fill='tozeroy'), row=2, col=1)

        daily_returns = equity_curve.pct_change().dropna()
        fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns.values * 100, mode='lines', name='Daily Returns'), row=3, col=1)

        fig.update_layout(title_text="Performance Dashboard", showlegend=False, template='plotly_dark', height=900)
        return fig


    def _plot_performance_matplotlib(self, equity_curve, drawdown, trades):
        """Matplotlib implementation of performance dashboard"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Equity Curve
        ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=2, label='Portfolio Value')
        ax1.set_title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2.fill_between(drawdown.index, 0, drawdown.values * 100, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1, label='Drawdown')
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig