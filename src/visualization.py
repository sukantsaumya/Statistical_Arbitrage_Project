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

    def __init__(self, style='plotly'):
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

        Parameters:
        -----------
        data : pd.DataFrame
            Price data with symbols as columns
        symbols : List[str]
            List of symbols to plot
        title : str
            Plot title
        normalize : bool
            Whether to normalize prices to start at 100
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

    def plot_spread_analysis(self, spread: pd.Series, mean: float,
                             upper_band: float, lower_band: float,
                             entry_points: pd.Series = None,
                             exit_points: pd.Series = None):
        """
        Plot spread with bands and trading signals

        Parameters:
        -----------
        spread : pd.Series
            Spread time series
        mean : float
            Mean of spread
        upper_band : float
            Upper band (entry threshold)
        lower_band : float
            Lower band (entry threshold)
        entry_points : pd.Series
            Entry signals
        exit_points : pd.Series
            Exit signals
        """
        if self.style == 'plotly':
            return self._plot_spread_plotly(spread, mean, upper_band, lower_band,
                                            entry_points, exit_points)
        else:
            return self._plot_spread_matplotlib(spread, mean, upper_band, lower_band,
                                                entry_points, exit_points)

    def _plot_spread_plotly(self, spread, mean, upper_band, lower_band,
                            entry_points, exit_points):
        """Plotly implementation of spread plot"""
        fig = go.Figure()

        # Plot spread
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=spread.values,
            mode='lines',
            name='Spread',
            line=dict(color='blue', width=1.5)
        ))

        # Plot mean and bands
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[mean] * len(spread),
            mode='lines',
            name='Mean',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[upper_band] * len(spread),
            mode='lines',
            name='Upper Band',
            line=dict(color='red', width=1, dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[lower_band] * len(spread),
            mode='lines',
            name='Lower Band',
            line=dict(color='green', width=1, dash='dot')
        ))

        # Add entry/exit points if provided
        if entry_points is not None and not entry_points.empty:
            fig.add_trace(go.Scatter(
                x=entry_points.index,
                y=spread.loc[entry_points.index],
                mode='markers',
                name='Entry',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))

        if exit_points is not None and not exit_points.empty:
            fig.add_trace(go.Scatter(
                x=exit_points.index,
                y=spread.loc[exit_points.index],
                mode='markers',
                name='Exit',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))

        fig.update_layout(
            title="Spread Analysis with Trading Bands",
            xaxis_title="Date",
            yaxis_title="Spread (Z-Score)",
            hovermode='x unified',
            template='plotly_dark',
            height=500
        )

        return fig

    def _plot_spread_matplotlib(self, spread, mean, upper_band, lower_band,
                                entry_points, exit_points):
        """Matplotlib implementation of spread plot"""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot spread
        ax.plot(spread.index, spread.values, label='Spread', color='blue', linewidth=1.5)

        # Plot mean and bands
        ax.axhline(y=mean, color='gray', linestyle='--', label='Mean', alpha=0.7)
        ax.axhline(y=upper_band, color='red', linestyle=':', label='Upper Band', alpha=0.7)
        ax.axhline(y=lower_band, color='green', linestyle=':', label='Lower Band', alpha=0.7)

        # Add entry/exit points if provided
        if entry_points is not None and not entry_points.empty:
            ax.scatter(entry_points.index, spread.loc[entry_points.index],
                       color='green', marker='^', s=100, label='Entry', zorder=5)

        if exit_points is not None and not exit_points.empty:
            ax.scatter(exit_points.index, spread.loc[exit_points.index],
                       color='red', marker='v', s=100, label='Exit', zorder=5)

        ax.set_title('Spread Analysis with Trading Bands', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Spread (Z-Score)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, data: pd.DataFrame, title: str = "Correlation Matrix"):
        """
        Plot correlation heatmap

        Parameters:
        -----------
        data : pd.DataFrame
            Price data with symbols as columns
        title : str
            Plot title
        """
        corr_matrix = data.corr()

        if self.style == 'plotly':
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))

            fig.update_layout(
                title=title,
                template='plotly_dark',
                height=600,
                width=700
            )

            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                        center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig

    def plot_performance_metrics(self, equity_curve: pd.Series,
                                 drawdown: pd.Series,
                                 trades: pd.DataFrame = None):
        """
        Create comprehensive performance dashboard

        Parameters:
        -----------
        equity_curve : pd.Series
            Portfolio value over time
        drawdown : pd.Series
            Drawdown series
        trades : pd.DataFrame
            Trade history
        """
        if self.style == 'plotly':
            return self._plot_performance_plotly(equity_curve, drawdown, trades)
        else:
            return self._plot_performance_matplotlib(equity_curve, drawdown, trades)

    def _plot_performance_plotly(self, equity_curve, drawdown, trades):
        """Plotly implementation of performance dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown',
                            'Daily Returns', 'Returns Distribution',
                            'Monthly Returns', 'Trade Analysis'),
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # Equity Curve
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values,
                       mode='lines', name='Equity', line=dict(color='blue')),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                       mode='lines', name='Drawdown',
                       fill='tozeroy', line=dict(color='red')),
            row=1, col=2
        )

        # Daily Returns
        daily_returns = equity_curve.pct_change().dropna()
        fig.add_trace(
            go.Scatter(x=daily_returns.index, y=daily_returns.values * 100,
                       mode='lines', name='Daily Returns', line=dict(color='green')),
            row=2, col=1
        )

        # Returns Distribution
        fig.add_trace(
            go.Histogram(x=daily_returns.values * 100, nbinsx=50,
                         name='Returns Dist', marker_color='purple'),
            row=2, col=2
        )

        # Monthly Returns Heatmap
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.pivot_table(
            pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            }),
            index='Month', columns='Year', values='Return'
        )

        fig.add_trace(
            go.Heatmap(z=monthly_pivot.values,
                       x=monthly_pivot.columns,
                       y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_pivot)],
                       colorscale='RdYlGn', zmid=0),
            row=3, col=1
        )

        # Trade Analysis
        if trades is not None and not trades.empty:
            trade_returns = trades['return'].values * 100
            fig.add_trace(
                go.Bar(x=list(range(len(trade_returns))), y=trade_returns,
                       name='Trade Returns',
                       marker_color=['green' if r > 0 else 'red' for r in trade_returns]),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Performance Dashboard",
            showlegend=False,
            template='plotly_dark',
            height=900
        )

        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Year", row=3, col=1)
        fig.update_xaxes(title_text="Trade #", row=3, col=2)

        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Month", row=3, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=2)

        return fig

    def _plot_performance_matplotlib(self, equity_curve, drawdown, trades):
        """Matplotlib implementation of performance dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # Equity Curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=2)
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.fill_between(drawdown.index, 0, drawdown.values * 100,
                         color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Daily Returns
        ax3 = fig.add_subplot(gs[1, 0])
        daily_returns = equity_curve.pct_change().dropna()
        ax3.plot(daily_returns.index, daily_returns.values * 100,
                 'g-', linewidth=1, alpha=0.7)
        ax3.set_title('Daily Returns', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)

        # Returns Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(daily_returns.values * 100, bins=50, color='purple',
                 alpha=0.7, edgecolor='black')
        ax4.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

        # Monthly Returns Heatmap
        ax5 = fig.add_subplot(gs[2, 0])
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.pivot_table(
            pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            }),
            index='Month', columns='Year', values='Return'
        )

        sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=0, ax=ax5, cbar_kws={'label': 'Return (%)'})
        ax5.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
        ax5.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_pivot)])

        # Trade Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        if trades is not None and not trades.empty:
            trade_returns = trades['return'].values * 100
            colors = ['green' if r > 0 else 'red' for r in trade_returns]
            ax6.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7)
            ax6.set_title('Individual Trade Returns', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Trade Number')
            ax6.set_ylabel('Return (%)')
            ax6.grid(True, alpha=0.3)

        plt.suptitle('Performance Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def plot_pair_relationship(self, symbol1_prices: pd.Series,
                               symbol2_prices: pd.Series,
                               hedge_ratio: float):
        """
        Plot relationship between pair prices

        Parameters:
        -----------
        symbol1_prices : pd.Series
            Prices for first symbol
        symbol2_prices : pd.Series
            Prices for second symbol
        hedge_ratio : float
            Hedge ratio between the pair
        """
        if self.style == 'plotly':
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Price Relationship', 'Scatter Plot')
            )

            # Normalized prices
            norm1 = symbol1_prices / symbol1_prices.iloc[0] * 100
            norm2 = symbol2_prices / symbol2_prices.iloc[0] * 100

            fig.add_trace(
                go.Scatter(x=symbol1_prices.index, y=norm1,
                           mode='lines', name=symbol1_prices.name),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=symbol2_prices.index, y=norm2,
                           mode='lines', name=symbol2_prices.name),
                row=1, col=1
            )

            # Scatter plot
            fig.add_trace(
                go.Scatter(x=symbol1_prices, y=symbol2_prices,
                           mode='markers', name='Price Points',
                           marker=dict(size=3, opacity=0.5)),
                row=1, col=2
            )

            # Add regression line
            x_line = np.array([symbol1_prices.min(), symbol1_prices.max()])
            y_line = hedge_ratio * x_line
            fig.add_trace(
                go.Scatter(x=x_line, y=y_line,
                           mode='lines', name=f'Hedge Ratio: {hedge_ratio:.4f}',
                           line=dict(color='red', dash='dash')),
                row=1, col=2
            )

            fig.update_layout(
                title_text=f"Pair Relationship Analysis",
                template='plotly_dark',
                height=400
            )

            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Normalized prices
            norm1 = symbol1_prices / symbol1_prices.iloc[0] * 100
            norm2 = symbol2_prices / symbol2_prices.iloc[0] * 100

            ax1.plot(norm1.index, norm1, label=symbol1_prices.name, linewidth=2)
            ax1.plot(norm2.index, norm2, label=symbol2_prices.name, linewidth=2)
            ax1.set_title('Price Relationship (Normalized)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Normalized Price (Base = 100)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Scatter plot
            ax2.scatter(symbol1_prices, symbol2_prices, alpha=0.5, s=10)

            # Add regression line
            x_line = np.array([symbol1_prices.min(), symbol1_prices.max()])
            y_line = hedge_ratio * x_line
            ax2.plot(x_line, y_line, 'r--',
                     label=f'Hedge Ratio: {hedge_ratio:.4f}', linewidth=2)

            ax2.set_title('Price Scatter Plot', fontsize=12, fontweight='bold')
            ax2.set_xlabel(symbol1_prices.name)
            ax2.set_ylabel(symbol2_prices.name)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.suptitle('Pair Relationship Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()

            return fig

    def plot_rolling_statistics(self, spread: pd.Series, lookback: int = 30):
        """
        Plot rolling statistics of spread

        Parameters:
        -----------
        spread : pd.Series
            Spread time series
        lookback : int
            Lookback period for rolling calculations
        """
        # Calculate rolling statistics
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std

        if self.style == 'plotly':
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Spread with Rolling Bands', 'Rolling Volatility'),
                vertical_spacing=0.15
            )

            # Spread with bands
            fig.add_trace(
                go.Scatter(x=spread.index, y=spread.values,
                           mode='lines', name='Spread', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=spread.index, y=rolling_mean,
                           mode='lines', name='Rolling Mean',
                           line=dict(color='orange', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=spread.index, y=upper_band,
                           mode='lines', name='Upper Band',
                           line=dict(color='red', dash='dot')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=spread.index, y=lower_band,
                           mode='lines', name='Lower Band',
                           line=dict(color='green', dash='dot')),
                row=1, col=1
            )

            # Rolling volatility
            fig.add_trace(
                go.Scatter(x=spread.index, y=rolling_std,
                           mode='lines', name='Rolling Std',
                           fill='tozeroy', line=dict(color='purple')),
                row=2, col=1
            )

            fig.update_layout(
                title_text=f"Rolling Statistics (Window: {lookback})",
                template='plotly_dark',
                height=600
            )

            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Spread with bands
            ax1.plot(spread.index, spread.values, 'b-', label='Spread', linewidth=1.5)
            ax1.plot(rolling_mean.index, rolling_mean, 'orange',
                     label='Rolling Mean', linestyle='--', linewidth=1)
            ax1.plot(upper_band.index, upper_band, 'r:',
                     label='Upper Band', linewidth=1)
            ax1.plot(lower_band.index, lower_band, 'g:',
                     label='Lower Band', linewidth=1)
            ax1.set_title(f'Spread with Rolling Bands (Window: {lookback})',
                          fontsize=12, fontweight='bold')
            ax1.set_ylabel('Spread Value')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Rolling volatility
            ax2.fill_between(rolling_std.index, 0, rolling_std,
                             color='purple', alpha=0.3)
            ax2.plot(rolling_std.index, rolling_std, 'purple', linewidth=2)
            ax2.set_title('Rolling Volatility', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Standard Deviation')
            ax2.grid(True, alpha=0.3)

            plt.suptitle('Rolling Statistics Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()

            return fig