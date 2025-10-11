"""
Visualization Module for Pairs Trading System
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional

class Visualizer:
    def __init__(self, style='plotly'):
        self.style = style

    def plot_performance_metrics(self, equity_curve: pd.Series,
                                 drawdown: pd.Series,
                                 trades: Optional[pd.DataFrame] = None):
        if self.style == 'plotly':
            return self._plot_performance_plotly(equity_curve, drawdown)
        else:
            return self._plot_performance_matplotlib(equity_curve, drawdown)

    def _plot_performance_plotly(self, equity_curve, drawdown):
        """Plotly implementation of performance dashboard"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Strategy Equity Curve', 'Drawdown')
        )

        # Equity Curve
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values,
                       mode='lines', name='Portfolio Value', line=dict(color='#3B82F6', width=2)),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                       mode='lines', name='Drawdown',
                       fill='tozeroy', line=dict(color='#EF4444', width=1)),
            row=2, col=1
        )

        # Update layout for the dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=50),
            height=600,
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            yaxis2_title="Drawdown (%)"
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#374151')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#374151')

        return fig

    def _plot_performance_matplotlib(self, equity_curve, drawdown):
        """Matplotlib implementation of performance dashboard"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=2, label='Portfolio Value')
        ax1.set_title('Strategy Equity Curve', fontsize=14)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(drawdown.index, 0, drawdown.values * 100, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig