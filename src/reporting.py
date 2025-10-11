import os
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import time
from typing import Dict, Any, List


class ReportGenerator:
    """Generates a detailed HTML report for a backtest run."""

    def __init__(self, config: Any, pair_stats: Any, metrics: Dict[str, Any],
                 plot_html: str, trades: List[Any]):
        """
        Initialize the report generator with all necessary data.
        """
        self.config = config
        self.pair_stats = pair_stats
        self.metrics = metrics
        self.plot_html = plot_html
        self.trades = trades

        # Set up Jinja2 environment
        template_dir = Path(__file__).resolve().parent.parent / 'templates'
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate_html_report(self) -> str:
        """
        Renders the data into an HTML report and saves it to a file.

        Returns:
            The path to the saved report file.
        """
        template = self.env.get_template('report_template.html')

        report_data = {
            'pair_name': f"{self.pair_stats.ticker1}-{self.pair_stats.ticker2}",
            'run_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config,
            'pair_stats': self.pair_stats,
            'metrics': self.metrics,
            'plot_html': self.plot_html,
            'trades': self.trades
        }

        html_content = template.render(report_data)

        # Save the report to a file
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        filename = f"report_{report_data['pair_name']}_{time.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)