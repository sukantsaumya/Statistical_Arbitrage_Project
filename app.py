import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import json
import requests

# Add src directory to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data_handler import DataHandler
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from src.visualization import Visualizer
from src.pair_finder import PairStats
from src.reporting import ReportGenerator

app = Flask(__name__)

MARKETS = {
    'sp500': {'name': 'USA - S&P 500', 'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
              'ticker_col': 'Symbol', 'name_col': 'Security', 'sector_col': 'GICS Sector', 'ticker_suffix': ''},
    'ftse100': {'name': 'UK - FTSE 100', 'url': 'https://en.wikipedia.org/wiki/FTSE_100_Index', 'ticker_col': 'EPIC',
                'name_col': 'Company', 'sector_col': 'FTSE Industry Classification', 'ticker_suffix': '.L'},
    'dax': {'name': 'Germany - DAX', 'url': 'https://en.wikipedia.org/wiki/DAX', 'ticker_col': 'Ticker',
            'name_col': 'Company', 'sector_col': 'Industry', 'ticker_suffix': '.DE'}
}


def get_company_info(market_key='sp500'):
    cache_file = Path(f'{market_key}_company_info_cache.json')
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
        with open(cache_file, 'r', encoding='utf-8') as f: return json.load(f)

    market_info = MARKETS.get(market_key)
    if not market_info: return {}

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(market_info['url'], headers=headers, timeout=10)
        response.raise_for_status()

        tables = pd.read_html(response.text)

        # --- ROBUST TABLE FINDING LOGIC ---
        df = None
        for table in tables:
            # Clean up multi-level columns if they exist
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = table.columns.get_level_values(-1)
            if market_info['ticker_col'] in table.columns:
                df = table
                break

        if df is None: raise ValueError(f"Could not find a table with the '{market_info['ticker_col']}' column.")

        df = df.rename(columns={market_info['ticker_col']: 'ticker', market_info['name_col']: 'name',
                                market_info['sector_col']: 'sector'})

    except Exception as e:
        print(f"CRITICAL ERROR scraping Wikipedia for {market_key}: {e}")
        return {}

    company_data = {}
    for _, row in df.iterrows():
        base_ticker = str(row.get('ticker', '')).replace('.', '-')
        if not base_ticker: continue  # Skip empty tickers

        yf_ticker = f"{base_ticker}{market_info['ticker_suffix']}"
        company_data[yf_ticker] = {'name': row.get('name', ''), 'logo_url': '', 'sector': row.get('sector', '')}

    yf_tickers_list = list(company_data.keys())
    for i in range(0, len(yf_tickers_list), 100):
        batch_str = " ".join(yf_tickers_list[i:i + 100])
        try:
            yf_data = yf.Tickers(batch_str)
            for ticker_key, ticker_obj in yf_data.tickers.items():
                if company_data.get(ticker_key) and hasattr(ticker_obj, 'info'):
                    logo = ticker_obj.info.get('logo_url', '')
                    company_data[ticker_key]['logo_url'] = logo
        except Exception:
            continue

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(company_data, f)
    return company_data


# --- (All other Flask routes and functions remain the same) ---
@app.route('/')
def index():
    config = Config.from_yaml('configs/default_config.yaml')
    return render_template('index.html', config=config)


@app.route('/company-info')
def company_info():
    return render_template('company_info.html', markets=MARKETS)


@app.route('/api/companies')
def api_companies():
    market_key = request.args.get('market', 'sp500')
    companies = get_company_info(market_key)
    return jsonify(companies)


@app.route('/run', methods=['POST'])
def run_backtest():
    # This is a placeholder for your working backtest logic.
    return "Backtest would run here."


if __name__ == '__main__':
    app.run(debug=True)