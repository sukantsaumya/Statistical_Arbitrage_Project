# app.py (with date-window/preset handling + robust scraping & normalization)
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename

# third-party libs used in pipeline
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Add src directory to path so "from src..." works when running app.py from repo root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import project modules (these must exist in src/)
from src.config import Config
from src.data_handler import DataHandler
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from src.visualization import Visualizer
from src.pair_finder import PairFinder
from src.reporting import ReportGenerator

# Flask app
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Results / reports directory used by ReportGenerator (ensure it exists)
RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Market Definitions ---
MARKETS = {
    'sp500': {
        'name': 'USA - S&P 500',
        'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'ticker_col': 'Symbol', 'name_col': 'Security', 'sector_col': 'GICS Sector', 'ticker_suffix': ''
    },
    'ftse100': {
        'name': 'UK - FTSE 100',
        'url': 'https://en.wikipedia.org/wiki/FTSE_100_Index',
        'ticker_col': 'EPIC', 'name_col': 'Company', 'sector_col': 'FTSE Industry Classification', 'ticker_suffix': '.L'
    },
    'dax': {
        'name': 'Germany - DAX',
        'url': 'https://en.wikipedia.org/wiki/DAX',
        'ticker_col': 'Ticker', 'name_col': 'Company', 'sector_col': 'Industry', 'ticker_suffix': '.DE'
    }
}


# --- Helper: flexible column check ---
def column_matches(table_cols, desired_col):
    """Return True if table_cols contains something matching desired_col or common aliases."""
    desired = desired_col.lower().strip()
    aliases = {desired, 'symbol', 'ticker', 'epic', 'code', 'stock'}
    for col in table_cols:
        if str(col).lower().strip() in aliases:
            return True
    return False


# --- Helper: company scraping with caching and fallback ---
def get_company_info(market_key='sp500'):
    """
    Fetch company list from Wikipedia and cache to a JSON file (1 day).
    Returns dict {ticker: {name, logo_url, sector}} or {} on failure.
    """
    cache_file = project_root / f"{market_key}_company_info_cache.json"
    try:
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
            app.logger.debug("Using cached company info for %s", market_key)
            return json.loads(cache_file.read_text(encoding='utf-8'))
    except Exception:
        app.logger.exception("Failed to read cache, will re-fetch")

    market_info = MARKETS.get(market_key)
    if not market_info:
        return {}

    company_data = {}
    try:
        headers = {
            'User-Agent': 'statarb-pro/1.0 (+https://github.com/yourname/Statistical_Arbitrage_Project)'
        }
        resp = requests.get(market_info['url'], headers=headers, timeout=12)
        resp.raise_for_status()

        tables = pd.read_html(resp.text)
        df = None
        for table in tables:
            # normalize multiindex columns
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = table.columns.get_level_values(-1)
            if column_matches(table.columns, market_info['ticker_col']):
                df = table
                break

        if df is None:
            raise ValueError(f"Could not find a table with a ticker-like column on source page.")

        # Try to guess correct name/sector columns by common headers
        cols = [str(c).lower().strip() for c in df.columns]
        # map name_col if exists, else pick first non-ticker column
        if market_info['name_col'].lower() in cols:
            name_col = market_info['name_col']
        else:
            name_candidates = [c for c in df.columns if str(c).lower().strip() not in ('symbol', 'ticker', 'epic', 'code')]
            name_col = name_candidates[0] if name_candidates else df.columns[0]

        sector_col = market_info.get('sector_col', None)
        if sector_col not in df.columns:
            # try to find any column mentioning 'sector' or 'industry'
            sector_candidates = [c for c in df.columns if 'sector' in str(c).lower() or 'industry' in str(c).lower()]
            sector_col = sector_candidates[0] if sector_candidates else None

        # rename with best-effort mapping
        rename_map = {}
        # find ticker column name in df
        ticker_col = None
        for c in df.columns:
            if str(c).lower().strip() in ('symbol', 'ticker', 'epic', 'code'):
                ticker_col = c
                break
        if ticker_col is None:
            # fallback: first column
            ticker_col = df.columns[0]

        rename_map[ticker_col] = 'ticker'
        if name_col:
            rename_map[name_col] = 'name'
        if sector_col:
            rename_map[sector_col] = 'sector'

        df = df.rename(columns=rename_map)
        keep_cols = [c for c in ['ticker', 'name', 'sector'] if c in df.columns]
        df = df[keep_cols].dropna(subset=['ticker'])

        for _, row in df.iterrows():
            scraped_ticker = str(row.get('ticker', '')).strip()
            if not scraped_ticker:
                continue
            base = scraped_ticker.split('.')[0].split(' ')[0].replace('.', '-').rstrip('.,')
            yf_ticker = f"{base}{market_info.get('ticker_suffix','')}"
            company_data[yf_ticker] = {
                'name': row.get('name', '') if 'name' in row.index else '',
                'sector': row.get('sector', '') if 'sector' in row.index else '',
                'logo_url': ''
            }

        # Enrich logos via yfinance in batches (best-effort)
        tickers = list(company_data.keys())
        for i in range(0, len(tickers), 100):
            batch = tickers[i:i+100]
            try:
                tstr = " ".join(batch)
                yf_batch = yf.Tickers(tstr)
                for tk, info in yf_batch.tickers.items():
                    if tk in company_data and getattr(info, 'info', None):
                        company_data[tk]['logo_url'] = info.info.get('logo_url', '') or ''
            except Exception:
                app.logger.debug("yfinance logo fetch failed for batch starting at %d", i)

        # Save cache
        try:
            cache_file.write_text(json.dumps(company_data), encoding='utf-8')
        except Exception:
            app.logger.debug("Failed to write cache file, continuing without cache.")

        # If we got nothing, fallback later
        if company_data:
            return company_data

    except Exception:
        app.logger.exception("Error while scraping companies for %s", market_key)

    # fallback to cached data if exists
    try:
        if cache_file.exists():
            app.logger.warning("Using stale cached company data for %s", market_key)
            return json.loads(cache_file.read_text(encoding='utf-8'))
    except Exception:
        app.logger.debug("Failed to read fallback cache.")

    # final small fallback to a few known tickers so UI isn't empty
    app.logger.warning("Company scraping failed; returning small default set")
    return {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "logo_url": ""},
        "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "logo_url": ""},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Communication Services", "logo_url": ""}
    }


# --- Routes ---
@app.route('/')
def index():
    # return default config for the form, but don't fail if YAML is missing
    try:
        cfg = Config.from_yaml('configs/default_config.yaml')
    except Exception as e:
        app.logger.warning("Failed to load config YAML: %s", e)
        cfg = None
    return render_template('index.html', config=cfg, current_year=time.gmtime().tm_year)


@app.route('/company-info')
def company_info():
    return render_template('company_info.html', markets=MARKETS, current_year=time.gmtime().tm_year)
@app.route('/about')
def about():
    return render_template('about.html', current_year=time.gmtime().tm_year)


@app.route('/api/companies')
def api_companies():
    market_key = request.args.get('market', 'sp500')
    try:
        companies = get_company_info(market_key)
        return jsonify(companies)
    except Exception:
        app.logger.exception("api_companies failure")
        return jsonify({}), 500


@app.route('/run', methods=['GET', 'POST'])
def run_backtest():
    """
    Run the pipeline:
      - parse form
      - download/calculate data
      - find cointegrated pairs
      - run backtest on the best pair
      - compute metrics, plot (plot_html) and a report
    """
    if request.method == 'GET':
        # friendly message if user browses to /run directly
        return render_template('error.html', message="Please submit the backtest form from the homepage."), 400

    # Parse and validate inputs
    universe_raw = (request.form.get('universe') or '').strip()
    if not universe_raw:
        return render_template('error.html', message="Missing required field: 'universe'. Provide comma-separated tickers."), 400

    # load config with safe fallbacks
    try:
        cfg = Config.from_yaml('configs/default_config.yaml')
    except Exception:
        app.logger.exception("Failed to load config; using defaults")
        cfg = None

    def cfg_get(path, default=None):
        """Utility to get nested attributes safely. path is 'a.b.c'"""
        if cfg is None:
            return default
        cur = cfg
        for part in path.split('.'):
            cur = getattr(cur, part, None)
            if cur is None:
                return default
        return cur

    # parse numeric inputs with fallbacks
    def safe_float(name, fallback):
        val = request.form.get(name)
        if val in (None, ''):
            return fallback
        try:
            return float(val)
        except Exception:
            return fallback

    entry_z = safe_float('entry_z', cfg_get('strategy.entry_z_threshold', 2.0))
    exit_z = safe_float('exit_z', cfg_get('strategy.exit_z_threshold', 0.5))
    stop_loss_z = safe_float('stop_loss_z', cfg_get('strategy.stop_loss_z_threshold', 3.0))
    test_method = request.form.get('test_method', 'engle_granger')

    # ----- NEW: parse date window inputs -----
    preset_years = request.form.get('preset_years')  # e.g. "3"
    start_date_str = request.form.get('start_date')  # "YYYY-MM-DD" or empty
    end_date_str = request.form.get('end_date')
    out_sample_years = request.form.get('out_sample_years')  # e.g. "1" or "0.5"

    # helper parsers
    def parse_date(s):
        try:
            return pd.to_datetime(s).to_pydatetime()
        except Exception:
            return None

    def parse_float_or_none(s):
        try:
            return float(s) if s is not None and s != '' else None
        except Exception:
            return None

    # compute end_dt and start_dt with fallbacks to config or defaults
    end_dt = parse_date(end_date_str) or datetime.utcnow()
    if start_date_str:
        start_dt = parse_date(start_date_str)
    elif preset_years:
        years = parse_float_or_none(preset_years) or 3.0
        # use pandas DateOffset for safer arithmetic
        try:
            start_dt = end_dt - pd.DateOffset(years=int(years))
            start_dt = pd.to_datetime(start_dt).to_pydatetime()
        except Exception:
            start_dt = end_dt.replace(year=end_dt.year - int(years))
    else:
        # default to config in_sample_start or 3 years
        cfg_start = cfg_get('data.in_sample_start', None)
        if cfg_start:
            start_dt = parse_date(cfg_start)
        else:
            start_dt = end_dt - pd.DateOffset(years=3)
            start_dt = pd.to_datetime(start_dt).to_pydatetime()

    out_sample_years_val = parse_float_or_none(out_sample_years)
    if out_sample_years_val and out_sample_years_val > 0:
        try:
            out_sample_dt = end_dt - pd.DateOffset(days=int(out_sample_years_val * 365))
            out_sample_dt = pd.to_datetime(out_sample_dt).to_pydatetime()
        except Exception:
            out_sample_dt = end_dt.replace(year=end_dt.year - int(out_sample_years_val))
    else:
        out_sample_dt = None

    # convert to ISO strings for DataHandler
    def safe_strftime(dt):
        """Return 'YYYY-MM-DD' if dt is a valid datetime, else None."""
        try:
            if pd.isna(dt) or dt is None:
                return None
            return pd.to_datetime(dt).strftime('%Y-%m-%d')
        except Exception:
            return None

    in_sample_start_str = safe_strftime(start_dt)
    in_sample_end_str = safe_strftime(out_sample_dt or end_dt)
    out_sample_start_str = safe_strftime(out_sample_dt + pd.Timedelta(days=1)) if out_sample_dt is not None else None
    out_sample_end_str = safe_strftime(end_dt)

    # normalize tickers list
    raw_tickers = [t.strip().upper() for t in universe_raw.split(',') if t.strip()]
    universe = []
    for t in raw_tickers:
        # normalize dots -> dashes, keep suffixes (very basic)
        nt = t.replace('.', '-')
        universe.append(nt)

    if len(universe) < 2:
        return render_template('error.html', message="Please provide at least two tickers."), 400

    # --- Data fetch ---
    try:
        cache_dir = cfg_get('data.cache_dir', None)
        data_handler = DataHandler(cache_dir=cache_dir) if cache_dir is not None else DataHandler()

        # Use computed date strings; if None, DataHandler should handle defaults
        price_data = data_handler.fetch_data(
            tickers=universe,
            start_date=in_sample_start_str,
            end_date=out_sample_end_str
        )
        # Ensure we have columns and a proper index
        if price_data is None or price_data.empty:
            return render_template('error.html', message="No price data returned for the given tickers/timeframe."), 500

        # If we computed an explicit out-sample start, use it. Otherwise pass None to split_data
        train_data, test_data = data_handler.split_data(price_data, out_sample_start_str) if out_sample_start_str else (price_data, price_data)
    except Exception:
        app.logger.exception("Data fetching or splitting failed")
        return render_template('error.html', message="Data loading failed. See server logs."), 500

    # --- Pair finding ---
    try:
        significance = cfg_get('strategy.significance_level', 0.05)
        pf = PairFinder(significance_level=significance)
        found_pairs = pf.find_all_pairs(train_data, method=test_method)
        if not found_pairs:
            return render_template('error.html', message=f"No statistically significant pairs found using {test_method}."), 200

        best_pair_stats = found_pairs[0]
        s1, s2 = best_pair_stats.ticker1, best_pair_stats.ticker2
        pair_name = f"{s1}-{s2}"
    except Exception:
        app.logger.exception("Pair finding failed")
        return render_template('error.html', message="Pair finding error. See server logs."), 500

    # verify test data contains the pair
    if s1 not in test_data.columns or s2 not in test_data.columns:
        app.logger.error("Pair tickers not present in test data: %s, %s", s1, s2)
        return render_template('error.html', message=f"Selected pair tickers not found in test data: {s1}, {s2}"), 500

    # --- Backtest ---
    try:
        backtester = Backtester(
            initial_capital=cfg_get('backtest.initial_capital', 100000),
            transaction_cost_bps=cfg_get('backtest.transaction_cost_bps', 0.001),
            slippage_bps=cfg_get('backtest.slippage_bps', 0.0)
        )

        spread = price_data[s1] - best_pair_stats.hedge_ratio * price_data[s2]
        z_scores = (spread - best_pair_stats.spread_mean) / best_pair_stats.spread_std
        z_test = z_scores.reindex(test_data.index).fillna(method='ffill').fillna(method='bfill')

        backtest_df = backtester.run_backtest(
            price_data1=test_data[s1],
            price_data2=test_data[s2],
            z_scores=z_test,
            pair_name=pair_name,
            hedge_ratio=best_pair_stats.hedge_ratio,
            entry_threshold=entry_z,
            exit_threshold=exit_z,
            stop_loss_threshold=stop_loss_z
        )
    except Exception:
        app.logger.exception("Backtest execution failed")
        return render_template('error.html', message="Backtest failed. See server logs."), 500

    # --- Performance & visualization ---
    try:
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(backtest_df['portfolio_value'])

        visualizer = Visualizer(style='plotly')
        equity_curve = backtest_df['portfolio_value']
        drawdown = (equity_curve / equity_curve.expanding().max()) - 1
        fig = visualizer.plot_performance_metrics(equity_curve, drawdown)

        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True, 'displayModeBar': False})
    except Exception:
        app.logger.exception("Visualization or metrics failed")
        metrics = {}
        plot_html = None

    # --- Report generation (optional) ---
    report_filename = None
    export_csv = None
    try:
        rg = ReportGenerator(config=cfg, pair_stats=best_pair_stats, metrics=metrics, plot_html=plot_html, trades=getattr(backtester, 'trades', []))
        report_path = rg.generate_html_report()  # should return absolute or relative path
        if report_path:
            report_filename = Path(report_path).name
        if hasattr(rg, 'generate_csv'):
            csv_path = rg.generate_csv()
            export_csv = Path(csv_path).name if csv_path else None
    except Exception:
        app.logger.warning("Report generation failed; continuing without report file")

    # --- Normalize trades for template consumption (ensure 'date' exists) ---
    raw_trades = getattr(backtester, 'trades', []) or []
    normalized_trades = []
    for t in raw_trades:
        # Accept dicts, objects, or pandas Series
        if isinstance(t, dict):
            date_val = t.get('date') or t.get('entry_date') or t.get('timestamp') or ''
            # format pandas Timestamp if present
            if hasattr(date_val, 'strftime'):
                try:
                    date_val = date_val.strftime('%Y-%m-%d')
                except Exception:
                    date_val = str(date_val)
            normalized_trades.append({
                "date": date_val,
                "side": t.get("side", ""),
                "entry_price": t.get("entry_price", "") if t.get("entry_price", None) is not None else "",
                "exit_price": t.get("exit_price", "") if t.get("exit_price", None) is not None else "",
                "pnl": t.get("pnl", "") if t.get("pnl", None) is not None else "",
                "notes": t.get("notes", "") or ""
            })
        else:
            # object-like (namedtuple or custom class)
            date_val = getattr(t, "date", None) or getattr(t, "entry_date", None) or ""
            if hasattr(date_val, 'strftime'):
                try:
                    date_val = date_val.strftime('%Y-%m-%d')
                except Exception:
                    date_val = str(date_val)
            normalized_trades.append({
                "date": date_val,
                "side": getattr(t, "side", ""),
                "entry_price": getattr(t, "entry_price", ""),
                "exit_price": getattr(t, "exit_price", ""),
                "pnl": getattr(t, "pnl", ""),
                "notes": getattr(t, "notes", "") or ""
            })

    # final render
    try:
        return render_template(
            'results.html',
            metrics=metrics,
            trades=normalized_trades,
            plot_html=plot_html,
            pair=pair_name,
            start_date=str(test_data.index.min().date()) if hasattr(test_data.index, 'min') else in_sample_start_str,
            end_date=str(test_data.index.max().date()) if hasattr(test_data.index, 'max') else out_sample_end_str,
            report_filename=report_filename,
            export_csv=export_csv,
            current_year=time.gmtime().tm_year,
            best_pair_stats=best_pair_stats
        )
    except Exception:
        app.logger.exception("Rendering results template failed")
        return render_template('error.html', message="Rendering results failed. See server logs."), 500


# Serve generated HTML report (view in new tab)
@app.route('/results/<path:filename>')
def serve_report(filename):
    safe = secure_filename(filename)
    file_path = RESULTS_DIR / safe
    if not file_path.exists():
        abort(404)
    return send_from_directory(RESULTS_DIR, safe, as_attachment=False)


# Download CSV or other generated files (if available)
@app.route('/download/<path:filename>')
def download_results(filename):
    safe = secure_filename(filename)
    path = RESULTS_DIR / safe
    if not path.exists():
        abort(404)
    return send_from_directory(RESULTS_DIR, safe, as_attachment=True)


# Run local dev server
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
