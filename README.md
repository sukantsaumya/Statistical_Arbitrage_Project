# 📊 Statistical Arbitrage — Pairs Trading Framework

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Quantitative Trading](https://img.shields.io/badge/strategy-pairs--trading-orange.svg)](#)

> **A modular, professional-grade quantitative trading system** implementing a market-neutral *Statistical Arbitrage (Pairs Trading)* strategy — complete with pair discovery, event-driven backtesting, and performance analytics.  
> Inspired by research frameworks like **Hudson & Thames’ mlfinlab** and **QuantConnect’s Alpha Framework**.

---

## 🚀 Overview

This repository houses a fully extensible Python framework for designing, testing, and visualizing **Statistical Arbitrage (Pairs Trading)** strategies.

The system:
- Identifies cointegrated stock pairs via rigorous statistical testing  
- Simulates realistic trades using an **event-driven backtester**  
- Evaluates portfolio performance using professional risk metrics  
- Provides both a **CLI interface** and an optional **Flask web app** for interactive exploration  

---

## 🧠 Core Strategy Concept

**Pairs Trading** is a market-neutral mean-reversion strategy that profits from temporary price divergence between two historically correlated assets.

1. **Find a pair:** Identify two stocks that move together (cointegrated pair, e.g., KO & PEP).  
2. **Monitor spread:** Compute the price spread and z-score to detect deviations.  
3. **Trade:**  
   - If spread widens → short the outperformer, long the underperformer  
   - If spread narrows → reverse the positions  
4. **Close:** Exit when spread reverts to the mean.

---

## 🧩 Architecture

```
.
├─ configs/                # YAML configs (parameters, thresholds, data universe)
├─ src/
│   ├─ data_handler.py     # Fetches & preprocesses historical data (via yfinance)
│   ├─ pair_finder.py      # Runs cointegration tests & selects valid pairs
│   ├─ backtester.py       # Event-driven backtesting engine
│   ├─ performance.py      # Performance analytics (Sharpe, Drawdown, CAGR, etc.)
│   ├─ visualization.py    # Generates plots & equity curves
│   └─ __init__.py
├─ templates/              # HTML files for Flask web app
├─ static/                 # Static assets (CSS/JS)
├─ main.py                 # CLI entrypoint
├─ app.py                  # Flask web server entrypoint
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Installation

Recommended: use a virtual environment.

```bash
# clone the repository
git clone https://github.com/sukantsaumya/Statistical_Arbitrage_Project.git
cd Statistical_Arbitrage_Project

# create virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# install dependencies
pip install -r requirements.txt
```

---

## 🧭 Quick Start

### 🔹 Find Cointegrated Pairs (CLI)
```bash
python main.py find-pairs --config configs/default_config.yaml
```

### 🔹 Backtest the Strategy
```bash
python main.py backtest --config configs/default_config.yaml
```

### 🔹 Launch the Flask Web Interface
```bash
python app.py
# open http://127.0.0.1:5000/
```

---

## 🧾 Example Configuration (`configs/default_config.yaml`)

```yaml
data:
  tickers: ["KO", "PEP", "MCD", "SBUX"]
  start_date: "2015-01-01"
  end_date: "2024-12-31"
  frequency: "1D"

pair_finder:
  pvalue_threshold: 0.05
  lookback_window: 252
  half_life_max: 50

backtest:
  entry_z: 2.0
  exit_z: 0.5
  stop_loss: 3.0
  transaction_cost: 0.001
  initial_capital: 100000

visualization:
  show_equity_curve: true
  show_drawdown: true
```

---

## 📈 Sample Output Metrics

| Metric | Description | Example |
|---------|--------------|----------|
| Sharpe Ratio | Risk-adjusted return | 1.84 |
| Sortino Ratio | Downside-risk adjusted | 2.12 |
| Max Drawdown | Largest equity drop | -12.3% |
| CAGR | Annualized return | 14.6% |
| Win Rate | % profitable trades | 63% |

---

## 🧩 Key Technologies

| Category | Libraries |
|-----------|------------|
| Data | `pandas`, `numpy`, `yfinance` |
| Statistics | `statsmodels`, `scipy` |
| Visualization | `matplotlib`, `seaborn` |
| Config / I/O | `PyYAML`, `argparse` |
| Web App | `Flask` |

---

## 🧪 Testing

Run all tests:
```bash
pytest -q
```

You can add coverage gradually for:
- `DataHandler`
- `PairFinder`
- `Backtester`
- `PerformanceAnalyzer`

---

## 💡 Potential Extensions

- Multi-pair portfolio optimization  
- Regime detection for volatility filters  
- Bayesian or ML-based cointegration discovery  
- Real-time live trading via Alpaca or IBKR APIs  
- Streamlit/Dash interactive analytics  

---

## 👤 Author

<div align="center">

<div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="dark" data-type="VERTICAL" data-vanity="sukantsaumya" data-version="v1">
<a class="badge-base__link LI-simple-link" href="https://in.linkedin.com/in/sukantsaumya?trk=profile-badge">
<b>Sukant Saumya</b>
</a>
</div>

</div>

---

## 🧾 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

---

## 🌐 Acknowledgements

Inspired by open-source frameworks like:
- [QuantConnect](https://www.quantconnect.com/)
- [Hudson & Thames – mlfinlab](https://hudsonthames.org/)
- Academic works on Cointegration and Statistical Arbitrage
