# Stock Market ML

Python research pipeline to study 26+ years of daily US market data and evaluate **experimental** buy/flat/sell signals. No automated trading or brokerage access.

## Start

Install Python 3.10+ and run in the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data --start 2000-01-01
python -m src.model --fee-bps 10
```

On GitHub, see **Actions → Stock market ML research** to download, test and train on GitHub's runner (the Pull Request also triggers it). Results, including validation settings, held-out test metrics and the latest research signal, are saved in `reports/` locally. The GitHub runner only publishes a *metrics* artifact; market data CSVs are deliberately excluded.

## Project structure

```text
config/tickers.txt           Initial 9 stocks and the SPY market ETF
src/data.py                  Download and validate adjusted daily OHLCV
src/model.py                 Features, train/validation/test, backtesting and signal
tests/test_model.py          Synthetic regression and leakage tests
.github/workflows/research.yml   Tests and model evaluation on GitHub Actions
data/raw/                   Downloaded one-CSV-per-ticker (not committed)
data/metadata/              Source/provenance metadata (not committed)
reports/                    JSON evaluations and predictions (not committed)
```

## Modeling design

1. **Signal timing:** all features use data at or before the completed close of trading session *t*. Any *BUY* means a **hypothetical** fill at next adjusted open (*t+1*); the return target is from adjusted open (*t+1*) to adjusted open (*t+2*).
2. **Inputs:** daily and multiweek returns, rolling volatility, gap, intraday range, moving-average deviation and volume ratio.
3. **Models:** regularized linear Ridge and nonlinear Histogram Gradient Boosting. Both are tested against a **zero-return forecasting baseline**.
4. **Evaluation:** chronological 60% training / 20% validation / 20% held-out test. Labels at the split boundaries are purged for two sessions. Model and buy threshold are selected **only on validation data**, not the test period. The final forecast model is refitted on all *known* labels after evaluation.
5. **Trading simulator:** single equity, **long or cash**, positions decided once per session, transaction cost default **10 bps per trade side** (0.10% to buy; 0.10% to sell), no leverage and final liquidation. Metrics include total/annualized return, Sharpe, max drawdown, turnover, and a passive *same-asset* buy-and-hold reference. The simulator approximates execution using adjusted open prices and does **not** include slippage, market impact, spread variation, dividend taxation or real order fills.

**Important:** `FLAT/SELL` is a signal to stay in cash or hypothetically sell an existing long position. It is **not** a signal to short. The latest forecast is one-step research output, **not** evidence of future profitability. A single split and validation grid can overfit; repeat with rolling walk-forward tests and truly untouched live paper trading before considering actual capital.

## Dataset coverage and limitations

Data are requested from `2000-01-01` through the latest available trading session. End dates in `yfinance.download` are exclusive; no ticker necessarily covers all 26 years (e.g. later IPOs). We use `yfinance`'s `auto_adjust=True` OHLCV series, adjusted for corporate actions, which is a **proxy**, not directly executable live opening quotes. The present-day ticker list omits delisted companies, creating **survivorship bias**. SPY is an ETF, not a market universe. We skip unavailable tickers and log failures; we never invent data. Official exchange data and a point-in-time delisted universe are needed for higher-quality research.

Yahoo Finance's unofficial interface is intended for personal research. Respect the terms and licensing; do not commit, publish, or redistribute raw downloaded prices without permission. More: https://ranaroussi.github.io/yfinance/

## Next improvements

Walk-forward/purged cross-validation; broader survivorship-safe universe; paper-trading journal; data vendor with historical index membership; slippage and market-impact modeling; compare SPY benchmark on matching dates; probability calibration and regime analysis.



## V3 weekly research experiment

Run `python -m src.v3 --fee-bps 10` after downloading historical quotes.
V3 predicts each stock's **five-session return relative to SPY**, applies a
two-model disagreement filter, and falls back to SPY if active rotation does
not clear a validation-only hurdle. Results: `reports/v3/summary.json`.

See [V3 methodology](docs/v3-research-design.md). V3's historical test overlaps
the period already examined in V2, so it **does not constitute fresh prospective
evidence**. Real-money trades are intentionally unsupported.
