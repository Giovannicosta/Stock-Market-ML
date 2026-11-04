# First historical out-of-sample evaluation — 2026-11-04

**Status:** GitHub Actions run [41816198677](https://github.com/Giovannicosta/Stock-Market-ML/actions/runs/41816198677) completed successfully. [Metrics artifact](https://github.com/Giovannicosta/Stock-Market-ML/actions/runs/41816198677#artifacts). **This is a historical backtest, not forward/live trading.**

## Data
- Yahoo Finance via yfinance; `auto_adjust=True`.
- Daily historical data requested from 2000-01-01 through 2026-11-04 (exclusive).
- 10/10 downloads succeeded (9 equities plus SPY ETF). 6,750 sessions for all but GOOGL (5,588; starts 2004-08-19). Data last observed 2026-11-03.
- Four unit tests passed on synthetic data. No live orders were placed.

## Methodology
- Features available at the completed close of session t; estimate adjusted open(t+1) → open(t+2) return.
- Chronological 60% train, 20% validation, 20% untouched test; two-session label purge across split boundaries.
- Compare Ridge Regression and HistGradientBoosting; choose model + buy threshold from validation only.
- Long or flat, with a modeled transaction cost of 10 basis points **per side**.
- *Test* ranges: 2021-07-07–2026-10-30 for nine tickers; GOOGL 2022-06-07–2026-10-30.
- The benchmark is hypothetical continuous buy-and-hold in each **same stock** over the corresponding test observations. The reported benchmark excludes modeled entry/exit fees; this is a directional reference, not a perfectly matched trading strategy.

## Test outcomes (cumulative total returns)

| Ticker | Chosen model | ML strategy after fees | Same-asset buy-and-hold | Test Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|
| AAPL | Ridge | +4.75% | +136.12% | 0.15 | -30.11% |
| MSFT | Ridge | +61.88% | +91.90% | 0.52 | -34.75% |
| GOOGL | Ridge | -9.38% | +142.29% | -0.09 | -20.27% |
| AMZN | Ridge | -21.70% | +36.45% | 0.01 | -40.06% |
| NVDA | Ridge | +698.85% | +952.11% | 1.05 | -62.83% |
| JPM | HistGradientBoosting | +55.16% | +155.41% | 0.50 | -33.64% |
| JNJ | Ridge | -0.02% | +84.10% | 0.07 | -30.79% |
| XOM | Ridge | +57.11% | +262.17% | 0.54 | -26.95% |
| PG | Ridge | -7.32% | +36.79% | -0.01 | -25.57% |
| SPY | HistGradientBoosting | +10.58% | +88.30% | 0.21 | -25.08% |

**Result:** 0/10 ML strategies beat their same-asset buy-and-hold reference in this test interval. For the prediction MAE metric, 5/10 beat a naive zero-return forecast, but the differences are tiny and not evidence of tradable advantage. E.g., AAPL model MAE 0.01283 vs zero baseline 0.01272 (worse), SPY model MAE 0.00756 vs zero baseline 0.00748 (worse).

## Interpretation and limitations
- These tests **do not validate** buy/sell recommendations for real capital. Large drawdowns occurred, including roughly 63% for NVDA.
- A single held-out chronological split is insufficient to prove robustness. Threshold/model selection can overfit the validation segment. Need repeated walk-forward windows.
- Performance is sensitive to spread/slippage/timing; 10 bps/side is a simplified assumption.
- Adjusted historical opening prices cannot be assumed to equal executable opening prices.
- Survivorship bias: the ticker list only includes currently selected companies; delisted symbols are missing.
- No portfolio-level risk management, regime calibration, broker integration, or live paper-trading evidence.

## Next research milestones
1. Rolling walk-forward validation with a truly untouched test segment.
2. Compare to fee-adjusted buy-and-hold and a fixed broad ETF benchmark on identical periods.
3. Multi-asset portfolio with volatility scaling and drawdown-aware limits, evaluated conservatively.
4. Better point-in-time data (including delisted companies), corporate-action validation.
5. Paper-trading journal with forward-only signals before any consideration of real trades.

_Inputs: reports/summary.json and per-ticker JSON within GitHub Actions artifact from the run linked above. Generated 2026-11-04._
