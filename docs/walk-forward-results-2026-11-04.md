# V2 historical walk-forward experiment — 2026-11-04

The CI run on [GitHub Actions](https://github.com/Giovannicosta/Stock-Market-ML/actions/runs/41817075137) completed data download, unit tests and the pooled walk-forward backtest on real Yahoo Finance historical prices.

## Method

- 9 individual securities; SPY is the market benchmark.
- Expanding four-fold validation, with label purging, and a final chronological holdout (2021-08-13 through 2026-10-30).
- Ridge vs Histogram Gradient Boosting. Validation-only selection picked Gradient Boosting with forecast threshold 0.002 (0.20%).
- Each day, a simulated top-three long/cash strategy fills at the following *adjusted* open, with 10bps per transaction side.
- Portfolio exposure averaged 21.53%; turnover was approximately 417.80 traded equity fractions.

## Holdout results

| Metric | ML strategy | SPY buy and hold |
| --- | ---: | ---: |
| Cumulative return | **-19.15%** | **+81.26%** |
| Annualized return | -4.01% | +12.12% |
| Sharpe (zero risk-free assumption) | -0.058 | +0.755 |
| Maximum drawdown | -46.94% | -26.29% |

Mean absolute next-open return prediction error: **0.0136543** for the ML model versus **0.0136444** for a zero-return prediction (ML slightly worse).

## Conditional regime comparison

Mean ML daily return minus benchmark daily return, evaluated on the final holdout:

| Market regime (using only lagged SPY signals) | Sessions | Average daily excess |
| --- | ---: | ---: |
| Bear, high volatility | 158 | -0.0871% |
| Bear, normal volatility | 116 | -0.1161% |
| Bull, high volatility | 200 | -0.2094% |
| Bull, normal volatility | 836 | -0.0060% |

These are non-contiguous conditional averages, not independently tradable compounded returns.

## Interpretation

**The strategy failed this holdout and should not be traded.** Validation Sharpe was +0.499 for the chosen strategy; out-of-sample Sharpe was -0.058. This gap indicates potential overfitting or regime instability. Low exposure and frequent trading costs hurt comparisons with a permanently invested SPY benchmark. The asset universe includes only present-day survivors, leaving survivorship bias.

Next hypotheses for a **new future or strictly separate evaluation period**: less frequent rebalance, benchmark-relative forecasting, conservative trading-cost sensitivity, diversified point-in-time universe, uncertainty-based abstention, and paper trading. **Do not retune these parameters repeatedly to maximize this same holdout.**

No live orders. Current adjusted opens are research proxies rather than tradable quoted fills. Slippage, fees beyond 10bps assumptions, taxes, and market impact are omitted.
