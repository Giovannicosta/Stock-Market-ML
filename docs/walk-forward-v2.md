# Walk-forward experiment v2

Run `python -m src.data --start 2000-01-01` followed by `python -m src.walk_forward --fee-bps 10`.

Four expanding validation folds select between pooled Ridge and Histogram Gradient Boosting and a return threshold, then evaluate the chosen strategy on the last 20% of sessions held out from selection. The long/cash portfolio holds up to three equities, rebalanced daily at hypothetical next adjusted opens. Compare returns and drawdowns versus passive SPY. The report also separates market regimes based on trailing SPY price trend and volatility.

Output: `reports/walk_forward/summary.json`; CI uploads this report, not the market-data CSVs.

Research limitations: survivorship bias; adjusted opens are not real execution quotes; slippage and taxes are omitted; the sealed test must not be reused to tune hyperparameters. No real orders are placed.
