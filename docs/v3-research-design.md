# V3: weekly SPY-relative rotation experiment

This version is an **experiment**, not a proven trading algorithm or a broker-connected service. We compare a stock-selection model against SPY, using the same historical price vendor and 10 basis points per transaction side.

## Run

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data --start 2000-01-01
python -m src.v3 --fee-bps 10
```

Results: `reports/v3/summary.json` locally or in the GitHub Actions `stock-ml-metrics` artifact.

## Key changes from V2

1. **Weekly rebalancing:** predict a five-trading-session return from adjusted open(t+1) to adjusted open(t+6); trades begin only after the signal day's close. A common SPY trading-session calendar prevents weekly horizons from overlapping.
2. **Relative prediction target:** stock five-session return **minus** SPY five-session return. This asks whether a stock could outperform the benchmark, not whether all stocks rise in a bull market.
3. **Conservative signal filter:** pool Ridge and Histogram Gradient Boosting. Only overweight a stock when *both* models forecast positive excess and the ensemble prediction minus a multiple of their disagreement exceeds a validated threshold. Disagreement is a heuristic, **not calibrated uncertainty or a confidence level**.
4. **SPY fallback:** invest in SPY when no candidate qualifies or when no validation candidate shows a meaningful advantage. Otherwise allocate equally to up to three selected equities. This tests whether active stock rotation can beat passive SPY rather than comparing a cash-heavy strategy with an invested benchmark.
5. **Expanding walk-forward:** four development folds, with training labels purged across boundaries. Threshold/disagreement penalty selected entirely from development fold predictions; the most recent 20% of eligible weeks is evaluated retrospectively, without refitting on that period.

### Important interpretation

The last 20% historical period **is not truly pristine for this project**: aggregate results over a similar era were examined in V2. Do not promote V3 performance in that era as prospective evidence. Protect the *next, genuinely unseen* weeks/months for paper trading without changing parameters.

The historical universe contains only nine surviving tickers; delisted firms are missing. Adjusted opens are price proxies, not executable live fills. Spreads, impact, taxes, outage handling, survivorship bias and brokerage safety checks remain unsolved.

Future research: walk-forward portfolios with point-in-time constituents, costs sensitivity across multiple broker assumptions, and a fully logged paper-trading journal with immutable signal timestamps.
