"""V3: weekly stock-vs-SPY forecasting with walk-forward gated rotation.

This is retrospective research. No brokerage calls or actionable trading orders.
All features are known at the close of signal date t. Hypothetical entry is
the following adjusted open; exit is the adjusted open H trading days later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from .data import load_tickers
from .model import FEATURES, prepare_features
from .walk_forward import EXTRA, spy_features

COLUMNS = list(FEATURES) + list(EXTRA)
HOLD_SESSIONS = 5
GATE_THRESHOLDS = (0.0, 0.001, 0.003)
DISAGREEMENT_PENALTIES = (0.0, 0.5, 1.0)
MODELS = ("ridge", "boosting")


def build_weekly_panel(prices: dict[str, pd.DataFrame],
                       hold_sessions: int = HOLD_SESSIONS):
    """A fixed calendar prevents overlapping rebalancing periods and date drift."""
    if "SPY" not in prices or len(prices) < 2:
        raise ValueError("Need SPY plus at least one stock")
    if hold_sessions < 1:
        raise ValueError("holding period must be positive")
    spy = prices["SPY"].sort_index()
    index = pd.DatetimeIndex(spy.index).sort_values().unique()
    if len(index) < 500:
        raise ValueError("Need at least 500 daily SPY trading rows")
    signal_dates = index[::hold_sessions]
    market = spy_features(spy).reindex(index)
    # t+1 is the next rebalance entry open, t+H+1 is the exit open.
    spy_open = pd.to_numeric(spy["Open"].reindex(index), errors="coerce")
    spy_return = spy_open.shift(-(hold_sessions + 1)) / spy_open.shift(-1) - 1
    exits = pd.Series(index, index=index).shift(-(hold_sessions + 1))
    market = market.assign(spy_period_return=spy_return, label_end=exits)
    pieces = []
    for ticker, raw in sorted(prices.items()):
        if ticker == "SPY":
            continue
        p = raw.sort_index().reindex(index)
        local = prepare_features(p)[list(FEATURES)]
        start_open = pd.to_numeric(p["Open"], errors="coerce").shift(-1)
        end_open = pd.to_numeric(p["Open"], errors="coerce").shift(-(hold_sessions + 1))
        frame = local.join(market[list(EXTRA) + [
            "spy_period_return", "label_end", "regime"
        ]], how="left")
        frame["stock_period_return"] = end_open / start_open - 1
        frame["target_excess"] = frame["stock_period_return"] - frame["spy_period_return"]
        frame = frame.loc[signal_dates]
        frame["ticker"] = ticker
        frame.index.name = "Date"
        pieces.append(frame.reset_index())
    full = pd.concat(pieces, ignore_index=True).sort_values(["Date", "ticker"])
    if full.duplicated(["Date", "ticker"]).any():
        raise ValueError("Duplicate stock and signal date")
    needed = COLUMNS + ["stock_period_return", "spy_period_return",
                        "target_excess", "label_end"]
    labeled = full.dropna(subset=needed).copy()
    return full, labeled, market, signal_dates


def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=20.0)),
        "boosting": HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=2,
            early_stopping=False, random_state=42
        ),
    }


def train_before(labeled: pd.DataFrame, first_validation_date: pd.Timestamp):
    """Purge every label requiring an opening quote in or after the new fold."""
    return labeled[(labeled["Date"] < first_validation_date) &
                   (labeled["label_end"] < first_validation_date)]


def rolling_folds(signal_dates, holdout_fraction=0.2, folds=4):
    dates = pd.DatetimeIndex(sorted(pd.unique(signal_dates)))
    if len(dates) < 180 or not (0.1 <= holdout_fraction <= 0.4) or folds < 2:
        raise ValueError("Insufficient weekly dates or invalid settings")
    holdout_idx = int(len(dates) * (1 - holdout_fraction))
    development = dates[:holdout_idx]
    first = int(0.4 * len(development))
    edges = np.linspace(first, len(development), folds + 1, dtype=int)
    parts = [(development[edges[i]], development[edges[i + 1]]
              if i + 1 < folds else dates[holdout_idx])
             for i in range(folds)]
    return parts, dates[holdout_idx]


def forecast_models(train, future):
    if len(train) < 200 or len(future) == 0:
        raise ValueError("Too few training or validation observations")
    result = future[["Date", "ticker", "stock_period_return",
                     "spy_period_return", "target_excess", "regime"]].copy()
    for name, model in make_models().items():
        model.fit(train[COLUMNS], train["target_excess"])
        result[name] = model.predict(future[COLUMNS])
    return result


def performance(returns, periods_per_year=252 / HOLD_SESSIONS):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid backtest returns")
    equity = np.cumprod(1 + r)
    pathway = np.r_[1.0, equity]
    drawdown = pathway / np.maximum.accumulate(pathway) - 1
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {
        "periods": int(len(r)),
        "total_return": float(equity[-1] - 1),
        "annualized_return": float(equity[-1] ** (periods_per_year / len(r)) - 1),
        "sharpe_rf0": float(np.sqrt(periods_per_year) * r.mean() / std)
                       if std > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def rotation_backtest(forecasts, threshold=0.001, disagreement_penalty=0.5,
                      fee_bps=10.0, top_k=3, spy_only=False,
                      hold_sessions=HOLD_SESSIONS):
    """Long stocks when relative forecast is convincing, otherwise hold SPY.

    Model disagreement is an uncertainty *heuristic*, not a confidence interval.
    This simulation assumes predictable adjusted open fills; no real orders.
    """
    if forecasts.empty or forecasts.duplicated(["Date", "ticker"]).any():
        raise ValueError("Missing or duplicated forecasts")
    if fee_bps < 0 or threshold < 0 or disagreement_penalty < 0 or top_k < 1:
        raise ValueError("Invalid strategy arguments")
    needed = ["ridge", "boosting", "stock_period_return", "spy_period_return"]
    if forecasts[needed].isna().any().any():
        raise ValueError("Missing forecasts or realized returns")
    stocks = sorted(forecasts["ticker"].unique())
    symbols = ["SPY"] + stocks
    previous_weights = np.zeros(len(symbols), dtype=float)
    previous_returns = np.zeros(len(symbols), dtype=float)
    fees = fee_bps / 10000
    entries = []
    for date, block in forecasts.sort_values(["Date", "ticker"]).groupby("Date", sort=True):
        market_return = block["spy_period_return"].to_numpy(dtype=float)
        if not np.allclose(market_return, market_return[0], rtol=1e-8, atol=1e-10):
            raise ValueError("Inconsistent SPY period returns")
        block = block.copy()
        block["ensemble"] = (block["ridge"] + block["boosting"]) / 2
        block["uncertainty_proxy"] = np.abs(block["ridge"] - block["boosting"])
        block["conservative_excess"] = (block["ensemble"] -
                                       disagreement_penalty * block["uncertainty_proxy"])
        allowed = block[(block["conservative_excess"] > threshold) &
                        (block["ridge"] > 0) & (block["boosting"] > 0)]
        selected = [] if spy_only else allowed.nlargest(top_k, "conservative_excess")["ticker"].tolist()
        weights = np.zeros(len(symbols), dtype=float)
        if selected:
            for ticker in selected:
                weights[symbols.index(ticker)] = 1 / len(selected)
        else:
            weights[0] = 1.0
        actual = np.zeros(len(symbols), dtype=float)
        actual[0] = market_return[0]
        local = block.set_index("ticker")["stock_period_return"]
        for ticker, value in local.items():
            actual[symbols.index(ticker)] = float(value)
        previous_gross = float(previous_weights @ previous_returns)
        drifted = (previous_weights * (1 + previous_returns) /
                   (1 + previous_gross)) if np.any(previous_weights) else previous_weights
        turnover = float(np.abs(weights - drifted).sum())
        net = float(weights @ actual - fees * turnover)
        if net <= -1:
            raise ValueError("Nonphysical strategy period return")
        entries.append({
            "Date": date, "net_return": net, "spy_return": actual[0],
            "regime": str(block["regime"].iloc[0]), "stocks_count": len(selected),
            "tickers": ",".join(selected), "turnover": turnover,
        })
        previous_weights, previous_returns = weights, actual
    # Assume all positions sold after the last historical period.
    previous_gross = float(previous_weights @ previous_returns)
    final_weights = previous_weights * (1 + previous_returns) / (1 + previous_gross)
    final_turnover = float(np.abs(final_weights).sum())
    entries[-1]["net_return"] -= fees * final_turnover
    entries[-1]["turnover"] += final_turnover
    dates = pd.DataFrame(entries).set_index("Date")
    benchmark = dates["spy_return"].to_numpy(dtype=float).copy()
    benchmark[0] -= fees
    benchmark[-1] -= fees
    dates["spy_net"] = benchmark
    stats = {
        "strategy": performance(dates["net_return"],
                                periods_per_year=252 / hold_sessions),
        "spy": performance(dates["spy_net"],
                           periods_per_year=252 / hold_sessions),
        "stock_rotation_fraction": float((dates["stocks_count"] > 0).mean()),
        "turnover_sides": float(dates["turnover"].sum()),
        "excess_log_return": float(np.log1p(dates["net_return"]).sum()
                                   - np.log1p(dates["spy_net"]).sum()),
    }
    return dates, stats


def regime_summary(periods):
    rows = []
    for regime, block in periods.groupby("regime"):
        rows.append({
            "regime": regime, "periods": len(block),
            "mean_strategy_period": float(block["net_return"].mean()),
            "mean_spy_period": float(block["spy_net"].mean()),
            "mean_excess_period": float((block["net_return"] - block["spy_net"]).mean()),
        })
    return rows


def experiment(prices, fee_bps=10.0, n_folds=4, hold_sessions=HOLD_SESSIONS):
    full, labeled, market, scheduled = build_weekly_panel(prices, hold_sessions)
    available = pd.DatetimeIndex(sorted(pd.unique(labeled["Date"])))
    boundaries, holdout_start = rolling_folds(available, folds=n_folds)
    validation_predictions, fold_ranges = [], []
    for first, end in boundaries:
        training = train_before(labeled, first)
        val = labeled[(labeled["Date"] >= first) & (labeled["Date"] < end) &
                      (labeled["label_end"] < holdout_start)]
        validation_predictions.append(forecast_models(training, val))
        fold_ranges.append({
            "start": str(first.date()), "end_exclusive": str(end.date()),
            "train_rows": len(training), "validation_rows": len(val)
        })
    all_validation = pd.concat(validation_predictions, ignore_index=True)
    choices = []
    for threshold in GATE_THRESHOLDS:
        for penalty in DISAGREEMENT_PENALTIES:
            _, result = rotation_backtest(
                all_validation, threshold=threshold,
                disagreement_penalty=penalty, fee_bps=fee_bps,
                hold_sessions=hold_sessions)
            choices.append({
                "threshold": threshold, "disagreement_penalty": penalty,
                "validation_excess_log_return": result["excess_log_return"],
                "validation_rotation_fraction": result["stock_rotation_fraction"],
                "validation_total_return": result["strategy"]["total_return"],
            })
    # SPY-only is the baseline fallback. Do not manufacture a trade if validation
    # shows no benefit over it. Require meaningful gross historical advantage.
    eligible = [c for c in choices if c["validation_rotation_fraction"] >= 0.05
                and c["validation_excess_log_return"] > 0.015]
    choice = max(eligible, key=lambda x: x["validation_excess_log_return"]) if eligible else None
    threshold = choice["threshold"] if choice else GATE_THRESHOLDS[0]
    penalty = choice["disagreement_penalty"] if choice else DISAGREEMENT_PENALTIES[-1]
    selection = {
        "mode": "weekly_stock_rotation" if choice else "SPY_fallback",
        "threshold": threshold, "disagreement_penalty": penalty,
        "reason": "positive walk-forward geometric excess after costs"
                  if choice else "no candidate beat minimum validation excess hurdle",
        "validation_grid": choices,
    }
    # Never train with labels that matured after the historical test began.
    final_train = train_before(labeled, holdout_start)
    test_rows = labeled[labeled["Date"] >= holdout_start]
    test_pred = forecast_models(final_train, test_rows)
    periods, result = rotation_backtest(
        test_pred, threshold, penalty, fee_bps,
        spy_only=choice is None, hold_sessions=hold_sessions)
    fold_stats = []
    for info, preds in zip(fold_ranges, validation_predictions):
        _, fold_result = rotation_backtest(
            preds, threshold, penalty, fee_bps,
            spy_only=choice is None, hold_sessions=hold_sessions)
        fold_stats.append({**info, "strategy": fold_result["strategy"],
                           "spy": fold_result["spy"]})
    # This is an *historical snapshot*, not a current market recommendation.
    last_signal = pd.Timestamp(scheduled[-1])
    current = full[(full["Date"] == last_signal)].dropna(subset=COLUMNS)
    snapshot = {"signal_date": str(last_signal.date()), "available": False}
    if not current.empty:
        matured = labeled[labeled["label_end"] <= last_signal]
        models = make_models()
        output = current[["ticker"]].copy()
        for name, model in models.items():
            model.fit(matured[COLUMNS], matured["target_excess"])
            output[name] = model.predict(current[COLUMNS])
        output["conservative_excess"] = ((output["ridge"] + output["boosting"]) / 2
                                          - penalty * (output["ridge"] - output["boosting"]).abs())
        picked = output[(output["conservative_excess"] > threshold) &
                        (output["ridge"] > 0) & (output["boosting"] > 0)]
        snapshot = {
            "signal_date": str(last_signal.date()), "available": True,
            "stale_if_more_recent_sessions_exist": bool(last_signal < market.index[-1]),
            "hypothetical_stock_rotation": (
                picked.nlargest(3, "conservative_excess")["ticker"].tolist()
                if choice else []),
            "fallback": "SPY", "not_an_order": True,
        }
    return {
        "version": 3,
        "methodology": "weekly 5-session, relative-return pooled expanding walk-forward",
        "historical_holdout_is_not_new_unseen_data": True,
        "assets": sorted(labeled["ticker"].unique().tolist()),
        "selection": selection, "walk_forward_folds": fold_stats,
        "heldout": {
            "start": str(holdout_start.date()),
            "end": str(test_pred["Date"].max().date()),
            "strategy": result, "beats_spy": bool(
                result["strategy"]["total_return"] > result["spy"]["total_return"]),
            "model_mae_excess": float(mean_absolute_error(
                test_pred["target_excess"],
                (test_pred["ridge"] + test_pred["boosting"]) / 2)),
            "zero_excess_forecast_mae": float(mean_absolute_error(
                test_pred["target_excess"],
                np.zeros(len(test_pred)))),
            "regimes": regime_summary(periods),
        },
        "latest_snapshot": snapshot,
        "cautions": [
            "V2 already exposed a similar historical test era; V3 holdout is retrospective.",
            "Disagreement is an uncertainty proxy, not calibrated probability.",
            "Present-day universe creates survivorship bias; adjusted prices are not executable.",
            "Need truly forward paper trading before real-money decisions.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--tickers-file", type=Path, default=Path("config/tickers.txt"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/v3"))
    ap.add_argument("--fee-bps", type=float, default=10.0)
    args = ap.parse_args()
    if args.fee_bps < 0:
        ap.error("fee-bps cannot be negative")
    prices = {}
    for ticker in load_tickers(args.tickers_file):
        path = args.raw_dir / f"{ticker}.csv"
        if path.exists():
            prices[ticker] = pd.read_csv(path, index_col="Date",
                                         parse_dates=["Date"])
    result = experiment(prices, fee_bps=args.fee_bps)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "summary.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False),
                   encoding="utf-8")
    test = result["heldout"]
    strategy, spy = test["strategy"]["strategy"], test["strategy"]["spy"]
    print(f"V3 selection: {result['selection']['mode']}, "
          f"threshold {result['selection']['threshold']:.3%}")
    print(f"Historical holdout {test['start']}..{test['end']}: "
          f"V3 {strategy['total_return']:+.2%} vs SPY {spy['total_return']:+.2%}")
    print(f"V3 Sharpe {strategy['sharpe_rf0']:.2f} vs SPY {spy['sharpe_rf0']:.2f}")
    print(f"Excess-return MAE model {test['model_mae_excess']:.6f}, "
          f"zero {test['zero_excess_forecast_mae']:.6f}")
    for r in test["regimes"]:
        print(f"{r['regime']}: {r['periods']} weekly periods; "
              f"mean excess {r['mean_excess_period']:+.3%}")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
