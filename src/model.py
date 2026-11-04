"""Leakage-conscious next-session forecasting and long/flat simulation.

Predictions use features known at the close of session t.
Entry is modeled at next session's adjusted open (t+1).
Exit/rebalance is modeled at following session's adjusted open (t+2).
No live orders are placed; adjusted prices are only a research proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ('return_1', 'return_5', 'return_21', 'return_63',
            'volatility_21', 'close_to_sma_20', 'daily_range',
            'volume_to_mean_20', 'overnight_gap')
GAP_SESSIONS = 2
THRESHOLDS = (0.0, 0.0005, 0.001, 0.002)


def prepare_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_index().copy()
    for name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        df[name] = pd.to_numeric(df[name], errors='coerce')
    o, h, l, c, v = (df[k] for k in ('Open', 'High', 'Low', 'Close', 'Volume'))
    ret = c.pct_change(fill_method=None)
    df['return_1'] = ret
    for n in (5, 21, 63):
        df[f'return_{n}'] = c.pct_change(n, fill_method=None)
    df['volatility_21'] = ret.rolling(21, min_periods=21).std()
    df['close_to_sma_20'] = c / c.rolling(20, min_periods=20).mean() - 1
    df['daily_range'] = (h - l) / c
    df['volume_to_mean_20'] = v / v.rolling(20, min_periods=20).mean() - 1
    df['overnight_gap'] = o / c.shift(1) - 1
    # Signal after close(t), fill at open(t+1), rebalance at open(t+2).
    df['target'] = o.shift(-2) / o.shift(-1) - 1
    return df[list(FEATURES) + ['target']].replace([np.inf, -np.inf], np.nan)


def split_chronological(labeled: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """60% train / 20% validation / 20% untouched test, with target embargo."""
    n = len(labeled)
    if n < 500:
        raise ValueError(f'Insufficient labeled rows ({n}); require >= 500')
    i, j = int(n * 0.60), int(n * 0.80)
    train = labeled.iloc[:i - GAP_SESSIONS]
    validation = labeled.iloc[i:j - GAP_SESSIONS]
    fit_for_test = labeled.iloc[:j - GAP_SESSIONS]
    test = labeled.iloc[j:]
    if min(len(train), len(validation), len(test)) < 50:
        raise ValueError('Too few samples in a split')
    return train, validation, fit_for_test, test


def make_candidates() -> dict:
    return {
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'hist_gradient_boosting': HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=40,
            learning_rate=0.035, l2_regularization=1.0, random_state=42,
        ),
    }


def errors(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        'mae': float(mean_absolute_error(actual, predicted)),
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def backtest(actual: np.ndarray, predicted: np.ndarray, threshold: float,
             fee_bps_per_side: float = 10.0) -> dict:
    """One asset, long or flat, rebalanced at next open; close final position."""
    if len(actual) != len(predicted):
        raise ValueError('Prediction/target lengths differ')
    if not len(actual):
        raise ValueError('Empty backtest')
    positions = (np.asarray(predicted) > threshold).astype(float)
    y = np.asarray(actual, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(predicted).all():
        raise ValueError('Backtest arrays must be finite')
    prior = np.concatenate(([0.0], positions[:-1]))
    changes = np.abs(positions - prior)
    fee = fee_bps_per_side / 10000.0
    daily_net = positions * y - changes * fee
    daily_net[-1] -= positions[-1] * fee
    if (1 + daily_net <= 0).any():
        raise ValueError('Nonphysical return <= -100%')
    equity = np.cumprod(1 + daily_net)
    benchmark = np.cumprod(1 + y)
    variance = np.std(daily_net, ddof=1) if len(y) > 1 else 0.0
    sharpe = float(np.sqrt(252) * daily_net.mean() / variance) if variance > 0 else 0.0
    running_high = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    drawdown = np.concatenate(([1.0], equity)) / running_high - 1
    n = len(y)
    return {
        'total_return': float(equity[-1] - 1),
        'annualized_return': float(equity[-1] ** (252 / n) - 1) if equity[-1] > 0 else -1.0,
        'sharpe_annualized': sharpe,
        'max_drawdown': float(drawdown.min()),
        'invested_fraction': float(positions.mean()),
        'buy_count': int(np.sum((positions == 1) & (prior == 0))),
        'sell_count': int(np.sum((positions == 0) & (prior == 1)) + positions[-1]),
        'turnover_sides': int(np.sum(changes) + positions[-1]),
        'asset_buy_hold_total_return_approx': float(benchmark[-1] - 1),
    }


def choose_strategy(validation: pd.DataFrame, predictions_by_model: dict,
                    fee_bps: float) -> tuple[str, float, list[dict]]:
    """Select entirely from validation, avoiding candidates with very few trades."""
    candidates = []
    actual = validation['target'].to_numpy()
    for name, pred in predictions_by_model.items():
        for threshold in THRESHOLDS:
            metrics = backtest(actual, pred, threshold, fee_bps)
            candidates.append(dict(model=name, threshold=threshold, **metrics))
    eligible = [c for c in candidates if c['buy_count'] >= 10 and c['invested_fraction'] >= 0.05]
    if eligible:
        chosen = max(eligible, key=lambda x: (x['sharpe_annualized'], x['annualized_return']))
    else:
        chosen_name = min(predictions_by_model, key=lambda name: errors(actual, predictions_by_model[name])['mae'])
        chosen = next(c for c in candidates if c['model'] == chosen_name and c['threshold'] == 0.0)
    return chosen['model'], chosen['threshold'], candidates


def evaluate_one(prices: pd.DataFrame, ticker: str, fee_bps: float = 10.0) -> dict:
    all_features = prepare_features(prices)
    labeled = all_features.dropna(subset=list(FEATURES) + ['target'])
    train, validation, fit_for_test, test = split_chronological(labeled)
    validation_pred = {}
    for name, model in make_candidates().items():
        model.fit(train[list(FEATURES)], train['target'])
        validation_pred[name] = model.predict(validation[list(FEATURES)])

    chosen_name, threshold, selection = choose_strategy(validation, validation_pred, fee_bps)
    final_model = make_candidates()[chosen_name]
    final_model.fit(fit_for_test[list(FEATURES)], fit_for_test['target'])
    test_pred = final_model.predict(test[list(FEATURES)])
    test_y = test['target'].to_numpy()
    test_backtest = backtest(test_y, test_pred, threshold, fee_bps)
    zero_pred = np.zeros_like(test_y)
    live_features = all_features.dropna(subset=list(FEATURES)).iloc[-1:]
    if live_features.empty:
        raise ValueError('No complete feature row for forecast')
    next_model = make_candidates()[chosen_name]
    next_model.fit(labeled[list(FEATURES)], labeled['target'])
    next_prediction = float(next_model.predict(live_features[list(FEATURES)])[0])
    signal = 'BUY' if next_prediction > threshold else 'FLAT/SELL'
    return {
        'ticker': ticker,
        'source_last_trading_date': str(prices.index[-1].date()),
        'target': 'adjusted_open(t+1) to adjusted_open(t+2) simple return',
        'features_known': 'at the completed close of trading session t',
        'model_selection': 'validation only; threshold chosen on validation Sharpe with minimum trades',
        'selected_model': chosen_name,
        'selected_threshold': threshold,
        'fee_bps_per_side': fee_bps,
        'rows': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'dates': {'train_end': str(train.index[-1].date()),
                  'validation_start': str(validation.index[0].date()),
                  'validation_end': str(validation.index[-1].date()),
                  'test_start': str(test.index[0].date()),
                  'test_end': str(test.index[-1].date())},
        'validation_grid': selection,
        'test_predictions': errors(test_y, test_pred),
        'test_zero_return_baseline': errors(test_y, zero_pred),
        'test_backtest': test_backtest,
        'forecast': {
            'signal_computed_after_close': str(live_features.index[-1].date()),
            'expected_next_open_to_following_open_return': next_prediction,
            'research_signal': signal,
            'note': 'Not a real-time order or an estimate of probability of profit.',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers-file', type=Path, default=Path('config/tickers.txt'))
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports'))
    parser.add_argument('--fee-bps', type=float, default=10.0,
                        help='Cost per buy OR sell, in basis points (10 = 0.10%%)')
    args = parser.parse_args()
    if args.fee_bps < 0:
        parser.error('--fee-bps cannot be negative')
    from .data import load_tickers
    tickers = load_tickers(args.tickers_file)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ticker in tickers:
        path = args.raw_dir / f'{ticker}.csv'
        if not path.exists():
            print(f'{ticker}: skipping: no CSV file')
            continue
        try:
            prices = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
            output = evaluate_one(prices, ticker, args.fee_bps)
            results.append(output)
            output_path = args.report_dir / f'{ticker}_evaluation.json'
            output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
            stats = output['test_backtest']
            print(f'{ticker}: {output["selected_model"]} test Sharpe {stats["sharpe_annualized"]:.2f} '
                  f'net return {stats["total_return"]:+.2%}, '
                  f'buy-hold {stats["asset_buy_hold_total_return_approx"]:+.2%}')
        except Exception as exc:
            print(f'{ticker}: evaluation failed: {exc}')
    summary = {
        'evaluated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'evaluated_tickers': len(results),
        'results': [{key: result[key] for key in ('ticker', 'selected_model',
                     'selected_threshold', 'test_predictions', 'test_zero_return_baseline',
                     'test_backtest', 'forecast')} for result in results],
    }
    (args.report_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not results:
        parser.exit(1, 'No tickers could be evaluated.\n')


if __name__ == '__main__':
    main()
