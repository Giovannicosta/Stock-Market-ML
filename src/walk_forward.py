"""Cross-sectional walk-forward research; no brokerage orders are placed."""
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
from .model import FEATURES, prepare_features
from .data import load_tickers

EXTRA = ('spy_return_21', 'spy_vol_21', 'spy_trend_200', 'spy_high_vol')
COLUMNS = list(FEATURES) + list(EXTRA)
THRESHOLDS = (0.0, .0005, .001, .002)

def spy_features(spy):
    close = spy['Close']
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=spy.index)
    out['spy_return_21'] = close.pct_change(21, fill_method=None)
    out['spy_vol_21'] = ret.rolling(21).std()
    out['spy_trend_200'] = close / close.rolling(200).mean() - 1
    vol_cutoff = out['spy_vol_21'].rolling(252, min_periods=126).quantile(.75)
    out['spy_high_vol'] = (out['spy_vol_21'] > vol_cutoff).astype(float)
    out['regime'] = np.where(out['spy_trend_200'] > 0, 'bull', 'bear') + '_' + np.where(
        out['spy_high_vol'] > 0, 'high_vol', 'normal_vol')
    out['spy_target'] = spy['Open'].shift(-2) / spy['Open'].shift(-1) - 1
    return out.replace([np.inf, -np.inf], np.nan)

def build_panel(prices):
    if 'SPY' not in prices:
        raise ValueError('SPY benchmark is required')
    market = spy_features(prices['SPY'].sort_index())
    pieces = []
    for ticker, p in sorted(prices.items()):
        if ticker == 'SPY':
            continue
        p = p.sort_index()
        frame = prepare_features(p).join(market, how='inner')
        frame['label_end'] = pd.Series(p.index, index=p.index).shift(-2)
        frame['ticker'] = ticker
        frame.index.name = 'Date'
        pieces.append(frame.reset_index())
    if not pieces:
        raise ValueError('Need at least one non-SPY asset')
    panel = pd.concat(pieces, ignore_index=True).sort_values(['Date', 'ticker'])
    if panel.duplicated(['Date', 'ticker']).any():
        raise ValueError('Duplicate signal date / ticker')
    labeled = panel.dropna(subset=COLUMNS + ['target', 'label_end', 'spy_target'])
    return panel, labeled, market

def candidates():
    return {
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=20)),
        'gradient_boosting': HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, learning_rate=.04,
            min_samples_leaf=100, l2_regularization=2, early_stopping=False,
            random_state=42),
    }

def score(returns):
    a = np.asarray(returns, dtype=float)
    if len(a) == 0 or not np.isfinite(a).all() or (a <= -1).any():
        raise ValueError('Invalid trading return series')
    wealth = np.cumprod(1 + a)
    drawdown = np.r_[1, wealth] / np.maximum.accumulate(np.r_[1, wealth]) - 1
    std = a.std(ddof=1) if len(a) > 1 else 0
    return dict(sessions=len(a), total_return=float(wealth[-1] - 1),
                annualized_return=float(wealth[-1] ** (252 / len(a)) - 1),
                sharpe_rf0=float(np.sqrt(252) * a.mean() / std) if std > 0 else 0.0,
                max_drawdown=float(drawdown.min()))

def portfolio(predictions, threshold, fee_bps=10, top_k=3):
    """Signal at close(t); equal-weight long/cash positions from open(t+1).

    Turnover includes opening purchases, rebalancing and final liquidation.
    """
    if predictions.empty or predictions.duplicated(['Date','ticker']).any():
        raise ValueError('Missing or duplicated predictions')
    if fee_bps < 0 or top_k < 1:
        raise ValueError('Invalid fee or top_k')
    symbols = sorted(predictions['ticker'].unique())
    w_prev = np.zeros(len(symbols))
    prior_returns = np.zeros(len(symbols))
    fee = fee_bps / 10000
    entries = []
    for date, block in predictions.sort_values(['Date','ticker']).groupby('Date', sort=True):
        eligible = block[block['prediction'] > threshold].nlargest(top_k,'prediction')
        weights = np.zeros(len(symbols))
        if not eligible.empty:
            for ticker in eligible['ticker']:
                weights[symbols.index(ticker)] = 1 / len(eligible)
        actual = block.set_index('ticker')['target'].reindex(symbols).fillna(0).to_numpy()
        drifted = w_prev * (1 + prior_returns) / (1 + float(w_prev @ prior_returns))
        turnover = float(abs(weights - drifted).sum())
        net = float(weights @ actual) - fee * turnover
        if net <= -1:
            raise ValueError('Invalid net return')
        entries.append(dict(Date=date, ml=net, spy=float(block['spy_target'].iloc[0]),
                            regime=str(block['regime'].iloc[0]),
                            exposure=float(weights.sum()), turnover=turnover))
        w_prev, prior_returns = weights, actual
    # Close all remaining positions at the end of the sampled period.
    remaining = float((w_prev * (1 + prior_returns) /
                       (1 + float(w_prev @ prior_returns))).sum())
    entries[-1]['ml'] -= fee * remaining
    entries[-1]['turnover'] += remaining
    days = pd.DataFrame(entries).set_index('Date')
    benchmark = days['spy'].to_numpy().copy()
    benchmark[0] -= fee
    benchmark[-1] -= fee
    days['spy_net'] = benchmark
    return days, dict(ml=score(days['ml']), spy=score(days['spy_net']),
                      avg_exposure=float(days['exposure'].mean()),
                      turnover_sides=float(days['turnover'].sum()))

def folds_for(dates, n=4, holdout=.2):
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if len(dates) < 600 or n < 2:
        raise ValueError('Need 600+ signal dates and 2+ folds')
    end = int(len(dates) * (1-holdout))
    dev = dates[:end]
    indices = np.linspace(int(.4 * len(dev)), len(dev), n+1, dtype=int)
    folds = [(dev[indices[i]], dev[indices[i+1]] if i+1 < n else dates[end])
             for i in range(n)]
    return folds, dates[end]

def train_before(labeled, cutoff):
    """Purge all examples with target endpoints at or after fold start."""
    return labeled[(labeled['Date'] < cutoff) & (labeled['label_end'] < cutoff)]

def predict(model, train, validation):
    model.fit(train[COLUMNS], train['target'])
    out = validation[['Date','ticker','target','spy_target','regime']].copy()
    out['prediction'] = model.predict(validation[COLUMNS])
    return out

def regimes(days):
    result = []
    for regime, sub in days.groupby('regime'):
        result.append(dict(regime=regime, sessions=len(sub),
                           mean_ml_day=float(sub['ml'].mean()),
                           mean_spy_day=float(sub['spy_net'].mean()),
                           mean_daily_excess=float((sub['ml']-sub['spy_net']).mean())))
    return result

def experiment(prices, fee_bps=10, n_folds=4):
    panel, labeled, market = build_panel(prices)
    boundaries, test_start = folds_for(labeled['Date'], n=n_folds)
    oof = {name: [] for name in candidates()}
    fold_info = []
    for first, end in boundaries:
        training = train_before(labeled, first)
        validation = labeled[(labeled['Date'] >= first) & (labeled['Date'] < end)]
        if len(training) < 500 or len(validation) < 100:
            raise ValueError('Fold with insufficient samples')
        fold_info.append(dict(train_rows=len(training), val_rows=len(validation),
                              validation_start=str(first.date()), validation_end_exclusive=str(end.date())))
        for name, model in candidates().items():
            oof[name].append(predict(model, training, validation))
    choices = []
    for name, slices in oof.items():
        val = pd.concat(slices, ignore_index=True)
        for threshold in THRESHOLDS:
            _, report = portfolio(val, threshold, fee_bps)
            choices.append(dict(model=name, threshold=threshold,
                                val_sharpe=report['ml']['sharpe_rf0'],
                                val_return=report['ml']['total_return'],
                                val_spy_return=report['spy']['total_return'],
                                val_exposure=report['avg_exposure']))
    eligible = [r for r in choices if r['val_exposure'] >= .1]
    if not eligible:
        raise ValueError('No validation candidate passed 10% minimum exposure')
    selected = max(eligible, key=lambda r:(r['val_sharpe'],r['val_return']))
    name, threshold = selected['model'], selected['threshold']
    fold_reports = []
    for info, forecast in zip(fold_info, oof[name]):
        _, metrics = portfolio(forecast, threshold, fee_bps)
        fold_reports.append(dict(**info, ml=metrics['ml'], spy=metrics['spy']))
    train = train_before(labeled, test_start)
    test = labeled[labeled['Date'] >= test_start]
    forecast = predict(candidates()[name], train, test)
    daily, backtest = portfolio(forecast, threshold, fee_bps)
    benchmark_mae = float(mean_absolute_error(test['target'], np.zeros(len(test))))
    model_mae = float(mean_absolute_error(test['target'], forecast['prediction']))
    last = market.index.max()
    current = panel[panel['Date'] == last].dropna(subset=COLUMNS).copy()
    signals = []
    if not current.empty:
        live_model = candidates()[name]
        live_model.fit(labeled[COLUMNS], labeled['target'])
        current['prediction'] = live_model.predict(current[COLUMNS])
        signals = [dict(ticker=str(row.ticker), predicted_return=float(row.prediction))
                   for row in current.sort_values('prediction',ascending=False).itertuples()]
    return {
        'method': 'expanding walk-forward; model and threshold chosen before sealed holdout',
        'assets': sorted(set(labeled['ticker'])),
        'selection': dict(model=name, threshold=threshold, validation_grid=choices),
        'folds': fold_reports,
        'heldout': dict(start=str(test_start.date()), end=str(test['Date'].max().date()),
                        model_mae=model_mae, zero_baseline_mae=benchmark_mae,
                        portfolio=backtest, regimes=regimes(daily),
                        beats_spy=backtest['ml']['total_return'] > backtest['spy']['total_return']),
        'latest_exploratory_signal': dict(date=str(last.date()),
                          top_buys=[x['ticker'] for x in signals
                                    if x['predicted_return'] > threshold][:3],
                          predictions=signals),
        'limitations': [
            'Survivorship bias: current ticker list omits delisted firms.',
            'Adjusted opening quotes are non-executable proxies.',
            'No live orders; taxes, slippage and market impact excluded.',
            'Do not use the heldout test for further hyperparameter selection.',
        ],
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
    ap.add_argument('--tickers-file',type=Path,default=Path('config/tickers.txt'))
    ap.add_argument('--fee-bps',type=float,default=10)
    ap.add_argument('--report-dir',type=Path,default=Path('reports/walk_forward'))
    args = ap.parse_args()
    prices = {}
    for ticker in load_tickers(args.tickers_file):
        path = args.raw_dir / f'{ticker}.csv'
        if path.exists():
            prices[ticker] = pd.read_csv(path,parse_dates=['Date'],index_col='Date')
    result = experiment(prices,fee_bps=args.fee_bps)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    target = args.report_dir/'summary.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    r = result['heldout']
    print(f"Walk-forward selection: {result['selection']['model']}, threshold={result['selection']['threshold']:.3%}")
    print(f"Holdout {r['start']} to {r['end']}: ML {r['portfolio']['ml']['total_return']:+.2%}, SPY {r['portfolio']['spy']['total_return']:+.2%}")
    print(f"Beats SPY: {r['beats_spy']}, ML MAE {r['model_mae']:.6f}, zero forecast MAE {r['zero_baseline_mae']:.6f}")
    for regime in r['regimes']:
        print(f"  {regime['regime']}: {regime['sessions']} sessions, daily excess {regime['mean_daily_excess']:+.4%}")
    print('Report:',target)

if __name__=='__main__':
    main()
