"""Historical OHLCV acquisition. Run after market close for complete daily candles."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DEFAULT_TICKERS = Path('config/tickers.txt')


def load_tickers(path: Path) -> list[str]:
    names = [s.strip().upper() for s in path.read_text(encoding='utf-8').splitlines()]
    return list(dict.fromkeys(s for s in names if s and not s.startswith('#')))


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end, interval='1d', auto_adjust=True,
        actions=False, progress=False, threads=False, timeout=30,
    )
    if df is None or df.empty:
        raise ValueError('No rows returned')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = df[needed].sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = 'Date'
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([float('inf'), -float('inf')], float('nan'))
    df = df.dropna(subset=needed)
    df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1) & (df['Volume'] >= 0)]
    if df.empty:
        raise ValueError('All rows failed OHLCV validation')
    return df


def download_all(tickers: list[str], start: str, end: str, root: Path, delay: float) -> dict:
    raw = root / 'data' / 'raw'
    metadata = root / 'data' / 'metadata'
    raw.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    results = []
    for index, ticker in enumerate(tickers):
        try:
            df = download_one(ticker, start, end)
            outfile = raw / f'{ticker}.csv'
            df.to_csv(outfile, date_format='%Y-%m-%d', float_format='%.10g')
            result = dict(ticker=ticker, status='ok', rows=len(df),
                          first_date=str(df.index[0].date()),
                          last_date=str(df.index[-1].date()), file=str(outfile))
            print(f'{ticker}: {len(df)} sessions, {result["first_date"]}–{result["last_date"]}')
        except Exception as exc:
            result = dict(ticker=ticker, status='error', error=str(exc))
            print(f'{ticker}: download failed: {exc}')
        results.append(result)
        if index != len(tickers) - 1 and delay:
            time.sleep(delay)
    summary = dict(source='Yahoo Finance (via unofficial yfinance)',
                   adjustment='auto_adjust=True; split/dividend adjusted OHLC',
                   requested_start=start, requested_end_exclusive=end,
                   recorded_at_utc=datetime.now(timezone.utc).isoformat(), results=results)
    (metadata / 'download_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2000-01-01')
    parser.add_argument('--end', default=(datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat(),
                        help='YYYY-MM-DD; exclusive')
    parser.add_argument('--tickers-file', type=Path, default=DEFAULT_TICKERS)
    parser.add_argument('--delay', type=float, default=2.0)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    try:
        if datetime.fromisoformat(args.end).date() <= datetime.fromisoformat(args.start).date():
            parser.error('--end must be after --start')
    except ValueError:
        parser.error('Use YYYY-MM-DD for --start and --end')
    if args.delay < 0:
        parser.error('--delay must be non-negative')
    tickers = load_tickers(args.tickers_file)
    if not tickers:
        parser.error('No tickers supplied')
    summary = download_all(tickers, args.start, args.end, args.root, args.delay)
    ok = sum(r['status'] == 'ok' for r in summary['results'])
    print(f'Downloaded {ok}/{len(tickers)} ticker histories')
    if not ok:
        parser.exit(1, 'No ticker data could be downloaded.\n')


if __name__ == '__main__':
    main()
