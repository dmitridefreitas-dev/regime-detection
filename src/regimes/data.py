"""Daily price data with a local cache (same pattern as the sibling repos)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(
    ticker: str = "SPY",
    start: str = "1993-02-01",
    cache_dir: str | Path = "data",
) -> pd.Series:
    """Adjusted daily closes for `ticker`, cached locally as CSV."""
    cache = Path(cache_dir) / f"{ticker}_{start}.csv"
    if cache.exists():
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        return frame["Close"].rename(ticker)

    import yfinance as yf

    frame = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if frame.empty:
        raise RuntimeError(f"no data returned for {ticker!r} — check ticker and network")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    cache.parent.mkdir(parents=True, exist_ok=True)
    frame[["Close"]].to_csv(cache)
    return frame["Close"].rename(ticker)
