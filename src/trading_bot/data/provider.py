from datetime import date

import pandas as pd
import yfinance as yf

from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


def download_ohlcv(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted OHLCV from Yahoo Finance for a list of symbols.

    Returns a long-format DataFrame indexed by (date, symbol) with columns
    open/high/low/close/adj_close/volume. Empty if all downloads failed.
    """
    if not symbols:
        return pd.DataFrame()

    logger.info(f"Downloading {len(symbols)} symbols from {start} to {end or 'today'}")
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        logger.warning("yfinance returned empty data")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if len(symbols) == 1:
        df = raw.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["symbol"] = symbols[0]
        df = df.reset_index().rename(columns={"Date": "dt", "index": "dt"})
        frames.append(df)
    else:
        for sym in symbols:
            if sym not in raw.columns.get_level_values(0):
                continue
            sub = raw[sym].copy()
            if sub.dropna(how="all").empty:
                continue
            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
            sub["symbol"] = sym
            sub = sub.reset_index().rename(columns={"Date": "dt", "index": "dt"})
            frames.append(sub)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"]).dt.date
    rename = {"adj_close": "adj_close"}
    out = out.rename(columns=rename)
    expected = ["dt", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    out = out[[c for c in expected if c in out.columns]].dropna(subset=["close"])
    logger.info(f"Downloaded {len(out)} rows for {out['symbol'].nunique()} symbols")
    return out


def last_stored_date(symbol: str) -> date | None:
    from sqlalchemy import func, select

    from trading_bot.data.storage import Candle, get_session

    with get_session() as session:
        result = session.execute(
            select(func.max(Candle.dt)).where(Candle.symbol == symbol)
        ).scalar_one_or_none()
        return result
