from datetime import date, timedelta

import pandas as pd

from trading_bot.data.provider import download_ohlcv, last_stored_date
from trading_bot.data.storage import Candle, get_session
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_symbols(symbols: list[str], start: str, end: str | None = None) -> int:
    """Download and persist OHLCV for a list of symbols.

    Skips dates already in the database (incremental ingest).
    Returns the number of new candles inserted.
    """
    df = download_ohlcv(symbols, start=start, end=end)
    if df.empty:
        return 0

    inserted = 0
    with get_session() as session:
        # Build a set of (symbol, dt) tuples already in db for the symbols at hand
        existing = set(
            session.query(Candle.symbol, Candle.dt)
            .filter(Candle.symbol.in_(symbols))
            .all()
        )

        rows_to_insert = []
        for r in df.itertuples(index=False):
            key = (r.symbol, r.dt)
            if key in existing:
                continue
            rows_to_insert.append(
                Candle(
                    symbol=r.symbol,
                    dt=r.dt,
                    open=float(r.open),
                    high=float(r.high),
                    low=float(r.low),
                    close=float(r.close),
                    adj_close=float(r.adj_close),
                    volume=float(r.volume),
                )
            )

        if rows_to_insert:
            session.bulk_save_objects(rows_to_insert)
            session.commit()
            inserted = len(rows_to_insert)

    logger.info(f"Inserted {inserted} new candles")
    return inserted


def incremental_update(symbols: list[str], default_start: str) -> int:
    """For each symbol, ingest from the day after its last stored date.

    Returns the total number of rows inserted.
    """
    today = date.today()
    total = 0
    # batch by start date — symbols with similar last-stored dates download together
    groups: dict[str, list[str]] = {}
    for sym in symbols:
        last = last_stored_date(sym)
        next_start = (last + timedelta(days=1)).isoformat() if last else default_start
        groups.setdefault(next_start, []).append(sym)

    for start, syms in groups.items():
        if pd.Timestamp(start).date() > today:
            continue
        total += ingest_symbols(syms, start=start, end=None)
    return total


def load_panel(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Load adjusted-close prices as a wide DataFrame (date index, symbol cols)."""
    from sqlalchemy import select

    with get_session() as session:
        q = select(Candle.dt, Candle.symbol, Candle.adj_close).where(
            Candle.symbol.in_(symbols),
            Candle.dt >= pd.Timestamp(start).date(),
        )
        if end is not None:
            q = q.where(Candle.dt <= pd.Timestamp(end).date())
        rows = session.execute(q).all()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["dt", "symbol", "adj_close"])
    panel = df.pivot(index="dt", columns="symbol", values="adj_close").sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel
