from datetime import date, timedelta

import pandas as pd

from trading_bot.data.provider import download_ohlcv, last_stored_date
from trading_bot.data.storage import Candle, get_session
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_symbols(
    symbols: list[str], start: str, end: str | None = None, chunk_size: int = 50
) -> int:
    """Download and persist OHLCV for a list of symbols, in chunks to avoid rate limits.

    Skips dates already in the database (incremental ingest).
    Returns the number of new candles inserted.
    """
    total_inserted = 0
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        df = download_ohlcv(chunk, start=start, end=end)
        if df.empty:
            continue

        with get_session() as session:
            existing = set(
                session.query(Candle.symbol, Candle.dt)
                .filter(Candle.symbol.in_(chunk))
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
                total_inserted += len(rows_to_insert)
        logger.info(f"Chunk {i // chunk_size + 1}: inserted {len(rows_to_insert)} new candles")

    logger.info(f"Inserted {total_inserted} new candles total")
    return total_inserted


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


FX_EURUSD = "EURUSD=X"   # yfinance: prezzo di 1 EUR in USD (USD per EUR)


def ingest_fx(start: str = "2005-01-01") -> int:
    """Scarica e persiste il cambio EUR/USD giornaliero (pseudo-simbolo).

    Volume 0 → il filtro di liquidità dell'universo lo esclude automaticamente,
    quindi non finisce mai tra i titoli tradabili. Incrementale come gli altri.
    """
    last = last_stored_date(FX_EURUSD)
    next_start = (last + timedelta(days=1)).isoformat() if last else start
    if pd.Timestamp(next_start).date() > date.today():
        logger.info("EUR/USD già aggiornato.")
        return 0
    return ingest_symbols([FX_EURUSD], start=next_start)


def load_fx_eurusd(start: str = "2005-01-01", end: str | None = None) -> pd.Series:
    """Serie giornaliera EUR/USD (USD per 1 EUR). Vuota se non ancora ingerita."""
    panel = load_panel([FX_EURUSD], start=start, end=end)
    if panel.empty or FX_EURUSD not in panel.columns:
        return pd.Series(dtype=float)
    return panel[FX_EURUSD].dropna()


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
