"""Universe management — S&P 500 / 400 / 600 (≈ S&P 1500 / Russell 3000 proxy).

Fetches constituents from Wikipedia and persists them to the Ticker table.
The combined S&P 1500 (large + mid + small cap) covers ~90% of US investable
market cap and is a practical substitute for Russell 3000 using free data.

Index coverage:
    S&P 500  — 503 stocks, large cap,  ~80% US market cap
    S&P 400  — 400 stocks, mid cap,   ~ 7% US market cap
    S&P 600  — 603 stocks, small cap, ~ 3% US market cap
    ─────────────────────────────────────────────────────
    Total    — ~1506 stocks (S&P 1500)
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests
from sqlalchemy import text

from trading_bot.data.storage import Ticker, get_session
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

_INDEX_URLS = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400":  "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600":  "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


@dataclass
class UniverseEntry:
    symbol: str
    name: str
    sector: str
    sub_industry: str
    index_name: str   # "sp500" | "sp400" | "sp600"


def _fetch_index(index_key: str) -> list[UniverseEntry]:
    """Download one index from Wikipedia and return UniverseEntry list."""
    url = _INDEX_URLS[index_key]
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), header=0)
    df = tables[0]
    df.columns = [c.strip() for c in df.columns]

    entries: list[UniverseEntry] = []
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", row.get("Ticker", ""))).replace(".", "-").strip()
        if not sym or sym == "nan":
            continue
        entries.append(
            UniverseEntry(
                symbol=sym,
                name=str(row.get("Security", row.get("Company", ""))).strip(),
                sector=str(row.get("GICS Sector", "Unknown")).strip(),
                sub_industry=str(row.get("GICS Sub-Industry", "")).strip(),
                index_name=index_key,
            )
        )
    logger.info(f"Fetched {len(entries)} constituents from {index_key}")
    return entries


def fetch_sp500_constituents() -> list[UniverseEntry]:
    return _fetch_index("sp500")


def fetch_sp400_constituents() -> list[UniverseEntry]:
    return _fetch_index("sp400")


def fetch_sp600_constituents() -> list[UniverseEntry]:
    return _fetch_index("sp600")


def fetch_all_constituents() -> list[UniverseEntry]:
    """Fetch S&P 500 + 400 + 600 (≈ S&P 1500 / Russell 3000 proxy).

    Deduplicates symbols — some stocks move between indices;
    highest index (500 > 400 > 600) wins in case of conflict.
    """
    seen: dict[str, UniverseEntry] = {}
    priority = {"sp500": 0, "sp400": 1, "sp600": 2}

    for key in ("sp500", "sp400", "sp600"):
        try:
            entries = _fetch_index(key)
        except Exception as e:
            logger.warning(f"Failed to fetch {key}: {e}")
            continue
        for e in entries:
            if e.symbol not in seen or priority[key] < priority[seen[e.symbol].index_name]:
                seen[e.symbol] = e

    result = list(seen.values())
    logger.info(
        f"Total S&P 1500 universe: {len(result)} unique symbols "
        f"(sp500={sum(1 for e in result if e.index_name=='sp500')}, "
        f"sp400={sum(1 for e in result if e.index_name=='sp400')}, "
        f"sp600={sum(1 for e in result if e.index_name=='sp600')})"
    )
    return result


def persist_universe(entries: list[UniverseEntry]) -> int:
    """Upsert entries into the Ticker table. Returns count of new insertions."""
    inserted = 0
    updated = 0
    with get_session() as session:
        for e in entries:
            existing = session.query(Ticker).filter_by(symbol=e.symbol).one_or_none()
            if existing is None:
                session.add(
                    Ticker(
                        symbol=e.symbol,
                        name=e.name,
                        sector=e.sector,
                        sub_industry=e.sub_industry,
                        index_name=e.index_name,
                    )
                )
                inserted += 1
            else:
                # Update metadata if changed (sector reclassifications happen)
                if (existing.sector != e.sector or existing.name != e.name
                        or existing.index_name != e.index_name):
                    existing.sector = e.sector
                    existing.name = e.name
                    existing.sub_industry = e.sub_industry
                    existing.index_name = e.index_name
                    updated += 1
        session.commit()
    logger.info(
        f"Universe persist: {inserted} inserted, {updated} updated "
        f"({inserted + updated} total changes)"
    )
    return inserted


def get_top_n_by_liquidity(
    n: int = 500,
    min_adv_usd: float = 1_000_000,   # $1M min avg daily dollar volume
    lookback_days: int = 90,
) -> list[str]:
    """Return top-N symbols by recent dollar volume from the candles table.

    Parameters
    ----------
    n : int
        Maximum number of symbols to return.
    min_adv_usd : float
        Minimum average daily dollar volume filter. Removes illiquid small caps
        that would have excessive bid-ask spread and market impact.
        Default $1M filters out micro-caps while keeping most S&P 1500 names.
    lookback_days : int
        How many calendar days back to compute ADV.

    Falls back to first N tickers alphabetically if no candle data exists.
    """
    with get_session() as session:
        rows = session.execute(
            text(
                "SELECT symbol, AVG(close * volume) AS adv "
                "FROM candles "
                "WHERE dt >= date('now', :neg_days) "
                "GROUP BY symbol "
                "HAVING AVG(close * volume) >= :min_adv "
                "ORDER BY adv DESC "
                "LIMIT :limit"
            ),
            {
                "neg_days": f"-{lookback_days} days",
                "min_adv":  min_adv_usd,
                "limit":    n,
            },
        ).all()

        if rows:
            return [r[0] for r in rows]

        # Fallback: alphabetical from tickers table (pre-ingest)
        symbols = (
            session.query(Ticker.symbol).order_by(Ticker.symbol).limit(n).all()
        )
        return [s[0] for s in symbols]


def get_universe_stats() -> dict:
    """Return a summary of what's in the universe DB."""
    with get_session() as session:
        total = session.query(Ticker).count()
        sectors = session.execute(
            text("SELECT sector, COUNT(*) as n FROM tickers GROUP BY sector ORDER BY n DESC")
        ).all()
        liquid = session.execute(
            text(
                "SELECT COUNT(DISTINCT symbol) FROM candles "
                "WHERE dt >= date('now', '-90 days')"
            )
        ).scalar()
    return {
        "total_tickers": total,
        "liquid_90d": liquid or 0,
        "by_sector": {row[0]: row[1] for row in sectors},
    }
