from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests
from sqlalchemy import text

from trading_bot.data.storage import Ticker, get_session
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class UniverseEntry:
    symbol: str
    name: str
    sector: str
    sub_industry: str


def fetch_sp500_constituents() -> list[UniverseEntry]:
    # Wikipedia blocks default urllib user-agents — use requests with a real UA
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(SP500_WIKIPEDIA_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), header=0)
    df = tables[0]
    df.columns = [c.strip() for c in df.columns]
    sym_col = "Symbol"
    name_col = "Security"
    sector_col = "GICS Sector"
    sub_col = "GICS Sub-Industry"

    entries: list[UniverseEntry] = []
    for _, row in df.iterrows():
        # Yahoo uses '-' instead of '.' for class shares (e.g. BRK.B -> BRK-B)
        symbol = str(row[sym_col]).replace(".", "-").strip()
        entries.append(
            UniverseEntry(
                symbol=symbol,
                name=str(row[name_col]).strip(),
                sector=str(row[sector_col]).strip(),
                sub_industry=str(row[sub_col]).strip(),
            )
        )
    logger.info(f"Fetched {len(entries)} S&P 500 constituents from Wikipedia")
    return entries


def persist_universe(entries: list[UniverseEntry]) -> int:
    inserted = 0
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
                    )
                )
                inserted += 1
        session.commit()
    logger.info(f"Persisted {inserted} new tickers (universe now contains all S&P 500)")
    return inserted


def get_top_n_by_liquidity(n: int = 100) -> list[str]:
    """Return top-N symbols by recent dollar volume from the candles table.

    Used as a proxy for liquidity ranking. Falls back to the first N alphabetical
    symbols if no price data is available yet.
    """
    from trading_bot.data.storage import Candle

    with get_session() as session:
        rows = session.execute(
            text(
                "SELECT symbol, AVG(close * volume) AS adv "
                "FROM candles "
                "WHERE dt >= date('now', '-90 days') "
                "GROUP BY symbol "
                "ORDER BY adv DESC "
                "LIMIT :limit"
            ),
            {"limit": n},
        ).all()
        if rows:
            return [r[0] for r in rows]

        # fallback: alphabetical first N from tickers
        symbols = (
            session.query(Ticker.symbol).order_by(Ticker.symbol).limit(n).all()
        )
        return [s[0] for s in symbols]
