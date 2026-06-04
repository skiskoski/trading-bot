"""Point-in-time S&P 500 membership reconstruction.

Strategy:
1. Scrape the current constituents (table 0 on the Wikipedia page).
2. Scrape the "Selected changes to the list" table (table 1).
3. Walk backwards from today: at each change date, undo the change to obtain
   the historical membership set.

This handles the most damaging form of survivorship bias for a US large-cap
strategy: backtesting on a universe that only contains today's winners.
"""

from datetime import date
from io import StringIO

import pandas as pd
import requests

from trading_bot.data.storage import IndexChange, get_session
from trading_bot.data.universe import SP500_WIKIPEDIA_URL
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)

_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def fetch_index_changes() -> pd.DataFrame:
    """Return a DataFrame of S&P 500 add/remove events from Wikipedia.

    Columns: dt (date), added (str|None), removed (str|None), reason (str|None).
    """
    resp = requests.get(SP500_WIKIPEDIA_URL, headers=_BROWSER_UA, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if len(tables) < 2:
        logger.warning("Wikipedia page returned < 2 tables; cannot extract changes")
        return pd.DataFrame()

    raw = tables[1].copy()
    # Wikipedia uses multi-level headers; flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            " ".join(c for c in col if c and not c.startswith("Unnamed")).strip()
            for col in raw.columns
        ]

    # Heuristically locate the columns we want
    col_date = next((c for c in raw.columns if "Date" in c), None)
    col_add = next((c for c in raw.columns if "Added" in c and "Ticker" in c), None)
    col_rem = next((c for c in raw.columns if "Removed" in c and "Ticker" in c), None)
    col_rsn = next((c for c in raw.columns if "Reason" in c), None)

    if col_date is None or (col_add is None and col_rem is None):
        logger.warning(
            f"Could not locate add/remove columns. Available: {list(raw.columns)}"
        )
        return pd.DataFrame()

    records: list[dict] = []
    for _, row in raw.iterrows():
        d = pd.to_datetime(row[col_date], errors="coerce")
        if pd.isna(d):
            continue
        added = str(row[col_add]).strip() if col_add and not pd.isna(row[col_add]) else None
        removed = str(row[col_rem]).strip() if col_rem and not pd.isna(row[col_rem]) else None
        reason = str(row[col_rsn]).strip() if col_rsn and not pd.isna(row[col_rsn]) else None
        # Normalise dot-style symbols (BRK.B -> BRK-B) for yfinance compatibility
        if added:
            added = added.replace(".", "-")
        if removed:
            removed = removed.replace(".", "-")
        records.append(
            {"dt": d.date(), "added": added, "removed": removed, "reason": reason}
        )

    df = pd.DataFrame(records)
    logger.info(f"Fetched {len(df)} S&P 500 index change events")
    return df


def persist_index_changes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with get_session() as session:
        # Wipe and re-insert to keep the table current
        session.query(IndexChange).delete()
        for _, r in df.iterrows():
            session.add(
                IndexChange(
                    dt=r["dt"], added=r["added"], removed=r["removed"], reason=r["reason"]
                )
            )
        session.commit()
    logger.info(f"Persisted {len(df)} index change rows")
    return len(df)


def ever_in_index(current_universe: set[str]) -> set[str]:
    """Return every symbol that was in the S&P 500 at any point in our history.

    Union of current members and all 'removed' tickers from the changes table.
    """
    with get_session() as session:
        removed = {
            r[0] for r in session.query(IndexChange.removed).filter(IndexChange.removed.is_not(None)).all()
        }
        added = {
            r[0] for r in session.query(IndexChange.added).filter(IndexChange.added.is_not(None)).all()
        }
    return set(current_universe) | removed | added


def build_membership_panel(
    dates: pd.DatetimeIndex,
    current_universe: set[str],
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame[date, symbol] of booleans indicating S&P 500 membership.

    Walks history forward: starting from the historical membership at ``dates[0]``,
    applies each change in order to obtain membership at every date.

    If ``symbols`` is provided, the panel columns are restricted to that set.
    """
    if len(dates) == 0:
        return pd.DataFrame()

    # Pull all changes once
    with get_session() as session:
        all_changes = (
            session.query(IndexChange).order_by(IndexChange.dt.asc()).all()
        )

    # Initial state at first date
    start_date = pd.Timestamp(dates[0]).date()
    members = get_constituents_at(start_date, current_universe)

    cols = sorted(symbols) if symbols else sorted(ever_in_index(current_universe))
    panel = pd.DataFrame(False, index=dates, columns=cols)
    # First row
    first_row = pd.Series({c: c in members for c in cols})
    panel.iloc[0] = first_row

    # Iterate forward: at each change date, flip membership
    change_idx = 0
    for i in range(1, len(dates)):
        current_date = pd.Timestamp(dates[i]).date()
        while change_idx < len(all_changes) and all_changes[change_idx].dt <= current_date:
            ch = all_changes[change_idx]
            if ch.added:
                members.add(ch.added)
            if ch.removed:
                members.discard(ch.removed)
            change_idx += 1
        panel.iloc[i] = pd.Series({c: c in members for c in cols})
    return panel


def get_constituents_at(asof: date, current_universe: set[str]) -> set[str]:
    """Return the set of S&P 500 symbols as of ``asof``.

    Algorithm: start with the current universe, then for every change AFTER
    ``asof`` (i.e., between now and ``asof``), undo it:
        - if a ticker was added after ``asof``, it wasn't a member at ``asof``
          → remove it
        - if a ticker was removed after ``asof``, it WAS a member at ``asof``
          → add it back
    """
    with get_session() as session:
        rows = (
            session.query(IndexChange)
            .filter(IndexChange.dt > asof)
            .order_by(IndexChange.dt.desc())
            .all()
        )

    members = set(current_universe)
    for r in rows:
        if r.added and r.added in members:
            members.discard(r.added)
        if r.removed:
            members.add(r.removed)
    return members
