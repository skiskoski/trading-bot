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

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)

_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


_INDEX_CHANGE_URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def _parse_changes_table(raw: pd.DataFrame, index_name: str) -> list[dict]:
    """Parse one Wikipedia 'Selected changes' table into change records."""
    raw = raw.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            " ".join(c for c in col if c and not c.startswith("Unnamed")).strip()
            for col in raw.columns
        ]

    col_date = next((c for c in raw.columns if "Date" in c), None)
    # "Ticker" sub-header for S&P 500; S&P 400/600 use plain "Added"/"Removed"
    col_add = (
        next((c for c in raw.columns if "Added" in c and "Ticker" in c), None)
        or next((c for c in raw.columns if c.strip() == "Added"), None)
        or next((c for c in raw.columns if "Added" in c and "Security" not in c), None)
    )
    col_rem = (
        next((c for c in raw.columns if "Removed" in c and "Ticker" in c), None)
        or next((c for c in raw.columns if c.strip() == "Removed"), None)
        or next((c for c in raw.columns if "Removed" in c and "Security" not in c), None)
    )
    col_rsn = next((c for c in raw.columns if "Reason" in c), None)

    if col_date is None or (col_add is None and col_rem is None):
        logger.warning(
            f"[{index_name}] Could not locate add/remove columns. "
            f"Available: {list(raw.columns)}"
        )
        return []

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
            if added in ("nan", "—", "-", ""):
                added = None
        if removed:
            removed = removed.replace(".", "-")
            if removed in ("nan", "—", "-", ""):
                removed = None
        if not added and not removed:
            continue
        records.append(
            {"dt": d.date(), "added": added, "removed": removed,
             "reason": reason, "index_name": index_name}
        )
    return records


def fetch_index_changes(indices: list[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame of add/remove events across S&P 500/400/600.

    Args:
        indices: which indices to scrape. Default: all three (S&P 1500).

    Columns: dt, added, removed, reason, index_name.
    """
    if indices is None:
        indices = ["sp500", "sp400", "sp600"]

    all_records: list[dict] = []
    for idx in indices:
        url = _INDEX_CHANGE_URLS[idx]
        try:
            resp = requests.get(url, headers=_BROWSER_UA, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
        except Exception as e:
            logger.warning(f"[{idx}] fetch failed: {e}")
            continue

        # Find the changes table: the one with Date + Added/Removed columns.
        # Flatten across ALL header levels (Added/Removed live in level 0,
        # Ticker/Security in level 1 for the multi-index Wikipedia tables).
        changes_table = None
        for t in tables[1:]:  # skip table 0 (current constituents)
            if isinstance(t.columns, pd.MultiIndex):
                flat = " ".join(
                    str(c) for lvl in range(t.columns.nlevels)
                    for c in t.columns.get_level_values(lvl)
                )
            else:
                flat = " ".join(str(c) for c in t.columns)
            if "Date" in flat and ("Added" in flat or "Removed" in flat):
                changes_table = t
                break
        if changes_table is None:
            logger.warning(f"[{idx}] no changes table found")
            continue

        recs = _parse_changes_table(changes_table, idx)
        all_records.extend(recs)
        logger.info(f"[{idx}] parsed {len(recs)} change events")

    df = pd.DataFrame(all_records)
    logger.info(f"Total index change events across {len(indices)} indices: {len(df)}")
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
                    dt=r["dt"], added=r["added"], removed=r["removed"],
                    reason=r.get("reason"), index_name=r.get("index_name"),
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

    # PER-INDEX membership (audit A5): a symbol promoted sp400→sp500 is
    # removed from one index and added to another around the same date.
    # Tracking ONE merged set made the apply-order decide whether the symbol
    # survived — transitions silently dropped stocks. We track one member set
    # per index and a symbol is a member if it belongs to ANY index.
    start_date = pd.Timestamp(dates[0]).date()
    members_by_index = _constituents_by_index_at(start_date, current_universe)

    def _is_member(sym: str) -> bool:
        return any(sym in s for s in members_by_index.values())

    cols = sorted(symbols) if symbols else sorted(ever_in_index(current_universe))
    panel = pd.DataFrame(False, index=dates, columns=cols)
    panel.iloc[0] = pd.Series({c: _is_member(c) for c in cols})

    change_idx = 0
    for i in range(1, len(dates)):
        current_date = pd.Timestamp(dates[i]).date()
        while change_idx < len(all_changes) and all_changes[change_idx].dt <= current_date:
            ch = all_changes[change_idx]
            idx_set = members_by_index.setdefault(ch.index_name or "sp500", set())
            if ch.added:
                idx_set.add(ch.added)
            if ch.removed:
                idx_set.discard(ch.removed)
            change_idx += 1
        panel.iloc[i] = pd.Series({c: _is_member(c) for c in cols})
    return panel


def _constituents_by_index_at(
    asof: date, current_universe: set[str]
) -> dict[str, set[str]]:
    """Per-index member sets as of ``asof`` (walking back from today).

    Current per-index membership comes from Ticker.index_name; symbols
    without one (ETFs like SPY/GLD, manual additions) sit in an 'unknown'
    bucket that has no change history — always members if currently present.
    """
    from trading_bot.data.storage import Ticker

    with get_session() as session:
        idx_map = {
            t.symbol: (t.index_name or "unknown")
            for t in session.query(Ticker).all()
        }
        rows = (
            session.query(IndexChange)
            .filter(IndexChange.dt > asof)
            .order_by(IndexChange.dt.desc())
            .all()
        )

    # Symbols absent from tickers (e.g. delisted before the current fetch)
    # inherit their index from the change tables — otherwise they'd sit in
    # an 'unknown' bucket the changes never touch and stay members forever.
    with get_session() as session:
        for r2 in session.query(IndexChange).all():
            for sym2 in (r2.added, r2.removed):
                if sym2 and sym2 not in idx_map:
                    idx_map[sym2] = r2.index_name or "sp500"

    members_by_index: dict[str, set[str]] = {}
    for sym in current_universe:
        members_by_index.setdefault(idx_map.get(sym, "unknown"), set()).add(sym)

    # Undo changes within each index independently
    for r in rows:
        idx_set = members_by_index.setdefault(r.index_name or "sp500", set())
        if r.added and r.added in idx_set:
            idx_set.discard(r.added)
        if r.removed:
            idx_set.add(r.removed)
    return members_by_index


def get_constituents_at(asof: date, current_universe: set[str]) -> set[str]:
    """Union of per-index constituents as of ``asof`` (S&P 1500 membership)."""
    by_index = _constituents_by_index_at(asof, current_universe)
    out: set[str] = set()
    for s in by_index.values():
        out |= s
    return out
