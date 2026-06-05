"""Cross-sectional transformations applied to feature values across symbols.

Used to convert raw feature outputs into ranked / standardised scores that
the strategy composer combines.
"""

from __future__ import annotations

import pandas as pd


def cross_section_rank(s: pd.Series) -> pd.Series:
    """Convert raw values into ranks in [0, 1] (higher value → higher rank).

    NaNs are dropped before ranking.
    """
    cleaned = s.dropna()
    if cleaned.empty:
        return pd.Series(dtype=float)
    n = len(cleaned)
    ranks = cleaned.rank(method="average", ascending=True)
    return (ranks - 0.5) / n  # midpoint of each rank cell, in [0,1)


def cross_section_zscore(s: pd.Series) -> pd.Series:
    cleaned = s.dropna()
    if cleaned.empty or cleaned.std() == 0:
        return pd.Series(dtype=float)
    return (cleaned - cleaned.mean()) / cleaned.std()


def top_decile(s: pd.Series, decile: float = 0.1) -> pd.Series:
    """Boolean mask: True for the top ``decile`` of the cross-section."""
    cleaned = s.dropna()
    if cleaned.empty:
        return pd.Series(dtype=bool)
    cutoff = cleaned.quantile(1 - decile)
    return cleaned >= cutoff
