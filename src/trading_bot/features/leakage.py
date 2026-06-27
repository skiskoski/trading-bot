"""Leakage check harness.

A feature is leakage-free iff its value at ``asof`` does not depend on data
strictly after ``asof``. We verify this empirically: compute the feature on
the full panel and again on the same panel truncated *to* ``asof``, then
assert the two are equal.

Any feature added to the library should be run through this harness once.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

FeatureFn = Callable[..., pd.Series]


def check_no_lookahead(
    feature_fn: FeatureFn,
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    tolerance: float = 1e-10,
    **kwargs: Any,
) -> bool:
    """Return True if ``feature_fn`` produces identical values whether or not
    post-asof data are present in the panel.
    """
    full = feature_fn(prices, asof, **kwargs)
    truncated = feature_fn(prices.loc[:asof], asof, **kwargs)
    if full.empty and truncated.empty:
        return True
    common = full.index.intersection(truncated.index)
    if len(common) == 0:
        return False
    diff = (full.loc[common] - truncated.loc[common]).abs()
    return bool((diff.max() if len(diff) else 0) <= tolerance)
