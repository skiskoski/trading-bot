"""Bayesian feature scorecard.

Tracks how each feature (and parameter combination) has performed across all
research runs. After each run the score is updated via a simple online
Bayesian update:

    score = mean_oos_sharpe * (1 - mean_pbo) * sqrt(times_used + 1)
            * promotion_bonus

This gives a prior probability for the next hypothesis generation step:
features that appeared in promoted strategies get a higher sampling weight.

The scorecard is persisted in the SQLite DB (FeatureScore table) so it
survives across sessions and accumulates knowledge over time.
"""

from __future__ import annotations

import json
import math

from sqlalchemy import select

from trading_bot.data.storage import FeatureScore, get_session, init_db

init_db()


def _key(feature: str, params: dict, universe: int | None = None) -> str:
    """Canonical string key for a (feature, params) pair.

    When ``universe`` is given, the key is namespaced by universe size so that
    learning from different universes (e.g. top-100 vs top-500) never mixes.
    """
    param_str = json.dumps(params, sort_keys=True)
    base = f"{feature}:{param_str}"
    return f"u{universe}|{base}" if universe is not None else base


def update_score(
    feature: str,
    params: dict,
    oos_sharpe: float,
    pbo: float,
    promoted: bool,
    universe: int | None = None,
) -> float:
    """Update the score for a (feature, params) combo after a completed run.

    Returns the new score.
    """
    key = _key(feature, params, universe)
    with get_session() as session:
        row = session.execute(
            select(FeatureScore).where(FeatureScore.feature_key == key)
        ).scalar_one_or_none()
        if row is None:
            row = FeatureScore(feature_key=key, times_used=0, times_promoted=0,
                               sum_oos_sharpe=0.0, sum_pbo=0.0, score=0.5)
            session.add(row)

        row.times_used += 1
        if promoted:
            row.times_promoted += 1
        row.sum_oos_sharpe += max(oos_sharpe, -2.0)  # clip extreme negatives
        row.sum_pbo += pbo

        n = row.times_used
        mean_oos = row.sum_oos_sharpe / n
        mean_pbo = row.sum_pbo / n
        promotion_bonus = 1.0 + 0.5 * (row.times_promoted / n)
        # Clip oos_sharpe contribution to [0, 2] so no single run dominates
        sharpe_contrib = max(0.0, min(mean_oos, 2.0))
        row.score = sharpe_contrib * (1.0 - mean_pbo) * math.sqrt(n) * promotion_bonus
        session.commit()
        return row.score


def get_score(feature: str, params: dict, universe: int | None = None) -> float:
    """Return current score for a feature/param combo. Default 0.5 if unseen."""
    key = _key(feature, params, universe)
    with get_session() as session:
        row = session.execute(
            select(FeatureScore).where(FeatureScore.feature_key == key)
        ).scalar_one_or_none()
        return float(row.score) if row else 0.5


def get_all_scores(universe: int | None = None) -> list[dict]:
    """Return feature scores sorted by score descending.

    When ``universe`` is given, only scores from that universe namespace are
    returned (keys prefixed ``u{universe}|``), so the bandit never learns from
    a different universe's results.
    """
    with get_session() as session:
        query = select(FeatureScore).order_by(FeatureScore.score.desc())
        if universe is not None:
            query = query.where(FeatureScore.feature_key.like(f"u{universe}|%"))
        rows = session.execute(query).scalars().all()

        def _strip(k: str) -> str:
            # Remove "u{n}|" namespace prefix so downstream feature:params parsing works
            return k.split("|", 1)[1] if k.startswith("u") and "|" in k else k

        return [
            {
                "key": _strip(r.feature_key),
                "times_used": r.times_used,
                "times_promoted": r.times_promoted,
                "mean_oos_sharpe": r.sum_oos_sharpe / r.times_used if r.times_used else 0.0,
                "mean_pbo": r.sum_pbo / r.times_used if r.times_used else 0.0,
                "score": r.score,
            }
            for r in rows
        ]
