"""Strategy composer: build long-only cross-sectional strategies declaratively.

A composed strategy is described by:
    - one or more ``signal`` features whose linear combination forms the ranking score
    - per-symbol eligibility filters (boolean features that must be true)
    - a regime gate (single-asset condition that, when False, forces all-cash)
    - a sizing rule (currently: top-N equal weight)

This makes new strategies cheap to add: write the dict, the rest is free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.features import price as price_features
from trading_bot.features.cross_section import cross_section_rank
from trading_bot.strategies.base import BaseStrategy, StrategyConfig


@dataclass(frozen=True)
class SignalSpec:
    feature: str  # name in price_features.REGISTRY
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    # If True, the feature's cross-sectional rank is used (robust to outliers);
    # otherwise the raw value is used directly.
    use_rank: bool = True
    # If True, the feature is inverted (e.g. for "low-volatility" anomaly).
    negate: bool = False


@dataclass(frozen=True)
class FilterSpec:
    feature: str
    params: dict[str, Any] = field(default_factory=dict)
    # The feature must be > threshold (default 0). For boolean features (above_sma)
    # this evaluates as "True" since they return 1.0/0.0.
    threshold: float = 0.0


@dataclass(frozen=True)
class RegimeSpec:
    """Regime filter using a single asset's price feature.

    When the feature value on ``symbol`` falls below ``threshold``, the
    strategy goes all-cash.
    """

    symbol: str
    feature: str
    params: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.0


@dataclass(frozen=True)
class ComposedConfig(StrategyConfig):
    name: str = "composed"
    rationale: str = ""
    signals: tuple[SignalSpec, ...] = ()
    filters: tuple[FilterSpec, ...] = ()
    regime: RegimeSpec | None = None
    top_n: int = 10


class ComposedStrategy(BaseStrategy):
    def __init__(self, config: ComposedConfig) -> None:
        super().__init__(config)
        self.cfg: ComposedConfig = config

    def _eval_feature(
        self, spec: SignalSpec | FilterSpec | RegimeSpec, prices: pd.DataFrame, asof: pd.Timestamp
    ) -> pd.Series:
        fn = price_features.REGISTRY[spec.feature]
        return fn(prices, asof, **spec.params)

    def rank(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        if not self.cfg.signals:
            return pd.Series(dtype=float)

        # Exclude regime symbol from the rankable universe
        tradable = prices
        if self.cfg.regime is not None and self.cfg.regime.symbol in tradable.columns:
            tradable = tradable.drop(columns=[self.cfg.regime.symbol])

        # Build composite score
        score = pd.Series(0.0, index=tradable.columns)
        score[:] = np.nan
        first = True
        for sig in self.cfg.signals:
            fn = price_features.REGISTRY[sig.feature]
            raw = fn(tradable, asof, **sig.params)
            if raw.empty:
                return pd.Series(dtype=float)
            if sig.negate:
                raw = -raw
            transformed = cross_section_rank(raw) if sig.use_rank else raw
            if first:
                score = transformed * sig.weight
                first = False
            else:
                score = score.add(transformed * sig.weight, fill_value=0.0)

        # Apply filters: each must be True
        for filt in self.cfg.filters:
            fn = price_features.REGISTRY[filt.feature]
            f = fn(tradable, asof, **filt.params)
            if f.empty:
                return pd.Series(dtype=float)
            score = score.where(f > filt.threshold)

        return score.dropna()

    def weights(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        # Regime gate
        if self.cfg.regime is not None and self.cfg.regime.symbol in prices.columns:
            fn = price_features.REGISTRY[self.cfg.regime.feature]
            r = fn(prices[[self.cfg.regime.symbol]], asof, **self.cfg.regime.params)
            if r.empty:
                return pd.Series(dtype=float)
            value = r.iloc[0] if isinstance(r, pd.Series) else r
            if value <= self.cfg.regime.threshold:
                return pd.Series(dtype=float)

        scores = self.rank(prices, asof)
        if scores.empty:
            return pd.Series(dtype=float)
        top = scores.sort_values(ascending=False).head(self.cfg.top_n)
        if top.empty:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(top), index=top.index)
