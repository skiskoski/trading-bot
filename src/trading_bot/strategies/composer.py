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
    # ── Gold sleeve as a SEARCHABLE STRATEGY PARAMETER ──────────────────
    # gold_weight: capital share allocated to GLD when the sleeve is ON.
    #   0.0 = no gold. Explored by the research loop like any other param.
    # gold_mode:
    #   "defensive" = gold held ONLY when max-drawdown risk is elevated
    #     (SPY below SMA200 or market vol > 25% ann.) — crisis ballast,
    #     not a permanent allocation.
    #   "always" = gold held whenever its own trend allows.
    # In both modes GLD must also be above its own SMA200 (don't catch a
    # falling gold knife — Baur-Lucey 2010 crisis-hedge evidence).
    gold_weight: float = 0.0
    gold_mode: str = "defensive"


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

        # Exclude regime symbol and GLD (sleeve asset) from the rankable universe
        tradable = prices
        drop_cols = []
        if self.cfg.regime is not None and self.cfg.regime.symbol in tradable.columns:
            drop_cols.append(self.cfg.regime.symbol)
        if "GLD" in tradable.columns and "GLD" not in drop_cols:
            drop_cols.append("GLD")
        if drop_cols:
            tradable = tradable.drop(columns=drop_cols)

        # Build composite score. A symbol must have ALL signals available:
        # add(fill_value=0) let partial-coverage symbols (e.g. recent IPOs
        # missing the long-lookback signal) sneak into the ranking with a
        # truncated score (audit finding M3).
        parts: list[pd.Series] = []
        for sig in self.cfg.signals:
            fn = price_features.REGISTRY[sig.feature]
            raw = fn(tradable, asof, **sig.params)
            if raw.empty:
                return pd.Series(dtype=float)
            if sig.negate:
                raw = -raw
            transformed = cross_section_rank(raw) if sig.use_rank else raw
            parts.append(transformed * sig.weight)
        score_df = pd.concat(parts, axis=1)
        score = score_df.sum(axis=1)
        score[score_df.isna().any(axis=1)] = np.nan   # require full coverage

        # Apply filters: each must be True
        for filt in self.cfg.filters:
            fn = price_features.REGISTRY[filt.feature]
            f = fn(tradable, asof, **filt.params)
            if f.empty:
                return pd.Series(dtype=float)
            score = score.where(f > filt.threshold)

        return score.dropna()

    def _gold_active(self, prices: pd.DataFrame, asof: pd.Timestamp) -> bool:
        """Should the gold sleeve be ON at ``asof``? Uses only data ≤ asof.

        defensive mode: ONLY when equity max-drawdown risk is elevated —
        SPY below SMA200 OR market 21d vol > 25% annualized.
        always mode: whenever gold's own trend allows.
        Both modes require GLD above its own SMA200.
        """
        if "GLD" not in prices.columns:
            return False
        gld = prices["GLD"].loc[:asof].dropna()
        if len(gld) < 200 or gld.iloc[-1] <= gld.iloc[-200:].mean():
            return False    # gold itself in downtrend — no knife catching
        if self.cfg.gold_mode == "always":
            return True
        # defensive: equity stress expected?
        if "SPY" not in prices.columns:
            return False
        spy = prices["SPY"].loc[:asof].dropna()
        if len(spy) < 200:
            return False
        below_sma = spy.iloc[-1] < spy.iloc[-200:].mean()
        vol21 = float(spy.pct_change().iloc[-21:].std() * np.sqrt(252))
        return bool(below_sma or vol21 > 0.25)

    def weights(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        # Regime gate (equity sleeve only — gold sleeve decided separately:
        # in a bear regime the equity book goes flat but defensive gold can
        # be ON; that is exactly its crisis-ballast job)
        equity_on = True
        if self.cfg.regime is not None and self.cfg.regime.symbol in prices.columns:
            fn = price_features.REGISTRY[self.cfg.regime.feature]
            r = fn(prices[[self.cfg.regime.symbol]], asof, **self.cfg.regime.params)
            if r.empty:
                equity_on = False
            else:
                value = r.iloc[0] if isinstance(r, pd.Series) else r
                if value <= self.cfg.regime.threshold:
                    equity_on = False

        out = pd.Series(dtype=float)
        if equity_on:
            scores = self.rank(prices, asof)
            if not scores.empty:
                top = scores.sort_values(ascending=False).head(self.cfg.top_n)
                if not top.empty:
                    out = pd.Series(1.0 / len(top), index=top.index)

        # Gold sleeve: a strategy parameter, not a fixed overlay
        gw = self.cfg.gold_weight
        if gw > 0 and self._gold_active(prices, asof):
            out = out * (1.0 - gw)
            out.loc["GLD"] = out.get("GLD", 0.0) + gw

        return out
