"""Catalogue of pre-built strategy configs.

Adding a new strategy is now a 5-line dict, not a 100-line class. Each entry
includes a ``rationale`` field that must reference the economic logic — this
is part of the honesty harness (pre-registration).
"""

from __future__ import annotations

from trading_bot.strategies.composer import (
    ComposedConfig,
    ComposedStrategy,
    FilterSpec,
    RegimeSpec,
    SignalSpec,
)

CATALOG: dict[str, ComposedConfig] = {
    "momentum_12_1": ComposedConfig(
        name="momentum_12_1",
        rationale=(
            "Cross-sectional momentum (Jegadeesh-Titman 1993, Carhart 1997). "
            "12-month formation, 1-month skip. Stock above SMA-100 filter and "
            "SPY above SMA-200 regime gate as in Clenow (2015)."
        ),
        signals=(
            SignalSpec(feature="momentum", params={"lookback": 252, "skip": 21}, use_rank=True),
        ),
        filters=(
            FilterSpec(feature="above_sma", params={"window": 100}, threshold=0.5),
        ),
        regime=RegimeSpec(
            symbol="SPY", feature="sma_distance", params={"window": 200}, threshold=0.0
        ),
        top_n=10,
    ),
    "rsi2_reversal": ComposedConfig(
        name="rsi2_reversal",
        rationale=(
            "Short-term mean reversion (Lehmann 1990, Connors-Alvarez 2009). "
            "RSI(2) < threshold on names in a longer-term uptrend (above SMA-200). "
            "Inverted so low RSI ranks higher."
        ),
        signals=(
            SignalSpec(feature="rsi", params={"period": 2}, use_rank=True, negate=True),
        ),
        filters=(
            FilterSpec(feature="above_sma", params={"window": 200}, threshold=0.5),
        ),
        regime=RegimeSpec(
            symbol="SPY", feature="sma_distance", params={"window": 200}, threshold=0.0
        ),
        top_n=10,
    ),
    "low_volatility": ComposedConfig(
        name="low_volatility",
        rationale=(
            "Low-volatility anomaly (Baker-Bradley-Wurgler 2011, Frazzini-Pedersen 2014 "
            "Betting Against Beta). Long the lowest-volatility names within an uptrending "
            "regime; defensive ballast for the portfolio."
        ),
        signals=(
            SignalSpec(feature="low_volatility", params={"lookback": 252}, use_rank=True),
        ),
        filters=(
            FilterSpec(feature="above_sma", params={"window": 200}, threshold=0.5),
        ),
        regime=RegimeSpec(
            symbol="SPY", feature="sma_distance", params={"window": 200}, threshold=0.0
        ),
        top_n=15,
    ),
    "combo_mom_lowvol": ComposedConfig(
        name="combo_mom_lowvol",
        rationale=(
            "Composite: 60% cross-sectional 12-1 momentum + 40% low-volatility. "
            "Asness, Frazzini, Pedersen (2019) and Asness-Moskowitz-Pedersen (2013) "
            "argue these factors are weakly to negatively correlated and combine to "
            "improve risk-adjusted returns."
        ),
        signals=(
            SignalSpec(
                feature="momentum",
                params={"lookback": 252, "skip": 21},
                use_rank=True,
                weight=0.6,
            ),
            SignalSpec(
                feature="low_volatility",
                params={"lookback": 252},
                use_rank=True,
                weight=0.4,
            ),
        ),
        filters=(FilterSpec(feature="above_sma", params={"window": 100}, threshold=0.5),),
        regime=RegimeSpec(
            symbol="SPY", feature="sma_distance", params={"window": 200}, threshold=0.0
        ),
        top_n=10,
    ),
}


def get_strategy(name: str) -> ComposedStrategy:
    if name not in CATALOG:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(CATALOG.keys())}"
        )
    return ComposedStrategy(CATALOG[name])


def list_strategies() -> list[str]:
    return sorted(CATALOG.keys())
