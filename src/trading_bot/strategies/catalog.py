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
from trading_bot.strategies.tsmom import TSMOMConfig, TSMOMStrategy

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
            SignalSpec(feature="volatility", params={"lookback": 252}, use_rank=True, negate=True),
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
                feature="volatility",
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


# TSMOM config stored as a separate object (not ComposedConfig) but keyed in
# the same CATALOG dict for CLI / GUI discoverability. The rationale field is
# pulled from TSMOMConfig.name; a pseudo-ComposedConfig wrapper gives the
# registry the fields it expects (name, rationale).
_TSMOM_RATIONALE = (
    "Time-series momentum on gold (GLD/IAU). "
    "Moskowitz-Ooi-Pedersen (2012) JFE: past 12-month return predicts future "
    "return for a single asset. Signal = price > 200d SMA AND 12-mo return > 0; "
    "sizing = vol-target 10% annualised (Hurst-Ooi-Pedersen 2017 AQR). "
    "Crisis alpha sleeve: corr -0.1 to +0.2 with equity (Baur-Lucey 2010)."
)

TSMOM_CONFIG = ComposedConfig(
    name="tsmom_gold",
    rationale=_TSMOM_RATIONALE,
    signals=(),   # not used — TSMOMStrategy overrides weights() directly
    top_n=1,
)

CATALOG["tsmom_gold"] = TSMOM_CONFIG


def get_strategy(name: str) -> ComposedStrategy | TSMOMStrategy:
    if name not in CATALOG:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(CATALOG.keys())}"
        )
    if name == "tsmom_gold":
        return TSMOMStrategy(TSMOMConfig())
    return ComposedStrategy(CATALOG[name])


def list_strategies() -> list[str]:
    return sorted(CATALOG.keys())
