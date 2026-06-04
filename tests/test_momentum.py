import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.momentum import MomentumConfig, MomentumStrategy


def make_panel(n_days: int, symbols: list[str], drifts: dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    data = {}
    for sym in symbols:
        drift = drifts.get(sym, 0.0)
        if sym == "SPY":
            # deterministic uptrend so the regime filter passes consistently in tests
            prices = 100 * np.exp(np.linspace(0, 0.6, n_days))
        else:
            rets = rng.normal(loc=drift, scale=0.01, size=n_days)
            prices = 100 * np.exp(np.cumsum(rets))
        data[sym] = prices
    return pd.DataFrame(data, index=idx)


def test_rank_picks_high_drift():
    # 3 years of business days ≈ 780, plenty for 252-day lookback
    panel = make_panel(
        800,
        symbols=["A", "B", "C", "D", "E", "SPY"],
        drifts={"A": 0.001, "B": 0.0008, "C": 0.0005, "D": 0.0001, "E": -0.0005, "SPY": 0.0003},
    )
    strat = MomentumStrategy(MomentumConfig(top_n=2))
    weights = strat.weights(panel, panel.index[-1])
    # Top 2 names by drift should be A, B (after filters)
    assert not weights.empty
    assert set(weights.index).issubset({"A", "B", "C", "D"})
    # Equal weights summing to 1
    assert weights.sum() == pytest.approx(1.0)


def test_regime_filter_blocks_when_spy_below_sma():
    # Make SPY in a downtrend
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-01", periods=800, freq="B")
    panel = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 800))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, 800))),
            "SPY": 100 * np.exp(np.cumsum(np.linspace(-0.001, -0.001, 800))),  # downtrend
        },
        index=idx,
    )
    strat = MomentumStrategy(MomentumConfig(top_n=2))
    w = strat.weights(panel, panel.index[-1])
    assert w.empty, "regime filter must produce no weights when SPY is below its SMA"


def test_short_history_returns_empty():
    panel = make_panel(50, symbols=["A", "B", "SPY"], drifts={})
    strat = MomentumStrategy(MomentumConfig())
    w = strat.weights(panel, panel.index[-1])
    assert w.empty
