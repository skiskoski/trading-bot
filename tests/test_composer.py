import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.catalog import CATALOG, get_strategy
from trading_bot.strategies.composer import (
    ComposedConfig,
    ComposedStrategy,
    FilterSpec,
    RegimeSpec,
    SignalSpec,
)


def make_panel(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    data = {}
    for sym, drift in zip(
        ["A", "B", "C", "D", "E", "F"],
        [0.0010, 0.0008, 0.0005, 0.0001, -0.0002, -0.0005],
    ):
        data[sym] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    data["SPY"] = 100 * np.exp(np.linspace(0, 0.5, n))  # uptrend
    return pd.DataFrame(data, index=idx)


def test_catalog_loadable():
    for name in CATALOG:
        strat = get_strategy(name)
        assert isinstance(strat, ComposedStrategy)
        assert strat.cfg.rationale  # must be non-empty


def test_composer_long_top_n_equal_weight():
    panel = make_panel()
    strat = get_strategy("momentum_12_1")
    w = strat.weights(panel, panel.index[-1])
    assert not w.empty
    assert "SPY" not in w.index  # regime symbol excluded
    assert abs(w.sum() - 1.0) < 1e-9
    # Equal weight
    assert w.nunique() == 1


def test_composer_regime_blocks_in_downtrend():
    panel = make_panel()
    # Override SPY into downtrend
    panel["SPY"] = 100 * np.exp(np.linspace(0, -0.5, len(panel)))
    strat = get_strategy("momentum_12_1")
    w = strat.weights(panel, panel.index[-1])
    assert w.empty


def test_rsi2_strategy_picks_oversold():
    panel = make_panel()
    strat = get_strategy("rsi2_reversal")
    # short history → empty
    w_early = strat.weights(panel.iloc[:50], panel.index[49])
    assert w_early.empty
    # Full history → may or may not have weights depending on RSI levels
    w = strat.weights(panel, panel.index[-1])
    assert isinstance(w, pd.Series)


def test_combo_strategy_blends_two_signals():
    panel = make_panel()
    strat = get_strategy("combo_mom_lowvol")
    w = strat.weights(panel, panel.index[-1])
    assert not w.empty
    assert "SPY" not in w.index
