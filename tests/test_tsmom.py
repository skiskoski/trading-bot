"""Tests for TSMOM gold sleeve strategy."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.features.price import realized_vol, tsmom_signal
from trading_bot.strategies.tsmom import TSMOMConfig, TSMOMStrategy


def make_gld_panel(n_days: int = 1500, trend: float = 0.0003) -> pd.DataFrame:
    """Uptrending GLD panel — should trigger long signal most of the time."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2005-01-01", periods=n_days, freq="B")
    rets = rng.normal(trend, 0.010, n_days)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"GLD": prices}, index=idx)


def make_gld_panel_bear(n_days: int = 1500) -> pd.DataFrame:
    """Downtrending GLD panel — should stay flat."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2005-01-01", periods=n_days, freq="B")
    rets = rng.normal(-0.0005, 0.010, n_days)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"GLD": prices}, index=idx)


# ── feature tests ──────────────────────────────────────────────────────────

def test_tsmom_signal_binary():
    panel = make_gld_panel()
    asof = panel.index[-1]
    sig = tsmom_signal(panel, asof)
    assert set(sig.values).issubset({0.0, 1.0})


def test_tsmom_signal_long_in_uptrend():
    panel = make_gld_panel(trend=0.0006)
    asof = panel.index[-1]
    sig = tsmom_signal(panel, asof)
    assert sig["GLD"] == 1.0


def test_tsmom_signal_flat_in_downtrend():
    panel = make_gld_panel_bear()
    asof = panel.index[-1]
    sig = tsmom_signal(panel, asof)
    assert sig["GLD"] == 0.0


def test_tsmom_signal_empty_on_short_history():
    panel = make_gld_panel(n_days=100)
    asof = panel.index[-1]
    sig = tsmom_signal(panel, asof, sma_window=200, mom_lookback=252)
    assert sig.empty


def test_realized_vol_positive():
    panel = make_gld_panel()
    asof = panel.index[-1]
    rv = realized_vol(panel, asof, lookback=66)
    assert rv["GLD"] > 0


# ── strategy tests ─────────────────────────────────────────────────────────

def test_tsmom_weights_in_uptrend():
    panel = make_gld_panel(trend=0.0006)
    strat = TSMOMStrategy(TSMOMConfig())
    asof = panel.index[-1]
    w = strat.weights(panel, asof)
    assert "GLD" in w.index
    assert 0.0 < w["GLD"] <= 1.0


def test_tsmom_weights_flat_in_downtrend():
    panel = make_gld_panel_bear()
    strat = TSMOMStrategy(TSMOMConfig())
    asof = panel.index[-1]
    w = strat.weights(panel, asof)
    assert w.empty


def test_tsmom_vol_targeting_caps_at_max_weight():
    """When realised vol is very low, weight should be capped at max_weight."""
    panel = make_gld_panel(trend=0.0006)
    # Artificially flatten price to make vol near zero
    flat = pd.DataFrame(
        {"GLD": np.linspace(100, 115, len(panel))}, index=panel.index
    )
    strat = TSMOMStrategy(TSMOMConfig(vol_target=0.10, max_weight=1.0))
    asof = flat.index[-1]
    w = strat.weights(flat, asof)
    if not w.empty:
        assert w["GLD"] <= 1.0


def test_tsmom_backtest_produces_equity():
    panel = make_gld_panel(n_days=1500)
    strat = TSMOMStrategy(TSMOMConfig())
    bt = CrossSectionalBacktester(strat, BacktestConfig())
    result = bt.run(panel)
    assert len(result.equity) == len(panel)
    assert result.equity.iloc[-1] > 0


def test_tsmom_no_leverage():
    """Vol-targeted weight must never exceed max_weight=1.0."""
    panel = make_gld_panel(trend=0.0006)
    strat = TSMOMStrategy(TSMOMConfig(max_weight=1.0))
    bt = CrossSectionalBacktester(strat, BacktestConfig())
    result = bt.run(panel)
    if not result.weights_history.empty:
        assert (result.weights_history.fillna(0) <= 1.0 + 1e-9).all().all()
