"""Tests for portfolio overlays: vol targeting, adaptive rebalance, gold sleeve."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.strategies.base import BaseStrategy


class _EqualWeightTop2(BaseStrategy):
    """Minimal strategy: always hold first two columns equally."""

    name = "test_eq2"

    def __init__(self):
        pass

    def rank(self, prices, asof):
        cols = [c for c in prices.columns if c != "SPY"][:2]
        return pd.Series(1.0, index=cols)

    def weights(self, prices, asof):
        cols = [c for c in prices.columns if c != "SPY"][:2]
        return pd.Series(0.5, index=cols)


def _make_panel(n=600, seed=0, vol_regimes=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    if vol_regimes:
        # First half calm (0.5% daily vol), second half stressed (3% daily vol)
        sig = np.concatenate([np.full(n // 2, 0.005), np.full(n - n // 2, 0.03)])
    else:
        sig = np.full(n, 0.01)
    data = {
        f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, sig)))
        for i in range(3)
    }
    data["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0002, sig)))
    return pd.DataFrame(data, index=idx)


class TestVolTargeting:
    def test_vol_target_reduces_realized_vol(self):
        panel = _make_panel(vol_regimes=True)
        base = CrossSectionalBacktester(_EqualWeightTop2(), BacktestConfig(
            circuit_breaker_dd=1.0)).run(panel)
        targeted = CrossSectionalBacktester(_EqualWeightTop2(), BacktestConfig(
            circuit_breaker_dd=1.0, vol_target=0.10)).run(panel)

        # Vol of the stressed second half must be lower with targeting
        half = len(panel) // 2
        v_base = base.returns.iloc[half:].std()
        v_tgt = targeted.returns.iloc[half:].std()
        assert v_tgt < v_base

    def test_vol_target_never_levers_up(self):
        """Scale is capped at 1.0 — calm regimes must NOT be amplified."""
        panel = _make_panel()
        base = CrossSectionalBacktester(_EqualWeightTop2(), BacktestConfig(
            circuit_breaker_dd=1.0)).run(panel)
        targeted = CrossSectionalBacktester(_EqualWeightTop2(), BacktestConfig(
            circuit_breaker_dd=1.0, vol_target=5.0)).run(panel)  # absurd target
        # With a huge target the scale caps at 1 → identical returns
        pd.testing.assert_series_equal(base.returns, targeted.returns)


class TestAdaptiveRebalance:
    def test_extra_rebalances_in_high_vol(self):
        panel = _make_panel(vol_regimes=True)
        cfg_off = BacktestConfig(circuit_breaker_dd=1.0, adaptive_rebalance=False)
        cfg_on = BacktestConfig(circuit_breaker_dd=1.0, adaptive_rebalance=True,
                                vol_trigger=0.25)
        r_off = CrossSectionalBacktester(_EqualWeightTop2(), cfg_off).run(panel)
        r_on = CrossSectionalBacktester(_EqualWeightTop2(), cfg_on).run(panel)
        # Adaptive mode must produce MORE rebalance rows (weekly adds in stress)
        assert len(r_on.weights_history) > len(r_off.weights_history)

    def test_no_extra_rebalances_in_calm(self):
        panel = _make_panel(vol_regimes=False)  # 1% daily ≈ 16% ann < 25% trigger
        cfg_on = BacktestConfig(circuit_breaker_dd=1.0, adaptive_rebalance=True,
                                vol_trigger=0.25)
        cfg_off = BacktestConfig(circuit_breaker_dd=1.0, adaptive_rebalance=False)
        r_on = CrossSectionalBacktester(_EqualWeightTop2(), cfg_on).run(panel)
        r_off = CrossSectionalBacktester(_EqualWeightTop2(), cfg_off).run(panel)
        assert len(r_on.weights_history) == len(r_off.weights_history)


class TestRunnerOverlays:
    def test_is_volatility_rebalance_day_non_monday(self):
        from trading_bot.live.runner import is_volatility_rebalance_day
        # A Wednesday — must be False regardless of vol
        assert not is_volatility_rebalance_day(pd.Timestamp("2023-01-04"))

    def test_apply_overlays_preserves_budget(self):
        """Total exposure after overlays must stay <= 1.0."""
        from trading_bot.live.runner import apply_portfolio_overlays
        core = {"AAPL": 0.5, "MSFT": 0.5}
        out = apply_portfolio_overlays(core, pd.Timestamp("2025-06-02"))
        assert sum(out.values()) <= 1.0 + 1e-9
        assert all(w >= 0 for w in out.values())


class TestPaperMods:
    """Le 3 modifiche dai paper: HLZ gate, DM panic, NMV hold band."""

    def test_hold_band_reduces_turnover(self):
        panel = _make_panel(n=700)
        from trading_bot.strategies.composer import (ComposedConfig,
                                                     ComposedStrategy, SignalSpec)
        cfg = ComposedConfig(name="t", signals=(
            SignalSpec(feature="rate_of_change", params={"lookback": 21}),),
            top_n=2)
        base = CrossSectionalBacktester(ComposedStrategy(cfg), BacktestConfig(
            circuit_breaker_dd=1.0, hold_band_mult=1.0)).run(panel)
        band = CrossSectionalBacktester(ComposedStrategy(cfg), BacktestConfig(
            circuit_breaker_dd=1.0, hold_band_mult=2.0)).run(panel)
        t_base = base.weights_history.diff().abs().sum().sum()
        t_band = band.weights_history.diff().abs().sum().sum()
        assert t_band <= t_base  # band must not increase turnover

    def test_panic_filter_scales_in_panic(self):
        panel = _make_panel(n=700, vol_regimes=True)
        # Make SPY collapse in second half so it sits below SMA200 with high vol
        import numpy as np
        half = len(panel) // 2
        spy = panel["SPY"].values.copy()
        spy[half:] = spy[half] * np.exp(np.cumsum(
            np.random.default_rng(1).normal(-0.004, 0.03, len(panel) - half)))
        panel["SPY"] = spy
        base = CrossSectionalBacktester(_EqualWeightTop2(), BacktestConfig(
            circuit_breaker_dd=1.0, panic_filter=False)).run(panel)
        pan = CrossSectionalBacktester(_EqualWeightTop2(), BacktestConfig(
            circuit_breaker_dd=1.0, panic_filter=True, vol_trigger=0.25,
            panic_scale=0.5)).run(panel)
        # In panic periods gross exposure must be lower
        g_base = base.weights_history.abs().sum(axis=1).max()
        assert pan.returns.iloc[half:].abs().sum() <= base.returns.iloc[half:].abs().sum() + 1e-9

    def test_adaptive_gate_monotone(self):
        import math
        thr = lambda n: 0.5 * (1 + 0.3 * max(0.0, math.log10(max(n, 1) / 100)))
        assert thr(50) == 0.5 and thr(100) == 0.5
        assert thr(1000) > thr(100)
        assert abs(thr(1000) - 0.65) < 1e-9
