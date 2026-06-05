import numpy as np
import pandas as pd
import pytest

from trading_bot.features import price as pf
from trading_bot.features.cross_section import cross_section_rank
from trading_bot.features.leakage import check_no_lookahead


def make_panel(n: int = 800, n_sym: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
            for i in range(n_sym)
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "fn,kw",
    [
        (pf.momentum, {"lookback": 252, "skip": 21}),
        (pf.short_term_reversal, {"lookback": 5}),
        (pf.volatility, {"lookback": 252}),
        (pf.low_volatility, {"lookback": 252}),
        (pf.sma_distance, {"window": 200}),
        (pf.above_sma, {"window": 100}),
        (pf.max_gap, {"lookback": 90}),
        (pf.drawdown, {"lookback": 252}),
    ],
)
def test_features_no_lookahead(fn, kw):
    panel = make_panel()
    asof = panel.index[-1]
    assert check_no_lookahead(fn, panel, asof, **kw)


def test_rsi_in_range():
    panel = make_panel()
    val = pf.rsi(panel, panel.index[-1], period=2)
    assert ((0 <= val) & (val <= 100)).all()


def test_cross_section_rank_in_unit_interval():
    s = pd.Series([3, 1, 2, 5, 4], index=list("ABCDE"))
    r = cross_section_rank(s)
    assert ((r >= 0) & (r <= 1)).all()
    # Ordering preserved
    assert r["D"] > r["E"] > r["A"] > r["C"] > r["B"]
