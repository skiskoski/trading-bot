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
        # Original features
        (pf.momentum,             {"lookback": 252, "skip": 21}),
        (pf.short_term_reversal,  {"lookback": 5}),
        (pf.volatility,           {"lookback": 252}),
        (pf.sma_distance,         {"window": 200}),
        (pf.above_sma,            {"window": 100}),
        (pf.max_gap,              {"lookback": 90}),
        (pf.drawdown,             {"lookback": 252}),
        (pf.tsmom_signal,         {"sma_window": 200, "mom_lookback": 252}),
        (pf.realized_vol,         {"lookback": 66}),
        # New features
        (pf.rate_of_change,       {"lookback": 63}),
        (pf.price_acceleration,   {"short": 21, "long": 63}),
        (pf.ema_ratio,            {"fast": 21, "slow": 63}),
        (pf.macd_histogram,       {"fast": 12, "slow": 26, "signal": 9}),
        (pf.relative_strength,    {"lookback": 126}),
        (pf.bollinger_position,   {"window": 20, "n_std": 2.0}),
        (pf.mean_reversion_score, {"lookback": 252, "short": 21}),
        (pf.downside_vol,         {"lookback": 126}),
        (pf.vol_trend,            {"short": 21, "long": 126}),
        (pf.autocorr_returns,     {"lookback": 126, "lag": 1}),
        (pf.quality_score,        {"lookback": 252}),
        (pf.trend_quality,        {"lookback": 126}),
        (pf.seasonal_month_return, {"years": 5}),
        # Information-theoretic / state-space features
        (pf.hurst_exponent,       {"lookback": 252}),
        (pf.return_entropy,       {"lookback": 126, "bins": 12}),
        (pf.kalman_trend,         {"lookback": 252, "q_ratio": 0.05}),
    ],
)
def test_features_no_lookahead(fn, kw):
    panel = make_panel(n=800)
    asof = panel.index[-1]
    assert check_no_lookahead(fn, panel, asof, **kw)


def test_hurst_range_and_discrimination():
    """Hurst must be in [0,1]; a strongly trending series must score higher
    than a strongly mean-reverting one."""
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Trending: persistent drift + small noise
    trend = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.005, n)))
    # Mean-reverting: Ornstein-Uhlenbeck around 100
    mr = np.empty(n); mr[0] = 100.0
    for i in range(1, n):
        mr[i] = mr[i-1] + 0.5 * (100.0 - mr[i-1]) + rng.normal(0, 1.0)
    panel = pd.DataFrame({"TREND": trend, "MR": mr}, index=idx)
    h = pf.hurst_exponent(panel, idx[-1], lookback=512)
    assert ((h.dropna() >= 0) & (h.dropna() <= 1)).all()
    assert h["TREND"] > h["MR"]


def test_return_entropy_range_and_discrimination():
    """Entropy score in [0,1]; uniform-chaos returns must score lower
    (less structure) than concentrated returns."""
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    # Structured: tiny constant drift, near-zero noise → returns concentrated
    structured = 100 * np.exp(np.cumsum(np.full(n, 0.001) + rng.normal(0, 0.0005, n)))
    # Chaotic: huge uniform noise → returns spread across all bins
    chaotic = 100 * np.exp(np.cumsum(rng.uniform(-0.05, 0.05, n)))
    panel = pd.DataFrame({"STRUCT": structured, "CHAOS": chaotic}, index=idx)
    e = pf.return_entropy(panel, idx[-1], lookback=252, bins=12)
    assert ((e.dropna() >= 0) & (e.dropna() <= 1)).all()
    assert e["STRUCT"] > e["CHAOS"]


def test_kalman_trend_sign():
    """Kalman slope must be positive for uptrend, negative for downtrend."""
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    up = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n)))
    down = 100 * np.exp(np.cumsum(rng.normal(-0.002, 0.01, n)))
    panel = pd.DataFrame({"UP": up, "DOWN": down}, index=idx)
    k = pf.kalman_trend(panel, idx[-1], lookback=252, q_ratio=0.05)
    assert k["UP"] > 0
    assert k["DOWN"] < 0
    assert k["UP"] > k["DOWN"]


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


def test_volume_features_on_real_data():
    """Le feature volume usano il DB reale; con simboli sintetici → vuote."""
    import pytest
    from trading_bot.features.price import (_volume_panel, amihud_illiquidity,
                                            price_volume_corr, volume_momentum)
    if _volume_panel() is None:
        pytest.skip("DB volume non disponibile")
    from trading_bot.data.ingest import load_panel
    panel = load_panel(["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY"],
                       start="2023-01-01").ffill()
    asof = panel.index[-1]
    for fn, kw in [(amihud_illiquidity, {"lookback": 63}),
                   (volume_momentum, {"short": 21, "long": 126}),
                   (price_volume_corr, {"lookback": 63})]:
        v = fn(panel, asof, **kw)
        assert v.notna().sum() >= 4, f"{fn.__name__} vuota su dati reali"
    # panel sintetico → vuote, non crash
    synth = make_panel(n=400)
    assert amihud_illiquidity(synth, synth.index[-1]).empty
