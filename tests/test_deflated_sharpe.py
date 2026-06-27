import numpy as np
import pandas as pd
import pytest

from trading_bot.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_psr_is_half_at_observed_sharpe():
    """PSR(SR*) is exactly 0.5 when SR* equals the observed annualised Sharpe."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 2000))
    # observed annualised sharpe
    obs_sr_annual = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
    psr = probabilistic_sharpe_ratio(rets, benchmark_sharpe=obs_sr_annual)
    assert abs(psr - 0.5) < 1e-3


def test_psr_monotone_in_benchmark():
    rng = np.random.default_rng(0)
    daily_mean = 1.0 / np.sqrt(252) * 0.01  # SR≈1 annualised
    rets = pd.Series(rng.normal(daily_mean, 0.01, 1500))
    psr_low = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0)
    psr_high = probabilistic_sharpe_ratio(rets, benchmark_sharpe=2.0)
    assert psr_low > psr_high


def test_psr_high_for_strong_positive_sharpe():
    rng = np.random.default_rng(1)
    # SR ~ 2 annualised: daily mean 2/sqrt(252)*0.01 ≈ 1.26e-3, sigma 1%
    daily_mean = 2.0 / np.sqrt(252) * 0.01
    rets = pd.Series(rng.normal(daily_mean, 0.01, 1500))
    psr = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0)
    assert psr > 0.95


def test_expected_max_sharpe_grows_with_n():
    e1 = expected_max_sharpe(2)
    e10 = expected_max_sharpe(10)
    e100 = expected_max_sharpe(100)
    assert e1 < e10 < e100


def test_dsr_below_psr_with_many_trials():
    rng = np.random.default_rng(2)
    daily_mean = 1.0 / np.sqrt(252) * 0.01
    rets = pd.Series(rng.normal(daily_mean, 0.01, 1500))
    psr = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0)
    dsr = deflated_sharpe_ratio(rets, n_trials=100)
    # With many trials, the bar is higher, so DSR <= PSR
    assert dsr <= psr
