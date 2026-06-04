"""Probabilistic and Deflated Sharpe Ratios.

Implementation follows Bailey & López de Prado (2012, 2014):
- PSR: probability that the observed Sharpe exceeds a benchmark SR*, taking
  into account sample length and higher moments (skew, kurtosis).
- DSR: PSR with SR* replaced by the *expected maximum Sharpe under the null*
  across N independent trials, so a positive DSR signal survives multiple
  testing.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sharpe: float = 0.0, periods_per_year: int = 252
) -> float:
    """Return PSR(SR*), the probability the *true* annualised Sharpe exceeds ``benchmark_sharpe``.

    Inputs in **per-period** returns; the annualised Sharpe is computed inside.
    """
    if len(returns) < 3:
        return float("nan")
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr_per_period = mu / sd
    sr_annualised = sr_per_period * math.sqrt(periods_per_year)

    g3 = float(returns.skew())
    g4 = float(returns.kurtosis())  # pandas returns excess kurtosis already
    n = len(returns)

    # SR* in per-period units
    sr_star_per_period = benchmark_sharpe / math.sqrt(periods_per_year)

    denom = math.sqrt(
        max(1e-12, 1.0 - g3 * sr_per_period + ((g4) / 4.0) * sr_per_period * sr_per_period)
    )
    numer = (sr_per_period - sr_star_per_period) * math.sqrt(n - 1)
    z = numer / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` standard-normal draws.

    Returned in **standardised** units (multiply by σ_SR — the cross-sectional
    standard deviation of the trials' Sharpe ratios — to get the expected max
    in the same scale as your observed Sharpe).

    Approximation (Bailey & López de Prado 2014):

        E[max] ≈ (1-γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e))

    where γ = Euler-Mascheroni constant.
    """
    if n_trials <= 1:
        return 0.0
    gamma = EULER_MASCHERONI
    a = (1 - gamma) * norm.ppf(1 - 1 / n_trials)
    b = gamma * norm.ppf(1 - 1 / (n_trials * math.e))
    return float(a + b)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sigma_sr_annualised: float = 0.5,
    periods_per_year: int = 252,
) -> float:
    """Return DSR — probability the observed Sharpe survives ``n_trials`` multiple testing.

    ``sigma_sr_annualised`` is the cross-sectional standard deviation of
    *annualised* Sharpes across the N trials. If you don't have it, use a
    conservative default in [0.4, 1.0]; smaller σ_SR → smaller benchmark →
    higher DSR. Bailey & López de Prado often use ~0.5 as a baseline.

    DSR > 0.95 ⇒ strong evidence the observed Sharpe is not the maximum of N
    noisy null trials.
    """
    expected_max_annualised = expected_max_sharpe(n_trials) * sigma_sr_annualised
    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=expected_max_annualised,
        periods_per_year=periods_per_year,
    )


def min_track_record_length(
    sharpe: float,
    benchmark_sharpe: float,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Minimum number of (per-period) observations required to conclude the
    Sharpe is statistically distinguishable from ``benchmark_sharpe`` at
    ``confidence``. Per-period Sharpes everywhere.
    """
    if sharpe <= benchmark_sharpe:
        return float("inf")
    z = norm.ppf(confidence)
    var_term = 1.0 - skewness * sharpe + ((excess_kurtosis) / 4.0) * sharpe * sharpe
    return 1.0 + var_term * (z / (sharpe - benchmark_sharpe)) ** 2
