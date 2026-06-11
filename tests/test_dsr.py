"""Tests for PSR/DSR: formula correctness (kurtosis term) and the
discriminating power that was lost when DSR collapsed to 0 for everything."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from trading_bot.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def _make_returns(mu: float, sigma: float = 0.01, n: int = 2520,
                  seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


# ── PSR formula (termine di curtosi B&LdP) ──────────────────────────────────

def test_psr_matches_bldp_formula_by_hand():
    """Il denominatore deve usare (γ4_raw − 1)/4 = (excess + 2)/4."""
    r = _make_returns(0.0006)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = float(r.skew()), float(r.kurtosis())
    n = len(r)
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4 + 2.0) / 4.0) * sr * sr))
    expected = float(norm.cdf(sr * math.sqrt(n - 1) / denom))
    got = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_psr_normal_returns_above_half_when_positive_sharpe():
    r = _make_returns(0.0008)
    assert probabilistic_sharpe_ratio(r) > 0.95


def test_psr_zero_mean_near_half():
    r = _make_returns(0.0)
    r = r - r.mean()  # media campionaria esattamente zero → PSR = 0.5
    assert probabilistic_sharpe_ratio(r) == pytest.approx(0.5, abs=1e-6)


# ── E[max] ──────────────────────────────────────────────────────────────────

def test_expected_max_sharpe_monotonic_in_trials():
    vals = [expected_max_sharpe(n) for n in (1, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert vals == sorted(vals)


# ── DSR: deve distinguere segnale da rumore ─────────────────────────────────

def test_dsr_nonzero_for_realistic_strategy():
    """Strategia con Sharpe ~1.4 su 10 anni, 50 trial, σ_SR empirico 0.3:
    il DSR deve essere alto — NON collassare a 0."""
    r = _make_returns(0.0012, 0.01)  # SR teorico ~1.9, realizzato ~1.4
    dsr = deflated_sharpe_ratio(r, n_trials=50, sigma_sr_annualised=0.3)
    assert dsr > 0.8


def test_dsr_near_zero_for_pure_noise():
    r = _make_returns(0.0, 0.01)
    dsr = deflated_sharpe_ratio(r, n_trials=50, sigma_sr_annualised=0.3)
    assert dsr < 0.5


def test_dsr_ordering_signal_beats_noise():
    sig = _make_returns(0.0008, 0.01, seed=1)
    noise = _make_returns(0.0, 0.01, seed=2)
    for n_trials in (10, 100, 800):
        d_sig = deflated_sharpe_ratio(sig, n_trials=n_trials,
                                      sigma_sr_annualised=0.3)
        d_noise = deflated_sharpe_ratio(noise, n_trials=n_trials,
                                        sigma_sr_annualised=0.3)
        assert d_sig > d_noise


def test_dsr_decreases_with_more_trials():
    r = _make_returns(0.0006, 0.01)
    d10 = deflated_sharpe_ratio(r, n_trials=10, sigma_sr_annualised=0.3)
    d1000 = deflated_sharpe_ratio(r, n_trials=1000, sigma_sr_annualised=0.3)
    assert d10 > d1000


# ── σ_SR empirico dal registry ──────────────────────────────────────────────

def test_estimate_sigma_sr_from_runs(tmp_path, monkeypatch):
    """Con run nel DB, σ_SR = std cross-sezionale degli Sharpe (clippata)."""
    import importlib

    import trading_bot.config as config_module
    import trading_bot.data.storage as storage_module
    import trading_bot.registry as registry_module

    monkeypatch.setenv("TRADEBOT_DB_PATH", str(tmp_path / "sigma.db"))
    try:
        importlib.reload(config_module)
        importlib.reload(storage_module)
        importlib.reload(registry_module)

        # Nessun run → fallback al default
        assert registry_module.estimate_sigma_sr(default=0.5) == 0.5

        # Inserisci run finti con Sharpe noti
        import json as _json
        rng = np.random.default_rng(3)
        sharpes = rng.normal(0.4, 0.25, 30)
        with storage_module.get_session() as session:
            for i, s in enumerate(sharpes):
                session.add(storage_module.Run(
                    strategy_id=1, strategy_name=f"s{i}",
                    start_date=pd.Timestamp("2020-01-01").date(),
                    end_date=pd.Timestamp("2021-01-01").date(),
                    universe_size=500, use_pit=1,
                    metrics_json=_json.dumps({"Sharpe": float(s)}),
                    equity_json="{}", returns_json="{}",
                ))
            session.commit()

        est = registry_module.estimate_sigma_sr()
        expected = float(np.std(sharpes, ddof=1))
        assert est == pytest.approx(expected, rel=1e-9)
        assert 0.1 <= est <= 1.0
    finally:
        # I reload lasciano l'engine agganciato al DB temporaneo: ripristina
        # i moduli sull'ambiente reale o i test successivi falliscono.
        monkeypatch.undo()
        importlib.reload(config_module)
        importlib.reload(storage_module)
        importlib.reload(registry_module)
