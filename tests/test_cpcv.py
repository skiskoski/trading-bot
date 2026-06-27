"""Tests for CPCV engine."""

import math

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestConfig
from trading_bot.strategies.momentum import MomentumConfig, MomentumStrategy
from trading_bot.validation.cpcv import CPCVConfig, _split_into_groups, run_cpcv


def make_panel(n_days: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2005-01-01", periods=n_days, freq="B")
    syms = {
        "A": 0.0008, "B": 0.0006, "C": 0.0004, "D": 0.0002,
        "E": 0.0000, "F": -0.0002, "G": -0.0004, "H": 0.0005,
        "I": 0.0003, "J": 0.0001, "SPY": 0.0003,
    }
    data = {}
    for sym, mu in syms.items():
        rets = rng.normal(mu, 0.012, n_days)
        data[sym] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def test_split_into_groups_correct_count():
    idx = pd.date_range("2005-01-01", periods=800, freq="B")
    groups = _split_into_groups(idx, k=8)
    assert len(groups) == 8
    # All dates accounted for
    total = sum(len(g) for g in groups)
    assert total == len(idx)


def test_split_into_groups_last_absorbs_remainder():
    idx = pd.date_range("2005-01-01", periods=805, freq="B")
    groups = _split_into_groups(idx, k=8)
    # Last group gets the 5 extra days
    assert len(groups[-1]) >= len(groups[0])


def test_cpcv_path_count():
    panel = make_panel(2000)
    strat = MomentumStrategy(MomentumConfig())
    cfg = CPCVConfig(k=6, n_test=2, purge_days=5)
    result = run_cpcv(strat, panel, bt_config=BacktestConfig(), cpcv_config=cfg)
    # C(6,2) = 15; early-fold combos may be skipped due to warmup, so allow -2
    max_paths = math.comb(6, 2)
    assert max_paths - 2 <= result.n_paths <= max_paths
    assert len(result.oos_sharpe_distribution) == result.n_paths


def test_cpcv_pbo_is_valid_probability():
    panel = make_panel(2000)
    strat = MomentumStrategy(MomentumConfig())
    cfg = CPCVConfig(k=6, n_test=2, purge_days=5)
    result = run_cpcv(strat, panel, cpcv_config=cfg)
    assert 0.0 <= result.pbo <= 1.0


def test_cpcv_fraction_positive_in_range():
    panel = make_panel(2000)
    strat = MomentumStrategy(MomentumConfig())
    cfg = CPCVConfig(k=6, n_test=2, purge_days=5)
    result = run_cpcv(strat, panel, cpcv_config=cfg)
    assert 0.0 <= result.fraction_positive <= 1.0


def test_cpcv_raises_on_insufficient_data():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    panel = pd.DataFrame({"A": 100.0, "SPY": 100.0}, index=idx)
    strat = MomentumStrategy(MomentumConfig())
    with pytest.raises(ValueError, match="at least"):
        run_cpcv(strat, panel, cpcv_config=CPCVConfig(k=8))


def test_cpcv_k8_produces_28_paths():
    panel = make_panel(2500)
    strat = MomentumStrategy(MomentumConfig())
    cfg = CPCVConfig(k=8, n_test=2, purge_days=5)
    result = run_cpcv(strat, panel, cpcv_config=cfg)
    # C(8,2) = 28; early-fold combos may be skipped due to strategy warmup
    max_paths = math.comb(8, 2)
    assert max_paths - 2 <= result.n_paths <= max_paths
    assert result.k == 8
    assert result.n_test == 2


def test_cpcv_default_k10_45_paths():
    """Default k=10 → C(10,2)=45 path: granularità PBO 1/45 invece di 1/15."""
    assert CPCVConfig().k == 10
    assert math.comb(10, 2) == 45


def test_cpcv_pbo_internally_consistent_and_neutral_for_honest_edge():
    """Verifica del CALCOLO del PBO (nessun bug di allineamento date/path).

    1) Coerenza interna: pbo deve essere esattamente la frazione di path
       OOS sotto la mediana IS ricomputata dalle distribuzioni ritornate.
    2) Semantica: in questa variante single-strategy IS e OOS della stessa
       strategia vengono dalla stessa distribuzione → per un edge onesto e
       stazionario il valore NEUTRO è ~0.5. Con k=6 i 15 path discretizzano
       a 9/15 = 0.60: il "PBO sempre 0.6" osservato era discretizzazione +
       neutralità, non un bug. PBO ≫ 0.5 = squilibrio IS/OOS sistematico
       (overfitting); il gate <0.4 richiede OOS sopra la mediana IS.
    """
    rng = np.random.default_rng(123)
    idx = pd.date_range("2005-01-01", periods=2500, freq="B")
    mus = np.linspace(-0.001, 0.002, 12)
    panel = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(mu, 0.008, len(idx))))
         for i, mu in enumerate(mus)},
        index=idx,
    )
    result = run_cpcv(MomentumStrategy(MomentumConfig()), panel,
                      cpcv_config=CPCVConfig(k=10, n_test=2, purge_days=5))

    med_is = float(np.median(result.is_sharpe_distribution))
    recomputed = float(np.mean(
        [1.0 if s < med_is else 0.0 for s in result.oos_sharpe_distribution]
    ))
    assert result.pbo == pytest.approx(recomputed, abs=1e-12)
    # Edge onesto e stazionario → banda neutra attorno a 0.5
    assert 0.30 <= result.pbo <= 0.65


def test_cpcv_oos_sharpe_discriminates_signal_from_noise():
    """La metrica che separa segnale da rumore nel gate è l'OOS Sharpe:
    drift persistenti catturati dal momentum → OOS alto; nessun drift →
    OOS vicino a zero."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2005-01-01", periods=2500, freq="B")
    mus = np.linspace(-0.001, 0.002, 12)
    signal_panel = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(mu, 0.008, len(idx))))
         for i, mu in enumerate(mus)},
        index=idx,
    )
    noise_panel = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
         for i in range(12)},
        index=idx,
    )
    strat = MomentumStrategy(MomentumConfig())
    cfg = CPCVConfig(k=10, n_test=2, purge_days=5)
    oos_signal = run_cpcv(strat, signal_panel, cpcv_config=cfg).mean_oos_sharpe
    oos_noise = run_cpcv(strat, noise_panel, cpcv_config=cfg).mean_oos_sharpe
    assert oos_signal > oos_noise + 1.0
    assert oos_signal > 1.0
