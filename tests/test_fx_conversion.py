"""EUR/USD overlay: un portafoglio USD denominato in euro.

Il cambio è un fattore comune a tutto il portafoglio — qui verifichiamo solo
la matematica di conversione (direzione + invarianza dell'edge relativo)."""

import numpy as np
import pandas as pd

from trading_bot.gui.data import to_eur_curve, to_eur_returns


def _fx(start_val: float, end_val: float, n: int = 500) -> pd.Series:
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(np.linspace(start_val, end_val, n), index=idx)


def test_usd_strengthening_lifts_eur_curve():
    """Se l'USD si rafforza (EUR/USD scende), 100 USD valgono più euro."""
    fx = _fx(1.35, 1.10)                      # USD per EUR cala → USD forte
    eq = pd.Series(100.0, index=fx.index)     # NAV USD costante
    eur = to_eur_curve(eq, fx)
    assert eur.iloc[0] == 100.0               # rinormalizzata all'inizio
    assert eur.iloc[-1] > eur.iloc[0]         # in EUR vali di più
    # rapporto atteso ≈ fx_0/fx_T
    assert abs(eur.iloc[-1] / 100.0 - 1.35 / 1.10) < 1e-6


def test_eur_weakening_lowers_eur_curve():
    fx = _fx(1.10, 1.35)                      # USD per EUR sale → USD debole
    eq = pd.Series(100.0, index=fx.index)
    eur = to_eur_curve(eq, fx)
    assert eur.iloc[-1] < eur.iloc[0]


def test_flat_fx_is_identity():
    fx = _fx(1.2, 1.2)
    eq = pd.Series(np.linspace(100, 180, len(fx)), index=fx.index)
    eur = to_eur_curve(eq, fx)
    assert np.allclose(eur.values, eq.values)


def test_returns_conversion_matches_curve():
    """to_eur_returns ricostruisce la stessa curva di to_eur_curve."""
    fx = _fx(1.30, 1.05)
    rng = np.random.default_rng(0)
    r_usd = pd.Series(rng.normal(0.0004, 0.01, len(fx)), index=fx.index)
    r_usd.iloc[0] = 0.0
    eq_usd = 100 * (1 + r_usd).cumprod()

    r_eur = to_eur_returns(r_usd, fx)
    eq_from_rets = 100 * (1 + r_eur).cumprod()
    eq_from_curve = to_eur_curve(eq_usd, fx)
    assert np.allclose(eq_from_rets.values, eq_from_curve.values, rtol=1e-6)


def test_empty_fx_is_passthrough():
    eq = pd.Series([100.0, 110.0, 120.0])
    assert to_eur_curve(eq, pd.Series(dtype=float)).equals(eq)
