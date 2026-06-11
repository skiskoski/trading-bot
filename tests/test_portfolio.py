"""Tests for the portfolio combiner and crisis alpha analysis."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.portfolio.combiner import (
    PortfolioConfig,
    SleeveSpec,
    combine_sleeves,
    crisis_alpha_table,
)


def make_returns(n: int = 500, mu: float = 0.0003, sigma: float = 0.010,
                 seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


# ── PortfolioConfig ──────────────────────────────────────────────────────────

def test_weights_normalised():
    cfg = PortfolioConfig(sleeves=[SleeveSpec("a", 3.0), SleeveSpec("b", 1.0)])
    weights = {s.name: s.weight for s in cfg.sleeves}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_single_sleeve_identity():
    """Combined with 100% one sleeve = same returns as that sleeve."""
    rets = make_returns()
    cfg = PortfolioConfig(sleeves=[SleeveSpec("eq", 1.0)])
    result = combine_sleeves({"eq": rets}, cfg)
    pd.testing.assert_series_equal(
        result.returns.reindex(rets.index).fillna(0),
        rets.fillna(0),
        check_names=False,
        atol=1e-10,
    )


# ── combine_sleeves ──────────────────────────────────────────────────────────

def test_combined_sharpe_improves_with_negcorr_asset():
    """Adding a negatively-correlated asset should improve or maintain Sharpe."""
    from trading_bot.backtest.metrics import sharpe
    rng = np.random.default_rng(42)
    idx = pd.date_range("2005-01-01", periods=2000, freq="B")
    eq_rets = pd.Series(rng.normal(0.0004, 0.012, 2000), index=idx)
    # Gold: slight positive drift, negatively correlated with equity
    gold_rets = pd.Series(-0.3 * eq_rets.values + rng.normal(0.0002, 0.006, 2000), index=idx)

    cfg = PortfolioConfig(sleeves=[SleeveSpec("eq", 0.80), SleeveSpec("gold", 0.20)])
    result = combine_sleeves({"eq": eq_rets, "gold": gold_rets}, cfg)
    # Combined Sharpe should be >= standalone equity Sharpe due to diversification
    assert result.metrics["Sharpe"] >= sharpe(eq_rets) - 0.05  # allow tiny tolerance


def test_combined_maxdd_lower_with_safe_haven():
    """A safe-haven asset (rises in crashes) reduces MaxDD of the combined portfolio."""
    from trading_bot.backtest.metrics import max_drawdown
    idx = pd.date_range("2005-01-01", periods=2000, freq="B")
    rng = np.random.default_rng(7)
    eq_rets = pd.Series(rng.normal(0.0003, 0.012, 2000), index=idx)
    # Safe haven: exactly negatively correlated
    safe_rets = pd.Series(-eq_rets.values * 0.5, index=idx)

    from trading_bot.backtest.engine import BacktestConfig
    cfg = PortfolioConfig(sleeves=[SleeveSpec("eq", 0.80), SleeveSpec("safe", 0.20)])
    result = combine_sleeves({"eq": eq_rets, "safe": safe_rets}, cfg)

    eq_equity = (1 + eq_rets).cumprod() * 100_000
    assert result.metrics["MaxDrawdown"] > max_drawdown(eq_equity)  # less negative


def test_rolling_correlation_shape():
    eq = make_returns(500, seed=1)
    gold = make_returns(500, seed=2)
    cfg = PortfolioConfig(sleeves=[SleeveSpec("eq", 0.80), SleeveSpec("gold", 0.20)])
    result = combine_sleeves({"eq": eq, "gold": gold}, cfg)
    assert result.rolling_correlation is not None
    assert len(result.rolling_correlation) == len(eq)


def test_equity_curve_starts_at_initial_capital():
    eq = make_returns(300)
    cfg = PortfolioConfig(sleeves=[SleeveSpec("eq", 1.0)], initial_capital=50_000)
    result = combine_sleeves({"eq": eq}, cfg)
    assert abs(result.equity.iloc[0] - 50_000 * (1 + eq.iloc[0])) < 1.0


# ── crisis_alpha_table ───────────────────────────────────────────────────────

def test_crisis_alpha_table_returns_n_rows():
    eq = make_returns(1000, seed=0)
    gold = make_returns(1000, seed=5)
    df = crisis_alpha_table({"eq": eq, "gold": gold},
                            equity_sleeve="eq", gold_sleeve="gold", n_worst=10)
    assert len(df) == 10


def test_crisis_alpha_table_columns():
    eq = make_returns(500, seed=0)
    gold = make_returns(500, seed=3)
    df = crisis_alpha_table({"eq": eq, "gold": gold},
                            equity_sleeve="eq", gold_sleeve="gold")
    assert "Month" in df.columns
    assert "Gold helped?" in df.columns


def test_crisis_alpha_table_missing_sleeve_returns_empty():
    eq = make_returns(300, seed=0)
    df = crisis_alpha_table({"eq": eq}, equity_sleeve="eq", gold_sleeve="gold")
    assert df.empty
