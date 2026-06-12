import numpy as np
import pandas as pd

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.backtest.metrics import max_drawdown, sharpe
from trading_bot.strategies.momentum import MomentumConfig, MomentumStrategy


def make_panel(n_days: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    drifts = {
        "A": 0.0008,
        "B": 0.0006,
        "C": 0.0004,
        "D": 0.0002,
        "E": 0.0000,
        "F": -0.0002,
        "G": -0.0004,
        "SPY": 0.0003,
    }
    data = {}
    for sym, mu in drifts.items():
        rets = rng.normal(mu, 0.012, n_days)
        data[sym] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def test_backtest_produces_equity_curve():
    panel = make_panel()
    strat = MomentumStrategy(MomentumConfig(top_n=2))
    bt = CrossSectionalBacktester(strat, BacktestConfig(initial_capital=10_000.0))
    result = bt.run(panel)

    assert not result.equity.empty
    assert result.equity.iloc[0] == 10_000.0
    assert (result.equity > 0).all()
    assert "Sharpe" in result.metrics
    assert isinstance(result.metrics["MaxDrawdown"], float)


def test_no_lookahead_through_weights():
    """Weights at day t must use prices at t-1 (shifted)."""
    panel = make_panel(800)
    strat = MomentumStrategy(MomentumConfig(top_n=2))
    bt = CrossSectionalBacktester(strat)
    result = bt.run(panel)
    # First non-zero return must come AFTER the first rebalance date
    first_rebal = result.weights_history.index.min()
    first_nonzero = result.returns[result.returns != 0].index.min()
    assert first_nonzero >= first_rebal


def test_metrics_consistency():
    panel = make_panel()
    strat = MomentumStrategy(MomentumConfig(top_n=3))
    bt = CrossSectionalBacktester(strat)
    result = bt.run(panel)
    # Sharpe computed via metrics module matches metrics dict
    sr = sharpe(result.returns)
    assert result.metrics["Sharpe"] == sr
    dd = max_drawdown(result.equity)
    assert result.metrics["MaxDrawdown"] == dd


# ── Commissione fissa per ordine (IBKR) ──────────────────────────────────────

def test_fixed_cost_disabled_by_default_is_unchanged():
    """capital_base=None (default) → backtest identico a prima del modello costi."""
    panel = make_panel()
    strat = MomentumStrategy(MomentumConfig(top_n=3))
    base = CrossSectionalBacktester(strat, BacktestConfig()).run(panel)
    # stessa identica config esplicita
    again = CrossSectionalBacktester(
        strat, BacktestConfig(capital_base=None, fixed_cost_per_trade=0.35)
    ).run(panel)
    assert base.metrics["CAGR"] == again.metrics["CAGR"]


def test_fixed_cost_reduces_return_monotonic_in_capital():
    """Costo fisso abbassa il CAGR; un conto più piccolo lo abbassa di più."""
    panel = make_panel()
    strat = MomentumStrategy(MomentumConfig(top_n=3))
    none = CrossSectionalBacktester(strat, BacktestConfig()).run(panel)
    big = CrossSectionalBacktester(
        strat, BacktestConfig(capital_base=100_000, fixed_cost_per_trade=0.35)
    ).run(panel)
    small = CrossSectionalBacktester(
        strat, BacktestConfig(capital_base=1_000, fixed_cost_per_trade=0.35)
    ).run(panel)
    assert big.metrics["CAGR"] < none.metrics["CAGR"]
    assert small.metrics["CAGR"] < big.metrics["CAGR"]


def test_fixed_cost_zero_fee_is_noop():
    """fixed_cost_per_trade=0 → nessun effetto anche con capital_base impostato."""
    panel = make_panel()
    strat = MomentumStrategy(MomentumConfig(top_n=3))
    base = CrossSectionalBacktester(strat, BacktestConfig()).run(panel)
    zero = CrossSectionalBacktester(
        strat, BacktestConfig(capital_base=10_000, fixed_cost_per_trade=0.0)
    ).run(panel)
    assert base.metrics["CAGR"] == zero.metrics["CAGR"]
