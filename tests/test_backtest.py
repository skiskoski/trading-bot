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
