import json

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.catalog import CATALOG
from trading_bot.validation.robustness import (
    MonteCarloResult,
    ParameterRun,
    ParameterStabilityResult,
    RobustnessReport,
    apply_parameter_overrides,
    default_parameter_grid,
    run_monte_carlo,
    save_report,
)


def test_default_grid_contains_local_momentum_parameters():
    cfg = CATALOG["momentum_12_1"]
    grid = default_parameter_grid(cfg)

    assert grid
    assert any(row.get("signal.momentum.lookback") == 252 for row in grid)
    assert any(row.get("signal.momentum.skip") == 21 for row in grid)
    assert any(row.get("filter.above_sma.window") == 100 for row in grid)
    assert any(row.get("regime.sma_distance.window") == 200 for row in grid)
    assert any(row.get("top_n") == cfg.top_n for row in grid)


def test_apply_parameter_overrides_keeps_original_config_immutable():
    cfg = CATALOG["momentum_12_1"]
    changed = apply_parameter_overrides(
        cfg,
        {
            "signal.momentum.lookback": 189,
            "signal.momentum.skip": 42,
            "filter.above_sma.window": 150,
            "regime.sma_distance.window": 250,
            "top_n": 20,
        },
    )

    assert cfg.signals[0].params["lookback"] == 252
    assert cfg.signals[0].params["skip"] == 21
    assert cfg.filters[0].params["window"] == 100
    assert cfg.regime.params["window"] == 200
    assert cfg.top_n == 10

    assert changed.signals[0].params["lookback"] == 189
    assert changed.signals[0].params["skip"] == 42
    assert changed.filters[0].params["window"] == 150
    assert changed.regime.params["window"] == 250
    assert changed.top_n == 20


def test_monte_carlo_is_deterministic_with_seed():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    returns = pd.Series(np.full(len(idx), 0.001), index=idx)

    a = run_monte_carlo(returns, cfg=None)
    b = run_monte_carlo(returns, cfg=None)

    assert a == b
    assert a.trials == 1000
    assert a.block_length >= 5
    assert a.loss_probability == pytest.approx(0.0)


def test_monte_carlo_rejects_tiny_samples():
    with pytest.raises(ValueError, match="at least 30"):
        run_monte_carlo(pd.Series([0.01] * 10))


def test_save_report_writes_json(tmp_path):
    ps = ParameterStabilityResult(
        strategy_name="demo",
        grid_name="unit",
        metric="Sharpe",
        runs=[
            ParameterRun(
                params={"top_n": 10},
                sharpe=0.5,
                cagr=0.1,
                max_drawdown=-0.2,
                ic_mean=0.03,
                ic_ir=0.7,
                periods=252,
            )
        ],
        best_params={"top_n": 10},
        best_metric=0.5,
        median_metric=0.5,
        plateau_threshold=0.15,
        plateau_fraction=1.0,
        best_neighbor_gap=0.0,
        passed=True,
    )
    mc = MonteCarloResult(
        trials=10,
        block_length=5,
        seed=1,
        sharpe_p05=0.1,
        sharpe_p50=0.5,
        sharpe_p95=1.0,
        max_drawdown_p50=0.1,
        max_drawdown_p95=0.2,
        terminal_return_p05=0.0,
        terminal_return_p50=0.1,
        terminal_return_p95=0.2,
        loss_probability=0.1,
        passed=True,
    )
    report = RobustnessReport(
        strategy_name="demo",
        generated_at="2026-06-18T00:00:00Z",
        parameter_stability=ps,
        monte_carlo=mc,
        walk_forward=None,
        hard_gates={"parameter_stability": True, "monte_carlo": True},
        decision="promote_to_next_validation",
        notes=[],
    )

    path = save_report(report, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.endswith("_demo_robustness.json")
    assert data["strategy_name"] == "demo"
    assert data["parameter_stability"]["best_params"] == {"top_n": 10}
    assert data["hard_gates"]["monte_carlo"] is True
