import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from trading_bot.cli import app
from trading_bot.risk.market_storm import (
    MarketStormConfig,
    StormReport,
    apply_storm_overlay,
    build_storm_report,
    compute_storm_history,
    save_storm_report,
    score_to_regime,
)


def _panel(vol: float = 0.006, crash: bool = False, n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    common = rng.normal(0.0002, vol, n)
    if crash:
        common[-80:] += rng.normal(-0.002, vol * 2.0, 80)
        common[-30:] -= 0.006
    data = {"SPY": 100 * np.exp(np.cumsum(common))}
    for i in range(12):
        noise = rng.normal(0, vol * (0.4 if crash else 1.0), n)
        data[f"S{i}"] = 50 * np.exp(np.cumsum(common * (0.8 if crash else 0.2) + noise))
    data["HYG"] = 80 * np.exp(np.cumsum(common * 0.35 + rng.normal(0, vol * 0.5, n)))
    return pd.DataFrame(data, index=idx)


def test_score_to_regime_boundaries():
    assert score_to_regime(0) == "calm"
    assert score_to_regime(25) == "unstable"
    assert score_to_regime(50) == "storm"
    assert score_to_regime(75) == "panic"


def test_storm_score_rises_in_stressed_synthetic_panel():
    calm = compute_storm_history(_panel(crash=False), MarketStormConfig(max_corr_assets=20))
    stressed = compute_storm_history(_panel(crash=True), MarketStormConfig(max_corr_assets=20))

    assert stressed["storm_score"].iloc[-1] > calm["storm_score"].iloc[-1]
    assert stressed["spy_drawdown"].iloc[-1] > calm["spy_drawdown"].iloc[-1]


def test_storm_overlay_uses_prior_day_scale_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = pd.Series([0.10, 0.10, 0.10, 0.10], index=idx)
    history = pd.DataFrame({"exposure_scale": [1.0, 0.5, 0.25, 0.25]}, index=idx)

    over = apply_storm_overlay(returns, history, lag_days=1)

    assert over.iloc[0] == 0.10
    assert over.iloc[1] == 0.10
    assert over.iloc[2] == 0.05
    assert over.iloc[3] == 0.025


def test_storm_report_json_serializable(tmp_path):
    panel = _panel(crash=True)
    strategy_returns = panel["S0"].pct_change().fillna(0.0)
    report, history = build_storm_report(
        "demo",
        panel,
        strategy_returns=strategy_returns,
        config=MarketStormConfig(max_corr_assets=20),
    )
    path = save_storm_report(report, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(report, StormReport)
    assert not history.empty
    assert data["strategy_name"] == "demo"
    assert "storm_score" in data["current"]
    assert data["overlay"] is not None


def test_market_storm_cli_help_imports():
    runner = CliRunner()
    result = runner.invoke(app, ["market-storm", "--help"])

    assert result.exit_code == 0
    assert "Market Storm" in result.output
