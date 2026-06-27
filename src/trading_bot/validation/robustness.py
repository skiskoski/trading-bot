"""Robustness validation layer.

This module sits after the normal research filters. It answers a different
question from "did this backtest look good?": did the edge survive nearby
parameters, temporal folds, and realistic resampling noise?
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.backtest.metrics import max_drawdown, sharpe
from trading_bot.strategies.composer import ComposedConfig, ComposedStrategy
from trading_bot.validation.walk_forward import WalkForwardConfig, run_walk_forward


@dataclass(frozen=True)
class RobustnessGateConfig:
    min_plateau_fraction: float = 0.50
    max_best_neighbor_gap: float = 0.35
    min_mc_p05_sharpe: float = 0.0
    max_mc_loss_probability: float = 0.35
    max_mc_p95_drawdown: float = 0.35
    min_wf_fraction_profitable: float = 0.65
    min_wf_worst_sharpe: float = -0.25


@dataclass(frozen=True)
class RobustnessConfig:
    metric: str = "Sharpe"
    output_dir: str = "data/validation_reports"
    mc_trials: int = 1000
    parameter_mc_trials: int = 100
    mc_block_length: int | None = None
    mc_seed: int = 42
    wf_folds: int = 8
    wf_test_years: float = 1.0
    gate: RobustnessGateConfig = RobustnessGateConfig()


@dataclass(frozen=True)
class ParameterRun:
    params: dict[str, Any]
    sharpe: float
    cagr: float
    max_drawdown: float
    ic_mean: float
    ic_ir: float
    periods: int
    mc_sharpe_p05: float | None = None
    mc_loss_probability: float | None = None
    mc_max_drawdown_p95: float | None = None


@dataclass(frozen=True)
class ParameterStabilityResult:
    strategy_name: str
    grid_name: str
    metric: str
    runs: list[ParameterRun]
    best_params: dict[str, Any]
    best_metric: float
    median_metric: float
    plateau_threshold: float
    plateau_fraction: float
    best_neighbor_gap: float
    passed: bool


@dataclass(frozen=True)
class MonteCarloResult:
    trials: int
    block_length: int
    seed: int
    sharpe_p05: float
    sharpe_p50: float
    sharpe_p95: float
    max_drawdown_p50: float
    max_drawdown_p95: float
    terminal_return_p05: float
    terminal_return_p50: float
    terminal_return_p95: float
    loss_probability: float
    passed: bool


@dataclass(frozen=True)
class WalkForwardRobustnessResult:
    rolling_summary: dict[str, Any]
    anchored_summary: dict[str, Any]
    rolling_folds: list[dict[str, Any]]
    anchored_folds: list[dict[str, Any]]
    worst_sharpe: float
    fraction_profitable: float
    passed: bool


@dataclass(frozen=True)
class RobustnessReport:
    strategy_name: str
    generated_at: str
    parameter_stability: ParameterStabilityResult | None
    monte_carlo: MonteCarloResult | None
    walk_forward: WalkForwardRobustnessResult | None
    hard_gates: dict[str, bool]
    decision: str
    notes: list[str]


def default_parameter_grid(config: ComposedConfig, grid_name: str = "auto") -> list[dict[str, Any]]:
    """Return a conservative local grid around the strategy's declared params.

    The grid is intentionally small enough for an MVP validation pass. It
    focuses on parameters already present in ``ComposedConfig`` instead of
    inventing a global optimiser.
    """
    if grid_name not in {"auto", "momentum_core"}:
        raise ValueError(f"Unknown grid '{grid_name}'. Available: auto | momentum_core")

    grids: list[dict[str, Any]] = [{}]

    def expand(key: str, values: Iterable[Any]) -> None:
        nonlocal grids
        grids = [{**base, key: value} for base in grids for value in values]

    if any(s.feature == "momentum" for s in config.signals):
        expand("signal.momentum.lookback", [189, 252, 315])
        expand("signal.momentum.skip", [21, 42])
    if any(f.feature == "above_sma" for f in config.filters):
        expand("filter.above_sma.window", [100, 200])
    if config.regime and config.regime.feature == "sma_distance":
        expand("regime.sma_distance.window", [200])
    expand("top_n", sorted({max(5, config.top_n // 2), config.top_n, config.top_n * 2}))
    return grids


def apply_parameter_overrides(config: ComposedConfig, overrides: dict[str, Any]) -> ComposedConfig:
    signals = list(config.signals)
    filters = list(config.filters)
    regime = config.regime
    top_n = config.top_n

    for key, value in overrides.items():
        if key == "top_n":
            top_n = int(value)
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        kind, feature, param = parts[0], parts[1], ".".join(parts[2:])
        if kind == "signal":
            signals = [
                replace(s, params={**s.params, param: value}) if s.feature == feature else s
                for s in signals
            ]
        elif kind == "filter":
            filters = [
                replace(f, params={**f.params, param: value}) if f.feature == feature else f
                for f in filters
            ]
        elif kind == "regime" and regime is not None and regime.feature == feature:
            regime = replace(regime, params={**regime.params, param: value})

    digest = hashlib.sha1(json.dumps(overrides, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    suffix = "_robust_" + digest
    return replace(
        config,
        name=(config.name + suffix)[:120],
        signals=tuple(signals),
        filters=tuple(filters),
        regime=regime,
        top_n=top_n,
    )


def run_parameter_stability(
    config: ComposedConfig,
    prices: pd.DataFrame,
    bt_config: BacktestConfig,
    grid: list[dict[str, Any]] | None = None,
    grid_name: str = "auto",
    gate: RobustnessGateConfig | None = None,
    metric: str = "Sharpe",
    mc_config: RobustnessConfig | None = None,
) -> ParameterStabilityResult:
    gate = gate or RobustnessGateConfig()
    mc_config = mc_config or RobustnessConfig(parameter_mc_trials=0)
    rows: list[ParameterRun] = []
    grid = grid or default_parameter_grid(config, grid_name=grid_name)

    for idx, overrides in enumerate(grid):
        cfg = apply_parameter_overrides(config, overrides)
        result = CrossSectionalBacktester(ComposedStrategy(cfg), bt_config).run(prices)
        m = result.metrics
        mc = None
        if mc_config.parameter_mc_trials > 0:
            per_set_cfg = replace(
                mc_config,
                mc_trials=mc_config.parameter_mc_trials,
                mc_seed=mc_config.mc_seed + idx,
            )
            mc = run_monte_carlo(result.returns, per_set_cfg)
        rows.append(
            ParameterRun(
                params=overrides,
                sharpe=float(m.get("Sharpe", 0.0)),
                cagr=float(m.get("CAGR", 0.0)),
                max_drawdown=float(m.get("MaxDrawdown", 0.0)),
                ic_mean=float(m.get("IC_mean", 0.0)),
                ic_ir=float(m.get("IC_IR", 0.0)),
                periods=int(m.get("Periods", 0)),
                mc_sharpe_p05=mc.sharpe_p05 if mc else None,
                mc_loss_probability=mc.loss_probability if mc else None,
                mc_max_drawdown_p95=mc.max_drawdown_p95 if mc else None,
            )
        )

    values = np.array([_metric_value(r, metric) for r in rows], dtype=float)
    if len(values) == 0:
        raise ValueError("Parameter stability grid produced no runs")

    best_idx = int(np.nanargmax(values))
    best = rows[best_idx]
    best_metric = float(values[best_idx])
    median_metric = float(np.nanmedian(values))
    plateau_threshold = max(0.0, best_metric - gate.max_best_neighbor_gap)
    plateau_fraction = float(np.mean(values >= plateau_threshold))
    neighbor_gap = _best_neighbor_gap(best.params, rows, values)
    passed = (
        plateau_fraction >= gate.min_plateau_fraction
        and neighbor_gap <= gate.max_best_neighbor_gap
    )
    mc_rows = [r for r in rows if r.mc_sharpe_p05 is not None]
    if mc_rows:
        passed = passed and (
            float(np.mean([r.mc_sharpe_p05 >= gate.min_mc_p05_sharpe for r in mc_rows]))
            >= gate.min_plateau_fraction
        )

    return ParameterStabilityResult(
        strategy_name=config.name,
        grid_name=grid_name,
        metric=metric,
        runs=rows,
        best_params=best.params,
        best_metric=best_metric,
        median_metric=median_metric,
        plateau_threshold=plateau_threshold,
        plateau_fraction=plateau_fraction,
        best_neighbor_gap=neighbor_gap,
        passed=passed,
    )


def run_monte_carlo(
    returns: pd.Series,
    cfg: RobustnessConfig | None = None,
) -> MonteCarloResult:
    cfg = cfg or RobustnessConfig()
    clean = returns.dropna().astype(float)
    if len(clean) < 30:
        raise ValueError("Monte Carlo requires at least 30 return observations")
    block_len = cfg.mc_block_length or max(5, int(round(len(clean) ** (1 / 3))))
    rng = np.random.default_rng(cfg.mc_seed)
    arr = clean.to_numpy()

    sharpes: list[float] = []
    dds: list[float] = []
    terminal_returns: list[float] = []
    for _ in range(cfg.mc_trials):
        sample = _moving_block_sample(arr, block_len, rng)
        s = pd.Series(sample)
        eq = (1.0 + s).cumprod()
        sharpes.append(sharpe(s))
        dds.append(abs(max_drawdown(eq)))
        terminal_returns.append(float(eq.iloc[-1] - 1.0))

    sh = np.array(sharpes, dtype=float)
    dd = np.array(dds, dtype=float)
    tr = np.array(terminal_returns, dtype=float)
    gate = cfg.gate
    passed = (
        float(np.quantile(sh, 0.05)) >= gate.min_mc_p05_sharpe
        and float(np.mean(tr < 0.0)) <= gate.max_mc_loss_probability
        and float(np.quantile(dd, 0.95)) <= gate.max_mc_p95_drawdown
    )
    return MonteCarloResult(
        trials=cfg.mc_trials,
        block_length=block_len,
        seed=cfg.mc_seed,
        sharpe_p05=float(np.quantile(sh, 0.05)),
        sharpe_p50=float(np.quantile(sh, 0.50)),
        sharpe_p95=float(np.quantile(sh, 0.95)),
        max_drawdown_p50=float(np.quantile(dd, 0.50)),
        max_drawdown_p95=float(np.quantile(dd, 0.95)),
        terminal_return_p05=float(np.quantile(tr, 0.05)),
        terminal_return_p50=float(np.quantile(tr, 0.50)),
        terminal_return_p95=float(np.quantile(tr, 0.95)),
        loss_probability=float(np.mean(tr < 0.0)),
        passed=passed,
    )


def run_walk_forward_robustness(
    strategy: ComposedStrategy,
    prices: pd.DataFrame,
    bt_config: BacktestConfig,
    cfg: RobustnessConfig | None = None,
) -> WalkForwardRobustnessResult:
    cfg = cfg or RobustnessConfig()
    rolling = run_walk_forward(
        strategy,
        prices,
        bt_config,
        WalkForwardConfig(n_folds=cfg.wf_folds, test_years=cfg.wf_test_years, anchored=False),
    )
    anchored = run_walk_forward(
        strategy,
        prices,
        bt_config,
        WalkForwardConfig(n_folds=cfg.wf_folds, test_years=cfg.wf_test_years, anchored=True),
    )
    folds = [*rolling.folds, *anchored.folds]
    sharpes = [f.sharpe for f in folds]
    worst = float(min(sharpes)) if sharpes else 0.0
    frac = float(np.mean([s > 0.0 for s in sharpes])) if sharpes else 0.0
    passed = (
        frac >= cfg.gate.min_wf_fraction_profitable
        and worst >= cfg.gate.min_wf_worst_sharpe
    )
    return WalkForwardRobustnessResult(
        rolling_summary=rolling.summary,
        anchored_summary=anchored.summary,
        rolling_folds=[_fold_to_dict(f) for f in rolling.folds],
        anchored_folds=[_fold_to_dict(f) for f in anchored.folds],
        worst_sharpe=worst,
        fraction_profitable=frac,
        passed=passed,
    )


def build_robustness_report(
    config: ComposedConfig,
    prices: pd.DataFrame,
    bt_config: BacktestConfig,
    base_returns: pd.Series,
    cfg: RobustnessConfig | None = None,
    grid_name: str = "auto",
    include_parameter_stability: bool = True,
    include_monte_carlo: bool = True,
    include_walk_forward: bool = True,
) -> RobustnessReport:
    cfg = cfg or RobustnessConfig()
    notes: list[str] = []
    ps = None
    mc = None
    wf = None

    if include_parameter_stability:
        ps = run_parameter_stability(
            config,
            prices,
            bt_config,
            grid_name=grid_name,
            gate=cfg.gate,
            metric=cfg.metric,
            mc_config=cfg,
        )
    else:
        notes.append("parameter_stability skipped")

    if include_monte_carlo:
        mc = run_monte_carlo(base_returns, cfg)
    else:
        notes.append("monte_carlo skipped")

    if include_walk_forward:
        wf = run_walk_forward_robustness(ComposedStrategy(config), prices, bt_config, cfg)
    else:
        notes.append("walk_forward skipped")

    gates = {
        "parameter_stability": bool(ps.passed) if ps else True,
        "monte_carlo": bool(mc.passed) if mc else True,
        "walk_forward": bool(wf.passed) if wf else True,
    }
    decision = "promote_to_paper_research" if all(gates.values()) else "reject_or_research_more"
    if decision != "promote_to_paper_research":
        notes.append("Do not deploy automatically; keep in research/paper review.")

    return RobustnessReport(
        strategy_name=config.name,
        generated_at=pd.Timestamp.now("UTC").isoformat(),
        parameter_stability=ps,
        monte_carlo=mc,
        walk_forward=wf,
        hard_gates=gates,
        decision=decision,
        notes=notes,
    )


def save_report(report: RobustnessReport, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}_{report.strategy_name}_robustness.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def load_reports(report_dir: str | Path = "data/validation_reports") -> list[dict[str, Any]]:
    path = Path(report_dir)
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    for p in sorted(path.glob("*_robustness.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["_path"] = str(p)
            reports.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    return reports


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _metric_value(row: ParameterRun, metric: str) -> float:
    if metric == "Sharpe":
        return row.sharpe
    if metric == "IC_mean":
        return row.ic_mean
    if metric == "IC_IR":
        return row.ic_ir
    if metric == "CAGR":
        return row.cagr
    raise ValueError(f"Unsupported metric '{metric}'")


def _best_neighbor_gap(best_params: dict[str, Any], rows: list[ParameterRun], values: np.ndarray) -> float:
    best_metric = float(np.nanmax(values))
    neighbor_values = [
        float(v)
        for row, v in zip(rows, values, strict=True)
        if row.params != best_params and _is_neighbor(best_params, row.params)
    ]
    if not neighbor_values:
        return float("inf")
    return best_metric - max(neighbor_values)


def _is_neighbor(a: dict[str, Any], b: dict[str, Any]) -> bool:
    differing = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    return 0 < len(differing) <= 1


def _moving_block_sample(arr: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    chunks: list[np.ndarray] = []
    while sum(len(c) for c in chunks) < n:
        start = int(rng.integers(0, max(1, n - block_len + 1)))
        chunks.append(arr[start : start + block_len])
    return np.concatenate(chunks)[:n]


def _fold_to_dict(f) -> dict[str, Any]:
    return {
        "fold_id": f.fold_id,
        "train_start": f.train_start.isoformat(),
        "train_end": f.train_end.isoformat(),
        "test_start": f.test_start.isoformat(),
        "test_end": f.test_end.isoformat(),
        "sharpe": float(f.sharpe),
        "max_dd": float(f.max_dd),
        "ic_mean": float(f.ic_mean) if f.ic_mean == f.ic_mean else None,
        "ic_count": int(f.ic_count),
        "n_periods": int(f.n_periods),
    }
