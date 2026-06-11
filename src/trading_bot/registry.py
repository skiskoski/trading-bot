"""Strategy registry and honest trial counter.

Every strategy run goes through ``persist_run()`` which:
  - registers the Strategy (or fetches its existing id) using its name
  - persists the equity curve and metrics
  - increments the global trial counter

The Deflated Sharpe in the GUI/CLI uses the global ``trial_count()`` so the
multiple-testing penalty reflects the *real* number of experiments performed
on this database, not just the ones that produced winners.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select

from trading_bot.backtest.engine import BacktestResult
from trading_bot.data.storage import Run, Strategy, TrialCounter, get_session, init_db
from trading_bot.strategies.composer import ComposedConfig
from trading_bot.validation.cpcv import CPCVResult
from trading_bot.utils.logging import get_logger
from trading_bot.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

logger = get_logger(__name__)

# Make sure the registry tables exist whenever this module is imported.
# Cheap idempotent call — SQLAlchemy only creates missing tables.
init_db()


def _config_to_json(config: ComposedConfig) -> str:
    payload = {
        "name": config.name,
        "rationale": config.rationale,
        "top_n": config.top_n,
        "gold_weight": getattr(config, "gold_weight", 0.0),
        "gold_mode": getattr(config, "gold_mode", "defensive"),
        "signals": [
            {
                "feature": s.feature,
                "params": s.params,
                "weight": s.weight,
                "use_rank": s.use_rank,
                "negate": s.negate,
            }
            for s in config.signals
        ],
        "filters": [
            {"feature": f.feature, "params": f.params, "threshold": f.threshold}
            for f in config.filters
        ],
        "regime": (
            {
                "symbol": config.regime.symbol,
                "feature": config.regime.feature,
                "params": config.regime.params,
                "threshold": config.regime.threshold,
            }
            if config.regime
            else None
        ),
    }
    return json.dumps(payload, default=str)


def register_strategy(config: ComposedConfig) -> int:
    if not config.rationale or len(config.rationale.strip()) < 20:
        raise ValueError(
            f"Strategy '{config.name}' rejected: rationale is required and must be substantive "
            "(>=20 chars). Pre-registration enforces honest hypothesis statement."
        )
    with get_session() as session:
        existing = session.execute(
            select(Strategy).where(Strategy.name == config.name)
        ).scalar_one_or_none()
        if existing is not None:
            return existing.id
        s = Strategy(
            name=config.name,
            rationale=config.rationale,
            config_json=_config_to_json(config),
            status="untested",
        )
        session.add(s)
        session.commit()
        return s.id


def increment_trial_count(n: int = 1) -> int:
    with get_session() as session:
        row = session.execute(select(TrialCounter)).scalar_one_or_none()
        if row is None:
            row = TrialCounter(total_trials=n)
            session.add(row)
        else:
            row.total_trials += n
        session.commit()
        return row.total_trials


def trial_count() -> int:
    with get_session() as session:
        row = session.execute(select(TrialCounter)).scalar_one_or_none()
        return int(row.total_trials) if row else 0


def estimate_sigma_sr(min_runs: int = 10, default: float = 0.5) -> float:
    """σ_SR empirico: deviazione standard cross-sezionale degli Sharpe
    annualizzati di tutti i run persistiti.

    Bailey & López de Prado definiscono il benchmark deflazionato come
    E[max] · σ_SR dove σ_SR è la dispersione REALE dei trial — non una
    costante. Con σ fisso a 0.5 e N nell'ordine delle centinaia il
    benchmark esplode (Sharpe ≈ 1.5+) e il DSR collassa a 0 per qualsiasi
    strategia: era il bug "DSR sempre 0". Clippato in [0.1, 1.0] contro
    campioni degeneri.
    """
    with get_session() as session:
        rows = session.execute(select(Run.metrics_json)).scalars().all()
    sharpes: list[float] = []
    for mj in rows:
        try:
            s = json.loads(mj).get("Sharpe")
            if s is not None and s == s:
                sharpes.append(float(s))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if len(sharpes) < min_runs:
        return default
    sd = float(np.std(sharpes, ddof=1))
    return float(min(max(sd, 0.1), 1.0))


def persist_run(
    config: ComposedConfig,
    result: BacktestResult,
    start: str,
    end: str | None,
    universe_size: int,
    use_pit: bool,
    sigma_sr_annualised: float | None = None,
    cpcv_result: CPCVResult | None = None,
) -> int:
    strategy_id = register_strategy(config)
    n = increment_trial_count(1)
    sigma_sr = (sigma_sr_annualised if sigma_sr_annualised is not None
                else estimate_sigma_sr())
    psr = probabilistic_sharpe_ratio(result.returns, benchmark_sharpe=0.0)
    dsr = deflated_sharpe_ratio(
        result.returns, n_trials=n, sigma_sr_annualised=sigma_sr
    )

    end_date = (
        pd.Timestamp(end).date()
        if end
        else result.equity.index[-1].date()
        if not result.equity.empty
        else date.today()
    )
    start_date = pd.Timestamp(start).date()
    eq = result.equity
    rt = result.returns
    eq_json = json.dumps(
        {
            "index": [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in eq.index],
            "values": [float(v) for v in eq.values],
        }
    )
    rt_json = json.dumps(
        {
            "index": [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in rt.index],
            "values": [float(v) for v in rt.values],
        }
    )
    # Promotion gate: DSR ≥ 0.95 AND (if CPCV available) PBO < 0.5
    pbo_ok = (cpcv_result is None) or (cpcv_result.pbo < 0.5)
    status = "promoted" if (dsr >= 0.95 and result.metrics.get("Sharpe", 0) > 0.5 and pbo_ok) else "tested"

    cpcv_json_str: str | None = None
    if cpcv_result is not None:
        cpcv_json_str = json.dumps({
            "k": cpcv_result.k,
            "n_test": cpcv_result.n_test,
            "n_paths": cpcv_result.n_paths,
            "pbo": cpcv_result.pbo,
            "mean_oos_sharpe": cpcv_result.mean_oos_sharpe,
            "std_oos_sharpe": cpcv_result.std_oos_sharpe,
            "median_oos_sharpe": cpcv_result.median_oos_sharpe,
            "fraction_positive": cpcv_result.fraction_positive,
            "oos_sharpe_distribution": cpcv_result.oos_sharpe_distribution,
            "is_sharpe_distribution": cpcv_result.is_sharpe_distribution,
        })

    with get_session() as session:
        run = Run(
            strategy_id=strategy_id,
            strategy_name=config.name,
            start_date=start_date,
            end_date=end_date,
            universe_size=universe_size,
            use_pit=1 if use_pit else 0,
            metrics_json=json.dumps({k: _json_safe(v) for k, v in result.metrics.items()}),
            equity_json=eq_json,
            returns_json=rt_json,
            psr=float(psr) if psr == psr else 0.0,
            dsr=float(dsr) if dsr == dsr else 0.0,
            n_trials_used=n,
            cpcv_json=cpcv_json_str,
        )
        session.add(run)
        # Update strategy status
        strat = session.get(Strategy, strategy_id)
        if strat is not None:
            strat.status = status
        session.commit()
        run_id = run.id

    logger.info(
        f"Persisted run for {config.name}: Sharpe={result.metrics.get('Sharpe', 0):.3f}, "
        f"PSR={psr:.3f}, DSR={dsr:.3f} (σ_SR={sigma_sr:.3f}), status={status}, N_trials={n}"
    )
    return run_id


def _json_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def list_runs() -> list[dict]:
    with get_session() as session:
        rows = session.execute(select(Run).order_by(Run.created_at.desc())).scalars().all()
        return [
            {
                "id": r.id,
                "strategy_name": r.strategy_name,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "universe_size": r.universe_size,
                "use_pit": bool(r.use_pit),
                "metrics": json.loads(r.metrics_json),
                "psr": r.psr,
                "dsr": r.dsr,
                "n_trials_used": r.n_trials_used,
                "cpcv": json.loads(r.cpcv_json) if r.cpcv_json else None,
                "created_at": r.created_at,
            }
            for r in rows
        ]


def get_run(run_id: int) -> dict:
    with get_session() as session:
        r = session.get(Run, run_id)
        if r is None:
            raise KeyError(f"Run {run_id} not found")
        return {
            "id": r.id,
            "strategy_name": r.strategy_name,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "metrics": json.loads(r.metrics_json),
            "equity": _deserialise_series(r.equity_json),
            "returns": _deserialise_series(r.returns_json),
            "psr": r.psr,
            "dsr": r.dsr,
            "n_trials_used": r.n_trials_used,
            "cpcv": json.loads(r.cpcv_json) if r.cpcv_json else None,
        }


def _deserialise_series(payload: str) -> pd.Series:
    obj = json.loads(payload)
    idx = pd.to_datetime(obj["index"])
    return pd.Series(obj["values"], index=idx)


def latest_run_per_strategy() -> list[dict]:
    """Return the most recent run for each strategy."""
    runs = list_runs()
    seen: set[str] = set()
    out: list[dict] = []
    for r in runs:
        if r["strategy_name"] in seen:
            continue
        seen.add(r["strategy_name"])
        # Hydrate equity/returns
        full = get_run(r["id"])
        r["equity"] = full["equity"]
        r["returns"] = full["returns"]
        out.append(r)
    return out


def list_strategies() -> list[dict]:
    with get_session() as session:
        rows = session.execute(select(Strategy).order_by(Strategy.created_at.desc())).scalars().all()
        return [
            {
                "id": s.id,
                "name": s.name,
                "rationale": s.rationale,
                "config": json.loads(s.config_json),
                "status": s.status,
                "created_at": s.created_at,
            }
            for s in rows
        ]
