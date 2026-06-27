"""Daily signal generator.

Loads the best promoted strategy from the registry, runs it on the
most recent prices, and returns ranked buy/sell targets.

Usage:
    from trading_bot.live.signals import generate_signals
    signals = generate_signals()          # uses best promoted strategy
    signals = generate_signals("my_strat")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import pandas as pd

from trading_bot.data.ingest import load_panel
from trading_bot.data.universe import get_top_n_by_liquidity
from trading_bot.registry import list_runs, list_strategies
from trading_bot.strategies.composer import (
    ComposedConfig,
    ComposedStrategy,
    FilterSpec,
    RegimeSpec,
    SignalSpec,
)
from trading_bot.strategies.tsmom import TSMOMConfig, TSMOMStrategy
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


class NoDeployableStrategyError(RuntimeError):
    """Raised when no strategy has passed the current live/paper gate."""


@dataclass
class SignalOutput:
    strategy_name: str
    asof: date
    positions: dict[str, float]      # symbol → target weight (0..1)
    ranked_universe: pd.Series       # full ranking, all symbols
    regime_active: bool | None       # None if no regime filter
    n_universe: int


def _load_strategy(name: str | None) -> tuple[str, ComposedConfig | TSMOMConfig]:
    """Load a ComposedConfig (or TSMOMConfig) from the registry.

    If name is None, picks the best promoted strategy by OOS Sharpe.
    Promoted status is read from research_log (source of truth), not registry.
    The loader is intentionally fail-closed: it must not fall back to arbitrary
    registry strategies when there are no current promotions.
    """
    from trading_bot.data.storage import ResearchLog, get_session
    from sqlalchemy import select

    strategies = list_strategies()

    # No registry and no runs → nothing to load (respects mocked/empty state)
    if not strategies and not list_runs():
        raise RuntimeError("No strategies in registry. Run `tradebot research` first.")

    strat_by_name = {s["name"]: s for s in strategies}

    # Use research_log as source of truth for promoted status
    with get_session() as sess:
        promoted_rows = sess.execute(
            select(ResearchLog)
            .where(ResearchLog.status == "promoted")
            .order_by(ResearchLog.oos_sharpe.desc())
        ).scalars().all()

    if name:
        row = next((r for r in promoted_rows if r.hypothesis_name == name), None)
        if row is None:
            raise NoDeployableStrategyError(
                f"Strategy '{name}' is not deployable: it has not passed the current promotion gate."
            )
        match = strat_by_name.get(name)
        chosen = match if match else {"name": row.hypothesis_name, "config_json": row.config_json}
    else:
        # Auto-select best promoted
        if not promoted_rows:
            raise NoDeployableStrategyError(
                "No deployable strategy found. Signals are disabled until a strategy passes "
                "the current validation/promotion gate."
            )
        else:
            best_row = promoted_rows[0]  # already sorted by oos_sharpe desc
            match = strat_by_name.get(best_row.hypothesis_name)
            if match:
                chosen = match
            else:
                # Reconstruct from research_log config_json
                chosen = {
                    "name": best_row.hypothesis_name,
                    "config_json": best_row.config_json,
                }

    # Support both registry format (config=dict) and research_log format (config_json=str)
    if "config" in chosen and isinstance(chosen["config"], dict):
        cfg_json = chosen["config"]
    elif "config_json" in chosen:
        cfg_json = json.loads(chosen["config_json"])
    else:
        raise ValueError(f"Strategy '{chosen.get('name')}' has no config data.")

    # Detect strategy type
    if cfg_json.get("strategy_type") == "tsmom" or "symbol" in cfg_json:
        return chosen["name"], TSMOMConfig(**cfg_json)

    # Build ComposedConfig
    sigs = tuple(
        SignalSpec(
            feature=s["feature"],
            params=s.get("params", {}),
            weight=s.get("weight", 1.0),
            use_rank=s.get("use_rank", True),
            negate=s.get("negate", False),
        )
        for s in cfg_json.get("signals", [])
    )
    filts = tuple(
        FilterSpec(
            feature=f["feature"],
            params=f.get("params", {}),
            threshold=f.get("threshold", 0.0),
        )
        for f in cfg_json.get("filters", [])
    )
    regime_d = cfg_json.get("regime")
    regime = RegimeSpec(**regime_d) if regime_d else None

    config = ComposedConfig(
        name=chosen["name"],
        rationale=cfg_json.get("rationale", ""),
        signals=sigs,
        filters=filts,
        regime=regime,
        top_n=cfg_json.get("top_n", 10),
        gold_weight=cfg_json.get("gold_weight", 0.0),
        gold_mode=cfg_json.get("gold_mode", "defensive"),
    )
    return chosen["name"], config


def generate_signals(
    strategy_name: str | None = None,
    universe_size: int = 500,
    asof: pd.Timestamp | None = None,
    lookback_days: int = 1000,  # ~4y calendar: copre lookback 378gg + warmup
                                # (audit A4: 504 produceva pesi diversi dal backtest)
) -> SignalOutput:
    """Generate today's position targets from the best promoted strategy.

    Args:
        strategy_name: specific strategy name, or None for best promoted.
        universe_size: how many top-liquidity stocks to consider.
        asof: date to generate signals for (default: today).
        lookback_days: how many calendar days of history to load.

    Returns:
        SignalOutput with target weights summing to ≤ 1.0.
    """
    if asof is None:
        asof = pd.Timestamp.today().normalize()

    logger.info(f"Generating signals as of {asof.date()}")

    # Load universe
    symbols = get_top_n_by_liquidity(universe_size)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]

    # Load recent price history
    start = asof - pd.DateOffset(days=lookback_days)
    panel = load_panel(symbols, start=start.strftime("%Y-%m-%d"))
    panel = panel.dropna(how="all", axis=1)

    # Trim to asof (no future data)
    panel = panel.loc[:asof]

    if panel.empty or len(panel) < 100:
        raise RuntimeError(f"Insufficient price data loaded (got {len(panel)} rows).")

    name, config = _load_strategy(strategy_name)

    # Build strategy
    if isinstance(config, TSMOMConfig):
        strat = TSMOMStrategy(config)
    else:
        strat = ComposedStrategy(config)

    # Get weights as of last available date
    last_date = panel.index[-1]
    weights = strat.weights(panel, last_date)

    # Get full ranking for reporting
    try:
        ranked = strat.rank(panel, last_date)
    except Exception:
        ranked = pd.Series(dtype=float)

    # Check regime
    regime_active = None
    if isinstance(config, ComposedConfig) and config.regime:
        try:
            from trading_bot.features import price as pf
            feat_fn = getattr(pf, config.regime.feature, None)
            if feat_fn and config.regime.symbol in panel.columns:
                regime_panel = panel[[config.regime.symbol]]
                regime_val = feat_fn(regime_panel, last_date, **config.regime.params)
                regime_active = bool(not regime_val.empty and regime_val.iloc[0] > config.regime.threshold)
        except Exception:
            pass

    # Normalize weights to ensure they sum to ≤ 1.0
    positions = weights.to_dict() if not weights.empty else {}
    total_w = sum(abs(v) for v in positions.values())
    if total_w > 1.0:
        positions = {k: v / total_w for k, v in positions.items()}

    logger.info(
        f"Signals generated: {len(positions)} positions, "
        f"regime_active={regime_active}, asof={last_date.date()}"
    )

    return SignalOutput(
        strategy_name=name,
        asof=last_date.date(),
        positions=positions,
        ranked_universe=ranked,
        regime_active=regime_active,
        n_universe=len(panel.columns) - 1,  # exclude SPY
    )
