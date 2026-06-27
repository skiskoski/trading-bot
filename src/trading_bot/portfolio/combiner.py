"""Multi-sleeve portfolio combiner.

Combines daily returns from multiple strategy sleeves using fixed capital
weights. The gold sleeve is not evaluated for its standalone Sharpe — its
value lies in the portfolio context: low/negative correlation with equity
during drawdowns provides crisis alpha and reduces the combined MaxDD.

Reference: Baur & Lucey (2010), Moskowitz-Ooi-Pedersen (2012),
           asset-allocation-decision-2026-06.md (in docs/).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_bot.backtest.metrics import cagr, max_drawdown, sharpe, sortino


@dataclass
class SleeveSpec:
    name: str
    weight: float       # capital weight, e.g. 0.80 for core equity


@dataclass
class PortfolioConfig:
    """Describes the sleeve mix.

    Default: core equity 80% + gold TSMOM 20%.
    Weights are normalised to sum to 1 at construction time.
    """
    sleeves: list[SleeveSpec] = field(default_factory=lambda: [
        SleeveSpec("momentum_12_1",  0.80),
        SleeveSpec("tsmom_gold",     0.20),
    ])
    name: str = "core80_gold20"
    initial_capital: float = 100_000.0

    def __post_init__(self):
        total = sum(s.weight for s in self.sleeves)
        if total > 0:
            for s in self.sleeves:
                s.weight = s.weight / total


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    sleeve_returns: pd.DataFrame   # daily returns per sleeve (unweighted)
    sleeve_equity: pd.DataFrame    # equity curves per sleeve normalised to 100
    metrics: dict
    rolling_correlation: pd.Series | None   # rolling 66d corr between first two sleeves
    drawdown_series: pd.Series


def combine_sleeves(
    sleeve_returns: dict[str, pd.Series],
    config: PortfolioConfig,
) -> PortfolioResult:
    """Combine sleeve daily returns into a single portfolio equity curve.

    ``sleeve_returns``: dict mapping sleeve name → daily return Series.
    Missing dates are forward-filled then filled with 0.
    """
    specs = {s.name: s.weight for s in config.sleeves}

    # Align on common calendar
    ret_df = pd.DataFrame(sleeve_returns).sort_index()
    ret_df = ret_df.reindex(sorted(set().union(*[r.index for r in sleeve_returns.values()])))
    ret_df = ret_df.ffill().fillna(0.0)

    # Weighted portfolio return
    portfolio_rets = pd.Series(0.0, index=ret_df.index)
    for name, weight in specs.items():
        if name in ret_df.columns:
            portfolio_rets = portfolio_rets + ret_df[name] * weight

    equity = (1 + portfolio_rets).cumprod() * config.initial_capital

    # Per-sleeve normalised equity (base 100 from common start)
    sleeve_equity = (1 + ret_df).cumprod() * 100.0

    # Drawdown series
    running_max = equity.cummax()
    dd_series = equity / running_max - 1.0

    # Rolling 66-day correlation between the first two sleeves (if both present)
    rolling_corr: pd.Series | None = None
    names = [s.name for s in config.sleeves if s.name in ret_df.columns]
    if len(names) >= 2:
        rolling_corr = (
            ret_df[names[0]]
            .rolling(66, min_periods=30)
            .corr(ret_df[names[1]])
            .rename(f"corr_{names[0]}_{names[1]}_66d")
        )

    metrics = {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(portfolio_rets),
        "Sortino": sortino(portfolio_rets),
        "MaxDrawdown": max_drawdown(equity),
        "Periods": int(len(portfolio_rets)),
    }
    # Per-sleeve standalone metrics
    for name in names:
        s_ret = ret_df[name]
        s_eq = sleeve_equity[name]
        metrics[f"{name}_Sharpe"] = sharpe(s_ret)
        metrics[f"{name}_MaxDD"] = max_drawdown(s_eq)

    return PortfolioResult(
        equity=equity,
        returns=portfolio_rets,
        sleeve_returns=ret_df[names],
        sleeve_equity=sleeve_equity[names],
        metrics=metrics,
        rolling_correlation=rolling_corr,
        drawdown_series=dd_series,
    )


def crisis_alpha_table(
    sleeve_returns: dict[str, pd.Series],
    equity_sleeve: str,
    gold_sleeve: str,
    n_worst: int = 10,
) -> pd.DataFrame:
    """Return a table of the N worst equity drawdown periods and gold's return.

    Demonstrates the safe-haven / crisis-alpha role: when equity falls hardest,
    gold tends to appreciate or at least lose less.
    """
    ret_df = pd.DataFrame(sleeve_returns).sort_index().fillna(0.0)
    if equity_sleeve not in ret_df or gold_sleeve not in ret_df:
        return pd.DataFrame()

    # Monthly aggregation for readability
    monthly = ret_df.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    equity_monthly = monthly[equity_sleeve].sort_values().head(n_worst)
    rows = []
    for dt, eq_ret in equity_monthly.items():
        gold_ret = monthly.loc[dt, gold_sleeve] if dt in monthly.index else float("nan")
        rows.append({
            "Month": dt.strftime("%Y-%m"),
            f"{equity_sleeve} return": eq_ret,
            f"{gold_sleeve} return": gold_ret,
            "Gold helped?": "✅" if gold_ret > eq_ret else "⚠️",
        })
    return pd.DataFrame(rows)
