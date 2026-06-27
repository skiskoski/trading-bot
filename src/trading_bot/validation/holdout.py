"""Holdout out-of-sample validation.

⚠ AUDIT FINDING A3 (2026-06-10): the research loop trains with end=None
(data through today), so the 2024→today window has NEVER been virgin and
this module's "holdout" is in-sample by construction. DECISION: the true
holdout is PAPER TRADING — future data is unseen by construction. This
module is kept for relative comparisons only; do NOT treat its output as
an unbiased OOS estimate.

De Prado principle: once you look at the holdout, the test is consumed.
Call this ONCE per strategy, only when ready to go live.

Usage:
    from trading_bot.validation.holdout import run_holdout_test
    result = run_holdout_test("my_strategy")
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.backtest.metrics import sharpe, max_drawdown, cagr

sharpe_ratio = sharpe
from trading_bot.data.ingest import load_panel
from trading_bot.data.pit_universe import build_membership_panel
from trading_bot.data.storage import get_session
from trading_bot.data.universe import get_top_n_by_liquidity
from trading_bot.live.signals import _load_strategy
from trading_bot.registry import list_strategies
from trading_bot.strategies.composer import ComposedStrategy
from trading_bot.strategies.tsmom import TSMOMStrategy
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class HoldoutResult:
    strategy_name: str
    holdout_start: str
    holdout_end: str
    sharpe: float
    cagr: float
    max_drawdown: float
    n_months: int
    equity_curve: pd.Series
    monthly_returns: pd.Series
    is_sharpe: float      # in-sample Sharpe for comparison
    passed: bool          # holdout Sharpe > 0 and > is_sharpe * 0.5

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "⚠️  WARN"
        return (
            f"{status} | Holdout Sharpe={self.sharpe:.3f}  "
            f"CAGR={self.cagr*100:.1f}%  MaxDD={self.max_drawdown*100:.1f}%  "
            f"(IS Sharpe={self.is_sharpe:.3f})"
        )


def run_holdout_test(
    strategy_name: str | None = None,
    holdout_start: str = "2024-01-01",
    holdout_end: str | None = None,
    universe_size: int = 100,
    use_pit: bool = True,
) -> HoldoutResult:
    """Run the final holdout test on a promoted strategy.

    This uses data from holdout_start → today that was never seen during
    training. The holdout is short (~18 months) so results are indicative,
    not definitive — the main validation remains CPCV on the full history.
    """
    if holdout_end is None:
        holdout_end = pd.Timestamp.today().strftime("%Y-%m-%d")

    console.print(Panel(
        f"[bold cyan]Holdout OOS Test[/bold cyan]\n"
        f"Period: {holdout_start} → {holdout_end}\n"
        f"Strategy: {strategy_name or 'best promoted'}\n"
        f"[yellow]⚠ This test consumes the holdout. Run only once per strategy.[/yellow]",
        title="Holdout",
    ))

    # Load universe
    symbols = get_top_n_by_liquidity(universe_size)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]

    # Load holdout data + 2 years of warm-up before it
    warmup_start = (pd.Timestamp(holdout_start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    panel = load_panel(symbols, start=warmup_start, end=holdout_end)
    panel = panel.dropna(how="all", axis=1)

    if panel.empty:
        raise RuntimeError("No price data for holdout period.")

    # Resolve strategy
    name, config = _load_strategy(strategy_name)

    # Get IS Sharpe from registry for comparison
    strategies = list_strategies()
    is_sharpe = 0.0
    for s in strategies:
        if s["name"] == name:
            # Get from latest run
            from trading_bot.registry import list_runs
            runs = [r for r in list_runs() if r.get("strategy") == name]
            if runs:
                runs.sort(key=lambda r: r.get("run_id", 0), reverse=True)
                is_sharpe = runs[0].get("sharpe", 0.0) or 0.0
            break

    # Build membership for PIT universe
    membership = None
    if use_pit:
        from trading_bot.data.storage import Ticker
        with get_session() as session:
            current_tickers = {s[0] for s in session.query(Ticker.symbol).all()}
        # PIT panel covering the holdout period
        holdout_panel = panel.loc[holdout_start:]
        membership = build_membership_panel(
            holdout_panel.index, current_tickers, symbols=list(panel.columns)
        )

    bt_cfg = BacktestConfig(membership=membership)

    # Build strategy
    if hasattr(config, "symbol"):  # TSMOMConfig
        strat = TSMOMStrategy(config)
    else:
        strat = ComposedStrategy(config)

    # Run backtest on FULL panel (warm-up + holdout)
    bt = CrossSectionalBacktester(strat, bt_cfg)
    full_result = bt.run(panel)

    # Extract only the holdout portion of returns
    eq = full_result.equity
    if eq.empty:
        raise RuntimeError("Backtest returned empty equity curve.")

    holdout_eq = eq.loc[holdout_start:]
    if holdout_eq.empty or len(holdout_eq) < 20:
        raise RuntimeError(
            f"Too few holdout data points ({len(holdout_eq)}). "
            f"Check that {holdout_start} is within the data range."
        )

    holdout_returns = holdout_eq.pct_change().dropna()
    monthly_returns = holdout_returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)

    h_sharpe = sharpe_ratio(holdout_returns)
    h_cagr   = cagr(holdout_eq)
    h_mdd    = max_drawdown(holdout_eq)
    n_months = len(monthly_returns)

    passed = h_sharpe > 0.0 and h_sharpe > is_sharpe * 0.5

    result = HoldoutResult(
        strategy_name=name,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        sharpe=h_sharpe,
        cagr=h_cagr,
        max_drawdown=h_mdd,
        n_months=n_months,
        equity_curve=holdout_eq,
        monthly_returns=monthly_returns,
        is_sharpe=is_sharpe,
        passed=passed,
    )

    _print_holdout_result(result)
    return result


def _print_holdout_result(r: HoldoutResult) -> None:
    color = "green" if r.passed else "yellow"
    table = Table(title=f"Holdout OOS — {r.strategy_name}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Holdout", justify="right")
    table.add_column("In-Sample (ref)", justify="right", style="dim")

    table.add_row("Sharpe Ratio",  f"[{color}]{r.sharpe:.3f}[/{color}]",  f"{r.is_sharpe:.3f}")
    table.add_row("CAGR",          f"{r.cagr*100:.1f}%",  "—")
    table.add_row("Max Drawdown",  f"{r.max_drawdown*100:.1f}%", "—")
    table.add_row("Months",        str(r.n_months),        "—")
    table.add_row(
        "Gate (Sharpe > IS×0.5)",
        f"[{color}]{'PASS ✓' if r.passed else 'WARN ⚠'}[/{color}]",
        f"IS×0.5 = {r.is_sharpe*0.5:.3f}",
    )

    console.print(table)

    if not r.passed:
        console.print(
            "[yellow]Warning: holdout Sharpe is below 50% of in-sample Sharpe. "
            "This may indicate regime change or mild overfitting. "
            "Review before going live.[/yellow]"
        )
    else:
        console.print(
            f"[green]✅ Strategy holds up in unseen data. "
            f"Ready for paper trading.[/green]"
        )
