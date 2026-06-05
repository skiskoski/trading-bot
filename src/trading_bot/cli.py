from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from trading_bot.config import settings
from trading_bot.utils.logging import get_logger

app = typer.Typer(help="Trading bot CLI — universe, ingest, backtest")
console = Console()
logger = get_logger(__name__)


@app.command("init-db")
def init_db_cmd() -> None:
    """Create all SQLite tables."""
    from trading_bot.data.storage import init_db

    init_db()
    console.print(f"[green]✓[/green] Initialized database at {settings.db_path}")


@app.command("fetch-universe")
def fetch_universe_cmd() -> None:
    """Fetch S&P 500 constituents from Wikipedia and persist."""
    from trading_bot.data.storage import init_db
    from trading_bot.data.universe import fetch_sp500_constituents, persist_universe

    init_db()
    entries = fetch_sp500_constituents()
    inserted = persist_universe(entries)
    console.print(f"[green]✓[/green] Universe: {len(entries)} total, {inserted} new")


@app.command("ingest")
def ingest_cmd(
    top: int = typer.Option(100, help="Top-N by liquidity"),
    start: str = typer.Option(settings.start_date, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    add_regime: bool = typer.Option(True, help="Also ingest SPY (regime filter)"),
) -> None:
    """Download adjusted OHLCV for the top-N tickers."""
    from trading_bot.data.ingest import ingest_symbols
    from trading_bot.data.storage import init_db
    from trading_bot.data.universe import get_top_n_by_liquidity

    init_db()
    symbols = get_top_n_by_liquidity(top)
    if add_regime and "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    console.print(f"Ingesting {len(symbols)} symbols from {start}")
    inserted = ingest_symbols(symbols, start=start, end=end)
    console.print(f"[green]✓[/green] {inserted} new candles inserted")


@app.command("ingest-all")
def ingest_all_cmd(
    start: str = typer.Option(settings.start_date, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    add_regime: bool = typer.Option(True, help="Also ingest SPY"),
) -> None:
    """Bootstrap ingest: download OHLCV for every ticker in the universe table.

    Run this once after ``fetch-universe`` to populate prices for all
    constituents. Subsequent ``ingest`` calls can then use liquidity ranking.
    """
    from trading_bot.data.ingest import ingest_symbols
    from trading_bot.data.storage import Ticker, get_session, init_db

    init_db()
    with get_session() as session:
        symbols = [s[0] for s in session.query(Ticker.symbol).order_by(Ticker.symbol).all()]
    if add_regime and "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    console.print(f"Ingesting ALL {len(symbols)} symbols from {start}")
    inserted = ingest_symbols(symbols, start=start, end=end)
    console.print(f"[green]✓[/green] {inserted} new candles inserted")


@app.command("fetch-changes")
def fetch_changes_cmd() -> None:
    """Scrape the S&P 500 'Selected changes' table for point-in-time membership."""
    from trading_bot.data.pit_universe import fetch_index_changes, persist_index_changes
    from trading_bot.data.storage import init_db

    init_db()
    df = fetch_index_changes()
    n = persist_index_changes(df)
    console.print(f"[green]✓[/green] Stored {n} S&P 500 index change rows")


@app.command("ingest-historic")
def ingest_historic_cmd(
    start: str = typer.Option(settings.start_date, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
) -> None:
    """Download OHLCV for symbols that were *ever* in the S&P 500 but no longer are.

    Eliminates the worst form of survivorship bias by ensuring removed tickers
    are present in the price panel.
    """
    from trading_bot.data.ingest import ingest_symbols
    from trading_bot.data.pit_universe import ever_in_index
    from trading_bot.data.storage import Ticker, get_session, init_db

    init_db()
    with get_session() as session:
        current = {s[0] for s in session.query(Ticker.symbol).all()}
    all_ever = ever_in_index(current)
    only_historic = sorted(all_ever - current)
    console.print(
        f"Found {len(all_ever)} ever-in-index symbols, {len(only_historic)} not in current universe"
    )
    if not only_historic:
        return
    inserted = ingest_symbols(only_historic, start=start, end=end)
    console.print(f"[green]✓[/green] {inserted} historic candles inserted")


@app.command("strategies")
def strategies_cmd() -> None:
    """List all strategies in the catalog."""
    from trading_bot.strategies.catalog import CATALOG

    table = Table(title="Strategy catalog", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Signals", justify="left")
    table.add_column("Top N", justify="right")
    table.add_column("Rationale (truncated)")
    for name, cfg in CATALOG.items():
        sigs = ", ".join(
            f"{s.feature}{'(neg)' if s.negate else ''}×{s.weight}" for s in cfg.signals
        )
        table.add_row(name, sigs, str(cfg.top_n), cfg.rationale[:90] + "...")
    console.print(table)


@app.command("run")
def run_cmd(
    name: str = typer.Argument(..., help="Strategy name from catalog"),
    start: str = typer.Option("2018-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(100),
    use_pit: bool = typer.Option(True, "--pit/--no-pit"),
) -> None:
    """Run a single catalog strategy and persist the run."""
    from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
    from trading_bot.data.ingest import load_panel
    from trading_bot.data.pit_universe import build_membership_panel
    from trading_bot.data.storage import Ticker, get_session
    from trading_bot.data.universe import get_top_n_by_liquidity
    from trading_bot.registry import persist_run
    from trading_bot.strategies.catalog import CATALOG, get_strategy

    if name not in CATALOG:
        console.print(f"[red]Unknown strategy: {name}[/red]")
        raise typer.Exit(code=1)
    cfg = CATALOG[name]

    symbols = get_top_n_by_liquidity(top)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    panel = load_panel(symbols, start=start, end=end).dropna(how="all", axis=1)
    if panel.empty:
        console.print("[red]No data — run fetch-universe + ingest-all first[/red]")
        raise typer.Exit(code=1)

    membership = None
    if use_pit:
        with get_session() as session:
            current = {s[0] for s in session.query(Ticker.symbol).all()}
        membership = build_membership_panel(
            panel.index, current, symbols=list(panel.columns)
        )

    strat = get_strategy(name)
    bt = CrossSectionalBacktester(strat, BacktestConfig(membership=membership))
    result = bt.run(panel)
    _print_metrics(result.metrics)

    run_id = persist_run(
        cfg, result, start=start, end=end, universe_size=top, use_pit=use_pit
    )
    console.print(f"[green]✓[/green] Persisted run #{run_id} for '{name}'")


@app.command("run-all")
def run_all_cmd(
    start: str = typer.Option("2018-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(100),
    use_pit: bool = typer.Option(True, "--pit/--no-pit"),
) -> None:
    """Run every strategy in the catalog and persist each run."""
    from trading_bot.strategies.catalog import CATALOG

    for name in CATALOG:
        console.print(f"\n[bold cyan]━━━ {name} ━━━[/bold cyan]")
        run_cmd(name=name, start=start, end=end, top=top, use_pit=use_pit)


@app.command("gui")
def gui_cmd(
    port: int = typer.Option(8501, help="Streamlit port"),
) -> None:
    """Launch the Streamlit GUI."""
    import os
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).resolve().parent / "gui" / "app.py"
    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    console.print(f"Launching GUI at http://localhost:{port}")
    subprocess.run(cmd, check=False)


@app.command("validate")
def validate_cmd(
    strategy: str = typer.Argument("momentum"),
    start: str = typer.Option("2018-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(100),
    use_pit: bool = typer.Option(True, "--pit/--no-pit", help="Use point-in-time membership"),
    n_folds: int = typer.Option(6, help="Walk-forward folds"),
    n_trials: int = typer.Option(1, help="Number of trials for DSR (1 = no multiple testing)"),
) -> None:
    """Run the full honest-validation pipeline: PIT-filtered backtest + walk-forward + DSR."""
    from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
    from trading_bot.data.ingest import load_panel
    from trading_bot.data.pit_universe import build_membership_panel
    from trading_bot.data.storage import Ticker, get_session
    from trading_bot.data.universe import get_top_n_by_liquidity
    from trading_bot.strategies.momentum import MomentumConfig, MomentumStrategy
    from trading_bot.validation.deflated_sharpe import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
    )
    from trading_bot.validation.walk_forward import WalkForwardConfig, run_walk_forward

    if strategy != "momentum":
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        raise typer.Exit(code=1)

    symbols = get_top_n_by_liquidity(top)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    panel = load_panel(symbols, start=start, end=end).dropna(how="all", axis=1)
    if panel.empty:
        console.print("[red]No data. Run fetch-universe + ingest-all first.[/red]")
        raise typer.Exit(code=1)

    membership = None
    if use_pit:
        with get_session() as session:
            current = {s[0] for s in session.query(Ticker.symbol).all()}
        membership = build_membership_panel(panel.index, current, symbols=list(panel.columns))
        console.print(
            f"PIT membership panel: {membership.shape[0]} dates × "
            f"{membership.shape[1]} symbols ({membership.sum(axis=1).iloc[-1]} live today)"
        )

    strat = MomentumStrategy(MomentumConfig())
    bt_cfg = BacktestConfig(membership=membership)
    bt = CrossSectionalBacktester(strat, bt_cfg)
    result = bt.run(panel)
    _print_metrics(result.metrics)

    psr = probabilistic_sharpe_ratio(result.returns, benchmark_sharpe=0.0)
    dsr = deflated_sharpe_ratio(result.returns, n_trials=n_trials)
    console.print(f"\n[bold]PSR (vs SR=0)[/bold]: {psr:.4f}")
    console.print(f"[bold]DSR  (N={n_trials})[/bold]: {dsr:.4f}")

    # Walk-forward
    wf = run_walk_forward(
        strat, panel, bt_cfg, WalkForwardConfig(n_folds=n_folds, test_years=1.0)
    )
    _print_walk_forward(wf)


def _print_walk_forward(wf) -> None:
    table = Table(title=f"Walk-forward folds (n={len(wf.folds)})", show_header=True)
    table.add_column("Fold")
    table.add_column("Test window")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("IC", justify="right")
    table.add_column("N", justify="right")
    for f in wf.folds:
        table.add_row(
            str(f.fold_id),
            f"{f.test_start.date()} → {f.test_end.date()}",
            f"{f.sharpe:.3f}",
            f"{f.max_dd * 100:.2f}%",
            f"{f.ic_mean:.4f}" if not np.isnan(f.ic_mean) else "n/a",
            str(f.n_periods),
        )
    console.print(table)
    s = wf.summary
    console.print(
        f"\n[bold]Summary[/bold]: "
        f"Sharpe mean={s['sharpe_mean']:.3f} (std {s['sharpe_std']:.3f}, "
        f"min {s['sharpe_min']:.3f}, max {s['sharpe_max']:.3f}); "
        f"IC mean={s['ic_mean']:.4f}; profitable folds={s['fraction_profitable'] * 100:.0f}%"
    )


@app.command("backtest")
def backtest_cmd(
    strategy: str = typer.Argument("momentum", help="Strategy name"),
    start: str = typer.Option("2015-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(100, help="Top-N universe by liquidity"),
    output: Path | None = typer.Option(None, help="Optional path to write equity curve CSV"),
) -> None:
    """Run a backtest of the given strategy on the stored data."""
    from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
    from trading_bot.data.ingest import load_panel
    from trading_bot.data.universe import get_top_n_by_liquidity
    from trading_bot.strategies.momentum import MomentumConfig, MomentumStrategy

    if strategy != "momentum":
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        raise typer.Exit(code=1)

    symbols = get_top_n_by_liquidity(top)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    console.print(f"Loading panel for {len(symbols)} symbols from {start}")
    panel = load_panel(symbols, start=start, end=end)
    if panel.empty:
        console.print(
            "[red]No data — run 'tradebot fetch-universe' and 'tradebot ingest' first[/red]"
        )
        raise typer.Exit(code=1)

    panel = panel.dropna(how="all", axis=1)
    console.print(f"Panel shape: {panel.shape[0]} rows × {panel.shape[1]} symbols")

    strat = MomentumStrategy(MomentumConfig())
    bt = CrossSectionalBacktester(strat, BacktestConfig())
    result = bt.run(panel)

    _print_metrics(result.metrics)

    if output is not None:
        result.equity.to_csv(output, header=["equity"])
        console.print(f"[green]✓[/green] Equity curve written to {output}")


def _print_metrics(metrics: dict) -> None:
    table = Table(title="Backtest summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        if isinstance(v, float):
            if "Drawdown" in k or "CAGR" in k or "HitRate" in k:
                table.add_row(k, f"{v * 100:.2f}%")
            else:
                table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
