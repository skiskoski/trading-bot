from pathlib import Path

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
