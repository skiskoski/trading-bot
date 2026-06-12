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
def fetch_universe_cmd(
    index: str = typer.Option(
        "all",
        help="Which index to fetch: sp500 | sp400 | sp600 | all (S&P 1500)",
    ),
) -> None:
    """Fetch S&P 500/400/600 constituents from Wikipedia (S&P 1500 ≈ Russell 3000).

    --index all  fetches all three indices (~1500 stocks, default)
    --index sp500  fetches only S&P 500 large caps (503 stocks)
    """
    from trading_bot.data.storage import init_db
    from trading_bot.data.universe import (
        fetch_all_constituents,
        fetch_sp400_constituents,
        fetch_sp500_constituents,
        fetch_sp600_constituents,
        get_universe_stats,
        persist_universe,
    )

    init_db()

    if index == "all":
        entries = fetch_all_constituents()
    elif index == "sp500":
        entries = fetch_sp500_constituents()
    elif index == "sp400":
        entries = fetch_sp400_constituents()
    elif index == "sp600":
        entries = fetch_sp600_constituents()
    else:
        console.print(f"[red]Unknown index '{index}'. Use: sp500 | sp400 | sp600 | all[/red]")
        raise typer.Exit(1)

    inserted = persist_universe(entries)
    stats = get_universe_stats()
    console.print(
        f"[green]✓[/green] Fetched {len(entries)} symbols, {inserted} new.\n"
        f"  Total in DB: {stats['total_tickers']} tickers\n"
        f"  Liquid (90d): {stats['liquid_90d']} have price data"
    )


@app.command("ingest")
def ingest_cmd(
    top: int = typer.Option(500, help="Top-N by liquidity"),
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
    chunk: int = typer.Option(50, help="Symbols per yfinance batch request"),
    skip_existing: bool = typer.Option(True, help="Skip symbols that already have recent data"),
) -> None:
    """Bootstrap ingest: download OHLCV for every ticker in the universe table.

    Optimised for large universes (S&P 1500+): uses chunked downloads with
    progress tracking and skips symbols that already have up-to-date data.
    Run once after fetch-universe, then use incremental-update for daily refresh.
    """
    import time

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from trading_bot.data.ingest import ingest_symbols
    from trading_bot.data.provider import last_stored_date
    from trading_bot.data.storage import Ticker, get_session, init_db

    init_db()
    with get_session() as session:
        all_symbols = [s[0] for s in session.query(Ticker.symbol).order_by(Ticker.symbol).all()]
    if add_regime and "SPY" not in all_symbols:
        all_symbols = ["SPY", *all_symbols]

    # Optionally skip symbols that already have data up to within 5 days of today
    if skip_existing:
        import pandas as pd
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=5)).date()
        todo = [s for s in all_symbols if (last_stored_date(s) or pd.Timestamp("2000-01-01").date()) < cutoff]
        skipped = len(all_symbols) - len(todo)
        console.print(
            f"Universe: {len(all_symbols)} symbols total, "
            f"[yellow]{skipped} already up-to-date[/yellow], "
            f"[cyan]{len(todo)} to ingest[/cyan]"
        )
    else:
        todo = all_symbols

    if not todo:
        console.print("[green]✓[/green] All symbols already up-to-date.")
        return

    total_inserted = 0
    n_chunks = (len(todo) + chunk - 1) // chunk

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Ingesting {len(todo)} symbols...", total=n_chunks)

        for i in range(0, len(todo), chunk):
            batch = todo[i : i + chunk]
            try:
                inserted = ingest_symbols(batch, start=start, end=end, chunk_size=len(batch))
                total_inserted += inserted
            except Exception as e:
                console.print(f"[yellow]Batch {i//chunk+1} error: {e}[/yellow]")
            progress.advance(task)
            # Gentle rate limiting — yfinance is fine with 50-100 symbols/batch
            time.sleep(0.3)

    console.print(
        f"[green]✓[/green] Ingest complete: {total_inserted:,} new candles "
        f"across {len(todo)} symbols."
    )


@app.command("universe-stats")
def universe_stats_cmd() -> None:
    """Show universe coverage: how many symbols have price data."""
    from trading_bot.data.universe import get_universe_stats

    stats = get_universe_stats()
    t = Table(title="Universe Coverage", show_header=True)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="white")
    t.add_row("Total tickers in DB", str(stats["total_tickers"]))
    t.add_row("Symbols with 90d price data", str(stats["liquid_90d"]))
    t.add_row("Coverage", f"{stats['liquid_90d']/max(stats['total_tickers'],1)*100:.0f}%")
    console.print(t)

    t2 = Table(title="By Sector", show_header=True)
    t2.add_column("Sector")
    t2.add_column("Count", justify="right")
    for sector, cnt in stats["by_sector"].items():
        t2.add_row(sector, str(cnt))
    console.print(t2)


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
    start: str = typer.Option("2005-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
    use_pit: bool = typer.Option(True, "--pit/--no-pit"),
    cpcv: bool = typer.Option(False, "--cpcv/--no-cpcv", help="Also run CPCV validation"),
    cpcv_k: int = typer.Option(10, help="CPCV folds (10 = 45 path OOS)"),
) -> None:
    """Run a single catalog strategy and persist the run."""
    _run_strategy(name=name, start=start, end=end, top=top, use_pit=use_pit,
                  run_cpcv=cpcv, cpcv_k=cpcv_k)


def _run_strategy(
    name: str,
    start: str,
    end: str | None,
    top: int,
    use_pit: bool,
    run_cpcv: bool = False,
    cpcv_k: int = 8,
) -> None:
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

    # TSMOM gold uses a single-asset panel (GLD only)
    if name == "tsmom_gold":
        from trading_bot.data.ingest import ingest_symbols
        ingest_symbols(["GLD"], start=start, end=end)
        panel = load_panel(["GLD"], start=start, end=end).dropna(how="all", axis=1)
        if panel.empty:
            console.print("[red]GLD data unavailable[/red]")
            raise typer.Exit(code=1)
        membership = None  # no PIT filter for single-asset

    bt_cfg = BacktestConfig(membership=membership)
    strat = get_strategy(name)
    bt = CrossSectionalBacktester(strat, bt_cfg)
    result = bt.run(panel)
    _print_metrics(result.metrics)

    # Optional CPCV
    cpcv_result = None
    if run_cpcv:
        from trading_bot.validation.cpcv import CPCVConfig, run_cpcv as _run_cpcv
        console.print(f"[cyan]Running CPCV k={cpcv_k} ({cpcv_k*(cpcv_k-1)//2} paths)…[/cyan]")
        try:
            cpcv_result = _run_cpcv(
                strat, panel, bt_cfg, CPCVConfig(k=cpcv_k, n_test=2)
            )
            _print_cpcv(cpcv_result)
        except Exception as e:
            console.print(f"[yellow]CPCV failed: {e}[/yellow]")

    run_id = persist_run(
        cfg, result, start=start, end=end, universe_size=top,
        use_pit=use_pit, cpcv_result=cpcv_result,
    )
    console.print(f"[green]✓[/green] Persisted run #{run_id} for '{name}'")


@app.command("run-all")
def run_all_cmd(
    start: str = typer.Option("2005-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
    use_pit: bool = typer.Option(True, "--pit/--no-pit"),
    cpcv: bool = typer.Option(False, "--cpcv/--no-cpcv", help="Run CPCV after each backtest"),
    cpcv_k: int = typer.Option(10, help="CPCV number of folds (10 = 45 path OOS)"),
) -> None:
    """Run every strategy in the catalog and persist each run."""
    from trading_bot.strategies.catalog import CATALOG

    for name in CATALOG:
        console.print(f"\n[bold cyan]━━━ {name} ━━━[/bold cyan]")
        _run_strategy(name=name, start=start, end=end, top=top, use_pit=use_pit,
                      run_cpcv=cpcv, cpcv_k=cpcv_k)


@app.command("research")
def research_cmd(
    start: str = typer.Option("2005-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
    rounds: int = typer.Option(5, help="Research rounds to run"),
    candidates: int = typer.Option(8, help="Candidates to test per round"),
    ic_threshold: float = typer.Option(0.02, help="Minimum IC to proceed to full backtest"),
    cpcv_k: int = typer.Option(10, help="CPCV folds (10=45 paths, 6=15 paths)"),
    target: int = typer.Option(3, help="Stop early after this many promotions"),
    pbo_gate: float = typer.Option(0.4, help="Max PBO to promote (CPCV non-parametric gate)"),
    min_oos_sharpe: float = typer.Option(0.5, help="Min OOS Sharpe to promote"),
    n_signals: int | None = typer.Option(None, help="Force signal count: 1=single, 2=dual, 3=triple (default: all)"),
) -> None:
    """Run the autonomous research loop.

    Each round: analyze existing runs → generate hypotheses guided by
    the Bayesian scorecard → IC pre-scan → CPCV backtest → update scorecard.
    Repeats until budget (rounds × candidates) is exhausted or target
    promotions are reached. Results are persisted so the loop is resumable.
    """
    from trading_bot.research.loop import LoopConfig, run_research_loop

    cfg = LoopConfig(
        start=start,
        end=end,
        top_n_universe=top,
        rounds=rounds,
        candidates_per_round=candidates,
        ic_prescan_threshold=ic_threshold,
        cpcv_k=cpcv_k,
        target_promotions=target,
        pbo_gate=pbo_gate,
        min_oos_sharpe=min_oos_sharpe,
        n_signals=n_signals,
    )
    run_research_loop(cfg)


@app.command("research-daemon")
def research_daemon_cmd(
    start: str = typer.Option("2005-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
    rounds: int = typer.Option(5, help="Round per batch (il daemon ricicla batch all'infinito)"),
    candidates: int = typer.Option(8, help="Candidati testati per round"),
    ic_threshold: float = typer.Option(0.02, help="IC minimo per passare al backtest completo"),
    cpcv_k: int = typer.Option(10, help="Fold CPCV (10 = 45 path OOS)"),
    pbo_gate: float = typer.Option(0.4, help="PBO massimo per la promozione"),
    min_oos_sharpe: float = typer.Option(0.5, help="OOS Sharpe attivo minimo"),
    n_signals: int | None = typer.Option(None, help="Forza n. segnali: 1/2/3 (default: tutti)"),
    workers: int = typer.Option(1, help="Candidati in parallelo (2 = ~1.5x più veloce su multi-core)"),
) -> None:
    """Avvia il research loop come daemon in background — gira finché non lo fermi.

    Il processo si stacca dal terminale: puoi chiudere la shell. Log su
    ~/.trading_bot/research.log. Segui in tempo reale con:
      tradebot research-follow
    Ferma con: tradebot research-stop.
    """
    from trading_bot.research.daemon import LOG_FILE, daemon_pid, start_daemon

    existing = daemon_pid()
    if existing is not None:
        console.print(f"[red]Daemon già attivo (pid {existing}). "
                      f"Fermalo prima con: tradebot research-stop[/red]")
        raise typer.Exit(code=1)

    try:
        pid = start_daemon({
            "start": start, "end": end, "top_n_universe": top,
            "rounds": rounds, "candidates_per_round": candidates,
            "ic_prescan_threshold": ic_threshold, "cpcv_k": cpcv_k,
            "pbo_gate": pbo_gate, "min_oos_sharpe": min_oos_sharpe,
            "n_signals": n_signals, "workers": workers,
            # nel daemon i batch si riciclano: mai fermarsi alle promozioni
            "target_promotions": 10_000,
        })
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"[green]✓[/green] Daemon avviato (pid {pid})"
                  + (f"  [dim]workers={workers}[/dim]" if workers > 1 else ""))
    console.print(f"  Log:    tradebot research-follow")
    console.print("  Stato:  tradebot research-status")
    console.print("  Stop:   tradebot research-stop")


@app.command("research-stop")
def research_stop_cmd(
    timeout: float = typer.Option(30.0, help="Secondi di attesa per la chiusura pulita"),
) -> None:
    """Ferma il daemon di ricerca (SIGTERM, poi SIGKILL dopo il timeout)."""
    from trading_bot.research.daemon import stop_daemon

    outcome, pid = stop_daemon(timeout=timeout)
    if outcome == "none":
        console.print("Nessun daemon attivo.")
    elif outcome == "stopped":
        console.print(f"[green]✓[/green] Daemon fermato pulito (pid {pid})")
    else:
        console.print(f"[yellow]⚠ Daemon non rispondeva — terminato forzatamente "
                      f"(pid {pid})[/yellow]")


@app.command("research-follow")
def research_follow_cmd(
    lines: int = typer.Option(40, help="Righe di storico da mostrare all'avvio"),
) -> None:
    """Segui il log del daemon in tempo reale — come tail -f con colori (Ctrl+C per uscire)."""
    import time as _time
    from trading_bot.research.daemon import LOG_FILE

    if not LOG_FILE.exists():
        console.print(f"[yellow]Nessun log trovato ({LOG_FILE}).[/yellow]")
        console.print("Avvia il daemon prima: [bold]tradebot research-daemon[/bold]")
        return

    def _colorize(line: str) -> str:
        if "★ PROMOTED" in line or "PROMOTED" in line:
            return f"[bold green]{line}[/bold green]"
        if "SKIP" in line:
            return f"[yellow]{line}[/yellow]"
        if "ERROR" in line or "Error" in line or "Traceback" in line:
            return f"[red]{line}[/red]"
        if "━━━ Batch" in line:
            return f"[bold cyan]{line}[/bold cyan]"
        if "Testing:" in line:
            return f"[cyan]{line}[/cyan]"
        if "OOS Sharpe" in line:
            return f"[bold]{line}[/bold]"
        if "PASS" in line:
            return f"[green]{line}[/green]"
        if "fail" in line:
            return f"[red]{line}[/red]"
        return line

    console.print(f"[dim]Seguendo {LOG_FILE} (Ctrl+C per uscire)[/dim]\n")
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            for ln in all_lines[-lines:]:
                console.print(_colorize(ln.rstrip()), markup=True)
            while True:
                ln = f.readline()
                if ln:
                    console.print(_colorize(ln.rstrip()), markup=True)
                else:
                    _time.sleep(0.3)
    except KeyboardInterrupt:
        console.print("\n[dim]Follow terminato.[/dim]")


@app.command("research-status")
def research_status_cmd() -> None:
    """Show research loop progress: scorecard, log summary, promoted strategies."""
    from datetime import datetime, timezone

    from sqlalchemy import select
    from trading_bot.data.storage import ResearchLog
    from trading_bot.research.daemon import LOG_FILE, daemon_pid, daemon_started_at
    from trading_bot.research.scorecard import get_all_scores
    from trading_bot.registry import trial_count

    # ── Stato daemon ─────────────────────────────────────────────────────
    pid = daemon_pid()
    if pid is not None:
        started = daemon_started_at()
        uptime, session_info = "", ""
        if started:
            delta = datetime.now(timezone.utc) - started
            h, rem = divmod(int(delta.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            uptime = f"  uptime {h}h {m:02d}m {s:02d}s"
            from sqlalchemy import func as _sqlfunc

            from trading_bot.data.storage import get_session as _gs
            with _gs() as _session:
                n_sess = _session.execute(
                    select(_sqlfunc.count(ResearchLog.id)).where(
                        ResearchLog.created_at >= started.replace(tzinfo=None))
                ).scalar() or 0
                last_trial = _session.execute(
                    select(_sqlfunc.max(ResearchLog.created_at))
                ).scalar()
            last_s = (last_trial.replace(tzinfo=timezone.utc).astimezone()
                      .strftime("%H:%M:%S") if last_trial else "—")
            session_info = f"  trial in sessione: {n_sess}  ultimo trial: {last_s}"
        console.print(f"[bold green]● Daemon ATTIVO[/bold green]  pid {pid}"
                      f"{uptime}{session_info}")
        console.print(f"  [dim]log: tail -f {LOG_FILE}  —  "
                      f"stop: tradebot research-stop[/dim]")
    else:
        console.print("[bold red]○ Daemon FERMO[/bold red]  "
                      "[dim]avvia con: tradebot research-daemon[/dim]")

    from trading_bot.research.search_space import estimate_search_space_size
    from trading_bot.data.storage import ResearchLog, get_session
    space = estimate_search_space_size()
    console.print(
        f"\n[bold]Search space:[/bold] ~{space['total_estimated']:,} unique configurations "
        f"({space['features']} features × params × weights × filters × regimes)"
    )
    console.print(
        f"  Single-signal: {space['single_signal_configs']:,}  "
        f"Dual: {space['dual_signal_configs']:,}  "
        f"Triple: {space['triple_signal_configs']:,}"
    )
    console.print(f"\n[bold]Trial counter (total):[/bold] {trial_count()}")

    with get_session() as session:
        logs = session.execute(
            select(ResearchLog).order_by(ResearchLog.created_at.desc())
        ).scalars().all()

    if not logs:
        console.print("No research runs yet. Run: tradebot research")
        return

    table = Table(title=f"Research log ({len(logs)} entries)", show_header=True)
    table.add_column("Round", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("IC scan", justify="right")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("PBO", justify="right")
    table.add_column("DSR", justify="right")

    for log in logs[:30]:
        status_color = {
            "promoted": "green", "tested": "yellow",
            "skipped": "dim", "error": "red", "pending": "white",
        }.get(log.status, "white")
        table.add_row(
            str(log.round_id),
            log.hypothesis_name[:40],
            f"[{status_color}]{log.status}[/{status_color}]",
            f"{log.ic_prescan:.4f}" if log.ic_prescan is not None else "—",
            f"{log.oos_sharpe:.3f}" if log.oos_sharpe is not None else "—",
            f"{log.pbo:.3f}" if log.pbo is not None else "—",
            f"{log.dsr:.3f}" if log.dsr is not None else "—",
        )
    console.print(table)

    scores = get_all_scores()
    if scores:
        st = Table(title="Feature scorecard (top 8)", show_header=True)
        st.add_column("Feature", style="cyan")
        st.add_column("Used", justify="right")
        st.add_column("OOS Sharpe avg", justify="right")
        st.add_column("PBO avg", justify="right")
        st.add_column("Score", justify="right")
        for s in scores[:8]:
            st.add_row(s["key"][:45], str(s["times_used"]),
                       f"{s['mean_oos_sharpe']:.3f}", f"{s['mean_pbo']:.3f}",
                       f"[bold]{s['score']:.3f}[/bold]")
        console.print(st)


@app.command("validate-cpcv")
def validate_cpcv_cmd(
    name: str = typer.Argument(..., help="Strategy name from catalog"),
    start: str = typer.Option("2005-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
    use_pit: bool = typer.Option(True, "--pit/--no-pit"),
    k: int = typer.Option(10, help="Number of CPCV folds (C(k,2) OOS paths)"),
) -> None:
    """Run CPCV on a catalog strategy and print the OOS Sharpe distribution + PBO."""
    _run_strategy(name=name, start=start, end=end, top=top, use_pit=use_pit,
                  run_cpcv=True, cpcv_k=k)


@app.command("run-portfolio")
def run_portfolio_cmd(
    start: str = typer.Option("2005-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
    core: str = typer.Option("momentum_12_1", help="Core equity sleeve strategy name"),
    gold_weight: float = typer.Option(0.20, help="Weight of gold TSMOM sleeve (0–1)"),
) -> None:
    """Combine core equity sleeve + TSMOM gold sleeve and show portfolio metrics.

    Demonstrates crisis alpha: gold's low/negative correlation with equity
    during drawdowns improves the combined Sharpe and reduces MaxDD.
    """
    from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
    from trading_bot.data.ingest import ingest_symbols, load_panel
    from trading_bot.data.pit_universe import build_membership_panel
    from trading_bot.data.storage import Ticker, get_session
    from trading_bot.data.universe import get_top_n_by_liquidity
    from trading_bot.portfolio.combiner import (
        PortfolioConfig,
        SleeveSpec,
        combine_sleeves,
        crisis_alpha_table,
    )
    from trading_bot.strategies.catalog import CATALOG, get_strategy
    from trading_bot.strategies.tsmom import TSMOMConfig, TSMOMStrategy

    # ── Core equity sleeve ───────────────────────────────────────────────
    if core not in CATALOG:
        console.print(f"[red]Unknown strategy: {core}[/red]")
        raise typer.Exit(code=1)

    symbols = get_top_n_by_liquidity(top)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    panel_eq = load_panel(symbols, start=start, end=end).dropna(how="all", axis=1)
    if panel_eq.empty:
        console.print("[red]No equity data — run ingest-all first[/red]")
        raise typer.Exit(code=1)

    with get_session() as session:
        current = {s[0] for s in session.query(Ticker.symbol).all()}
    membership = build_membership_panel(panel_eq.index, current, symbols=list(panel_eq.columns))

    strat_eq = get_strategy(core)
    result_eq = CrossSectionalBacktester(strat_eq, BacktestConfig(membership=membership)).run(panel_eq)
    console.print(f"\n[bold cyan]Core sleeve ({core})[/bold cyan]")
    _print_metrics(result_eq.metrics)

    # ── Gold TSMOM sleeve ────────────────────────────────────────────────
    ingest_symbols(["GLD"], start=start, end=end)
    panel_gld = load_panel(["GLD"], start=start, end=end).dropna(how="all", axis=1)
    strat_gld = TSMOMStrategy(TSMOMConfig())
    result_gld = CrossSectionalBacktester(strat_gld, BacktestConfig()).run(panel_gld)
    console.print(f"\n[bold yellow]Gold sleeve (tsmom_gold)[/bold yellow]")
    _print_metrics(result_gld.metrics)

    # ── Combined portfolio ───────────────────────────────────────────────
    core_weight = 1.0 - gold_weight
    cfg = PortfolioConfig(
        sleeves=[SleeveSpec(core, core_weight), SleeveSpec("tsmom_gold", gold_weight)],
        name=f"{core}_{int(core_weight*100)}_gold{int(gold_weight*100)}",
    )
    combined = combine_sleeves(
        {core: result_eq.returns, "tsmom_gold": result_gld.returns}, cfg
    )
    console.print(f"\n[bold green]Combined portfolio ({int(core_weight*100)}% core + {int(gold_weight*100)}% gold)[/bold green]")
    _print_metrics(combined.metrics)

    # ── Crisis alpha table ───────────────────────────────────────────────
    cat = crisis_alpha_table(
        {core: result_eq.returns, "tsmom_gold": result_gld.returns},
        equity_sleeve=core, gold_sleeve="tsmom_gold", n_worst=10,
    )
    if not cat.empty:
        console.print("\n[bold]Top-10 worst equity months — gold as safe haven[/bold]")
        table = Table(show_header=True)
        for col in cat.columns:
            table.add_column(col)
        for _, row in cat.iterrows():
            table.add_row(
                str(row["Month"]),
                f"{row[f'{core} return']:.2%}",
                f"{row['tsmom_gold return']:.2%}" if row['tsmom_gold return'] == row['tsmom_gold return'] else "n/a",
                str(row["Gold helped?"]),
            )
        console.print(table)


def _kill_stale_gui(port: int) -> None:
    """Termina qualsiasi server Streamlit già in ascolto sulla porta.

    La GUI parte headless e SOPRAVVIVE alla chiusura del terminale: senza
    questo, riavviare `tradebot gui` si limita a riconnettersi al vecchio
    processo, che serve ancora il codice/CSS congelato in memoria → "le
    modifiche non si vedono mai". Qui lo ammazziamo prima di ripartire.
    """
    import os
    import signal
    import subprocess
    import time

    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return  # lsof assente (raro su macOS) — pazienza, niente kill

    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    my_pid = os.getpid()
    for pid in pids:
        if pid == my_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[dim]Chiudo il server GUI precedente (pid {pid}) "
                          f"sulla porta {port}…[/dim]")
        except ProcessLookupError:
            continue
        except PermissionError:
            console.print(f"[yellow]Non posso terminare il pid {pid} "
                          f"(permessi). Chiudilo a mano.[/yellow]")
            continue
    if pids:
        # Attendi il rilascio della porta (SIGTERM → SIGKILL se ostinato).
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            still = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True,
            ).stdout.split()
            if not still:
                return
            time.sleep(0.4)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)


@app.command("gui")
def gui_cmd(
    port: int = typer.Option(8501, help="Streamlit port"),
    keep_existing: bool = typer.Option(
        False, "--keep-existing",
        help="NON terminare un eventuale server GUI già attivo sulla porta"),
    stop: bool = typer.Option(
        False, "--stop", help="Ferma la GUI in ascolto sulla porta ed esci"),
) -> None:
    """Avvia la GUI Streamlit in background (non blocca il terminale).

    Gira staccata: il prompt torna subito libero, così puoi lanciare anche
    `tradebot research-daemon` nello stesso terminale. Sopravvive alla
    chiusura della shell. Riavviando, termina prima il server vecchio sulla
    porta (le modifiche al codice si vedono sempre). Ferma con: gui --stop.
    """
    import os
    import subprocess
    import time
    import urllib.request
    from pathlib import Path

    if stop:
        _kill_stale_gui(port)
        console.print(f"[green]✓[/green] GUI fermata (porta {port}).")
        return

    if not keep_existing:
        _kill_stale_gui(port)

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
    # macOS fork-safety: streamlit phones home (telemetry/version check)
    # BEFORE running app.py — that initializes Apple's Network framework and
    # any later fork() SIGSEGVs the child. Disable telemetry and propagate
    # the no_proxy guard into the child environment.
    env = {
        **os.environ,
        "no_proxy": "*",
        "NO_PROXY": "*",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }

    # Lancio DETACHED, non bloccante: prima `subprocess.run` teneva occupato il
    # terminale finché non facevi Ctrl+C, quindi qualunque comando digitato dopo
    # (es. `tradebot research-daemon`) non partiva mai. Ora la GUI gira in
    # background con output su log e il prompt torna subito libero.
    log_dir = Path.home() / ".trading_bot"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "gui.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )

    # Aspetta che il server risponda (o che muoia in avvio) per dare un esito.
    url = f"http://localhost:{port}"
    deadline = time.monotonic() + 20.0
    up = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            console.print(f"[red]La GUI è morta in avvio (exit {proc.returncode}). "
                          f"Controlla: {log_path}[/red]")
            raise typer.Exit(code=1)
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1) as r:
                if r.status == 200:
                    up = True
                    break
        except Exception:
            time.sleep(0.4)

    if up:
        console.print(f"[green]✓[/green] GUI attiva su [bold]{url}[/bold] "
                      f"(pid {proc.pid}) — il terminale è libero.")
        console.print(f"  Log:   {log_path}")
        console.print(f"  Stop:  tradebot gui --stop")
    else:
        console.print(f"[yellow]GUI avviata (pid {proc.pid}) ma non ha ancora "
                      f"risposto. Apri {url} tra qualche secondo. Log: {log_path}"
                      f"[/yellow]")


@app.command("validate")
def validate_cmd(
    strategy: str = typer.Argument("momentum"),
    start: str = typer.Option("2018-01-01"),
    end: str | None = typer.Option(None),
    top: int = typer.Option(500),
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
    top: int = typer.Option(500, help="Top-N universe by liquidity"),
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


def _print_cpcv(r) -> None:
    from rich.panel import Panel
    pbo_color = "red" if r.pbo > 0.5 else "green"
    console.print(
        Panel(
            f"[bold]CPCV Results[/bold]  k={r.k}, paths={r.n_paths}\n\n"
            f"OOS Sharpe:  mean={r.mean_oos_sharpe:.3f}  "
            f"median={r.median_oos_sharpe:.3f}  "
            f"std={r.std_oos_sharpe:.3f}\n"
            f"Fraction positive: {r.fraction_positive * 100:.1f}%\n"
            f"[{pbo_color}]PBO = {r.pbo:.3f}[/{pbo_color}]  "
            f"({'PASS ✓' if r.pbo < 0.5 else 'FAIL ✗ — likely overfit'})",
            title="CPCV",
            border_style=pbo_color,
        )
    )


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


# ── Live trading commands ─────────────────────────────────────────────────────

@app.command("generate-signals")
def generate_signals_cmd(
    strategy: str | None = typer.Option(None, "--strategy", "-s", help="Strategy name (default: best promoted)"),
    universe: int = typer.Option(500, help="Universe size"),
) -> None:
    """Generate today's trading signals from the best promoted strategy."""
    from trading_bot.live.signals import generate_signals

    try:
        signals = generate_signals(strategy_name=strategy, universe_size=universe)
        console.print(f"\n[bold cyan]Signals as of {signals.asof}[/bold cyan]")
        console.print(f"Strategy: [yellow]{signals.strategy_name}[/yellow]")
        console.print(f"Universe: {signals.n_universe} stocks")
        if signals.regime_active is not None:
            regime_str = "[green]ACTIVE[/green]" if signals.regime_active else "[red]OFF — no trades[/red]"
            console.print(f"Regime filter: {regime_str}")

        if signals.positions:
            table = Table(title="Target Positions", show_header=True)
            table.add_column("Symbol", style="cyan")
            table.add_column("Weight", justify="right")
            table.add_column("Allocation ($100k)", justify="right")
            for sym, w in sorted(signals.positions.items(), key=lambda x: -x[1]):
                bar = "█" * int(w * 30)
                table.add_row(sym, f"{w*100:.1f}%", f"${w*100_000:,.0f}  {bar}")
            console.print(table)
            console.print(f"\nTotal allocation: {sum(signals.positions.values())*100:.1f}%")
        else:
            console.print("[yellow]No positions — regime filter may be blocking all trades.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("run-daily")
def run_daily_cmd(
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Simulate without placing orders"),
    force_rebalance: bool = typer.Option(False, "--force-rebalance", help="Rebalance even if not first day of month"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip price data update"),
    universe: int = typer.Option(500, help="Universe size"),
) -> None:
    """Run the full daily cycle: ingest → signals → rebalance → notify.

    Default is --dry-run (safe). Use --live to place real paper orders on Alpaca.
    """
    from trading_bot.live.runner import run_daily
    result = run_daily(
        dry_run=dry_run,
        force_rebalance=force_rebalance,
        skip_ingest=skip_ingest,
        universe_size=universe,
    )
    if result.errors:
        raise typer.Exit(code=1)


@app.command("holdout-test")
def holdout_test_cmd(
    strategy: str | None = typer.Option(None, "--strategy", "-s", help="Strategy name (default: best promoted)"),
    holdout_start: str = typer.Option("2024-01-01", help="Start of holdout period"),
    universe: int = typer.Option(500, help="Universe size"),
) -> None:
    """Run the final holdout OOS test on a promoted strategy.

    WARNING: This consumes the holdout. Run only once per strategy before going live.
    """
    from trading_bot.validation.holdout import run_holdout_test

    console.print(
        "[yellow]⚠  This test consumes the holdout period. "
        "Run only once per strategy, when ready to go live.[/yellow]\n"
    )
    confirm = typer.confirm("Continue?")
    if not confirm:
        raise typer.Exit()

    result = run_holdout_test(
        strategy_name=strategy,
        holdout_start=holdout_start,
        universe_size=universe,
    )
    console.print(f"\n[bold]{result.summary()}[/bold]")


@app.command("alpaca-status")
def alpaca_status_cmd() -> None:
    """Show current Alpaca paper trading account status and positions."""
    from rich.panel import Panel
    from trading_bot.live.alpaca import get_alpaca_client

    alpaca = get_alpaca_client()
    if alpaca is None:
        console.print(
            "[yellow]Alpaca not configured.[/yellow]\n"
            "Add to .env:\n"
            "  TRADEBOT_ALPACA_API_KEY=PKxxx\n"
            "  TRADEBOT_ALPACA_SECRET_KEY=xxx\n"
            "(Get free paper trading keys at alpaca.markets)"
        )
        return

    account = alpaca.get_account()
    market_open = alpaca.is_market_open()

    console.print(Panel(
        f"[bold]Alpaca Paper Trading Account[/bold]\n\n"
        f"Portfolio value:  [cyan]${account.portfolio_value:>12,.2f}[/cyan]\n"
        f"Equity:           [cyan]${account.equity:>12,.2f}[/cyan]\n"
        f"Cash:             [cyan]${account.cash:>12,.2f}[/cyan]\n"
        f"Buying power:     [cyan]${account.buying_power:>12,.2f}[/cyan]\n"
        f"Market:           {'[green]OPEN[/green]' if market_open else '[red]CLOSED[/red]'}",
        title="Account",
    ))

    positions = alpaca.get_positions()
    if positions:
        table = Table(title="Current Positions", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Market Value", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        for p in sorted(positions, key=lambda x: -x.market_value):
            pl_color = "green" if p.unrealized_pl >= 0 else "red"
            table.add_row(
                p.symbol,
                f"{p.qty:.2f}",
                f"${p.market_value:,.2f}",
                f"[{pl_color}]${p.unrealized_pl:+,.2f}[/{pl_color}]",
                f"[{pl_color}]{p.unrealized_plpc*100:+.2f}%[/{pl_color}]",
            )
        console.print(table)
    else:
        console.print("[dim]No open positions.[/dim]")


@app.command("setup-scheduler")
def setup_scheduler_cmd(
    time_str: str = typer.Option("21:30", help="Local time to run daily (HH:MM, 24h)"),
) -> None:
    """Install a macOS launchd job to run the bot daily at the given time.

    This creates a plist in ~/Library/LaunchAgents/ and loads it.
    The bot will run every day at the specified time, even when this terminal is closed.
    """
    import os
    import shutil
    import subprocess

    # shutil.which avoids fork() — safer on macOS multi-threaded processes
    bot_path = shutil.which("tradebot")
    python_path = shutil.which("python3") or shutil.which("python")
    if not bot_path or not python_path:
        console.print("[red]tradebot or python not found on PATH[/red]")
        raise typer.Exit(1)
    log_dir = Path.home() / "Library" / "Logs" / "tradebot"
    log_dir.mkdir(parents=True, exist_ok=True)

    hour, minute = time_str.split(":")
    plist_label = "com.tradebot.daily"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{plist_label}.plist"

    # Working directory (project root)
    project_root = str(Path(__file__).resolve().parents[2])
    env_file = str(Path(project_root) / ".env")

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bot_path}</string>
        <string>run-daily</string>
        <string>--live</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_root}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/daily.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/daily_err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{os.path.dirname(bot_path)}</string>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
"""

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    console.print(f"[green]✓[/green] Plist written to {plist_path}")

    # Unload old version if exists
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    # Load new
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)

    if result.returncode == 0:
        console.print(f"[green]✓[/green] Scheduler installed. Bot will run daily at [cyan]{time_str}[/cyan].")
        console.print(f"  Logs: {log_dir}/daily.log")
        console.print(f"  To uninstall: launchctl unload {plist_path}")
    else:
        console.print(f"[red]launchctl error:[/red] {result.stderr}")
        console.print(f"Plist saved at {plist_path} — load manually with:")
        console.print(f"  launchctl load {plist_path}")


if __name__ == "__main__":
    app()
