"""Daily runner — the bot's main execution loop.

Called once per day (typically after market close ~22:00 IT) by the scheduler.

What it does:
  1. Ingest fresh price data
  2. Generate signals from best promoted strategy
  3. If rebalance day (first trading day of month): execute paper trades on Alpaca
  4. Check for drawdown alerts
  5. Send Telegram summary
  6. (Weekly) run research loop to find better strategies

Schedule:
  - Daily:   ingest + signals + drawdown check + notify
  - Monthly: rebalance on first trading day of month
  - Weekly:  research loop (configurable rounds)

Usage:
    tradebot run-daily          # full daily cycle
    tradebot run-daily --dry-run  # simulate without placing orders
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from trading_bot.data.ingest import ingest_symbols, incremental_update
from trading_bot.live.alpaca import AlpacaClient, get_alpaca_client
from trading_bot.live.notifier import NotifyLevel, TelegramNotifier, get_notifier, notify
from trading_bot.live.signals import SignalOutput, generate_signals
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

DRAWDOWN_ALERT_THRESHOLD = 0.15   # alert if portfolio DD > 15%


@dataclass
class DailyRunResult:
    date: date
    signals: SignalOutput | None = None
    rebalanced: bool = False
    orders_count: int = 0
    portfolio_value: float = 0.0
    current_drawdown: float = 0.0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


def is_rebalance_day(asof: pd.Timestamp | None = None) -> bool:
    """True if today is the first trading day of the month."""
    if asof is None:
        asof = pd.Timestamp.today().normalize()

    # First trading day of month = day 1-3 of month that is a weekday
    if asof.day > 5:
        return False
    # Check if it's a weekday
    if asof.weekday() >= 5:
        return False
    # Check no earlier weekday this month
    for d in range(1, asof.day):
        candidate = asof.replace(day=d)
        if candidate.weekday() < 5:
            return False  # there was an earlier trading day this month
    return True


def is_volatility_rebalance_day(asof: pd.Timestamp | None = None) -> bool:
    """Adaptive overlay: True on Mondays when market vol is elevated.

    Weekly rebalancing kicks in ONLY during stressed regimes (SPY 21-day
    realized vol > settings.vol_trigger, annualized). In calm markets the
    monthly schedule applies — no extra turnover cost when it isn't needed.
    """
    from trading_bot.config import settings

    if not settings.adaptive_rebalance:
        return False
    if asof is None:
        asof = pd.Timestamp.today().normalize()
    if asof.weekday() != 0:        # only Mondays
        return False
    try:
        from trading_bot.data.ingest import load_panel
        start = (asof - pd.DateOffset(days=60)).strftime("%Y-%m-%d")
        spy = load_panel(["SPY"], start=start)
        spy = spy.loc[:asof]
        if len(spy) < 22:
            return False
        vol = float(spy["SPY"].pct_change().iloc[-21:].std() * (252 ** 0.5))
        if vol > settings.vol_trigger:
            logger.info(f"Volatility rebalance triggered: SPY 21d vol={vol:.1%} "
                        f"> {settings.vol_trigger:.0%}")
            return True
    except Exception as e:
        logger.warning(f"Vol-trigger check failed: {e}")
    return False


def apply_portfolio_overlays(
    core_positions: dict[str, float],
    asof: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Apply portfolio-level overlays to raw strategy weights.

    Volatility targeting: scale total exposure by vol_target / realized
    portfolio vol (capped at 1.0, floored at 0.2). Moreira-Muir (2017).
    NOTE: the gold sleeve is NOT applied here — it is a strategy parameter
    (gold_weight/gold_mode) searched by the loop and applied inside
    ComposedStrategy.weights(), held only when drawdown risk is elevated.

    Research loop tests raw signal alpha; overlays are applied here, at the
    portfolio level — mirroring how the backtest engine applies them.
    """
    from trading_bot.config import settings

    if asof is None:
        asof = pd.Timestamp.today().normalize()
    positions = dict(core_positions)

    # Gold sleeve: NON è più un overlay fisso — è un parametro della
    # strategia (gold_weight/gold_mode in ComposedConfig), esplorato dal
    # loop e applicato da ComposedStrategy.weights(). Qui resta solo il
    # vol targeting.
    # ── 2. Volatility targeting ───────────────────────────────────────────
    if settings.vol_target > 0:
        try:
            from trading_bot.data.ingest import load_panel

            syms = [s for s in positions if positions[s] > 0]
            if syms:
                start = (asof - pd.DateOffset(days=160)).strftime("%Y-%m-%d")
                panel = load_panel(syms, start=start).loc[:asof].ffill()
                rets = panel.pct_change().dropna(how="all").iloc[-63:]
                w = pd.Series(positions).reindex(rets.columns).fillna(0.0)
                port_rets = (rets * w).sum(axis=1)
                realized = float(port_rets.std() * (252 ** 0.5))
                if realized > 0:
                    scale = min(1.0, max(0.2, settings.vol_target / realized))
                    positions = {s: w_ * scale for s, w_ in positions.items()}
                    logger.info(f"Vol targeting: realized={realized:.1%} "
                                f"target={settings.vol_target:.0%} scale={scale:.2f}")
        except Exception as e:
            logger.warning(f"Vol targeting failed (unscaled): {e}")

    # ── Panic filter (Daniel-Moskowitz 2016) ──────────────────────────────
    # Nei panic state (SPY sotto SMA200 + alta vol) il momentum storicamente
    # crasha: dimezza l'esposizione equity. L'oro difensivo della strategia
    # entra esattamente in questi stati — i due meccanismi si completano.
    try:
        from trading_bot.data.ingest import load_panel as _lp
        _spy = _lp(["SPY"], start=(asof - pd.DateOffset(days=420)).strftime("%Y-%m-%d")).loc[:asof]["SPY"].dropna()
        if len(_spy) >= 200:
            _below = _spy.iloc[-1] < _spy.iloc[-200:].mean()
            _vol = float(_spy.pct_change().iloc[-21:].std() * (252 ** 0.5))
            if _below and _vol > settings.vol_trigger:
                scale_p = settings.panic_scale
                positions = {s_: w_ * scale_p for s_, w_ in positions.items() if s_ != "GLD"} |                             {s_: w_ for s_, w_ in positions.items() if s_ == "GLD"}
                logger.info(f"Panic state (DM2016): equity scalata a {scale_p:.0%}, oro intatto")
    except Exception as e:
        logger.warning(f"Panic check fallito: {e}")

    # Budget clamp (audit M2): total exposure must never exceed 100%.
    total = sum(positions.values())
    if total > 1.0:
        positions = {s_: w_ / total for s_, w_ in positions.items()}

    return positions


def run_daily(
    dry_run: bool = False,
    force_rebalance: bool = False,
    skip_ingest: bool = False,
    universe_size: int = 500,
) -> DailyRunResult:
    """Execute the full daily cycle.

    Args:
        dry_run: simulate everything, don't place real orders.
        force_rebalance: execute rebalance even if not first day of month.
        skip_ingest: skip price data ingestion (use cached data).
        universe_size: universe size for signal generation.
    """
    today = pd.Timestamp.today().normalize()
    result = DailyRunResult(date=today.date(), dry_run=dry_run)

    console.print(Panel(
        f"[bold cyan]Daily Run — {today.date()}[/bold cyan]\n"
        f"dry_run={dry_run}  force_rebalance={force_rebalance}  skip_ingest={skip_ingest}",
        title="TradingBot",
    ))

    notifier = get_notifier()
    alpaca = get_alpaca_client()

    # ── 1. Ingest fresh data ──────────────────────────────────────────────────
    if not skip_ingest:
        console.print("[cyan]Step 1/5: Ingesting price data...[/cyan]")
        try:
            from trading_bot.data.universe import get_top_n_by_liquidity
            symbols = get_top_n_by_liquidity(universe_size)
            if "SPY" not in symbols:
                symbols = ["SPY", *symbols]
            n = incremental_update(symbols, default_start="2003-01-01")
            console.print(f"  ✅ Data ingested ({n} new candles)")
        except Exception as e:
            msg = f"Ingest failed: {e}"
            logger.error(msg)
            result.errors.append(msg)
            if notifier:
                notifier.notify_error("data ingest", e)
            # Continue — use cached data

    # ── 2. Generate signals ───────────────────────────────────────────────────
    console.print("[cyan]Step 2/5: Generating signals...[/cyan]")
    try:
        signals = generate_signals(universe_size=universe_size)
        result.signals = signals

        console.print(f"  ✅ {len(signals.positions)} positions generated")
        console.print(f"  Strategy: {signals.strategy_name}")
        for sym, w in sorted(signals.positions.items(), key=lambda x: -x[1]):
            console.print(f"    {sym:<8} {w*100:.1f}%")

        if notifier:
            notifier.notify_signals(
                strategy_name=signals.strategy_name,
                asof=signals.asof,
                positions=signals.positions,
                regime_active=signals.regime_active,
            )

    except Exception as e:
        msg = f"Signal generation failed: {e}"
        logger.error(msg)
        result.errors.append(msg)
        if notifier:
            notifier.notify_error("signal generation", e)
        console.print(f"  [red]✗ {msg}[/red]")

    # ── 3. Rebalance (first trading day of month) ─────────────────────────────
    should_rebalance = (
        force_rebalance
        or is_rebalance_day(today)
        or is_volatility_rebalance_day(today)   # weekly only in stressed regimes
    )

    if should_rebalance and result.signals:
        # ── Buy/hold band (Novy-Marx-Velikov 2016) ─────────────────────────
        # Tieni le posizioni già detenute finché restano nel band (2× top_n):
        # stesso comportamento del backtest engine → meno turnover, meno costi.
        try:
            if alpaca and result.signals.ranked_universe is not None \
                    and len(result.signals.ranked_universe) > 0:
                held = {p.symbol for p in alpaca.get_positions()} - {"GLD"}
                ranked = result.signals.ranked_universe.sort_values(ascending=False)
                pos = result.signals.positions
                gld_w = float(pos.get("GLD", 0.0))
                top_n = len([s for s in pos if s != "GLD"])
                if top_n > 0 and held:
                    band = set(ranked.head(int(top_n * 2)).index)
                    keep = [s for s in held if s in band and s in ranked.index]
                    fill = [s for s in ranked.index if s not in keep][:max(0, top_n - len(keep))]
                    sel = keep + fill
                    if sel:
                        eq_w = (1.0 - gld_w) / len(sel)
                        new_pos = {s: eq_w for s in sel}
                        if gld_w > 0:
                            new_pos["GLD"] = gld_w
                        result.signals.positions = new_pos
                        console.print(f"  Band buy/hold: tenute {len(keep)}, nuove {len(fill)}")
        except Exception as e:
            logger.warning(f"Hold-band fallita (uso pesi raw): {e}")

        # Portfolio overlays: gold TSMOM sleeve + volatility targeting.
        # Applied to the raw strategy weights before execution.
        try:
            result.signals.positions = apply_portfolio_overlays(
                result.signals.positions, today
            )
            console.print(
                f"  Overlays applied: total exposure "
                f"{sum(result.signals.positions.values())*100:.0f}%"
            )
        except Exception as e:
            logger.warning(f"Overlay application failed (raw weights used): {e}")
        console.print("[cyan]Step 3/5: Rebalancing portfolio...[/cyan]")

        if alpaca is None:
            console.print(
                "  [yellow]⚠ Alpaca not configured — skipping execution. "
                "Add keys to .env to enable paper trading.[/yellow]"
            )
        else:
            try:
                account = alpaca.get_account()
                result.portfolio_value = account.portfolio_value

                if not alpaca.is_market_open() and not dry_run:
                    console.print(
                        "  [yellow]Market is closed. Orders will be queued for next open.[/yellow]"
                    )

                orders = alpaca.rebalance(
                    target_weights=result.signals.positions,
                    min_trade_pct=0.01,
                    dry_run=dry_run,
                )
                result.rebalanced = True
                result.orders_count = len(orders)

                console.print(f"  ✅ {len(orders)} orders {'simulated' if dry_run else 'submitted'}")

                if notifier and not dry_run:
                    notifier.notify_rebalance(orders, account.portfolio_value, signals.strategy_name)

            except Exception as e:
                msg = f"Rebalance failed: {e}"
                logger.error(msg)
                result.errors.append(msg)
                if notifier:
                    notifier.notify_error("rebalance", e)
                console.print(f"  [red]✗ {msg}[/red]")
    else:
        reason = ("not a rebalance day (monthly or vol-triggered)"
                  if not should_rebalance else "no signals")
        console.print(f"[dim]Step 3/5: Skipping rebalance ({reason})[/dim]")

    # ── 4. Drawdown check ─────────────────────────────────────────────────────
    console.print("[cyan]Step 4/5: Checking drawdown...[/cyan]")
    if alpaca and result.portfolio_value > 0:
        try:
            history = alpaca.get_portfolio_history(period="6M", timeframe="1D")
            equity = history.get("equity", [])
            if equity and len(equity) > 2:
                import numpy as np
                eq_arr = [v for v in equity if v is not None]
                if eq_arr:
                    peak = max(eq_arr)
                    current = eq_arr[-1]
                    dd = (peak - current) / peak if peak > 0 else 0
                    result.current_drawdown = dd

                    console.print(f"  Current drawdown: {dd*100:.1f}%")
                    if dd > DRAWDOWN_ALERT_THRESHOLD and notifier:
                        notifier.notify_drawdown_alert(dd, DRAWDOWN_ALERT_THRESHOLD, current)
                        console.print(f"  [yellow]⚠ Drawdown alert sent[/yellow]")
        except Exception as e:
            logger.warning(f"Drawdown check failed: {e}")
    else:
        console.print("[dim]  Skipping — Alpaca not connected[/dim]")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    console.print("[cyan]Step 5/5: Summary[/cyan]")
    status = "✅ OK" if not result.errors else f"⚠ {len(result.errors)} error(s)"
    console.print(Panel(
        f"Date: {result.date}\n"
        f"Status: {status}\n"
        f"Signals: {len(result.signals.positions) if result.signals else 0} positions\n"
        f"Rebalanced: {'yes' if result.rebalanced else 'no'}\n"
        f"Orders: {result.orders_count}\n"
        f"Portfolio: ${result.portfolio_value:,.0f}\n"
        f"Drawdown: {result.current_drawdown*100:.1f}%\n"
        + (f"\nErrors:\n" + "\n".join(f"  • {e}" for e in result.errors) if result.errors else ""),
        title="Daily Run Complete",
        border_style="green" if not result.errors else "yellow",
    ))

    return result
