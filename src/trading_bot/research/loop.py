"""Autonomous research loop.

Full cycle per round:
  1. Analyze all existing runs (what worked / why)
  2. Generate N candidate hypotheses guided by the scorecard
  3. IC pre-scan (fast: last 2 years, reject IC < threshold)
  4. Full backtest + CPCV on surviving candidates
  5. Persist result, update scorecard, log in research_log table
  6. Repeat until budget exhausted or target promotions reached

The loop is interruptible: press Ctrl-C and progress is saved.
Resume by running again — it skips already-tested configs.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.backtest.metrics import information_coefficient
from trading_bot.data.ingest import load_panel
from trading_bot.data.pit_universe import build_membership_panel
from trading_bot.data.storage import ResearchLog, get_session, init_db
from trading_bot.registry import list_runs, persist_run, trial_count
from trading_bot.research.analyzer import analyze_runs
from trading_bot.research.generator import Candidate, generate
from trading_bot.research.scorecard import get_all_scores, update_score
from trading_bot.strategies.composer import ComposedConfig, ComposedStrategy
from trading_bot.utils.logging import get_logger
from trading_bot.validation.cpcv import CPCVConfig, run_cpcv

logger = get_logger(__name__)
console = Console()

init_db()


@dataclass
class LoopConfig:
    start: str = "2005-01-01"
    end: str | None = None
    top_n_universe: int = 500          # default to top-500 liquid stocks (S&P 1500)
    rounds: int = 5
    candidates_per_round: int = 10
    ic_prescan_threshold: float = 0.02      # min IC to proceed to full backtest
    ic_prescan_years: float = 3.0           # years of recent data for IC scan
    cpcv_k: int = 10                        # 45 path OOS — PBO a granularità 1/45
    pbo_gate: float = 0.4                   # PBO < this required (stricter than 0.5)
    min_oos_sharpe: float = 0.5             # absolute OOS Sharpe floor
    # DSR removed as hard gate — mathematically collapses with N>10 trials.
    # PBO from CPCV is non-parametric and more reliable. DSR still reported.
    target_promotions: int = 3              # stop early if reached
    use_pit: bool = True
    n_signals: int | None = None            # None=all, 1=single, 2=dual, 3=triple


def run_research_loop(cfg: LoopConfig, stop_event=None) -> list[str]:
    """Run the autonomous research loop. Returns list of promoted strategy names.

    ``stop_event`` (threading.Event-like): se settato (es. SIGTERM dal daemon),
    il loop chiude il trial corrente, persiste tutto ed esce pulito.
    """
    from trading_bot.data.storage import Ticker
    from trading_bot.data.universe import get_top_n_by_liquidity

    console.print(Panel(
        f"[bold cyan]Autonomous Research Loop[/bold cyan]\n"
        f"rounds={cfg.rounds}  candidates/round={cfg.candidates_per_round}  "
        f"IC_threshold={cfg.ic_prescan_threshold}  "
        f"CPCV k={cfg.cpcv_k}  PBO gate={cfg.pbo_gate}\n"
        f"Data: {cfg.start} → {cfg.end or 'today'}  universe={cfg.top_n_universe}",
        title="Research loop",
    ))

    # Load universe + panel once
    symbols = get_top_n_by_liquidity(cfg.top_n_universe)
    if "SPY" not in symbols:
        symbols = ["SPY", *symbols]
    for service in ("GLD", "HYG"):
        if service not in symbols:
            symbols = [service, *symbols]   # gold sleeve + credit regime
    panel = load_panel(symbols, start=cfg.start, end=cfg.end).dropna(how="all", axis=1)
    if panel.empty:
        console.print("[red]No data — run ingest-all first[/red]")
        return []

    from sqlalchemy import select
    with get_session() as session:
        current_tickers = {s[0] for s in session.query(Ticker.symbol).all()}

    membership = None
    if cfg.use_pit:
        membership = build_membership_panel(panel.index, current_tickers, symbols=list(panel.columns))
    bt_cfg = BacktestConfig(membership=membership)

    # Equal-weight universe daily returns — benchmark for ACTIVE Sharpe
    # (audit C2: absolute Sharpe gates pass on beta alone in bull samples).
    benchmark_returns = (
        panel.drop(columns=["SPY"], errors="ignore")
        .pct_change()
        .mean(axis=1)
        .fillna(0.0)
    )

    # IC pre-scan uses only the most recent N years
    prescan_start = pd.Timestamp(cfg.end or pd.Timestamp.today()) - pd.DateOffset(years=cfg.ic_prescan_years)
    panel_prescan = panel.loc[prescan_start:]

    # Already-tested fingerprints (only from THIS universe — same config on a
    # different universe is a different experiment)
    tested_fps = _load_tested_fingerprints(universe=cfg.top_n_universe)

    promoted: list[str] = []
    total_rounds = 0

    for round_id in range(1, cfg.rounds + 1):
        if stop_event is not None and stop_event.is_set():
            console.print("[yellow]Stop richiesto — uscita pulita.[/yellow]")
            break
        total_rounds = round_id
        n_trials = trial_count()
        console.rule(f"[bold]Round {round_id}/{cfg.rounds}  |  trials so far: {n_trials}[/bold]")

        # ── 1. Analyze existing runs ──────────────────────────────────────
        runs = list_runs()
        report = analyze_runs(runs)
        console.print(
            f"Analyzed {report.total_runs} runs → "
            f"{len(report.promoted_runs)} promoted, "
            f"avg OOS Sharpe={report.avg_oos_sharpe:.3f}\n"
            f"Best features so far: {report.best_features[:3]}"
        )

        # ── 2. Generate candidates: UCB1 arms + diverse archetypes ──────
        from trading_bot.research.search_space import generate_ordered_arms
        from trading_bot.research.archetypes import generate_diverse_batch

        # Half budget: UCB1 parametric search (exploitation of known-good signals)
        ucb1_budget = max(1, cfg.candidates_per_round // 2)
        arms = generate_ordered_arms(
            already_tested=tested_fps,
            max_arms=ucb1_budget * 5,
            n_total_experiments=trial_count(),
            n_signals=cfg.n_signals,
            universe=cfg.top_n_universe,
        )

        # Half budget: archetype-based generation (structural diversity)
        arch_budget = cfg.candidates_per_round - ucb1_budget
        already_names = {r[1].signals[0][0] for r in arms} if arms else set()
        arch_candidates = generate_diverse_batch(arch_budget, already_names)

        console.print(
            f"Candidates: {len(arms)} UCB1 arms + {len(arch_candidates)} archetype arms"
        )

        # Interleave UCB1 arms and archetype candidates
        # UCB1 arms first (exploitation), then archetype (exploration)
        ucb1_iter = iter(arms)
        arch_iter = iter(arch_candidates)
        candidate_queue = []
        ucb1_done, arch_done = False, False
        while len(candidate_queue) < cfg.candidates_per_round * 2:
            # Alternate: 1 UCB1, 1 archetype
            if not ucb1_done:
                try:
                    candidate_queue.append(("ucb1", next(ucb1_iter)))
                except StopIteration:
                    ucb1_done = True
            if not arch_done:
                try:
                    candidate_queue.append(("arch", next(arch_iter)))
                except StopIteration:
                    arch_done = True
            if ucb1_done and arch_done:
                break

        round_tested = 0
        for source, candidate in candidate_queue:
            if round_tested >= cfg.candidates_per_round:
                break

            # Unpack depending on source
            if source == "ucb1":
                ucb_score, arm = candidate
                fp = arm.fingerprint
                if fp in tested_fps:
                    continue
                tested_fps.add(fp)
                feat_names = [s[0] for s in arm.signals]
                name = (
                    f"r{round_id:02d}_{'+'.join(f[:3] for f in feat_names)}"
                    f"_t{arm.top_n}_{'_'.join(arm.regime_key.split('_')[-1:])}"
                )[:60]
                cfg_composed = arm.to_composed_config(name)
                arch_label = "pure_factor"
                arm_obj = arm
            else:
                # Archetype candidate
                arch_result = candidate
                cfg_composed = arch_result.config
                name = f"r{round_id:02d}_{arch_result.archetype[:6]}_{round_tested:02d}"[:60]
                cfg_composed = ComposedConfig(
                    name=name,
                    rationale=cfg_composed.rationale,
                    signals=cfg_composed.signals,
                    filters=cfg_composed.filters,
                    regime=cfg_composed.regime,
                    top_n=cfg_composed.top_n,
                )
                ucb_score = 0.0
                arch_label = arch_result.archetype
                feat_names = [s.feature for s in cfg_composed.signals]
                arm_obj = None

            console.print(
                f"\n  [cyan]Testing:[/cyan] {name}  "
                f"[dim]{arch_label}[/dim]  UCB1={ucb_score:.3f}"
            )
            console.print(f"    Signals: {feat_names}")

            # Log with archetype info embedded in rationale
            if arm_obj is not None:
                log_entry = _log_arm(round_id, arm_obj, cfg_composed,
                                     universe_size=cfg.top_n_universe)
            else:
                log_entry = _log_hypothesis_cfg(round_id, cfg_composed, arch_label,
                                                universe_size=cfg.top_n_universe)

            # ── 3. IC pre-scan ────────────────────────────────────────────
            ic_mean = _ic_prescan_config(cfg_composed, panel_prescan, bt_cfg)
            _update_log(log_entry["id"], ic_prescan=ic_mean)
            console.print(f"    IC pre-scan: {ic_mean:.4f}", end="")

            if ic_mean < cfg.ic_prescan_threshold:
                console.print(f"  [yellow]→ SKIP (IC < {cfg.ic_prescan_threshold})[/yellow]")
                _update_log(log_entry["id"], status="skipped",
                            skip_reason=f"IC={ic_mean:.4f} < threshold")
                for sig in cfg_composed.signals:
                    update_score(sig.feature, sig.params, oos_sharpe=-0.1, pbo=0.8,
                                 promoted=False, universe=cfg.top_n_universe)
                round_tested += 1
                continue

            console.print(f"  [green]→ proceed to full backtest[/green]")

            # ── 4. Full backtest + CPCV ───────────────────────────────────
            try:
                strat = ComposedStrategy(cfg_composed)
                bt = CrossSectionalBacktester(strat, bt_cfg)
                result = bt.run(panel)

                cpcv_result = run_cpcv(
                    strat, panel, bt_cfg,
                    CPCVConfig(k=cfg.cpcv_k, n_test=2, purge_days=21),
                    benchmark_returns=benchmark_returns,   # ACTIVE Sharpe (audit C2)
                )

                oos_sharpe = cpcv_result.mean_oos_sharpe
                pbo = cpcv_result.pbo

                run_id = persist_run(
                    cfg_composed, result,
                    start=cfg.start, end=cfg.end,
                    universe_size=cfg.top_n_universe,
                    use_pit=cfg.use_pit,
                    cpcv_result=cpcv_result,
                )

                dsr = _get_run_dsr(run_id)
                # Gate (audit C2 + M8 + Harvey-Liu-Zhu 2016): PBO non-parametric
                # + ACTIVE OOS Sharpe floor che SALE col numero di trial
                # (multiple-testing: con centinaia di test la soglia classica
                # produce falsi positivi; HLZ richiedono t>3 ≈ +30-60% hurdle)
                # + path robustness (≥65% of CPCV paths positive).
                import math as _math
                n_tr = trial_count()
                eff_min_oos = cfg.min_oos_sharpe * (
                    1.0 + 0.3 * max(0.0, _math.log10(max(n_tr, 1) / 100))
                )
                gate_pass = (
                    pbo < cfg.pbo_gate
                    and oos_sharpe >= eff_min_oos
                    and cpcv_result.fraction_positive >= 0.65
                )

                _update_log(log_entry["id"],
                            oos_sharpe=oos_sharpe, pbo=pbo, dsr=dsr,
                            status="promoted" if gate_pass else "tested")

                _print_result(name, oos_sharpe, pbo, dsr, gate_pass)
                console.print(f"    [dim]soglia OOS adattiva: {eff_min_oos:.3f} (N={n_tr} trial)[/dim]")

                # ── 5. Update scorecard (namespaced by universe) ──────────
                for sig in cfg_composed.signals:
                    update_score(sig.feature, sig.params,
                                 oos_sharpe=oos_sharpe, pbo=pbo, promoted=gate_pass,
                                 universe=cfg.top_n_universe)

                if gate_pass:
                    promoted.append(name)
                    console.print(f"  [bold green]★ PROMOTED: {name}[/bold green]")
                    if cfg_composed.gold_weight > 0:
                        console.print(
                            f"    [cyan]gold param:[/cyan] {cfg_composed.gold_weight:.0%} "
                            f"({cfg_composed.gold_mode})"
                        )
                    # Telegram notification
                    try:
                        from trading_bot.live.notifier import get_notifier
                        n = get_notifier()
                        if n:
                            n.notify_promotion(name, oos_sharpe, pbo, dsr)
                    except Exception:
                        pass
                    if len(promoted) >= cfg.target_promotions:
                        console.print(
                            f"\n[green]Target of {cfg.target_promotions} promotions reached![/green]"
                        )
                        _print_summary(promoted, total_rounds)
                        return promoted

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted — progress saved.[/yellow]")
                _print_summary(promoted, total_rounds)
                return promoted
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                logger.debug(traceback.format_exc())
                _update_log(log_entry["id"], status="error", skip_reason=str(e)[:200])

            round_tested += 1

            if stop_event is not None and stop_event.is_set():
                console.print("[yellow]Stop richiesto — trial corrente "
                              "persistito, uscita pulita.[/yellow]")
                _print_summary(promoted, total_rounds)
                return promoted

    _print_summary(promoted, total_rounds)
    return promoted


# ── Helpers ────────────────────────────────────────────────────────────────

def _ic_prescan_config(cfg_composed, panel: pd.DataFrame, bt_cfg: BacktestConfig) -> float:
    """Quick IC estimate on recent data only. Returns mean IC or 0 on failure."""
    try:
        strat = ComposedStrategy(cfg_composed)
        bt = CrossSectionalBacktester(strat, bt_cfg)
        result = bt.run(panel)
        if result.ic is not None and len(result.ic) >= 3:
            return float(result.ic.mean())
        if not result.rank_scores.empty and not result.forward_returns.empty:
            ic = information_coefficient(result.rank_scores, result.forward_returns)
            return float(ic.mean()) if len(ic) > 0 else 0.0
    except Exception:
        pass
    return 0.0


def _log_arm(round_id: int, arm, cfg_composed, universe_size: int | None = None) -> dict:
    """Persist arm to research_log before testing."""
    from trading_bot.registry import _config_to_json
    with get_session() as session:
        log = ResearchLog(
            round_id=round_id,
            hypothesis_name=cfg_composed.name,
            config_json=_config_to_json(cfg_composed),
            rationale=cfg_composed.rationale[:2000],
            status="pending",
            arm_fingerprint=arm.fingerprint,
            universe_size=universe_size,
        )
        session.add(log)
        session.commit()
        return {"id": log.id}


def _log_hypothesis_cfg(round_id: int, cfg_composed: ComposedConfig, archetype: str = "unknown",
                        universe_size: int | None = None) -> dict:
    """Log an archetype-generated config to research_log."""
    from trading_bot.registry import _config_to_json
    with get_session() as session:
        log = ResearchLog(
            round_id=round_id,
            hypothesis_name=cfg_composed.name,
            config_json=_config_to_json(cfg_composed),
            rationale=f"[{archetype}] " + cfg_composed.rationale[:1990],
            status="pending",
            arm_fingerprint=None,  # archetype arms don't have UCB1 fingerprints
            universe_size=universe_size,
        )
        session.add(log)
        session.commit()
        return {"id": log.id}


# Keep old name for backward compatibility
def _log_hypothesis(round_id: int, candidate) -> dict:
    from trading_bot.registry import _config_to_json
    with get_session() as session:
        log = ResearchLog(
            round_id=round_id,
            hypothesis_name=candidate.config.name,
            config_json=_config_to_json(candidate.config),
            rationale=candidate.config.rationale[:2000],
            status="pending",
        )
        session.add(log)
        session.commit()
        return {"id": log.id}


def _update_log(log_id: int, **kwargs) -> None:
    from sqlalchemy import select
    with get_session() as session:
        row = session.get(ResearchLog, log_id)
        if row:
            for k, v in kwargs.items():
                setattr(row, k, v)
            session.commit()


def _get_run_dsr(run_id: int) -> float:
    from trading_bot.registry import get_run
    try:
        return get_run(run_id).get("dsr", 0.0)
    except Exception:
        return 0.0


def _load_tested_fingerprints(universe: int | None = None) -> set[str]:
    """Load fingerprints of already-tested configs from research log.

    Prefers the stored arm_fingerprint (SearchArm format). Falls back to
    reconstructing from config_json for legacy rows.

    When ``universe`` is given, only fingerprints tested on that universe are
    returned — the same config on a different universe is a different
    experiment and should be re-tested.
    """
    from sqlalchemy import select
    fps: set[str] = set()
    try:
        with get_session() as session:
            query = select(ResearchLog)
            if universe is not None:
                query = query.where(ResearchLog.universe_size == universe)
            # Quarantined rows (tested on flawed PIT membership) must be re-tested
            query = query.where(ResearchLog.status.notlike("%_prepit"),
                                ResearchLog.status.notlike("%_enginev1"))
            rows = session.execute(query).scalars().all()
            for row in rows:
                # Fast path: stored arm fingerprint (new format)
                if row.arm_fingerprint:
                    fps.add(row.arm_fingerprint)
                    continue
                # Legacy path: reconstruct from config_json
                try:
                    from trading_bot.research.generator import _config_fingerprint
                    from trading_bot.strategies.composer import (
                        ComposedConfig, SignalSpec, FilterSpec, RegimeSpec
                    )
                    cfg_d = json.loads(row.config_json)
                    sigs = tuple(
                        SignalSpec(feature=s["feature"], params=s.get("params", {}),
                                   weight=s.get("weight", 1.0), use_rank=True,
                                   negate=s.get("negate", False))
                        for s in cfg_d.get("signals", [])
                    )
                    filts = tuple(
                        FilterSpec(feature=f["feature"], params=f.get("params", {}),
                                   threshold=f.get("threshold", 0.0))
                        for f in cfg_d.get("filters", [])
                    )
                    regime_d = cfg_d.get("regime")
                    regime = RegimeSpec(**regime_d) if regime_d else None
                    cfg = ComposedConfig(
                        name=cfg_d.get("name", ""),
                        rationale=cfg_d.get("rationale", "x" * 25),
                        signals=sigs, filters=filts, regime=regime,
                        top_n=cfg_d.get("top_n", 10),
                    )
                    fps.add(_config_fingerprint(cfg))
                except Exception:
                    pass
    except Exception:
        pass
    return fps


def _print_result(name: str, oos_sharpe: float, pbo: float, dsr: float, gate_pass: bool) -> None:
    color = "green" if gate_pass else "yellow" if pbo < 0.5 else "red"
    console.print(
        f"  [{color}]OOS Sharpe={oos_sharpe:.3f}  PBO={pbo:.3f}  DSR={dsr:.3f}  "
        f"{'★ PASS' if gate_pass else 'fail'}[/{color}]"
    )


def _print_summary(promoted: list[str], rounds: int) -> None:
    console.print(Panel(
        f"[bold]Research loop complete[/bold]\n"
        f"Rounds: {rounds}  |  Promoted strategies: {len(promoted)}\n"
        + ("\n".join(f"  ★ {n}" for n in promoted) if promoted else "  (none)"),
        title="Summary",
        border_style="green" if promoted else "yellow",
    ))

    scores = get_all_scores()
    if scores:
        table = Table(title="Feature scorecard (top 10)", show_header=True)
        table.add_column("Feature key", style="cyan")
        table.add_column("Used", justify="right")
        table.add_column("Promoted", justify="right")
        table.add_column("OOS Sharpe (avg)", justify="right")
        table.add_column("PBO (avg)", justify="right")
        table.add_column("Score", justify="right")
        for s in scores[:10]:
            table.add_row(
                s["key"][:50],
                str(s["times_used"]),
                str(s["times_promoted"]),
                f"{s['mean_oos_sharpe']:.3f}",
                f"{s['mean_pbo']:.3f}",
                f"{s['score']:.3f}",
            )
        console.print(table)
