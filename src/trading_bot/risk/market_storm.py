"""Market Storm regime overlay.

The storm layer is a transparent risk overlay, not an alpha strategy. It turns
observable market stress into a 0-100 score, a regime label, and an exposure
scale that can be tested against existing strategies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.backtest.metrics import cagr, hit_rate, max_drawdown, sharpe


@dataclass(frozen=True)
class MarketStormConfig:
    vol_window_fast: int = 21
    vol_window_slow: int = 63
    drawdown_window: int = 252
    corr_window: int = 63
    breadth_sma: int = 200
    dispersion_window: int = 21
    max_corr_assets: int = 80
    output_dir: str = "data/market_storm_reports"
    exposure_by_regime: dict[str, float] | None = None

    def exposures(self) -> dict[str, float]:
        return self.exposure_by_regime or {
            "calm": 1.00,
            "unstable": 0.75,
            "storm": 0.50,
            "panic": 0.25,
        }


@dataclass(frozen=True)
class StormPoint:
    dt: str
    storm_score: float
    regime: str
    recommended_action: str
    exposure_scale: float
    components: dict[str, float]
    rationale: str


@dataclass(frozen=True)
class OverlayComparison:
    base_metrics: dict[str, float]
    overlay_metrics: dict[str, float]
    deltas: dict[str, float]
    worst_months: list[dict[str, Any]]
    passed: bool
    gates: dict[str, bool]


@dataclass(frozen=True)
class StormReport:
    strategy_name: str
    generated_at: str
    current: StormPoint
    history_tail: list[StormPoint]
    overlay: OverlayComparison | None
    decision: str
    notes: list[str]


def score_to_regime(score: float) -> str:
    if score < 25:
        return "calm"
    if score < 50:
        return "unstable"
    if score < 75:
        return "storm"
    return "panic"


def regime_to_action(regime: str) -> str:
    return {
        "calm": "normal",
        "unstable": "reduce_exposure",
        "storm": "defensive_only",
        "panic": "block_new_entries",
    }.get(regime, "reduce_exposure")


def compute_storm_history(
    prices: pd.DataFrame,
    config: MarketStormConfig | None = None,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute daily storm score history using only data up to each date."""
    cfg = config or MarketStormConfig()
    px = prices.sort_index().ffill()
    if "SPY" not in px.columns:
        raise ValueError("Market Storm requires SPY in the price panel")

    returns = px.pct_change().fillna(0.0)
    spy_ret = returns["SPY"]
    spy = px["SPY"].dropna()

    vol_fast = spy_ret.rolling(cfg.vol_window_fast).std() * np.sqrt(252)
    vol_slow = spy_ret.rolling(cfg.vol_window_slow).std() * np.sqrt(252)
    vol_score = _score_between(vol_fast, low=0.12, high=0.40)
    vol_spike_score = _score_between(vol_fast / vol_slow.replace(0, np.nan), low=1.0, high=2.2)

    rolling_peak = spy.rolling(cfg.drawdown_window, min_periods=30).max()
    drawdown = (spy / rolling_peak - 1.0).fillna(0.0)
    drawdown_score = _score_between(drawdown.abs(), low=0.03, high=0.25)

    tradable = px.drop(columns=[c for c in ("SPY", "GLD", "HYG") if c in px.columns], errors="ignore")
    tradable = tradable.dropna(how="all", axis=1)
    if tradable.shape[1] > cfg.max_corr_assets:
        # Stable deterministic subset: most complete symbols first.
        coverage = tradable.notna().sum().sort_values(ascending=False)
        tradable = tradable[coverage.head(cfg.max_corr_assets).index]
    tradable_ret = tradable.pct_change().fillna(0.0)

    corr_score = _rolling_average_corr_score(tradable_ret, cfg.corr_window)

    xsec_dispersion = tradable_ret.std(axis=1).rolling(cfg.dispersion_window).mean() * np.sqrt(252)
    dispersion_score = _score_between(xsec_dispersion, low=0.12, high=0.35)

    above_sma = tradable > tradable.rolling(cfg.breadth_sma, min_periods=50).mean()
    breadth = above_sma.sum(axis=1) / above_sma.count(axis=1).replace(0, np.nan)
    breadth_score = _score_between(1.0 - breadth, low=0.35, high=0.75)

    downside_breadth = (tradable_ret < 0).sum(axis=1) / tradable_ret.count(axis=1).replace(0, np.nan)
    downside_score = _score_between(downside_breadth.rolling(5).mean(), low=0.55, high=0.80)

    credit_score = pd.Series(0.0, index=px.index)
    if "HYG" in px.columns:
        hyg = px["HYG"].dropna()
        hyg_peak = hyg.rolling(cfg.drawdown_window, min_periods=30).max()
        hyg_dd = (hyg / hyg_peak - 1.0).abs()
        credit_score = _score_between(hyg_dd.reindex(px.index).ffill(), low=0.02, high=0.12)

    sector_score = _sector_stress_score(tradable_ret, sector_map, cfg.vol_window_fast)

    components = pd.DataFrame(
        {
            "spy_vol": vol_score,
            "vol_spike": vol_spike_score,
            "spy_drawdown": drawdown_score,
            "correlation": corr_score,
            "breadth": breadth_score,
            "downside_breadth": downside_score,
            "dispersion": dispersion_score,
            "sector_stress": sector_score,
            "credit": credit_score,
        },
        index=px.index,
    ).fillna(0.0)

    weights = {
        "spy_vol": 0.18,
        "vol_spike": 0.08,
        "spy_drawdown": 0.18,
        "correlation": 0.16,
        "breadth": 0.16,
        "downside_breadth": 0.08,
        "dispersion": 0.10,
        "sector_stress": 0.04,
        "credit": 0.02,
    }
    score = sum(components[k] * w for k, w in weights.items()).clip(0, 100)
    out = components.copy()
    out["storm_score"] = score
    out["regime"] = [score_to_regime(float(x)) for x in score]
    exposures = cfg.exposures()
    out["recommended_action"] = [regime_to_action(r) for r in out["regime"]]
    out["exposure_scale"] = [float(exposures.get(r, 0.5)) for r in out["regime"]]
    return out


def current_storm_point(history: pd.DataFrame) -> StormPoint:
    if history.empty:
        raise ValueError("Storm history is empty")
    row = history.dropna(subset=["storm_score"]).iloc[-1]
    components = {
        k: float(row[k])
        for k in history.columns
        if k not in {"storm_score", "regime", "recommended_action", "exposure_scale"}
    }
    top = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:3]
    rationale = "Top stress: " + ", ".join(f"{k}={v:.0f}" for k, v in top)
    return StormPoint(
        dt=row.name.date().isoformat() if hasattr(row.name, "date") else str(row.name),
        storm_score=float(row["storm_score"]),
        regime=str(row["regime"]),
        recommended_action=str(row["recommended_action"]),
        exposure_scale=float(row["exposure_scale"]),
        components=components,
        rationale=rationale,
    )


def apply_storm_overlay(
    returns: pd.Series,
    storm_history: pd.DataFrame,
    lag_days: int = 1,
) -> pd.Series:
    """Scale strategy returns by the prior known storm exposure.

    ``lag_days=1`` enforces no lookahead: today's strategy return is scaled by
    yesterday's regime/exposure.
    """
    scale = (
        storm_history["exposure_scale"]
        .reindex(returns.index)
        .ffill()
        .shift(lag_days)
        .fillna(1.0)
    )
    return returns.fillna(0.0) * scale


def compare_overlay(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    storm_history: pd.DataFrame,
    cagr_tolerance: float = 0.35,
) -> OverlayComparison:
    base = strategy_returns.dropna().astype(float)
    overlay = apply_storm_overlay(base, storm_history)
    bench = benchmark_returns.reindex(base.index).fillna(0.0)

    base_m = _return_metrics(base)
    over_m = _return_metrics(overlay)
    deltas = {k: over_m[k] - base_m[k] for k in base_m if k in over_m}
    worst = _worst_month_table(base, overlay, bench)

    base_dd = abs(base_m["MaxDrawdown"])
    over_dd = abs(over_m["MaxDrawdown"])
    cagr_floor = base_m["CAGR"] * (1.0 - cagr_tolerance) if base_m["CAGR"] > 0 else base_m["CAGR"]
    gates = {
        "max_drawdown_not_worse": over_dd <= base_dd + 1e-9,
        "worst_month_improved": over_m["WorstMonth"] >= base_m["WorstMonth"] - 1e-9,
        "cagr_not_destroyed": over_m["CAGR"] >= cagr_floor,
        "volatility_not_higher": over_m["Volatility"] <= base_m["Volatility"] + 1e-9,
    }
    return OverlayComparison(
        base_metrics=base_m,
        overlay_metrics=over_m,
        deltas=deltas,
        worst_months=worst,
        passed=all(gates.values()),
        gates=gates,
    )


def build_storm_report(
    strategy_name: str,
    prices: pd.DataFrame,
    strategy_returns: pd.Series | None = None,
    config: MarketStormConfig | None = None,
    sector_map: dict[str, str] | None = None,
) -> tuple[StormReport, pd.DataFrame]:
    cfg = config or MarketStormConfig()
    history = compute_storm_history(prices, cfg, sector_map=sector_map)
    current = current_storm_point(history)
    overlay = None
    if strategy_returns is not None:
        bench = prices["SPY"].pct_change().fillna(0.0)
        overlay = compare_overlay(strategy_returns, bench, history)
    decision = "overlay_candidate" if overlay is not None and overlay.passed else "research_only"
    notes = [
        "Market Storm is a risk overlay, not an alpha strategy.",
        "Exposure uses prior-day regime to avoid lookahead.",
    ]
    if overlay is not None and not overlay.passed:
        notes.append("Overlay gates did not pass; keep in research/paper-review.")
    tail = [
        _row_to_point(dt, row)
        for dt, row in history.tail(252).iterrows()
    ]
    return (
        StormReport(
            strategy_name=strategy_name,
            generated_at=pd.Timestamp.now("UTC").isoformat(),
            current=current,
            history_tail=tail,
            overlay=overlay,
            decision=decision,
            notes=notes,
        ),
        history,
    )


def save_storm_report(report: StormReport, output_dir: str | Path) -> Path:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{stamp}_{report.strategy_name}_storm_report.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def load_storm_reports(report_dir: str | Path = "data/market_storm_reports") -> list[dict[str, Any]]:
    path = Path(report_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(path.glob("*_storm_report.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["_path"] = str(p)
            out.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _score_between(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series - low) / (high - low) * 100.0).clip(0.0, 100.0)


def _rolling_average_corr_score(returns: pd.DataFrame, window: int) -> pd.Series:
    if returns.shape[1] < 3:
        return pd.Series(0.0, index=returns.index)
    values: list[float] = []
    arr_index = returns.index
    for i in range(len(returns)):
        if i + 1 < window:
            values.append(np.nan)
            continue
        sub = returns.iloc[i + 1 - window : i + 1].dropna(axis=1, how="any")
        if sub.shape[1] < 3:
            values.append(np.nan)
            continue
        corr = sub.corr().to_numpy()
        tri = corr[np.triu_indices_from(corr, k=1)]
        avg = float(np.nanmean(tri)) if len(tri) else np.nan
        values.append(avg)
    avg_corr = pd.Series(values, index=arr_index)
    return _score_between(avg_corr, low=0.25, high=0.75)


def _sector_stress_score(
    returns: pd.DataFrame,
    sector_map: dict[str, str] | None,
    window: int,
) -> pd.Series:
    if not sector_map:
        return pd.Series(0.0, index=returns.index)
    sector_returns = {}
    for sector in sorted(set(sector_map.values())):
        cols = [s for s, sec in sector_map.items() if sec == sector and s in returns.columns]
        if len(cols) >= 2:
            sector_returns[sector] = returns[cols].mean(axis=1)
    if not sector_returns:
        return pd.Series(0.0, index=returns.index)
    sec = pd.DataFrame(sector_returns)
    sec_vol = sec.rolling(window).std() * np.sqrt(252)
    stressed = (sec_vol > 0.30).sum(axis=1) / sec_vol.count(axis=1).replace(0, np.nan)
    return _score_between(stressed, low=0.20, high=0.70).fillna(0.0)


def _return_metrics(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna().astype(float)
    eq = (1.0 + r).cumprod()
    monthly = r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    return {
        "CAGR": cagr(eq),
        "Sharpe": sharpe(r),
        "MaxDrawdown": max_drawdown(eq),
        "WorstMonth": float(monthly.min()) if len(monthly) else 0.0,
        "Volatility": float(r.std() * np.sqrt(252)) if len(r) else 0.0,
        "HitRate": hit_rate(r),
    }


def _worst_month_table(
    base: pd.Series,
    overlay: pd.Series,
    benchmark: pd.Series,
    n: int = 10,
) -> list[dict[str, Any]]:
    df = pd.DataFrame({"base": base, "overlay": overlay, "benchmark": benchmark}).fillna(0.0)
    monthly = df.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    if monthly.empty:
        return []
    worst_idx = monthly["benchmark"].sort_values().head(n).index
    rows = []
    for dt in worst_idx:
        rows.append({
            "month": dt.strftime("%Y-%m"),
            "benchmark": float(monthly.loc[dt, "benchmark"]),
            "base": float(monthly.loc[dt, "base"]),
            "overlay": float(monthly.loc[dt, "overlay"]),
            "overlay_helped": bool(monthly.loc[dt, "overlay"] >= monthly.loc[dt, "base"]),
        })
    return rows


def _row_to_point(dt: pd.Timestamp, row: pd.Series) -> StormPoint:
    components = {
        k: float(row[k])
        for k in row.index
        if k not in {"storm_score", "regime", "recommended_action", "exposure_scale"}
    }
    top = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return StormPoint(
        dt=dt.date().isoformat(),
        storm_score=float(row["storm_score"]),
        regime=str(row["regime"]),
        recommended_action=str(row["recommended_action"]),
        exposure_scale=float(row["exposure_scale"]),
        components=components,
        rationale="Top stress: " + ", ".join(f"{k}={v:.0f}" for k, v in top),
    )
