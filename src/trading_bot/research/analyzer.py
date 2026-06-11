"""Analyzes existing strategy runs to extract insights for hypothesis generation.

For each completed run it extracts:
- Which features were used and at what parameters
- OOS Sharpe, PBO, DSR (from CPCV + registry)
- Whether the strategy was promoted
- IC stability (mean / std)

The output is a structured AnalysisReport used by the HypothesisGenerator
to set informed priors on the next round of search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class RunInsight:
    name: str
    features_used: list[str]
    feature_params: dict[str, dict]    # feature → params dict
    signal_weights: dict[str, float]
    top_n: int
    oos_sharpe: float
    pbo: float
    dsr: float
    ic_mean: float
    promoted: bool


@dataclass
class AnalysisReport:
    total_runs: int
    promoted_runs: list[RunInsight]
    failed_runs: list[RunInsight]       # pbo >= 0.5 or dsr < 0.5
    all_runs: list[RunInsight]

    # Aggregate insights
    best_features: list[str]            # sorted by avg OOS Sharpe when used
    worst_features: list[str]           # features that consistently underperform
    best_top_n: int                     # top_n value that appears most in good runs
    untested_combinations: list[tuple]  # feature pairs not yet explored

    @property
    def avg_oos_sharpe(self) -> float:
        oss = [r.oos_sharpe for r in self.all_runs if r.oos_sharpe == r.oos_sharpe]
        return sum(oss) / len(oss) if oss else 0.0

    @property
    def promotion_rate(self) -> float:
        return len(self.promoted_runs) / max(self.total_runs, 1)


def analyze_runs(runs: list[dict]) -> AnalysisReport:
    """Build an AnalysisReport from a list of registry run dicts."""
    from trading_bot.research.knowledge_base import FEATURES

    insights: list[RunInsight] = []

    for r in runs:
        # Parse config
        try:
            cfg = json.loads(r.get("metrics_json", "{}")) if "metrics_json" in r else r.get("metrics", {})
        except Exception:
            cfg = {}

        # Extract feature info from run config via registry
        features_used: list[str] = []
        feature_params: dict[str, dict] = {}
        signal_weights: dict[str, float] = {}
        top_n = 10

        # Try to get config from the strategy registry
        try:
            from trading_bot.registry import list_strategies
            strats = {s["name"]: s for s in list_strategies()}
            strat_cfg = strats.get(r["strategy_name"], {}).get("config", {})
            for sig in strat_cfg.get("signals", []):
                feat = sig.get("feature", "")
                if feat:
                    features_used.append(feat)
                    feature_params[feat] = sig.get("params", {})
                    signal_weights[feat] = sig.get("weight", 1.0)
            top_n = strat_cfg.get("top_n", 10)
        except Exception:
            pass

        # CPCV data
        cpcv = r.get("cpcv") or {}
        oos_sharpe = cpcv.get("mean_oos_sharpe", float("nan"))
        pbo = cpcv.get("pbo", 1.0)

        metrics = r.get("metrics", {})
        ic_mean = metrics.get("IC_mean", 0.0) or 0.0

        promoted = r.get("dsr", 0.0) >= 0.95 and pbo < 0.5

        insights.append(RunInsight(
            name=r["strategy_name"],
            features_used=features_used,
            feature_params=feature_params,
            signal_weights=signal_weights,
            top_n=top_n,
            oos_sharpe=oos_sharpe if oos_sharpe == oos_sharpe else 0.0,
            pbo=pbo,
            dsr=r.get("dsr", 0.0),
            ic_mean=ic_mean,
            promoted=promoted,
        ))

    promoted = [r for r in insights if r.promoted]
    failed = [r for r in insights if r.pbo >= 0.5 or r.dsr < 0.3]

    # Feature importance: avg OOS Sharpe when feature was present
    feature_sharpe: dict[str, list[float]] = {}
    for ins in insights:
        for feat in ins.features_used:
            feature_sharpe.setdefault(feat, []).append(ins.oos_sharpe)

    feat_avg = {
        f: sum(v) / len(v)
        for f, v in feature_sharpe.items() if v
    }
    best_features = sorted(feat_avg, key=lambda x: feat_avg[x], reverse=True)
    worst_features = sorted(feat_avg, key=lambda x: feat_avg[x])[:3]

    # Best top_n from promoted/good runs
    good_top_ns = [r.top_n for r in insights if r.oos_sharpe > 0.5 and r.top_n > 0]
    best_top_n = max(set(good_top_ns), key=good_top_ns.count) if good_top_ns else 10

    # Feature pairs not yet tested together
    all_features = list(FEATURES.keys())
    tested_pairs: set[frozenset] = set()
    for ins in insights:
        if len(ins.features_used) >= 2:
            for i, a in enumerate(ins.features_used):
                for b in ins.features_used[i + 1:]:
                    tested_pairs.add(frozenset([a, b]))
    untested = [
        (a, b)
        for i, a in enumerate(all_features)
        for b in all_features[i + 1:]
        if frozenset([a, b]) not in tested_pairs
    ]

    return AnalysisReport(
        total_runs=len(insights),
        promoted_runs=promoted,
        failed_runs=failed,
        all_runs=insights,
        best_features=best_features,
        worst_features=worst_features,
        best_top_n=best_top_n,
        untested_combinations=untested,
    )
