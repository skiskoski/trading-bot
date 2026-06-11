"""Hypothesis generator for autonomous strategy research.

Generates new ComposedConfig candidates using:
1. Bayesian scorecard weights (features that worked before get higher priority)
2. Knowledge base synergies (known good/bad combinations from literature)
3. Novelty constraint (never re-test an already-tested config)
4. Economic rationale (every hypothesis has a paper citation)

The generator is stateless — all state lives in the DB via the scorecard.
Calling generate() multiple times with the same report will produce different
configurations because the random seed is based on round_id.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass

from trading_bot.research.knowledge_base import (
    FEATURES,
    FILTERS,
    REGIMES,
    FeatureSpec,
    build_rationale,
    get_synergy,
)
from trading_bot.research.scorecard import get_score
from trading_bot.strategies.composer import (
    ComposedConfig,
    FilterSpec,
    RegimeSpec,
    SignalSpec,
)


@dataclass
class Candidate:
    config: ComposedConfig
    expected_score: float   # prior estimate before testing
    novelty: float          # 0–1, how different from existing runs
    economic_strength: float  # 0–1, strength of academic backing


def _config_fingerprint(cfg: ComposedConfig) -> str:
    """Deterministic hash of a config for deduplication."""
    d = {
        "signals": sorted([
            (s.feature, json.dumps(s.params, sort_keys=True), s.weight, s.negate)
            for s in cfg.signals
        ]),
        "filters": sorted([
            (f.feature, json.dumps(f.params, sort_keys=True))
            for f in cfg.filters
        ]),
        "regime": (cfg.regime.symbol if cfg.regime else None),
        "top_n": cfg.top_n,
    }
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def generate(
    report,            # AnalysisReport from analyzer.py
    round_id: int,
    n_candidates: int = 20,
    already_tested: set[str] | None = None,
) -> list[Candidate]:
    """Generate up to n_candidates novel ComposedConfig hypotheses.

    Uses:
    - Scorecard scores as sampling weights
    - Synergy matrix from knowledge base
    - Novelty check against already_tested fingerprints
    - Varied combination strategies (single, dual, triple signal)
    """
    rng = random.Random(round_id * 31337 + 42)
    tested_fp = already_tested or set()
    candidates: list[Candidate] = []

    feat_names = list(FEATURES.keys())

    # ── Strategy 1: Single-signal with varied parameters ───────────────────
    for feat_name, spec in FEATURES.items():
        for params in _iter_param_combos(spec.param_grid, rng, max_combos=3):
            score = get_score(feat_name, params)
            for top_n in _sample_top_n(report, rng):
                for filter_name, filter_params in _sample_filter(rng):
                    for regime_key, regime_cfg in _sample_regime(rng):
                        cfg = _build_config(
                            signals=[(feat_name, params, 1.0, False)],
                            filter_name=filter_name,
                            filter_params=filter_params,
                            regime=regime_cfg,
                            top_n=top_n,
                            round_id=round_id,
                            counter=len(candidates),
                        )
                        fp = _config_fingerprint(cfg)
                        if fp in tested_fp:
                            continue
                        tested_fp.add(fp)
                        candidates.append(Candidate(
                            config=cfg,
                            expected_score=score,
                            novelty=_novelty(feat_name, params, report),
                            economic_strength=_econ_strength([feat_name], []),
                        ))
                        if len(candidates) >= n_candidates * 2:
                            break

    # ── Strategy 2: Dual-signal (scored by synergy + individual scores) ────
    for a, b in itertools.combinations(feat_names, 2):
        synergy = get_synergy(a, b)
        if synergy < -0.1:
            continue  # skip known anti-synergistic pairs
        for params_a in _iter_param_combos(FEATURES[a].param_grid, rng, max_combos=2):
            for params_b in _iter_param_combos(FEATURES[b].param_grid, rng, max_combos=2):
                score_a = get_score(a, params_a)
                score_b = get_score(b, params_b)
                combined_score = (score_a + score_b) / 2.0 + synergy * 0.3
                # Vary weights: 50/50, 60/40, 70/30
                for wa, wb in [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]:
                    for top_n in _sample_top_n(report, rng, n=1):
                        filter_name, filter_params = rng.choice(
                            list(_sample_filter(rng, n=1))
                        )
                        _, regime_cfg = rng.choice(list(_sample_regime(rng, n=1)))
                        cfg = _build_config(
                            signals=[(a, params_a, wa, False), (b, params_b, wb, False)],
                            filter_name=filter_name,
                            filter_params=filter_params,
                            regime=regime_cfg,
                            top_n=top_n,
                            round_id=round_id,
                            counter=len(candidates),
                        )
                        fp = _config_fingerprint(cfg)
                        if fp in tested_fp:
                            continue
                        tested_fp.add(fp)
                        candidates.append(Candidate(
                            config=cfg,
                            expected_score=combined_score,
                            novelty=_novelty_pair(a, b, report),
                            economic_strength=_econ_strength([a, b], []),
                        ))
                        if len(candidates) >= n_candidates * 3:
                            break

    # ── Strategy 3: Exploit untested combinations flagged by analyzer ──────
    for feat_a, feat_b in (report.untested_combinations or [])[:5]:
        synergy = get_synergy(feat_a, feat_b)
        params_a = _best_params(feat_a, report)
        params_b = _best_params(feat_b, report)
        _, regime_cfg = next(_sample_regime(rng, n=1))
        filter_name, filter_params = next(_sample_filter(rng, n=1))
        cfg = _build_config(
            signals=[(feat_a, params_a, 0.5, False), (feat_b, params_b, 0.5, False)],
            filter_name=filter_name,
            filter_params=filter_params,
            regime=regime_cfg,
            top_n=report.best_top_n or 10,
            round_id=round_id,
            counter=len(candidates),
        )
        fp = _config_fingerprint(cfg)
        if fp not in tested_fp:
            tested_fp.add(fp)
            candidates.append(Candidate(
                config=cfg,
                expected_score=0.5 + synergy,
                novelty=1.0,
                economic_strength=_econ_strength([feat_a, feat_b], []),
            ))

    # ── Rank by composite priority score and return top N ──────────────────
    candidates.sort(
        key=lambda c: c.expected_score * 0.5 + c.novelty * 0.3 + c.economic_strength * 0.2,
        reverse=True,
    )
    return candidates[:n_candidates]


# ── Helpers ────────────────────────────────────────────────────────────────

def _iter_param_combos(
    param_grid: dict[str, list], rng: random.Random, max_combos: int = 3
) -> list[dict]:
    keys = list(param_grid.keys())
    if not keys:
        return [{}]
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
    rng.shuffle(all_combos)
    return [dict(zip(keys, c)) for c in all_combos[:max_combos]]


def _sample_top_n(report, rng: random.Random, n: int = 2) -> list[int]:
    options = [5, 8, 10, 12, 15, 20]
    # Bias towards best_top_n seen in good runs
    weights = [2.0 if o == report.best_top_n else 1.0 for o in options]
    return rng.choices(options, weights=weights, k=n)


def _sample_filter(rng: random.Random, n: int = 2):
    import itertools as _it
    items = []
    for fname, finfo in FILTERS.items():
        grid = finfo["param_grid"]
        if not grid:
            items.append((fname, {}))    # "none" filter option
            continue
        keys = list(grid.keys())
        for vals in _it.product(*[grid[k] for k in keys]):
            items.append((fname, dict(zip(keys, vals))))
    rng.shuffle(items)
    for item in items[:n]:
        yield item


def _sample_regime(rng: random.Random, n: int = 2):
    items = list(REGIMES.items())
    rng.shuffle(items)
    for key, rcfg in items[:n]:
        yield key, rcfg


def _build_config(
    signals: list[tuple[str, dict, float, bool]],
    filter_name: str,
    filter_params: dict,
    regime: dict | None,
    top_n: int,
    round_id: int,
    counter: int,
) -> ComposedConfig:
    sig_specs = tuple(
        SignalSpec(feature=feat, params=params, weight=w, use_rank=True, negate=neg)
        for feat, params, w, neg in signals
    )
    filt_specs = (FilterSpec(feature=filter_name, params=filter_params, threshold=0.5),)
    regime_spec = None
    if regime:
        regime_spec = RegimeSpec(
            symbol=regime["symbol"],
            feature=regime["feature"],
            params=regime["params"],
            threshold=regime.get("threshold", 0.0),
        )

    # Build rationale from knowledge base
    rationale = build_rationale(
        signals=[(s[0], s[1]) for s in signals],
        filters=[(filter_name, filter_params)],
        regime=(
            {"symbol": regime["symbol"], "feature": regime["feature"],
             "params": regime["params"], "threshold": regime.get("threshold", 0.0)}
            if regime else None
        ),
        top_n=top_n,
    )

    # Unique name: rXX_c{counter}_{features}
    feat_abbr = "_".join(s[0][:3] for s in signals)
    name = f"r{round_id:02d}_c{counter:03d}_{feat_abbr}_top{top_n}"

    return ComposedConfig(
        name=name,
        rationale=rationale,
        signals=sig_specs,
        filters=filt_specs,
        regime=regime_spec,
        top_n=top_n,
    )


def _novelty(feat: str, params: dict, report) -> float:
    """1.0 if feature never used, decreases with usage."""
    used = sum(1 for r in report.all_runs if feat in r.features_used)
    return 1.0 / (1.0 + used)


def _novelty_pair(a: str, b: str, report) -> float:
    used = sum(
        1 for r in report.all_runs
        if a in r.features_used and b in r.features_used
    )
    return 1.0 / (1.0 + used)


def _econ_strength(features: list[str], _filters: list) -> float:
    """Score based on number of paper citations in the knowledge base."""
    score = 0.0
    for f in features:
        spec = FEATURES.get(f)
        if spec and spec.citation:
            score += 0.5
    return min(score, 1.0)


def _best_params(feat: str, report) -> dict:
    """Return the param set that produced best OOS Sharpe for this feature."""
    best_sharpe = -999.0
    best_params: dict = {}
    for run in report.all_runs:
        if feat in run.features_used:
            if run.oos_sharpe > best_sharpe:
                best_sharpe = run.oos_sharpe
                best_params = run.feature_params.get(feat, {})
    if not best_params:
        spec = FEATURES.get(feat)
        if spec:
            best_params = {k: v[0] for k, v in spec.param_grid.items()}
    return best_params
