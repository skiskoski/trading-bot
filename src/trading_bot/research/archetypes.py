"""Strategy archetypes — structural blueprints for strategy generation.

An archetype defines HOW a strategy is built, not just which features it uses.
The generator picks an archetype first, then instantiates it with specific
features and parameters from the knowledge base.

This ensures structural diversity: the bot explores fundamentally different
strategy types, not just parameter permutations of the same idea.

Archetypes:
    pure_factor       — single or multi-signal cross-sectional ranking (existing)
    defensive_quality — quality pre-filter + defensive signals; works in crashes
    regime_adaptive   — different signals in bull vs bear market
    factor_blend      — 2-3 signals from DIFFERENT families + vol-parity weights
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field

from trading_bot.research.knowledge_base import (
    FEATURES,
    FILTERS,
    REGIMES,
    FeatureSpec,
    get_synergy,
)
from trading_bot.strategies.composer import (
    ComposedConfig,
    FilterSpec,
    RegimeSpec,
    SignalSpec,
)


# ── Feature family groupings ──────────────────────────────────────────────────

FAMILY_MAP: dict[str, list[str]] = {}
for fname, fspec in FEATURES.items():
    FAMILY_MAP.setdefault(fspec.family, []).append(fname)

TREND_FEATURES    = FAMILY_MAP.get("momentum", []) + FAMILY_MAP.get("structural", [])
REVERSAL_FEATURES = FAMILY_MAP.get("reversal", [])
RISK_FEATURES     = FAMILY_MAP.get("risk", [])
QUALITY_FEATURES  = FAMILY_MAP.get("quality", [])

# Features known to work well as defensive signals
DEFENSIVE_SIGNALS = ["volatility", "quality_score", "trend_quality",
                     "downside_vol", "drawdown", "return_entropy",
                     "amihud_illiquidity"]
TREND_SIGNALS     = ["momentum", "ema_ratio", "rate_of_change",
                     "relative_strength", "kalman_trend", "hurst_exponent",
                     "volume_momentum"]


@dataclass
class ArchetypeResult:
    """A fully-specified strategy config generated from an archetype."""
    config: ComposedConfig
    archetype: str
    description: str


# ── Archetype generators ───────────────────────────────────────────────────────

def _random_params(feat: str) -> dict:
    """Pick a random valid parameter set for a feature."""
    spec = FEATURES.get(feat)
    if not spec or not spec.param_grid:
        return {}
    return {k: random.choice(v) for k, v in spec.param_grid.items()}


def _random_filters() -> tuple[FilterSpec, ...]:
    """Filtro come grado di libertà: trend, quality o NESSUNO."""
    fname = random.choice(list(FILTERS.keys()))
    finfo = FILTERS[fname]
    if not finfo["param_grid"]:          # "none"
        return ()
    params = {k: random.choice(v) for k, v in finfo["param_grid"].items()}
    return (FilterSpec(feature=fname, params=params,
                       threshold=finfo.get("threshold", 0.5)),)


def _random_regime(prefer_bull: bool = True) -> RegimeSpec | None:
    """Pick a regime spec. Returns None (~20% of time) to test without regime."""
    if random.random() < 0.15:
        return None   # no regime filter
    key = random.choice([k for k in REGIMES if k != "no_regime"])
    cfg = REGIMES[key]
    if cfg is None:
        return None
    return RegimeSpec(
        symbol=cfg["symbol"],
        feature=cfg["feature"],
        params=cfg["params"],
        threshold=cfg.get("threshold", 0.0),
    )


def _top_n() -> int:
    return random.choice([5, 8, 10, 12, 15, 20])


def _gold_params(defensive_bias: bool = False) -> dict:
    """Gold sleeve come parametro: peso esplorato, modalità quasi sempre
    'defensive' (oro in portafoglio SOLO quando il rischio drawdown è alto)."""
    if defensive_bias:
        w = random.choice([0.0, 0.1, 0.2, 0.3])
    else:
        w = random.choice([0.0, 0.0, 0.0, 0.1, 0.2])
    mode = random.choice(["defensive", "defensive", "defensive", "always"])
    return {"gold_weight": w, "gold_mode": mode}


# ── Archetype 1: Pure factor (enhanced version of existing) ──────────────────

def generate_pure_factor(name: str, n_signals: int = 1) -> ArchetypeResult:
    """Single or multi-signal ranking. Baseline archetype."""
    # Pick n_signals features, preferring those with known synergies
    all_feats = [f for f in FEATURES if f not in ("tsmom_signal", "max_gap")]
    chosen = random.sample(all_feats, min(n_signals, len(all_feats)))

    signals = tuple(
        SignalSpec(
            feature=f,
            params=_random_params(f),
            weight=round(1.0 / n_signals, 2),
            use_rank=True,
            negate=FEATURES[f].negate_default,
        )
        for f in chosen
    )
    regime = _random_regime()
    top_n = _top_n()

    cfg = ComposedConfig(
        name=name,
        rationale=(
            f"Pure factor: {[s.feature for s in signals]}. "
            f"Cross-sectional ranking, equal weight, top-{top_n}. "
            f"Regime: {regime.feature if regime else 'none'}."
        ),
        signals=signals,
        filters=_random_filters(),
        regime=regime,
        top_n=top_n,
        **_gold_params(),
    )
    return ArchetypeResult(cfg, "pure_factor",
                           f"Pure {n_signals}-signal: {[s.feature for s in signals]}")


# ── Archetype 2: Defensive quality ───────────────────────────────────────────

def generate_defensive_quality(name: str) -> ArchetypeResult:
    """Quality pre-filter + defensive signal.

    Logic:
    - MANDATORY quality filter: only trade stocks above median quality_score
    - Primary signal: low-vol, trend_quality, or drawdown (defensive)
    - Optional secondary: momentum (trend overlay)
    - Smaller top_n (5-8) — concentration in best-quality names
    - SPY SMA200 regime — go cash if market is bearish

    Works when: market stress, high volatility, momentum crashes (2022)
    Fails when: strong momentum rallies where low-quality stocks lead
    """
    # Primary defensive signal
    primary = random.choice([f for f in DEFENSIVE_SIGNALS if f in FEATURES])
    primary_params = _random_params(primary)

    signals_list = [
        SignalSpec(feature=primary, params=primary_params,
                   weight=0.7, use_rank=True,
                   negate=FEATURES[primary].negate_default)
    ]

    # 50% chance: add momentum overlay
    if random.random() < 0.5:
        sec = random.choice([f for f in TREND_SIGNALS if f in FEATURES])
        signals_list.append(
            SignalSpec(feature=sec, params=_random_params(sec),
                       weight=0.3, use_rank=True,
                       negate=FEATURES[sec].negate_default)
        )

    # Mandatory quality filter
    quality_filter = FilterSpec(
        feature="quality_score",
        params={"lookback": random.choice([126, 252])},
        threshold=0.3,   # only stocks in top 70% quality
    )
    # Secondary filter (may be empty — filter is a degree of freedom)
    extra_filters = _random_filters()

    top_n = random.choice([5, 8, 10])   # concentrated

    regime = RegimeSpec(
        symbol="SPY", feature="sma_distance",
        params={"window": 200}, threshold=0.0
    )

    cfg = ComposedConfig(
        name=name,
        rationale=(
            f"Defensive quality archetype. Primary: {primary}. "
            f"Quality pre-filter active (top 70% by quality_score). "
            f"Concentrated top-{top_n}. Regime: SPY SMA200. "
            f"Literature: Novy-Marx QMJ (2013), Frazzini-Pedersen BAB (2014). "
            f"Designed to preserve capital during market stress."
        ),
        signals=tuple(signals_list),
        filters=(quality_filter, *extra_filters),
        regime=regime,
        top_n=top_n,
        **_gold_params(defensive_bias=True),
    )
    return ArchetypeResult(cfg, "defensive_quality",
                           f"Defensive: {primary} + quality filter")


# ── Archetype 3: Regime adaptive ─────────────────────────────────────────────

def generate_regime_adaptive(name: str) -> ArchetypeResult:
    """Different signal weights in bull vs bear regime.

    Approximation using ComposedConfig: in bull regime (SPY > SMA200),
    the primary signal is a trend feature. In bear regime, the strategy
    is more defensive by adding a strong quality filter threshold.

    True regime-switching requires RegimeAdaptiveStrategy (future work).
    Here we build the closest approximation within ComposedConfig:
    - Blend trend signal (0.5) + defensive signal (0.5)
    - Strong quality filter
    - SPY SMA regime gate (all-cash if very bearish)

    This naturally behaves better than pure momentum in mixed regimes.
    Literature: Asness (2014) "Our Model Goes to Six and Saves Value",
    Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"
    """
    trend_feat    = random.choice([f for f in TREND_SIGNALS if f in FEATURES])
    defense_feat  = random.choice([f for f in DEFENSIVE_SIGNALS if f in FEATURES])

    signals = (
        SignalSpec(feature=trend_feat, params=_random_params(trend_feat),
                   weight=0.55, use_rank=True,
                   negate=FEATURES[trend_feat].negate_default),
        SignalSpec(feature=defense_feat, params=_random_params(defense_feat),
                   weight=0.45, use_rank=True,
                   negate=FEATURES[defense_feat].negate_default),
    )

    # Quality filter — moderately strict
    quality_filter = FilterSpec(
        feature="quality_score",
        params={"lookback": 252},
        threshold=0.2,
    )

    top_n = random.choice([8, 10, 12])
    regime = RegimeSpec(
        symbol="SPY", feature="sma_distance",
        params={"window": random.choice([100, 200])},
        threshold=0.0,
    )

    cfg = ComposedConfig(
        name=name,
        rationale=(
            f"Regime-adaptive blend: {trend_feat} (55%) + {defense_feat} (45%). "
            f"Trend signal leads in bull markets; defensive signal provides "
            f"stability in drawdowns. Quality filter removes fragile stocks. "
            f"Regime gate: all-cash when SPY below SMA. "
            f"Literature: Asness (2014), Faber (2007) TAA."
        ),
        signals=signals,
        filters=(quality_filter,),
        regime=regime,
        top_n=top_n,
        **_gold_params(defensive_bias=True),
    )
    return ArchetypeResult(cfg, "regime_adaptive",
                           f"Adaptive: {trend_feat} + {defense_feat}")


# ── Archetype 4: Factor blend (cross-family) ─────────────────────────────────

def generate_factor_blend(name: str) -> ArchetypeResult:
    """2-3 signals from DIFFERENT feature families.

    Diversification across factor types reduces family-specific risk.
    E.g. momentum + low_vol + quality_score spans three independent
    return premia (Jegadeesh/Titman, Frazzini/Pedersen, Novy-Marx).

    Weight allocation: largest to highest-scoring feature per UCB1.
    Literature: Blitz-van Vliet (2007), Asness-Frazzini-Pedersen (2015 AQR).
    """
    families = list(FAMILY_MAP.keys())
    random.shuffle(families)

    chosen_feats = []
    chosen_families = []
    for fam in families:
        candidates = [f for f in FAMILY_MAP[fam]
                      if f in FEATURES and f not in ("tsmom_signal", "max_gap")]
        if candidates:
            f = random.choice(candidates)
            # Check synergy with already chosen
            ok = all(get_synergy(f, cf) >= -0.05 for cf in chosen_feats)
            if ok:
                chosen_feats.append(f)
                chosen_families.append(fam)
        if len(chosen_feats) >= random.choice([2, 2, 3, 4]):
            break

    if len(chosen_feats) < 2:
        # Fallback: just pick 2 random features
        chosen_feats = random.sample(
            [f for f in FEATURES if f not in ("tsmom_signal", "max_gap")], 2
        )

    n = len(chosen_feats)
    weights = [0.4, 0.3, 0.2, 0.1][:n]
    # Normalize
    total = sum(weights[:n])
    weights = [round(w / total, 2) for w in weights[:n]]
    # Fix rounding
    weights[0] += round(1.0 - sum(weights), 2)

    signals = tuple(
        SignalSpec(feature=f, params=_random_params(f),
                   weight=w, use_rank=True,
                   negate=FEATURES[f].negate_default)
        for f, w in zip(chosen_feats, weights)
    )

    top_n = random.choice([8, 10, 12, 15])
    regime = _random_regime()

    cfg = ComposedConfig(
        name=name,
        rationale=(
            f"Cross-family factor blend: {chosen_feats} from families {chosen_families}. "
            f"Weights: {dict(zip(chosen_feats, weights))}. "
            f"Diversification across return premia reduces correlated drawdowns. "
            f"Literature: Blitz-van Vliet (2007), Asness-Frazzini-Pedersen (2015)."
        ),
        signals=signals,
        filters=_random_filters(),
        regime=regime,
        top_n=top_n,
        **_gold_params(),
    )
    return ArchetypeResult(cfg, "factor_blend",
                           f"Blend {chosen_families}: {chosen_feats}")


# ── Main generation function ──────────────────────────────────────────────────

ARCHETYPE_WEIGHTS = {
    "pure_factor":        0.25,   # reduce from dominant
    "defensive_quality":  0.30,   # more defensive strategies
    "regime_adaptive":    0.25,   # adaptive strategies
    "factor_blend":       0.20,   # cross-family
}


def generate_diverse_arm(name: str, archetype: str | None = None) -> ArchetypeResult:
    """Generate a strategy config from a randomly chosen archetype.

    Args:
        name: strategy name
        archetype: force a specific archetype, or None for weighted random

    Returns:
        ArchetypeResult with ComposedConfig and metadata
    """
    if archetype is None:
        archetypes = list(ARCHETYPE_WEIGHTS.keys())
        weights = list(ARCHETYPE_WEIGHTS.values())
        archetype = random.choices(archetypes, weights=weights, k=1)[0]

    n_sig = random.choice([1, 1, 2, 2, 3, 4])  # bias single/dual, allow up to 4

    if archetype == "pure_factor":
        return generate_pure_factor(name, n_signals=n_sig)
    elif archetype == "defensive_quality":
        return generate_defensive_quality(name)
    elif archetype == "regime_adaptive":
        return generate_regime_adaptive(name)
    elif archetype == "factor_blend":
        return generate_factor_blend(name)
    else:
        return generate_pure_factor(name, n_signals=1)


def generate_diverse_batch(
    n: int,
    already_tested_names: set[str],
    seed: int | None = None,
) -> list[ArchetypeResult]:
    """Generate n diverse strategy candidates ensuring archetype variety.

    Guarantees at least one candidate per archetype family if n >= 4.
    """
    if seed is not None:
        random.seed(seed)

    results = []
    seen_descriptions = set(already_tested_names)

    # Phase 1: guarantee one per archetype
    for arch in ARCHETYPE_WEIGHTS:
        if len(results) >= n:
            break
        candidate_name = f"arch_{arch[:4]}_{len(results):03d}"
        result = generate_diverse_arm(candidate_name, archetype=arch)
        desc = result.description
        if desc not in seen_descriptions:
            seen_descriptions.add(desc)
            results.append(result)

    # Phase 2: fill remaining slots with weighted random
    attempts = 0
    while len(results) < n and attempts < n * 10:
        attempts += 1
        candidate_name = f"arch_rnd_{len(results):03d}"
        result = generate_diverse_arm(candidate_name)
        desc = result.description
        if desc not in seen_descriptions:
            seen_descriptions.add(desc)
            results.append(result)

    return results
