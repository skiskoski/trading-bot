"""Exhaustive combinatorial search space with UCB1 ordering.

Enumerates ALL unique (signals × params × weights × filters × regime)
combinations, then orders them from most promising to least using the
UCB1 bandit algorithm:

    UCB1_score(arm) = mean_reward(arm) + C * sqrt(ln(N_total) / N_arm)

where:
  - mean_reward = avg OOS Sharpe × (1 - avg PBO) from scorecard
  - C = exploration constant (higher → explore more aggressively)
  - N_total = total experiments run so far
  - N_arm = times this (feature, params) combo has been tested

This guarantees:
  1. Every arm gets tested at least once (exploration)
  2. Arms with high mean reward are tested more often (exploitation)
  3. The order adapts as the scorecard learns

The space is estimated at ~340 000 unique configs — effectively infinite
relative to what we can test in a session.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

from trading_bot.research.knowledge_base import (
    FEATURES,
    FILTERS,
    REGIMES,
    build_rationale,
    get_synergy,
)
from trading_bot.research.scorecard import get_all_scores, get_score
from trading_bot.strategies.composer import (
    ComposedConfig,
    FilterSpec,
    RegimeSpec,
    SignalSpec,
)


# UCB1 exploration constant. Higher = more exploration of untested combos.
UCB1_C = 1.4


@dataclass(frozen=True)
class SearchArm:
    """A single testable configuration (one 'arm' of the bandit)."""
    signals: tuple            # ((feat, params_json, weight, negate), ...)
    filter_feat: str
    filter_params_json: str
    regime_key: str
    top_n: int
    # Gold sleeve as a searchable parameter (user requirement): weight 0=off,
    # mode "defensive"=only when max-DD risk is elevated / "always".
    gold_weight: float = 0.0
    gold_mode: str = "defensive"

    @property
    def fingerprint(self) -> str:
        d = {
            "signals": sorted(self.signals),
            "filter": (self.filter_feat, self.filter_params_json),
            "regime": self.regime_key,
            "top_n": self.top_n,
            "gold": (self.gold_weight, self.gold_mode),
        }
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def to_composed_config(self, name: str) -> ComposedConfig:
        sig_specs = tuple(
            SignalSpec(
                feature=feat,
                params=json.loads(params_json),
                weight=weight,
                use_rank=True,
                negate=negate,
            )
            for feat, params_json, weight, negate in self.signals
        )
        filt_params = json.loads(self.filter_params_json)
        if self.filter_feat == "none":
            filt_spec = ()
        else:
            thr = FILTERS.get(self.filter_feat, {}).get("threshold", 0.5)
            filt_spec = (FilterSpec(feature=self.filter_feat,
                                    params=filt_params, threshold=thr),)

        regime_cfg = REGIMES.get(self.regime_key)
        regime_spec = None
        if regime_cfg:
            regime_spec = RegimeSpec(
                symbol=regime_cfg["symbol"],
                feature=regime_cfg["feature"],
                params=regime_cfg["params"],
                threshold=regime_cfg.get("threshold", 0.0),
            )

        signal_tuples = [(feat, json.loads(params_json))
                         for feat, params_json, _, _ in self.signals]
        filter_tuples = [(self.filter_feat, filt_params)]
        rationale = build_rationale(
            signals=signal_tuples,
            filters=filter_tuples,
            regime=regime_cfg,
            top_n=self.top_n,
        )

        if self.gold_weight > 0:
            rationale += (f" | Gold sleeve {self.gold_weight:.0%} "
                          f"({self.gold_mode}: in portafoglio solo quando il "
                          f"rischio drawdown è elevato)" if self.gold_mode == "defensive"
                          else f" | Gold sleeve {self.gold_weight:.0%} (always-on)")
        return ComposedConfig(
            name=name,
            rationale=rationale,
            signals=sig_specs,
            filters=filt_spec,
            regime=regime_spec,
            top_n=self.top_n,
            gold_weight=self.gold_weight,
            gold_mode=self.gold_mode,
        )


def _param_combos(feat: str) -> list[dict]:
    """All parameter combinations for a feature from the knowledge base."""
    spec = FEATURES.get(feat)
    if not spec or not spec.param_grid:
        return [{}]
    keys = list(spec.param_grid.keys())
    return [
        dict(zip(keys, vals))
        for vals in itertools.product(*[spec.param_grid[k] for k in keys])
    ]


def _filter_options() -> list[tuple[str, dict]]:
    opts = []
    for fname, finfo in FILTERS.items():
        grid = finfo["param_grid"]
        if not grid:
            opts.append((fname, {}))     # "none"
            continue
        keys = list(grid.keys())
        for vals in itertools.product(*[grid[k] for k in keys]):
            opts.append((fname, dict(zip(keys, vals))))
    return opts


def _regime_options() -> list[str]:
    return list(REGIMES.keys())


def _weight_splits_1() -> list[tuple[float]]:
    return [(1.0,)]


def _weight_splits_2() -> list[tuple[float, float]]:
    return [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]


def _weight_splits_3() -> list[tuple[float, float, float]]:
    return [(0.34, 0.33, 0.33), (0.5, 0.3, 0.2), (0.6, 0.2, 0.2)]


def _weight_splits_4() -> list[tuple[float, float, float, float]]:
    return [(0.25, 0.25, 0.25, 0.25), (0.4, 0.3, 0.2, 0.1), (0.4, 0.2, 0.2, 0.2)]


def _top_n_options() -> list[int]:
    return [5, 8, 10, 12, 15, 20]


def ucb1_score(
    feature_key: str,
    n_total: int,
    scores_map: dict[str, dict],
    c: float = UCB1_C,
) -> float:
    """Compute UCB1 score for a feature key.

    feature_key = "feature_name:params_json" (scorecard key format)
    """
    s = scores_map.get(feature_key)
    if s is None or s["times_used"] == 0:
        # Never tested → infinite UCB1 (explore first)
        return float("inf")
    mean_r = max(0.0, s["mean_oos_sharpe"]) * (1.0 - s["mean_pbo"])
    n_arm = s["times_used"]
    exploration = c * math.sqrt(math.log(max(n_total, 1)) / n_arm)
    return mean_r + exploration


def arm_ucb1_score(
    arm: SearchArm,
    n_total: int,
    scores_map: dict[str, dict],
) -> float:
    """UCB1 score for a full arm = mean of constituent feature UCB1 scores
    plus synergy bonus from knowledge base."""
    feat_scores = []
    feat_names = []
    for feat, params_json, _, _ in arm.signals:
        key = f"{feat}:{params_json}"
        feat_scores.append(ucb1_score(key, n_total, scores_map))
        feat_names.append(feat)

    base = sum(feat_scores) / len(feat_scores) if feat_scores else 0.5

    # Add synergy bonuses/penalties
    for i, a in enumerate(feat_names):
        for b in feat_names[i + 1:]:
            base += get_synergy(a, b) * 0.2

    return base


def generate_ordered_arms(
    already_tested: set[str],
    max_arms: int = 500,
    n_signals: int | None = None,
    n_total_experiments: int = 0,
    universe: int | None = None,
) -> list[tuple[float, SearchArm]]:
    """Generate up to max_arms novel SearchArms ordered by UCB1 score.

    Guarantees diversity: samples one representative arm per feature family
    first, then fills remaining slots with best UCB1 scores across all combos.
    This prevents the generator from getting stuck on a single feature.

    When ``universe`` is given, the UCB1 priors are read only from that
    universe's scorecard namespace, so learning never mixes across universes.

    Returns list of (ucb1_score, arm) sorted descending.
    """
    import random
    scores_map = {s["key"]: s for s in get_all_scores(universe=universe)}
    feat_names = list(FEATURES.keys())
    filter_opts = _filter_options()
    regime_opts = _regime_options()
    top_ns = _top_n_options()

    # ── Phase 1: Diversity guarantee ─────────────────────────────────────────
    # Pick one untested arm per feature (single-signal) so every feature
    # gets at least one shot per round.
    diversity_arms: list[tuple[float, SearchArm]] = []
    default_filter = filter_opts[0]
    default_regime = "SPY_sma_200"
    default_top_n = 10

    for feat in feat_names:
        params_list = _param_combos(feat)
        # Pick best-scored params for this feature, or first if unseen
        best_params = params_list[0]
        best_score = -1.0
        for p in params_list[:4]:
            key = f"{feat}:{json.dumps(p, sort_keys=True)}"
            s = scores_map.get(key)
            sc = ucb1_score(key, n_total_experiments, scores_map)
            if sc > best_score:
                best_score = sc
                best_params = p

        signal_tuple = ((feat, json.dumps(best_params, sort_keys=True), 1.0,
                         FEATURES[feat].negate_default),)
        arm = SearchArm(
            signals=signal_tuple,
            filter_feat=default_filter[0],
            filter_params_json=json.dumps(default_filter[1], sort_keys=True),
            regime_key=default_regime,
            top_n=default_top_n,
        )
        if arm.fingerprint not in already_tested:
            score = arm_ucb1_score(arm, n_total_experiments, scores_map)
            diversity_arms.append((score, arm))

    # ── Phase 2: Broad random sampling across all combos ─────────────────────
    # Randomly sample feature combos to avoid sequential bias.
    signal_counts = [1, 2, 3, 4] if n_signals is None else [n_signals]
    all_feat_combos: list[tuple[int, tuple]] = []

    for n_sig in signal_counts:
        if n_sig == 1:
            combos = [(f,) for f in feat_names]
        elif n_sig == 2:
            combos = [
                c for c in itertools.combinations(feat_names, 2)
                if get_synergy(c[0], c[1]) >= -0.1
            ]
        elif n_sig == 3:
            combos = list(itertools.combinations(feat_names, 3))
        else:
            # 4-signal combos: synergy-screened to keep the space tractable
            # (C(24,4) ≈ 10k raw → keep only combos with net-positive synergy)
            combos = [
                c for c in itertools.combinations(feat_names, 4)
                if sum(get_synergy(a, b)
                       for a, b in itertools.combinations(c, 2)) > 0.2
            ]
        for c in combos:
            all_feat_combos.append((n_sig, c))

    # Shuffle to ensure all features get sampled, not just the first ones
    random.shuffle(all_feat_combos)

    results: list[tuple[float, SearchArm]] = list(diversity_arms)
    seen_fps = set(already_tested)
    for _, arm in diversity_arms:
        seen_fps.add(arm.fingerprint)

    budget = max_arms * 30   # sample this many candidates then trim
    checked = 0

    for n_sig, feat_combo in all_feat_combos:
        if len(results) >= budget:
            break

        if n_sig == 1:
            weight_splits = _weight_splits_1()
        elif n_sig == 2:
            weight_splits = _weight_splits_2()
        elif n_sig == 3:
            weight_splits = _weight_splits_3()
        else:
            weight_splits = _weight_splits_4()

        param_options = [_param_combos(f) for f in feat_combo]
        param_options = [opts[:6] for opts in param_options]  # cap per feature

        # Random sample of param combos (don't enumerate all)
        all_param_combos = list(itertools.product(*param_options))
        sample_size = min(len(all_param_combos), 3)
        sampled_params = random.sample(all_param_combos, sample_size)

        for param_combo in sampled_params:
            for weights in weight_splits[:2]:  # limit weight variants
                signal_tuple = tuple(
                    (feat, json.dumps(p, sort_keys=True), w,
                     FEATURES[feat].negate_default)   # canonical sign (audit A2)
                    for feat, p, w in zip(feat_combo, param_combo, weights)
                )
                # Sample one filter and one regime per combo
                filt = random.choice(filter_opts)
                regime_key = random.choice(regime_opts)
                top_n = random.choice(top_ns)

                # Gold come parametro qualunque: peso 0 (bias) o 10-30%,
                # modalità prevalentemente difensiva (solo in stress di mercato)
                gw = random.choice([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
                gm = random.choice(["defensive", "defensive", "always"])
                arm = SearchArm(
                    signals=signal_tuple,
                    filter_feat=filt[0],
                    filter_params_json=json.dumps(filt[1], sort_keys=True),
                    regime_key=regime_key,
                    top_n=top_n,
                    gold_weight=gw,
                    gold_mode=gm,
                )
                fp = arm.fingerprint
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                score = arm_ucb1_score(arm, n_total_experiments, scores_map)
                results.append((score, arm))
                checked += 1

    results.sort(key=lambda x: x[0], reverse=True)

    # Balance the pool by signal count (audit M7): the 4-signal combinatorial
    # space dwarfs the others (~88% of arms otherwise). Quota the final pool
    # so simpler strategies keep getting explored: 1-sig 30%, 2-sig 30%,
    # 3-sig 25%, 4-sig 15%.
    quotas = {1: 0.30, 2: 0.30, 3: 0.25, 4: 0.15}
    buckets: dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for score, arm in results:
        buckets.setdefault(len(arm.signals), []).append((score, arm))
    balanced: list[tuple[float, SearchArm]] = []
    for n_sig, frac in quotas.items():
        balanced.extend(buckets.get(n_sig, [])[: max(1, int(max_arms * frac))])
    # Fill any remaining capacity by global score order
    if len(balanced) < max_arms:
        chosen = {a.fingerprint for _, a in balanced}
        for score, arm in results:
            if len(balanced) >= max_arms:
                break
            if arm.fingerprint not in chosen:
                balanced.append((score, arm))
                chosen.add(arm.fingerprint)
    balanced.sort(key=lambda x: x[0], reverse=True)
    return balanced[:max_arms]


def estimate_search_space_size() -> dict:
    """Estimate total number of unique configurations."""
    feat_names = list(FEATURES.keys())
    n_f = len(feat_names)
    # Ogni filtro contribuisce col prodotto delle sue griglie (grid vuota = 1
    # variante): non tutti i filtri hanno il parametro "window".
    n_filters = sum(
        max(1, len(list(itertools.product(*v["param_grid"].values()))))
        for v in FILTERS.values()
    )
    n_regimes = len(REGIMES)
    n_top_n = len(_top_n_options())

    single_params = sum(
        len(list(itertools.product(*[FEATURES[f].param_grid[k]
                                     for k in FEATURES[f].param_grid])))
        for f in feat_names
    )
    dual_pairs = len(list(itertools.combinations(feat_names, 2)))
    triple_combos = len(list(itertools.combinations(feat_names, 3)))

    single = n_f * (single_params // n_f) * len(_weight_splits_1())
    dual   = dual_pairs * 8 * len(_weight_splits_2())
    triple = triple_combos * 4 * len(_weight_splits_3())

    base = (single + dual + triple) * n_filters * n_regimes * n_top_n
    return {
        "features": n_f,
        "single_signal_configs": single * n_filters * n_regimes * n_top_n,
        "dual_signal_configs":   dual   * n_filters * n_regimes * n_top_n,
        "triple_signal_configs": triple * n_filters * n_regimes * n_top_n,
        "total_estimated": base,
    }
