"""Tests for the autonomous research loop components."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.research.knowledge_base import FEATURES, build_rationale, get_synergy
from trading_bot.research.scorecard import get_score, update_score
from trading_bot.research.analyzer import analyze_runs, RunInsight, AnalysisReport
from trading_bot.research.generator import Candidate, generate, _config_fingerprint


# ── Knowledge base ─────────────────────────────────────────────────────────

def test_all_features_have_citation():
    for name, spec in FEATURES.items():
        assert spec.citation, f"{name} missing citation"


def test_all_features_have_param_grid():
    for name, spec in FEATURES.items():
        assert spec.param_grid, f"{name} missing param_grid"


def test_synergy_matrix_symmetric_or_absent():
    """get_synergy should return same value regardless of arg order."""
    assert get_synergy("momentum", "low_volatility") == get_synergy("low_volatility", "momentum")


def test_build_rationale_nonempty():
    r = build_rationale(
        signals=[("momentum", {"lookback": 252, "skip": 21})],
        filters=[("above_sma", {"window": 100})],
        regime={"symbol": "SPY", "feature": "sma_distance",
                "params": {"window": 200}, "threshold": 0.0},
        top_n=10,
    )
    assert len(r) > 20


# ── Scorecard ──────────────────────────────────────────────────────────────

def test_scorecard_update_and_read():
    feat, params = "momentum", {"lookback": 999, "skip": 99}  # unique params for test
    initial = get_score(feat, params)
    update_score(feat, params, oos_sharpe=1.2, pbo=0.2, promoted=True)
    new_score = get_score(feat, params)
    assert new_score > initial  # should increase after a good run


def test_scorecard_penalises_bad_run():
    feat, params = "rsi", {"period": 997}  # unique
    update_score(feat, params, oos_sharpe=-0.5, pbo=0.8, promoted=False)
    score = get_score(feat, params)
    assert score < 0.5  # bad run → low score


# ── Analyzer ──────────────────────────────────────────────────────────────

def _make_mock_run(name, features, oos_sharpe, pbo, dsr):
    return {
        "strategy_name": name,
        "metrics": {"IC_mean": 0.03, "Sharpe": oos_sharpe},
        "dsr": dsr,
        "cpcv": {"mean_oos_sharpe": oos_sharpe, "pbo": pbo},
    }


def test_analyze_runs_promoted_count():
    runs = [
        _make_mock_run("good", ["momentum"], 1.2, 0.2, 0.97),
        _make_mock_run("bad",  ["rsi"],      0.1, 0.7, 0.3),
    ]
    report = analyze_runs(runs)
    assert report.total_runs == 2
    assert len(report.promoted_runs) == 1
    assert report.promoted_runs[0].name == "good"


def test_analyze_runs_avg_oos_sharpe():
    runs = [
        _make_mock_run("a", ["momentum"], 1.0, 0.3, 0.96),
        _make_mock_run("b", ["rsi"],      0.5, 0.4, 0.96),
    ]
    report = analyze_runs(runs)
    assert abs(report.avg_oos_sharpe - 0.75) < 0.01


# ── Generator ──────────────────────────────────────────────────────────────

def _mock_report():
    runs = [
        _make_mock_run("momentum_12_1", ["momentum"], 0.8, 0.3, 0.97),
        _make_mock_run("rsi2_reversal", ["rsi"],      0.9, 0.33, 0.84),
    ]
    return analyze_runs(runs)


def test_generator_produces_candidates():
    report = _mock_report()
    candidates = generate(report, round_id=1, n_candidates=5)
    assert len(candidates) > 0
    assert all(isinstance(c, Candidate) for c in candidates)


def test_generator_no_duplicate_fingerprints():
    report = _mock_report()
    candidates = generate(report, round_id=1, n_candidates=15)
    fps = [_config_fingerprint(c.config) for c in candidates]
    assert len(fps) == len(set(fps)), "Duplicate configs generated"


def test_generator_rationale_nonempty():
    report = _mock_report()
    candidates = generate(report, round_id=1, n_candidates=5)
    for c in candidates:
        assert len(c.config.rationale) >= 20, f"Short rationale: {c.config.rationale}"


def test_generator_skips_tested_configs():
    """With a large enough already_tested set, round 2 produces novel configs."""
    report = _mock_report()
    # Generate a big set in round 1 to fill the space
    candidates_round1 = generate(report, round_id=1, n_candidates=30)
    fps_round1 = {_config_fingerprint(c.config) for c in candidates_round1}
    # Round 2 — all round-1 fingerprints are forbidden
    candidates_round2 = generate(report, round_id=99, n_candidates=10,
                                  already_tested=set(fps_round1))
    fps_round2 = {_config_fingerprint(c.config) for c in candidates_round2}
    # Every config returned in round 2 must NOT appear in the forbidden set
    overlap = fps_round1 & fps_round2
    assert len(overlap) == 0, f"Generator returned {len(overlap)} already-tested configs"
