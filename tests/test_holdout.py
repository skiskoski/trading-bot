"""Tests for holdout OOS validation."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestHoldoutResult:
    def test_holdout_result_pass_condition(self):
        from trading_bot.validation.holdout import HoldoutResult
        r = HoldoutResult(
            strategy_name="test",
            holdout_start="2024-01-01",
            holdout_end="2025-01-01",
            sharpe=0.8,
            cagr=0.12,
            max_drawdown=-0.10,
            n_months=12,
            equity_curve=pd.Series(dtype=float),
            monthly_returns=pd.Series(dtype=float),
            is_sharpe=1.0,
            passed=True,
        )
        assert r.passed
        assert "PASS" in r.summary()

    def test_holdout_result_fail_condition(self):
        from trading_bot.validation.holdout import HoldoutResult
        r = HoldoutResult(
            strategy_name="test",
            holdout_start="2024-01-01",
            holdout_end="2025-01-01",
            sharpe=-0.2,
            cagr=-0.05,
            max_drawdown=-0.30,
            n_months=12,
            equity_curve=pd.Series(dtype=float),
            monthly_returns=pd.Series(dtype=float),
            is_sharpe=1.0,
            passed=False,
        )
        assert not r.passed
        assert "WARN" in r.summary()

    def test_pass_gate_is_half_is_sharpe(self):
        """Gate: holdout Sharpe > 0 AND > IS Sharpe * 0.5"""
        from trading_bot.validation.holdout import HoldoutResult

        def make(h_sharpe, is_sharpe):
            return HoldoutResult(
                strategy_name="t", holdout_start="2024-01-01",
                holdout_end="2025-01-01", sharpe=h_sharpe, cagr=0.0,
                max_drawdown=0.0, n_months=12,
                equity_curve=pd.Series(dtype=float),
                monthly_returns=pd.Series(dtype=float),
                is_sharpe=is_sharpe,
                passed=(h_sharpe > 0 and h_sharpe > is_sharpe * 0.5),
            )

        assert make(0.6, 1.0).passed      # 0.6 > 0.5 * 1.0 = 0.5 ✓
        assert not make(0.4, 1.0).passed  # 0.4 < 0.5 ✗
        assert not make(-0.1, 1.0).passed # negative ✗
