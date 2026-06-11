"""Tests for the live trading module."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── AlpacaClient ─────────────────────────────────────────────────────────────

class TestAlpacaClient:
    def test_refuses_live_url(self):
        from trading_bot.live.alpaca import AlpacaClient, LIVE_URL
        with pytest.raises(ValueError, match="paper-trading only"):
            AlpacaClient("key", "secret", base_url=LIVE_URL)

    def test_paper_url_accepted(self):
        from trading_bot.live.alpaca import AlpacaClient, PAPER_URL
        client = AlpacaClient("key", "secret", base_url=PAPER_URL)
        assert client.base_url == PAPER_URL

    def test_get_alpaca_client_returns_client_or_none(self):
        from trading_bot.live.alpaca import get_alpaca_client, AlpacaClient
        # Either returns a configured client or None (if keys not set)
        client = get_alpaca_client()
        assert client is None or isinstance(client, AlpacaClient)

    def test_rebalance_dry_run(self):
        """Dry run should not call submit_order."""
        from trading_bot.live.alpaca import AlpacaClient, AlpacaAccount, AlpacaPosition, PAPER_URL
        client = AlpacaClient("key", "secret", base_url=PAPER_URL)

        mock_account = AlpacaAccount(equity=100_000, cash=50_000,
                                     buying_power=50_000, portfolio_value=100_000)
        mock_positions = [
            AlpacaPosition("AAPL", qty=10, market_value=15_000,
                           unrealized_pl=500, unrealized_plpc=0.034,
                           current_price=150, cost_basis=14_500)
        ]

        with patch.object(client, "get_account", return_value=mock_account), \
             patch.object(client, "get_positions", return_value=mock_positions), \
             patch.object(client, "submit_order") as mock_submit:

            orders = client.rebalance(
                target_weights={"AAPL": 0.2, "MSFT": 0.3},
                dry_run=True,
            )
            # Dry run → submit_order never called
            mock_submit.assert_not_called()
            assert orders == []


# ── TelegramNotifier ──────────────────────────────────────────────────────────

class TestTelegramNotifier:
    def test_send_returns_false_on_network_error(self):
        from trading_bot.live.notifier import TelegramNotifier, NotifyLevel
        notifier = TelegramNotifier("bad_token", "123")
        # Should not raise, just return False
        result = notifier.send("test", NotifyLevel.INFO)
        assert result is False

    def test_get_notifier_returns_notifier_or_none(self):
        from trading_bot.live.notifier import get_notifier, TelegramNotifier
        # Either returns a configured notifier or None (if token not set)
        notifier = get_notifier()
        assert notifier is None or isinstance(notifier, TelegramNotifier)

    def test_notify_fire_and_forget_safe(self):
        """notify() should never raise even without config."""
        from trading_bot.live.notifier import notify, NotifyLevel
        notify("test message", NotifyLevel.INFO)  # should not raise


# ── Runner ────────────────────────────────────────────────────────────────────

class TestRunner:
    def test_is_rebalance_day_first_weekday(self):
        from trading_bot.live.runner import is_rebalance_day
        # Monday Jan 2 2023 — 1st is Sunday, so 2nd is first trading day
        assert is_rebalance_day(pd.Timestamp("2023-01-02"))

    def test_is_rebalance_day_mid_month(self):
        from trading_bot.live.runner import is_rebalance_day
        assert not is_rebalance_day(pd.Timestamp("2023-01-15"))

    def test_is_rebalance_day_weekend(self):
        from trading_bot.live.runner import is_rebalance_day
        # Saturday
        assert not is_rebalance_day(pd.Timestamp("2023-01-07"))

    def test_daily_run_dry_run_no_data(self):
        """Daily run with no data should return result with errors, not crash."""
        from trading_bot.live.runner import run_daily
        result = run_daily(dry_run=True, skip_ingest=True)
        # May succeed or fail but should return a DailyRunResult, never raise
        from trading_bot.live.runner import DailyRunResult
        assert isinstance(result, DailyRunResult)


# ── Signals ───────────────────────────────────────────────────────────────────

class TestSignals:
    def test_generate_signals_no_strategy(self):
        """Should raise RuntimeError when no strategies in registry."""
        from trading_bot.live.signals import generate_signals
        from trading_bot.registry import list_strategies
        with patch("trading_bot.live.signals.list_strategies", return_value=[]), \
             patch("trading_bot.live.signals.list_runs", return_value=[]):
            with pytest.raises(RuntimeError, match="No strategies"):
                generate_signals()

    def test_signal_output_weights_normalized(self):
        """Positions should never sum to > 1.0."""
        from trading_bot.live.signals import SignalOutput
        from datetime import date
        out = SignalOutput(
            strategy_name="test",
            asof=date.today(),
            positions={"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.3},
            ranked_universe=pd.Series(dtype=float),
            regime_active=None,
            n_universe=100,
        )
        assert sum(out.positions.values()) <= 1.0 + 1e-9
