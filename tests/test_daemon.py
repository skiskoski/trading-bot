"""Tests for the research daemon: PID file lifecycle, stop semantics,
logger redirection, and stop_event interruptibility of the loop."""

import json
import logging
import os
import threading
from datetime import datetime, timezone

import pytest

from trading_bot.research import daemon


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point the daemon's state files at a temp dir so tests never touch
    (or kill) a real running daemon."""
    monkeypatch.setattr(daemon, "STATE_DIR", tmp_path)
    monkeypatch.setattr(daemon, "PID_FILE", tmp_path / "research.pid")
    monkeypatch.setattr(daemon, "LOG_FILE", tmp_path / "research.log")
    return tmp_path


# ── PID file ────────────────────────────────────────────────────────────────

def test_daemon_pid_none_without_pidfile(isolated_state):
    assert daemon.daemon_pid() is None
    assert daemon.daemon_started_at() is None


def test_daemon_pid_alive_process(isolated_state):
    daemon.PID_FILE.write_text(json.dumps({
        "pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat(),
    }))
    assert daemon.daemon_pid() == os.getpid()
    assert daemon.daemon_started_at() is not None


def test_daemon_pid_cleans_stale_pidfile(isolated_state):
    # Spawn-and-reap a child so its PID is guaranteed dead.
    import subprocess
    proc = subprocess.Popen(["true"])
    proc.wait()
    pid = proc.pid
    daemon.PID_FILE.write_text(json.dumps({
        "pid": pid, "started": datetime.now(timezone.utc).isoformat(),
    }))
    assert daemon.daemon_pid() is None
    assert not daemon.PID_FILE.exists()


def test_daemon_pid_corrupt_pidfile(isolated_state):
    daemon.PID_FILE.write_text("not json at all")
    assert daemon.daemon_pid() is None


# ── start/stop semantics ────────────────────────────────────────────────────

def test_start_daemon_refuses_double_start(isolated_state):
    daemon.PID_FILE.write_text(json.dumps({
        "pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat(),
    }))
    with pytest.raises(RuntimeError, match="già attivo"):
        daemon.start_daemon({})


def test_stop_daemon_none_when_not_running(isolated_state):
    outcome, pid = daemon.stop_daemon(timeout=1.0)
    assert outcome == "none"
    assert pid is None


# ── _LoggerWriter ───────────────────────────────────────────────────────────

def test_logger_writer_forwards_complete_lines(caplog):
    log = logging.getLogger("test.daemon.writer")
    w = daemon._LoggerWriter(log)
    with caplog.at_level(logging.INFO, logger="test.daemon.writer"):
        w.write("riga uno\nriga ")
        w.write("due\n")
        w.flush()
    messages = [r.message for r in caplog.records]
    assert "riga uno" in messages
    assert "riga due" in messages


def test_logger_writer_flush_partial_buffer(caplog):
    log = logging.getLogger("test.daemon.writer2")
    w = daemon._LoggerWriter(log)
    with caplog.at_level(logging.INFO, logger="test.daemon.writer2"):
        w.write("senza newline finale")
        w.flush()
    assert any("senza newline finale" in r.message for r in caplog.records)
    assert not w.isatty()


# ── stop_event interrompe il loop ───────────────────────────────────────────

def test_loop_returns_immediately_when_stop_preset(monkeypatch):
    """Con stop_event già settato il loop non deve eseguire nessun round."""
    from trading_bot.research.loop import LoopConfig, run_research_loop

    # Panel fittizio non vuoto così superiamo il guard iniziale e arriviamo
    # al check dello stop_event in testa al primo round.
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    fake_panel = pd.DataFrame(
        {s: 100 + np.arange(300) * 0.1 for s in ["SPY", "GLD", "HYG", "AAA"]},
        index=idx,
    )
    monkeypatch.setattr("trading_bot.research.loop.load_panel",
                        lambda *a, **k: fake_panel)
    monkeypatch.setattr(
        "trading_bot.data.universe.get_top_n_by_liquidity",
        lambda n: ["AAA"],
    )
    monkeypatch.setattr(
        "trading_bot.data.pit_universe.build_membership_panel",
        lambda *a, **k: None,
    )

    ev = threading.Event()
    ev.set()
    promoted = run_research_loop(
        LoopConfig(rounds=3, candidates_per_round=2, use_pit=False),
        stop_event=ev,
    )
    assert promoted == []
