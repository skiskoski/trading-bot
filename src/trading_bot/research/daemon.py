"""Daemon per il research loop autonomo.

Avvia il loop in un processo detached che ricicla batch di round a oltranza,
finché non riceve SIGTERM (``tradebot research-stop``). Stato condiviso via
PID file; output su log rotante, così la shell resta libera.

    ~/.trading_bot/research.pid   {"pid": int, "started": iso8601}
    ~/.trading_bot/research.log   RotatingFileHandler 10MB × 5
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

STATE_DIR = Path.home() / ".trading_bot"
PID_FILE = STATE_DIR / "research.pid"
LOG_FILE = STATE_DIR / "research.log"

# Se un batch termina più in fretta di così (es. DB vuoto, nessun candidato
# nuovo) aspetta prima del successivo per non girare a vuoto a CPU piena.
_MIN_BATCH_SECONDS = 10.0
_IDLE_SLEEP_SECONDS = 60.0


# ── PID file ────────────────────────────────────────────────────────────────

def read_pidfile() -> dict | None:
    try:
        return json.loads(PID_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def daemon_pid() -> int | None:
    """PID del daemon vivo, o None. Rimuove PID file orfani."""
    info = read_pidfile()
    if info is None:
        return None
    pid = int(info.get("pid", -1))
    if pid > 0 and _pid_alive(pid):
        return pid
    PID_FILE.unlink(missing_ok=True)
    return None


def daemon_started_at() -> datetime | None:
    info = read_pidfile()
    if info is None or daemon_pid() is None:
        return None
    try:
        return datetime.fromisoformat(info["started"])
    except (KeyError, ValueError):
        return None


# ── Start / stop (lato controller) ──────────────────────────────────────────

def start_daemon(loop_kwargs: dict) -> int:
    """Lancia il daemon detached e ritorna il suo PID.

    Raises RuntimeError se un daemon è già attivo o se l'avvio fallisce.
    """
    existing = daemon_pid()
    if existing is not None:
        raise RuntimeError(f"daemon già attivo (pid {existing})")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "trading_bot.research.daemon", json.dumps(loop_kwargs)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Il figlio scrive il PID file appena pronto: attendi conferma.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"il daemon è morto in avvio (exit {proc.returncode}) — "
                f"controlla {LOG_FILE}"
            )
        info = read_pidfile()
        if info is not None and int(info.get("pid", -1)) == proc.pid:
            return proc.pid
        time.sleep(0.2)
    raise RuntimeError(f"timeout in avvio daemon — controlla {LOG_FILE}")


def stop_daemon(timeout: float = 30.0) -> tuple[str, int | None]:
    """Ferma il daemon. Ritorna (esito, pid): esito in {none, stopped, killed}."""
    pid = daemon_pid()
    if pid is None:
        return "none", None

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            PID_FILE.unlink(missing_ok=True)
            return "stopped", pid
        time.sleep(0.5)

    os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    return "killed", pid


# ── Processo daemon (lato figlio) ───────────────────────────────────────────

class _LoggerWriter:
    """File-like che inoltra righe complete a un logger — usato per
    reindirizzare la rich Console del loop dentro il log rotante."""

    def __init__(self, log: logging.Logger, level: int = logging.INFO):
        self._log = log
        self._level = level
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log.log(self._level, line.rstrip())
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._log.log(self._level, self._buf.rstrip())
        self._buf = ""

    def isatty(self) -> bool:
        return False


def _setup_daemon_logging() -> logging.Logger:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Solo il file rotante: il daemon non ha terminale.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    return logging.getLogger("trading_bot.research.daemon")


def main() -> None:
    loop_kwargs: dict = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

    log = _setup_daemon_logging()
    stop_event = threading.Event()

    def _on_sigterm(signum, frame):  # noqa: ARG001
        log.info("SIGTERM ricevuto — chiusura pulita dopo il trial corrente…")
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    PID_FILE.write_text(json.dumps({
        "pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat(),
    }))

    # La rich Console del loop scrive nel log invece che su stdout (chiuso).
    from rich.console import Console

    from trading_bot.research import loop as loop_module
    from trading_bot.research.loop import LoopConfig, run_research_loop

    writer = _LoggerWriter(logging.getLogger("trading_bot.research.loop"))
    loop_module.console = Console(file=writer, force_terminal=False,
                                  width=110, highlight=False)
    sys.stdout = _LoggerWriter(logging.getLogger("stdout"))
    sys.stderr = _LoggerWriter(logging.getLogger("stderr"), logging.WARNING)

    cfg = LoopConfig(**loop_kwargs)
    log.info(f"Daemon avviato (pid {os.getpid()}) — batch infiniti di "
             f"{cfg.rounds} round × {cfg.candidates_per_round} candidati")

    batch = 0
    try:
        while not stop_event.is_set():
            batch += 1
            log.info(f"━━━ Batch {batch} ━━━")
            t0 = time.monotonic()
            try:
                promoted = run_research_loop(cfg, stop_event=stop_event)
                log.info(f"Batch {batch} completo — {len(promoted)} promozioni")
            except Exception:
                log.exception(f"Batch {batch} fallito — riprovo dopo pausa")
                stop_event.wait(_IDLE_SLEEP_SECONDS)
                continue
            if time.monotonic() - t0 < _MIN_BATCH_SECONDS:
                log.info(f"Batch troppo rapido (niente da testare?) — "
                         f"pausa {_IDLE_SLEEP_SECONDS:.0f}s")
                stop_event.wait(_IDLE_SLEEP_SECONDS)
    finally:
        PID_FILE.unlink(missing_ok=True)
        log.info("Daemon terminato.")


if __name__ == "__main__":
    main()
