from datetime import date, datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from trading_bot.config import settings


class Base(DeclarativeBase):
    pass


class Ticker(Base):
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(64))
    sub_industry: Mapped[str | None] = mapped_column(String(128))
    index_name: Mapped[str | None] = mapped_column(String(16))  # sp500 | sp400 | sp600
    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    dt: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "dt", name="uq_candle_symbol_dt"),
        Index("ix_candle_symbol_dt", "symbol", "dt"),
    )


class IndexChange(Base):
    """A single index add/remove event (S&P 500 / 400 / 600).

    Used to reconstruct point-in-time membership and avoid survivorship bias.
    """

    __tablename__ = "index_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dt: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    added: Mapped[str | None] = mapped_column(String(16))
    removed: Mapped[str | None] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(255))
    index_name: Mapped[str | None] = mapped_column(String(16))  # sp500 | sp400 | sp600


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    rationale: Mapped[str] = mapped_column(String(2048), nullable=False)
    config_json: Mapped[str] = mapped_column(String(8192), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="untested")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    universe_size: Mapped[int] = mapped_column(Integer, nullable=False)
    use_pit: Mapped[int] = mapped_column(Integer, default=1)
    metrics_json: Mapped[str] = mapped_column(String(4096), nullable=False)
    equity_json: Mapped[str] = mapped_column(String(2_000_000), nullable=False)
    returns_json: Mapped[str] = mapped_column(String(2_000_000), nullable=False)
    psr: Mapped[float] = mapped_column(Float, default=0.0)
    dsr: Mapped[float] = mapped_column(Float, default=0.0)
    n_trials_used: Mapped[int] = mapped_column(Integer, default=1)
    cpcv_json: Mapped[str | None] = mapped_column(String(65536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FeatureScore(Base):
    """Bayesian scorecard for each feature/param combo.

    Updated after every research run. The score drives hypothesis generation:
    features that appear in high-OOS / low-PBO strategies get a higher prior.
    """

    __tablename__ = "feature_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_key: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    times_promoted: Mapped[int] = mapped_column(Integer, default=0)
    sum_oos_sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    sum_pbo: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=0.5)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ResearchLog(Base):
    """One row per hypothesis tested by the autonomous research loop."""

    __tablename__ = "research_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    hypothesis_name: Mapped[str] = mapped_column(String(128), nullable=False)
    config_json: Mapped[str] = mapped_column(String(8192), nullable=False)
    rationale: Mapped[str] = mapped_column(String(2048), nullable=False)
    ic_prescan: Mapped[float | None] = mapped_column(Float, nullable=True)
    oos_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    pbo: Mapped[float | None] = mapped_column(Float, nullable=True)
    dsr: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    skip_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    arm_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    universe_size: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TrialCounter(Base):
    """Single-row table that counts every backtest ever run, including failures.

    Used by the Deflated Sharpe Ratio to honestly correct for multiple testing.
    """

    __tablename__ = "trial_counter"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_trials: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


engine = create_engine(settings.db_url, future=True)


if settings.db_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_concurrency_pragmas(dbapi_conn, _record):  # noqa: ANN001
        """WAL + busy_timeout: letture (GUI) e scritture (daemon) convivono.

        Senza WAL, SQLite blocca l'intero DB durante ogni scrittura e una
        lettura concorrente fallisce con 'database is locked'. WAL consente
        più lettori e uno scrittore in parallelo; busy_timeout fa attendere
        invece di fallire subito.
        """
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=10000")   # 10s
        cur.execute("PRAGMA synchronous=NORMAL")   # sicuro in WAL, più veloce
        cur.close()


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine, future=True)
