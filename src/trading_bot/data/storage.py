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
    """A single S&P 500 add/remove event.

    Used to reconstruct point-in-time membership.
    """

    __tablename__ = "index_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dt: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    added: Mapped[str | None] = mapped_column(String(16))
    removed: Mapped[str | None] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(255))


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


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine, future=True)
