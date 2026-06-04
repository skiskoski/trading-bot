from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture()
def session(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("TRADEBOT_DB_PATH", str(db_path))
    # Reset cached settings & engine
    import importlib

    import trading_bot.config as cfg
    importlib.reload(cfg)
    import trading_bot.data.storage as storage
    importlib.reload(storage)
    storage.init_db()
    with Session(storage.engine, future=True) as s:
        yield s, storage


def test_insert_ticker(session):
    s, storage = session
    s.add(storage.Ticker(symbol="AAPL", name="Apple", sector="Tech"))
    s.commit()
    rows = s.query(storage.Ticker).all()
    assert len(rows) == 1
    assert rows[0].symbol == "AAPL"


def test_candle_unique_constraint(session):
    s, storage = session
    c1 = storage.Candle(
        symbol="AAPL",
        dt=date(2024, 1, 2),
        open=100,
        high=101,
        low=99,
        close=100.5,
        adj_close=100.5,
        volume=1_000_000,
    )
    s.add(c1)
    s.commit()
    # Duplicate (symbol, dt) should fail
    c2 = storage.Candle(
        symbol="AAPL",
        dt=date(2024, 1, 2),
        open=100,
        high=101,
        low=99,
        close=100.5,
        adj_close=100.5,
        volume=1_000_000,
    )
    s.add(c2)
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        s.commit()
