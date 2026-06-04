from datetime import date

import pandas as pd
import pytest


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "pit.db"
    monkeypatch.setenv("TRADEBOT_DB_PATH", str(db_path))
    import importlib

    import trading_bot.config as cfg
    importlib.reload(cfg)
    import trading_bot.data.storage as storage
    importlib.reload(storage)
    import trading_bot.data.pit_universe as pit
    importlib.reload(pit)
    storage.init_db()
    return storage, pit


def test_get_constituents_undoes_changes(fresh_db):
    storage, pit = fresh_db

    # 2020: TSLA added, XYZ removed
    # 2022: AAPL removed (hypothetical), NEW added
    with storage.get_session() as s:
        s.add_all(
            [
                storage.IndexChange(dt=date(2020, 1, 1), added="TSLA", removed="XYZ"),
                storage.IndexChange(dt=date(2022, 1, 1), added="NEW", removed="AAPL"),
            ]
        )
        s.commit()

    current = {"TSLA", "AAPL", "NEW", "MSFT"}  # XYZ no longer in current set

    # Today: matches current
    members_now = pit.get_constituents_at(date(2024, 1, 1), current)
    assert members_now == current

    # 2021: NEW not yet added, AAPL still in
    members_2021 = pit.get_constituents_at(date(2021, 6, 1), current)
    assert "NEW" not in members_2021
    assert "AAPL" in members_2021
    assert "TSLA" in members_2021
    assert "XYZ" not in members_2021

    # 2019: TSLA not yet added, XYZ still in, AAPL still in
    members_2019 = pit.get_constituents_at(date(2019, 6, 1), current)
    assert "TSLA" not in members_2019
    assert "XYZ" in members_2019
    assert "AAPL" in members_2019


def test_build_membership_panel(fresh_db):
    storage, pit = fresh_db

    with storage.get_session() as s:
        s.add_all(
            [
                storage.IndexChange(dt=date(2020, 6, 15), added="TSLA", removed="XYZ"),
            ]
        )
        s.commit()

    dates = pd.date_range("2019-01-01", "2021-12-31", freq="MS")
    current = {"TSLA", "AAPL"}
    panel = pit.build_membership_panel(dates, current, symbols=["TSLA", "AAPL", "XYZ"])

    # Pre-2020-06-15: XYZ in, TSLA out
    pre = panel.loc[panel.index < pd.Timestamp("2020-06-15")]
    assert pre["XYZ"].all()
    assert not pre["TSLA"].any()
    assert pre["AAPL"].all()

    # Post: XYZ out, TSLA in
    post = panel.loc[panel.index > pd.Timestamp("2020-07-01")]
    assert not post["XYZ"].any()
    assert post["TSLA"].all()
