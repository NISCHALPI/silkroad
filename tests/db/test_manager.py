"""Tests for MarketDataManager logic."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from silkroad.core.enums import Horizon, AssetClass, Sector, Exchange
from silkroad.core.data_models import Asset
from silkroad.db.manager import DataManager
from silkroad.db.duckdb_store import DuckDBStore
from silkroad.db.backends import DataBackendProvider


class MockBackend(DataBackendProvider):
    def __init__(self):
        self.data_to_return = pd.DataFrame()

    def fetch_data(self, assets, start, end, horizon, **kwargs):
        return self.data_to_return


@pytest.fixture
def db(tmp_path):
    return DuckDBStore(str(tmp_path / "mgr.db"))


@pytest.fixture
def backend():
    return MockBackend()


@pytest.fixture
def manager(db, backend):
    return DataManager(db, backend)


def test_sync_incremental(manager, db, backend):
    # Setup: DB has partial data
    symbol = "TSLA"
    asset = Asset(
        ticker=symbol,
        asset_class=AssetClass.STOCK,
        sector=Sector.TECHNOLOGY,
        exchange=Exchange.NASDAQ,
    )

    dates_old = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    data_old = pd.DataFrame(
        {
            "close": [10.0] * 5,
            "open": [10.0] * 5,
            "high": [10.0] * 5,
            "low": [10.0] * 5,
            "volume": [100] * 5,
            "vwap": [10.0] * 5,
            "trade_count": [1.0] * 5,
        },
        index=dates_old,
    )
    db.save_market_data(symbol, data_old, Horizon.DAILY)

    # Backend provides new data (overlapping + new)
    dates_new = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    data_new = pd.DataFrame(
        {
            "close": [10.0] * 10,  # Same values for overlap
            "open": [10.0] * 10,
            "high": [10.0] * 10,
            "low": [10.0] * 10,
            "volume": [100] * 10,
            "vwap": [10.0] * 10,
            "trade_count": [1.0] * 10,
            "symbol": [symbol] * 10,
        },
        index=dates_new,
    )
    # Ensure multiindex for backend return
    data_new.set_index(["symbol"], append=True, inplace=True)
    data_new = data_new.swaplevel(0, 1)

    backend.data_to_return = data_new

    manager.sync(
        [asset], Horizon.DAILY, end_date=datetime(2023, 1, 10, tzinfo=timezone.utc)
    )

    # Verify DB now has 10 records
    final_df = db.get_market_data(
        [symbol],
        Horizon.DAILY,
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 20, tzinfo=timezone.utc),
    )
    assert len(final_df) == 10


def test_sync_split_detection(manager, db, backend):
    # Setup: DB has pre-split data
    symbol = "AAPL"
    asset = Asset(
        ticker=symbol,
        asset_class=AssetClass.STOCK,
        sector=Sector.TECHNOLOGY,
        exchange=Exchange.NASDAQ,
    )

    dates_old = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    data_old = pd.DataFrame(
        {
            "close": [100.0] * 5,  # Price 100
            "open": [100.0] * 5,
            "high": [100.0] * 5,
            "low": [100.0] * 5,
            "volume": [100] * 5,
            "vwap": [100.0] * 5,
            "trade_count": [1.0] * 5,
        },
        index=dates_old,
    )
    db.save_market_data(symbol, data_old, Horizon.DAILY)

    # Backend provides SPLIT data (price 50) for the SAME DATES
    dates_new = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    data_new = pd.DataFrame(
        {
            "close": [50.0] * 10,  # Price 50 (Mismatch!)
            "open": [50.0] * 10,
            "high": [50.0] * 10,
            "low": [50.0] * 10,
            "volume": [200] * 10,
            "vwap": [50.0] * 10,
            "trade_count": [1.0] * 10,
            "symbol": [symbol] * 10,
        },
        index=dates_new,
    )
    data_new.set_index(["symbol"], append=True, inplace=True)
    data_new = data_new.swaplevel(0, 1)

    backend.data_to_return = data_new

    # Run sync
    manager.sync(
        [asset], Horizon.DAILY, end_date=datetime(2023, 1, 10, tzinfo=timezone.utc)
    )

    # Verify DB now has updated values (50.0) - indicating a Replace happened
    final_df = db.get_market_data(
        [symbol],
        Horizon.DAILY,
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 5, tzinfo=timezone.utc),
    )
    assert final_df.iloc[0]["close"] == 50.0
