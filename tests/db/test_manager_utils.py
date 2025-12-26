import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List

from silkroad.core.enums import Horizon, Sector, Exchange
from silkroad.core.data_models import Asset, UniformBarSet, UniformBarCollection
from silkroad.db.duckdb_store import DuckDBStore
from silkroad.db.manager import DataManager
from silkroad.db.backends import DataBackendProvider


class MockBackend(DataBackendProvider):
    """Minimal mock backend."""

    def fetch_data(self, assets, start, end, horizon):
        return pd.DataFrame()

    def get_latest_bar(self, asset):
        return None


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_utils.duckdb"
    return DuckDBStore(str(db_path))


@pytest.fixture
def manager(store):
    backend = MockBackend()
    return DataManager(db=store, backend=backend)


def test_filter_symbols(manager, store):
    """Test filtering symbols by metadata."""
    # Setup data with metadata
    horizon = Horizon.DAILY

    # Symbol 1: Tech, NASDAQ
    s1 = "AAPL"
    df1 = pd.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
            "trade_count": [10],
            "vwap": [100.2],
        }
    )
    meta1 = {"sector": "Technology", "exchange": "NASDAQ"}
    store.save_market_data(s1, df1, horizon, metadata=meta1)

    # Symbol 2: Finance, NYSE
    s2 = "JPM"
    df2 = pd.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1)],
            "open": [50.0],
            "high": [51.0],
            "low": [49.0],
            "close": [50.5],
            "volume": [2000],
            "trade_count": [20],
            "vwap": [50.2],
        }
    )
    meta2 = {"sector": "Finance", "exchange": "NYSE"}
    store.save_market_data(s2, df2, horizon, metadata=meta2)

    # Symbol 3: Tech, NYSE
    s3 = "IBM"
    df3 = pd.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1)],
            "open": [120.0],
            "high": [121.0],
            "low": [119.0],
            "close": [120.5],
            "volume": [1500],
            "trade_count": [15],
            "vwap": [120.2],
        }
    )
    meta3 = {"sector": "Technology", "exchange": "NYSE"}
    store.save_market_data(s3, df3, horizon, metadata=meta3)

    # Test Filter
    tech = manager.filter_symbols(horizon, sector="Technology")
    assert sorted(tech) == ["AAPL", "IBM"]

    nyse = manager.filter_symbols(horizon, exchange="NYSE")
    assert sorted(nyse) == ["IBM", "JPM"]

    tech_nyse = manager.filter_symbols(horizon, sector="Technology", exchange="NYSE")
    assert tech_nyse == ["IBM"]

    none = manager.filter_symbols(horizon, sector="Energy")
    assert none == []


def test_read_barset(manager, store):
    """Test reading a UniformBarSet."""
    symbol = "TEST"
    horizon = Horizon.DAILY
    dates = pd.date_range("2023-01-01", periods=5, tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0] * 5,
            "high": [105.0] * 5,
            "low": [95.0] * 5,
            "close": [102.0] * 5,
            "volume": [1000.0] * 5,
            "trade_count": [10.0] * 5,
            "vwap": [101.0] * 5,
        }
    )

    store.save_market_data(symbol, df, horizon)

    bs = manager.read_barset(symbol, horizon)
    assert isinstance(bs, UniformBarSet)
    assert bs.symbol == symbol
    assert bs.horizon == horizon
    assert len(bs) == 5
    assert not bs.df.empty

    # Test range
    bs_range = manager.read_barset(
        symbol,
        horizon,
        start=datetime(2023, 1, 2, tzinfo=timezone.utc),
        end=datetime(2023, 1, 3, tzinfo=timezone.utc),
    )
    assert len(bs_range) == 2


def test_read_collection(manager, store):
    """Test reading a UniformBarCollection."""
    horizon = Horizon.DAILY
    dates = pd.date_range("2023-01-01", periods=5, tz="UTC")

    # S1: Full range
    s1 = "S1"
    df1 = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [10] * 5,
            "high": [11] * 5,
            "low": [9] * 5,
            "close": [10.5] * 5,
            "volume": [100] * 5,
            "trade_count": [1] * 5,
            "vwap": [10.2] * 5,
        }
    )
    store.save_market_data(s1, df1, horizon)

    # S2: Partial range (only last 3 days)
    s2 = "S2"
    dates2 = dates[2:]
    df2 = pd.DataFrame(
        {
            "timestamp": dates2,
            "open": [20] * 3,
            "high": [21] * 3,
            "low": [19] * 3,
            "close": [20.5] * 3,
            "volume": [200] * 3,
            "trade_count": [2] * 3,
            "vwap": [20.2] * 3,
        }
    )
    store.save_market_data(s2, df2, horizon)

    # Read Collection
    # Intersection should be the common 3 days
    col = manager.read_collection([s1, s2], horizon)
    assert isinstance(col, UniformBarCollection)
    assert len(col.bar_map) == 2
    assert col.n_bars == 3
    assert col.timestamps.equals(dates2)
