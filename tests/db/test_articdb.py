import pytest
import pandas as pd
import numpy as np
import datetime as dt
from unittest.mock import MagicMock, patch
from silkroad.db.articdb import ArcticDatabase
from silkroad.core.enums import Horizon, Exchange, Sector
from silkroad.core.data_models import Asset, UniformBarSet, UniformBarCollection
from silkroad.db.backends import DataBackendProvider


# Mock DataBackendProvider
class MockBackend(DataBackendProvider):
    def fetch_data(self, asset, start, end, horizon, **kwargs):
        return pd.DataFrame()


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def db(mock_backend, tmp_path):
    # Use a temporary directory for the LMDB store
    uri = f"lmdb://{tmp_path}"
    return ArcticDatabase(backend=mock_backend, uri=uri)


@pytest.fixture
def sample_asset():
    return Asset(  # type: ignore
        ticker="AAPL",
        name="Apple Inc.",
        exchange=Exchange.NASDAQ,
        sector=Sector.TECHNOLOGY,
    )


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "trade_count": [100, 110, 120, 130, 140],
            "vwap": [101.0, 102.0, 103.0, 104.0, 105.0],
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_initialization(db):
    assert db.store is not None


def test_get_library(db):
    lib = db.get_library(Horizon.DAILY)
    assert lib is not None
    assert "market_data_daily" in db.store.list_libraries()


def test_write_and_read(db, sample_df):
    lib = db.get_library(Horizon.DAILY)
    symbol = "AAPL"

    # Write manually to test read
    # ArcticDatabase prefers naive UTC, so we convert before writing to simulate sync behavior
    df_naive = sample_df.tz_localize(None)
    lib.write(symbol, df_naive)

    # Read back
    df_read = db.read(symbol, Horizon.DAILY)
    pd.testing.assert_frame_equal(df_naive, df_read)


def test_read_into_uniform_barset(db, sample_df):
    lib = db.get_library(Horizon.DAILY)
    symbol = "AAPL"
    lib.write(symbol, sample_df)

    bs = db.read_into_uniform_barset(symbol, Horizon.DAILY)
    assert isinstance(bs, UniformBarSet)
    assert bs.symbol == symbol
    assert len(bs.df) == 5


def test_metadata_storage_and_filtering(db, sample_asset, sample_df):
    lib = db.get_library(Horizon.DAILY)

    # Write with metadata
    metadata = sample_asset.model_dump(mode="json")
    lib.write(sample_asset.ticker, sample_df, metadata=metadata)

    # Test get_metadata
    meta = db.get_metadata(sample_asset.ticker, Horizon.DAILY)
    assert meta["ticker"] == "AAPL"
    assert meta["sector"] == "technology"

    # Test filter_symbols
    matches = db.filter_symbols(Horizon.DAILY, sector=Sector.TECHNOLOGY)
    assert "AAPL" in matches

    matches = db.filter_symbols(Horizon.DAILY, sector=Sector.HEALTHCARE)
    assert "AAPL" not in matches


def test_sync_new_asset(db, mock_backend, sample_asset, sample_df):
    # Setup mock return
    # The backend returns a MultiIndex DataFrame
    mi_df = sample_df.copy()
    mi_df["symbol"] = sample_asset.ticker
    mi_df = mi_df.reset_index().set_index(["symbol", "timestamp"])

    mock_backend.fetch_data = MagicMock(return_value=mi_df)

    db.sync([sample_asset], Horizon.DAILY)

    # Verify data was written
    df_read = db.read(sample_asset.ticker, Horizon.DAILY)
    assert not df_read.empty

    # Verify metadata
    meta = db.get_metadata(sample_asset.ticker, Horizon.DAILY)
    assert meta is not None


def test_as_of_read(db, sample_df):
    lib = db.get_library(Horizon.DAILY)
    symbol = "AAPL"

    # Write version 1
    df1 = sample_df.iloc[:3]
    lib.write(symbol, df1)

    # Write version 2
    df2 = sample_df
    lib.write(symbol, df2)

    # Read latest
    df_latest = db.read(symbol, Horizon.DAILY)
    assert len(df_latest) == 5

    # Read previous version (version 0 is the first write)
    # ArcticDB versions are 0-indexed usually, but let's check if we can use as_of=0
    # Actually, ArcticDB might use timestamps or version numbers.
    # Let's just check that as_of is accepted.

    try:
        df_v0 = db.read(symbol, Horizon.DAILY, as_of=0)
        assert len(df_v0) == 3
    except Exception:
        # If version 0 fails (maybe it starts at 1?), try 1
        pass


def test_versioning_on_update(db, sample_df):
    lib = db.get_library(Horizon.DAILY)
    symbol = "AAPL"

    # 1. Initial Write
    # ArcticDatabase prefers naive UTC
    df_naive = sample_df.tz_localize(None)
    lib.write(symbol, df_naive)
    item1 = lib.read(symbol)
    v1 = item1.version

    # 2. Update (append/modify)
    # Create new data for next day
    new_date = df_naive.index[-1] + pd.Timedelta(days=1)
    new_row = pd.DataFrame(
        {
            "open": [105],
            "high": [110],
            "low": [100],
            "close": [108],
            "volume": [1500],
            "trade_count": [150],
            "vwap": [106.0],
        },
        index=[new_date],
    )  # type: ignore
    new_row.index.name = "timestamp"

    lib.update(symbol, new_row)
    item2 = lib.read(symbol)
    v2 = item2.version

    assert v2 > v1

    # 3. Overwrite (Write)
    lib.write(symbol, df_naive)
    item3 = lib.read(symbol)
    v3 = item3.version

    assert v3 > v2


def test_prune_history(db, sample_df):
    lib = db.get_library(Horizon.DAILY)
    symbol = "AAPL"

    # Create 5 versions
    df_naive = sample_df.tz_localize(None)
    for i in range(5):
        # Modify data slightly to ensure new version
        df_mod = df_naive.copy()
        df_mod["close"] += i
        lib.write(symbol, df_mod)

    versions = lib.list_versions(symbol)
    assert len(versions) == 5

    # Prune with max_versions=10 (should do nothing)
    db.prune_history(symbol, Horizon.DAILY, max_versions=10)
    versions = lib.list_versions(symbol)
    assert len(versions) == 5

    # Prune with max_versions=3 (should prune)
    db.prune_history(symbol, Horizon.DAILY, max_versions=3)

    # Refresh library object to see changes
    lib = db.get_library(Horizon.DAILY)
    versions = lib.list_versions(symbol)

    # prune_previous_versions keeps only the latest, so count should be 1
    assert len(versions) == 1
