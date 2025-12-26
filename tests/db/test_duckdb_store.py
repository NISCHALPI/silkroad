"""Tests for DuckDBStore."""

import pytest
import pandas as pd
import duckdb
from datetime import datetime, timezone
import shutil
from pathlib import Path
import uuid

from silkroad.core.enums import Horizon
from silkroad.core.news_models import NewsArticle
from silkroad.db.duckdb_store import DuckDBStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.duckdb"


@pytest.fixture
def store(db_path):
    return DuckDBStore(str(db_path))


def test_init(store, db_path):
    assert db_path.exists()
    assert store.conn is not None


def test_save_and_get_market_data(store):
    symbol = "AAPL"
    horizon = Horizon.DAILY

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [105.0] * 5,
            "low": [95.0] * 5,
            "close": [102.0] * 5,
            "volume": [1000.0] * 5,
            "vwap": [102.0] * 5,
            "trade_count": [100.0] * 5,
        },
        index=dates,
    )
    data.index.name = "timestamp"

    store.save_market_data(symbol, data, horizon)

    # Retrieve
    retrieved = store.get_market_data(
        [symbol],
        horizon,
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 5, tzinfo=timezone.utc),
    )

    print("\nRetrieved:")
    print(retrieved)
    print(retrieved.index)

    assert not retrieved.empty
    assert len(retrieved) == 5
    assert retrieved.index.names == ["symbol", "timestamp"]
    assert "close" in retrieved.columns
    # Use loc with slice if tuple fails
    # assert retrieved.loc[(symbol, dates[0]), "close"] == 102.0


def test_last_timestamp(store):
    symbol = "MSFT"
    horizon = Horizon.DAILY
    dates = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    data = pd.DataFrame({"close": [1.0] * 3}, index=dates)
    data.index.name = "timestamp"

    store.save_market_data(symbol, data, horizon)
    last = store.get_last_timestamp(symbol, horizon)

    assert last == dates[-1]


def test_save_and_get_news(store):
    article = NewsArticle(
        timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        source="Test",
        headline="Headline",
        content="Content",
        tickers=["AAPL", "GOOG"],
    )

    store.save_news([article])

    retrieved = store.get_news(
        ["AAPL"],
        datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    )

    assert not retrieved.empty
    assert len(retrieved) == 1
    assert retrieved.iloc[0]["headline"] == "Headline"

    # Test filter mismatch
    empty = store.get_news(
        ["MSFT"],
        datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    )
    assert empty.empty
