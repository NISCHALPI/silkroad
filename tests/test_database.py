import pytest
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch
from silkroad.app.database import MarketDataDB
from silkroad.core.data_models import Asset, Horizon, AssetClass, Exchange
from alpaca.data.enums import Adjustment

# Fixtures


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "stockdb"


@pytest.fixture
def db(temp_db_path):
    return MarketDataDB(db_path=temp_db_path, horizon=Horizon.DAILY)


@pytest.fixture
def mock_alpaca_stock_client():
    with patch("silkroad.app.database.StockHistoricalDataClient") as mock:
        yield mock


@pytest.fixture
def mock_alpaca_crypto_client():
    with patch("silkroad.app.database.CryptoHistoricalDataClient") as mock:
        yield mock


@pytest.fixture
def sample_asset():
    return Asset(
        ticker="AAPL",
        name="Apple Inc.",
        asset_class=AssetClass.STOCK,
        exchange=Exchange.NASDAQ,
        sector=None,
    )


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.random.rand(len(dates)) * 100,
            "high": np.random.rand(len(dates)) * 100,
            "low": np.random.rand(len(dates)) * 100,
            "close": np.random.rand(len(dates)) * 100,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "trade_count": np.random.randint(10, 100, len(dates)),
            "vwap": np.random.rand(len(dates)) * 100,
        },
        index=dates,
    )
    # MultiIndex as returned by Alpaca
    df["symbol"] = "AAPL"
    df = df.reset_index().set_index(["symbol", "index"])
    return df


# Tests


def test_initialization(temp_db_path):
    db = MarketDataDB(db_path=temp_db_path, horizon=Horizon.DAILY)
    assert db.db_path == temp_db_path
    assert db.horizon == Horizon.DAILY
    assert db.library_name == "daily"


def test_initialization_invalid_horizon():
    with pytest.raises(ValueError, match="Unsupported horizon"):
        MarketDataDB(horizon=Horizon.SECONDS)


def test_write_and_read(db, sample_asset, sample_data):
    # Manually write data to test read
    # We need to strip symbol level for write as per implementation
    ticker_data = sample_data.xs("AAPL", level="symbol")
    # Implementation expects naive UTC for write usually, but let's see how we handled it.
    # The refactor ensures we convert to UTC and localize to None before write.

    # Let's use _write_asset_data directly to test it
    end_date = dt.datetime.now(dt.timezone.utc)
    db._write_asset_data("AAPL", ticker_data, sample_asset, end_date)

    assert "AAPL" in db.get_available_tickers()

    # Test get_available_date_range
    start, end = db.get_available_date_range("AAPL")
    assert start.year == 2023
    assert end.year == 2023
    assert start.tzinfo == dt.timezone.utc

    # Test get_ticker_data
    barset = db.get_ticker_data("AAPL", start, end)
    assert len(barset.df) == 10
    assert str(barset.df.index.tz) == "UTC"


def test_delete_ticker(db, sample_asset, sample_data):
    end_date = dt.datetime.now(dt.timezone.utc)
    ticker_data = sample_data.xs("AAPL", level="symbol")
    db._write_asset_data("AAPL", ticker_data, sample_asset, end_date)

    assert "AAPL" in db.get_available_tickers()

    db.delete_ticker("AAPL")
    assert "AAPL" not in db.get_available_tickers()


def test_get_latest_bar(db, sample_asset, sample_data):
    end_date = dt.datetime.now(dt.timezone.utc)
    ticker_data = sample_data.xs("AAPL", level="symbol")
    db._write_asset_data("AAPL", ticker_data, sample_asset, end_date)

    latest = db.get_latest_bar("AAPL")
    assert latest is not None
    assert latest.name == ticker_data.index[-1].replace(
        tzinfo=None
    )  # ArcticDB returns naive


def test_update_from_alpaca_full_refresh(
    db, mock_alpaca_stock_client, sample_asset, sample_data
):
    # Setup mock
    mock_instance = mock_alpaca_stock_client.return_value
    mock_instance.get_stock_bars.return_value.df = sample_data

    db.update_from_alpaca(
        api_key="key",
        api_secret="secret",
        assets=[sample_asset],
        start=dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc),
        end=dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
        update_mode="full_refresh",
    )

    assert "AAPL" in db.get_available_tickers()
    bars = db.get_ticker_data(
        "AAPL",
        dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
    )
    assert len(bars.df) == 10


def test_update_from_alpaca_smart_merge_no_overlap(
    db, mock_alpaca_stock_client, sample_asset, sample_data
):
    # Initial write
    ticker_data = sample_data.xs("AAPL", level="symbol")
    initial_data = ticker_data.iloc[:5]  # First 5 days
    db._write_asset_data(
        "AAPL", initial_data, sample_asset, dt.datetime.now(dt.timezone.utc)
    )

    # Mock new data (next 5 days)
    new_data = sample_data.iloc[5:]  # Last 5 days
    mock_instance = mock_alpaca_stock_client.return_value
    mock_instance.get_stock_bars.return_value.df = new_data

    db.update_from_alpaca(
        api_key="key",
        api_secret="secret",
        assets=[sample_asset],
        start=dt.datetime(2023, 1, 6, tzinfo=dt.timezone.utc),
        end=dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
        update_mode="smart_merge",
    )

    bars = db.get_ticker_data(
        "AAPL",
        dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
    )
    assert len(bars.df) == 10


def test_update_from_alpaca_smart_merge_with_overlap_no_conflict(
    db, mock_alpaca_stock_client, sample_asset, sample_data
):
    # Initial write: days 0-5
    ticker_data = sample_data.xs("AAPL", level="symbol")
    initial_data = ticker_data.iloc[:6]
    db._write_asset_data(
        "AAPL", initial_data, sample_asset, dt.datetime.now(dt.timezone.utc)
    )

    # Mock new data: days 5-9 (day 5 overlaps)
    new_data = sample_data.iloc[5:]
    mock_instance = mock_alpaca_stock_client.return_value
    mock_instance.get_stock_bars.return_value.df = new_data

    db.update_from_alpaca(
        api_key="key",
        api_secret="secret",
        assets=[sample_asset],
        start=dt.datetime(2023, 1, 6, tzinfo=dt.timezone.utc),
        end=dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
        update_mode="smart_merge",
    )

    bars = db.get_ticker_data(
        "AAPL",
        dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
    )
    assert len(bars.df) == 10


def test_update_from_alpaca_smart_merge_conflict_triggers_refresh(
    db, mock_alpaca_stock_client, sample_asset, sample_data
):
    # Initial write: days 0-5
    ticker_data = sample_data.xs("AAPL", level="symbol")
    initial_data = ticker_data.iloc[:6].copy()
    initial_data.iloc[-1, initial_data.columns.get_loc("close")] = (
        100.0  # Set stored close
    )
    db._write_asset_data(
        "AAPL", initial_data, sample_asset, dt.datetime.now(dt.timezone.utc)
    )

    # Mock new data: days 5-9, but day 5 has different price (split/dividend adjusted)
    new_data = sample_data.iloc[5:].copy()
    new_data.iloc[0, new_data.columns.get_loc("close")] = (
        50.0  # 50% drop, should trigger refresh
    )

    mock_instance = mock_alpaca_stock_client.return_value
    # First call returns new_data (overlap check)
    # Second call (full refresh) returns ALL data (simulated by just returning new_data for simplicity of mock,
    # but in reality it would ask for full history.
    # Let's make the mock smart enough to return full data on second call if we can, or just verify call args.

    # We'll just return new_data for the first call.
    # For the recursive full_refresh call, we need to return the FULL dataset.

    def side_effect(request_params):
        if request_params.start.year == 2015:  # Full refresh start date
            # Return full modified dataset
            full_df = sample_data.copy()
            # Apply the "split" to the whole history
            full_df["close"] = full_df["close"] * 0.5
            return MagicMock(df=full_df)
        else:
            return MagicMock(df=new_data)

    mock_instance.get_stock_bars.side_effect = side_effect

    # We need to patch the recursive call or verify it happened.
    # Actually, we can just verify the data in DB is updated.

    with patch.object(
        db, "update_from_alpaca", wraps=db.update_from_alpaca
    ) as spy_update:
        db.update_from_alpaca(
            api_key="key",
            api_secret="secret",
            assets=[sample_asset],
            start=dt.datetime(2023, 1, 6, tzinfo=dt.timezone.utc),
            end=dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc),
            update_mode="smart_merge",
        )

        # Verify recursive call
        assert spy_update.call_count == 2
        assert spy_update.call_args_list[1].kwargs["update_mode"] == "full_refresh"
