import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from silkroad.core.data_models import UniformBarSet, Horizon
from alpaca.data.models import Bar


@pytest.fixture
def start_time():
    return pd.Timestamp("2023-01-01 09:30:00", tz="UTC")


@pytest.fixture
def symbol():
    return "AAPL"


def create_dummy_bar(symbol, timestamp, price=100.0):
    return Bar(
        symbol=symbol,
        raw_data={
            "t": timestamp,
            "o": price,
            "h": price + 1,
            "l": price - 1,
            "c": price,
            "v": 1000,
            "n": 10,
            "vw": price,
        },
    )


@pytest.fixture
def sample_bars(symbol, start_time):
    return [
        create_dummy_bar(symbol, start_time + timedelta(minutes=i), 100 + i)
        for i in range(5)
    ]


def test_initialization(symbol, sample_bars):
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, initial_bars=sample_bars)
    assert len(ubs) == 5
    assert len(ubs._df) == 5
    assert len(ubs._buffer) == 0
    assert ubs.symbol == symbol
    assert ubs.horizon == Horizon.MINUTE


def test_push(symbol, sample_bars, start_time):
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, initial_bars=sample_bars)
    new_bar = create_dummy_bar(symbol, start_time + timedelta(minutes=5), 105)

    ubs.push(new_bar)

    assert len(ubs) == 6
    assert len(ubs._df) == 5
    assert len(ubs._buffer) == 1
    assert ubs.peek.timestamp == new_bar.timestamp


def test_dataframe_access(symbol, sample_bars, start_time):
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, initial_bars=sample_bars)
    new_bar = create_dummy_bar(symbol, start_time + timedelta(minutes=5), 105)
    ubs.push(new_bar)

    df = ubs.df
    assert len(df) == 6
    assert df.index[-1] == new_bar.timestamp
    # Verify columns exist
    expected_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for col in expected_cols:
        assert col in df.columns


def test_pop(symbol, sample_bars, start_time):
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, initial_bars=sample_bars)
    new_bar = create_dummy_bar(symbol, start_time + timedelta(minutes=5), 105)
    ubs.push(new_bar)

    # Pop from history
    first_bar = ubs.pop()
    assert first_bar.timestamp == sample_bars[0].timestamp
    assert len(ubs) == 5
    assert len(ubs._df) == 4

    # Pop remaining history
    for _ in range(4):
        ubs.pop()

    assert len(ubs) == 1
    assert len(ubs._df) == 0
    assert len(ubs._buffer) == 1

    # Pop from buffer - this should fail because we can't empty the set
    with pytest.raises(IndexError, match="Cannot pop the last bar"):
        ubs.pop()

    assert len(ubs) == 1
    assert ubs.peek.timestamp == new_bar.timestamp


def test_resample(symbol, start_time):
    # Create enough data for resampling (2 hours worth of minute bars)
    bars_resample = [
        create_dummy_bar(symbol, start_time + timedelta(minutes=i), 100 + i)
        for i in range(120)
    ]
    ubs = UniformBarSet(
        symbol=symbol, horizon=Horizon.MINUTE, initial_bars=bars_resample
    )

    resampled = ubs.resample(Horizon.HOURLY)

    # Depending on start time alignment, we might get 2 or 3 bars.
    # 09:30 -> 10:00 (1 bar), 10:00 -> 11:00 (1 bar), 11:00 -> 11:30 (1 bar)
    # Pandas resample '1h' usually buckets by hour start.
    # 09:30-09:59 -> 09:00 bucket
    # 10:00-10:59 -> 10:00 bucket
    # 11:00-11:29 -> 11:00 bucket

    print(f"\nResampled DF:\n{resampled.df}")

    assert len(resampled) >= 2
    assert resampled.horizon == Horizon.HOURLY

    # Check values
    df = resampled.df
    assert df.iloc[0]["open"] == 100.0  # First bar open
    assert df.iloc[-1]["close"] == 100 + 119  # Last bar close


def test_empty_initialization(symbol):
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE)
    assert len(ubs) == 0
    assert ubs._df.empty
    assert not ubs._buffer


def test_invalid_push(symbol, sample_bars, start_time):
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, initial_bars=sample_bars)

    # Wrong symbol
    bad_bar = create_dummy_bar("WRONG", start_time + timedelta(minutes=10))
    with pytest.raises(ValueError, match="symbol"):
        ubs.push(bad_bar)

    # Out of order
    old_bar = create_dummy_bar(symbol, start_time)
    with pytest.raises(ValueError, match="chronological"):
        ubs.push(old_bar)
