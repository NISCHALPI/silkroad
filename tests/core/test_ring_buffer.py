import pytest
import pandas as pd
from datetime import timedelta
from silkroad.core.data_models import UniformBarSet, Horizon
from alpaca.data.models import Bar


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
def start_time():
    return pd.Timestamp("2023-01-01 09:30:00", tz="UTC")


def test_ring_buffer_initialization(start_time):
    symbol = "AAPL"
    bars = [
        create_dummy_bar(symbol, start_time + timedelta(minutes=i), 100 + i)
        for i in range(10)
    ]

    # Init with max_bars=5
    ubs = UniformBarSet(
        symbol=symbol, horizon=Horizon.MINUTE, initial_bars=bars, max_bars=5
    )

    assert len(ubs) == 5
    # Should have kept the last 5 bars
    assert ubs.df.index[0] == bars[5].timestamp
    assert ubs.df.index[-1] == bars[9].timestamp


def test_ring_buffer_push(start_time):
    symbol = "AAPL"
    bars = [
        create_dummy_bar(symbol, start_time + timedelta(minutes=i), 100 + i)
        for i in range(5)
    ]

    # Init with max_bars=5 (full)
    ubs = UniformBarSet(
        symbol=symbol, horizon=Horizon.MINUTE, initial_bars=bars, max_bars=5
    )
    assert len(ubs) == 5

    # Push new bar
    new_bar = create_dummy_bar(symbol, start_time + timedelta(minutes=5), 105)
    ubs.push(new_bar)

    # Length should remain 5
    assert len(ubs) == 5
    # Oldest bar should be removed
    assert ubs.df.index[0] == bars[1].timestamp
    # Newest bar should be present
    assert ubs.peek.timestamp == new_bar.timestamp


def test_infinite_capacity(start_time):
    symbol = "AAPL"
    bars = [
        create_dummy_bar(symbol, start_time + timedelta(minutes=i), 100 + i)
        for i in range(5)
    ]

    # Init with max_bars=None
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, initial_bars=bars)

    new_bar = create_dummy_bar(symbol, start_time + timedelta(minutes=5), 105)
    ubs.push(new_bar)

    assert len(ubs) == 6
