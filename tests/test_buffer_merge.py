import pytest
import pandas as pd
from datetime import timedelta
from silkroad.core.base_models import UniformBarSet, Horizon
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


def test_buffer_merging():
    start_time = pd.Timestamp("2023-01-01 09:30:00", tz="UTC")
    symbol = "AAPL"

    # Initialize with small buffer limit
    ubs = UniformBarSet(symbol=symbol, horizon=Horizon.MINUTE, buffer_limit=3)

    # Push 2 bars (under limit)
    for i in range(2):
        bar = create_dummy_bar(symbol, start_time + timedelta(minutes=i))
        ubs.push(bar)

    assert len(ubs._buffer) == 2
    assert len(ubs._df) == 0

    # Push 3rd bar (reaches limit)
    bar3 = create_dummy_bar(symbol, start_time + timedelta(minutes=2))
    ubs.push(bar3)

    # Should have flushed
    assert len(ubs._buffer) == 0
    assert len(ubs._df) == 3
    assert len(ubs) == 3

    # Verify data integrity
    df = ubs.df
    assert len(df) == 3
    assert df.index[-1] == bar3.timestamp
