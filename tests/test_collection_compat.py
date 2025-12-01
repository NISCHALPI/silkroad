import pytest
import pandas as pd
from datetime import timedelta
from silkroad.core.data_models import UniformBarSet, UniformBarCollection, Horizon
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


def test_collection_compatibility(start_time):
    # Create two compatible bar sets
    bars1 = [
        create_dummy_bar("AAPL", start_time + timedelta(minutes=i), 100 + i)
        for i in range(5)
    ]
    bars2 = [
        create_dummy_bar("GOOGL", start_time + timedelta(minutes=i), 200 + i)
        for i in range(5)
    ]

    ubs1 = UniformBarSet(symbol="AAPL", horizon=Horizon.MINUTE, initial_bars=bars1)
    ubs2 = UniformBarSet(symbol="GOOGL", horizon=Horizon.MINUTE, initial_bars=bars2)

    # Initialize Collection
    collection = UniformBarCollection(bar_map={"AAPL": ubs1, "GOOGL": ubs2})

    assert len(collection) == 2
    assert collection.horizon == Horizon.MINUTE
    assert collection.n_bars == 5

    # Test DF access
    df = collection.df
    assert isinstance(df.index, pd.MultiIndex)
    assert len(df) == 10  # 5 bars * 2 assets

    # Test Push
    new_bar1 = create_dummy_bar("AAPL", start_time + timedelta(minutes=5), 105)
    new_bar2 = create_dummy_bar("GOOGL", start_time + timedelta(minutes=5), 205)

    new_bars = {
        "AAPL": UniformBarSet(
            symbol="AAPL", horizon=Horizon.MINUTE, initial_bars=[new_bar1]
        ),
        "GOOGL": UniformBarSet(
            symbol="GOOGL", horizon=Horizon.MINUTE, initial_bars=[new_bar2]
        ),
    }

    collection.push(new_bars)
    assert collection.n_bars == 6

    # Test Pop
    popped = collection.pop()
    assert popped["AAPL"].timestamp == bars1[0].timestamp
    assert popped["GOOGL"].timestamp == bars2[0].timestamp
    assert collection.n_bars == 5


def test_collection_caching(start_time):
    # Setup
    bars = [
        create_dummy_bar("AAPL", start_time + timedelta(minutes=i), 100 + i)
        for i in range(5)
    ]
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.MINUTE, initial_bars=bars)
    collection = UniformBarCollection(bar_map={"AAPL": ubs})

    # First access - should cache
    df1 = collection.df
    assert collection._df_cache is not None
    assert df1 is collection._df_cache

    # Second access - should return same object
    df2 = collection.df
    assert df2 is df1

    # Push - should invalidate
    new_bar = create_dummy_bar("AAPL", start_time + timedelta(minutes=5), 105)
    collection.push(
        {
            "AAPL": UniformBarSet(
                symbol="AAPL", horizon=Horizon.MINUTE, initial_bars=[new_bar]
            )
        }
    )
    assert collection._df_cache is None

    # Access again - should rebuild and cache
    df3 = collection.df
    assert collection._df_cache is not None
    assert df3 is not df1

    # Pop - should invalidate
    collection.pop()
    assert collection._df_cache is None
