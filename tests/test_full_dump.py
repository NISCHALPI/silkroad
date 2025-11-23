import pytest
import pandas as pd
from silkroad.core.base_models import UniformBarSet, Horizon
from alpaca.data.models import Bar


def test_full_dump_inclusion():
    bar = Bar(
        symbol="AAPL",
        raw_data={
            "t": pd.Timestamp("2023-01-01", tz="UTC"),
            "o": 100,
            "h": 101,
            "l": 99,
            "c": 100,
            "v": 1000,
            "n": 10,
            "vw": 100,
        },
    )
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, bars=[bar])

    # Default dump should include bars
    dump = ubs.model_dump()
    assert "bars" in dump
    assert isinstance(dump["bars"], list)
    assert len(dump["bars"]) == 1
    # Check if it's a dict (serialized) or object
    # Pydantic model_dump usually dumps sub-models as dicts
    # Bar is not a Pydantic model, it's a class from alpaca.
    # Wait, UniformBarSet uses Bar which is NOT a Pydantic model.
    # Pydantic might not serialize non-Pydantic objects automatically unless they have __dict__ or similar.
    # Let's check what it actually dumps.


def test_dump_exclusion():
    bar = Bar(
        symbol="AAPL",
        raw_data={
            "t": pd.Timestamp("2023-01-01", tz="UTC"),
            "o": 100,
            "h": 101,
            "l": 99,
            "c": 100,
            "v": 1000,
            "n": 10,
            "vw": 100,
        },
    )
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, bars=[bar])

    # Exclude bars
    dump = ubs.model_dump(exclude={"bars"})
    assert "bars" not in dump
