import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import jax
from silkroad.core.data_models import UniformBarSet, UniformBarCollection
from silkroad.core.enums import Horizon
from alpaca.data.models.bars import Bar


@pytest.fixture
def dummy_bars():
    symbol = "AAPL"
    start_price = 150.0
    n_bars = 100
    bars = []
    price = start_price
    start_time = datetime(2023, 1, 1)

    # Create deterministic random walk for reproducibility in tests
    np.random.seed(42)

    for i in range(n_bars):
        t = start_time + timedelta(days=i)
        price = price * (1 + np.random.normal(0, 0.01))
        data = {
            "t": t,
            "o": price,
            "h": price * 1.01,
            "l": price * 0.99,
            "c": price,
            "v": 1000,
            "n": 100,
            "vw": price,
        }
        bars.append(Bar(symbol=symbol, raw_data=data))
    return bars


@pytest.fixture
def dummy_collection(dummy_bars):
    bars1 = dummy_bars

    # Create second asset
    symbol2 = "GOOG"
    start_price2 = 2800.0
    n_bars = len(bars1)
    bars2 = []
    price = start_price2
    start_time = datetime(2023, 1, 1)

    np.random.seed(43)

    for i in range(n_bars):
        t = start_time + timedelta(days=i)
        price = price * (1 + np.random.normal(0, 0.01))
        data = {
            "t": t,
            "o": price,
            "h": price * 1.01,
            "l": price * 0.99,
            "c": price,
            "v": 1000,
            "n": 100,
            "vw": price,
        }
        bars2.append(Bar(symbol=symbol2, raw_data=data))

    ubs1 = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, initial_bars=bars1)
    ubs2 = UniformBarSet(symbol="GOOG", horizon=Horizon.DAILY, initial_bars=bars2)

    return UniformBarCollection(bar_map={"AAPL": ubs1, "GOOG": ubs2})


def test_uniform_bar_set_sample_gbm(dummy_bars):
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, initial_bars=dummy_bars)
    n_paths = 5

    samples = ubs.sample(n_paths=n_paths, method="gbm", key=42)

    assert len(samples) == n_paths
    assert isinstance(samples[0], UniformBarSet)
    assert len(samples[0]) == len(dummy_bars)
    assert samples[0].symbol == "AAPL"

    # Check timestamps match
    assert samples[0].df.index.equals(ubs.df.index)

    # Check that prices are different from original (it's a simulation)
    # But start price should be close to original start price (since we set init_price=prices[0])
    # Actually, GBM paths start at init_price.
    original_start = ubs.df["close"].iloc[0]
    sample_start = samples[0].df["close"].iloc[0]
    assert np.isclose(original_start, sample_start)


def test_uniform_bar_set_sample_mbb(dummy_bars):
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, initial_bars=dummy_bars)
    n_paths = 5

    samples = ubs.sample(n_paths=n_paths, method="mbb", key=42)

    assert len(samples) == n_paths
    assert len(samples[0]) == len(dummy_bars)
    # Timestamps match
    assert samples[0].df.index.equals(ubs.df.index)


def test_uniform_bar_collection_sample_gbm(dummy_collection):
    n_paths = 5
    samples = dummy_collection.sample(n_paths=n_paths, method="gbm", key=42)

    assert len(samples) == n_paths
    assert isinstance(samples[0], UniformBarCollection)
    assert len(samples[0].bar_map) == 2
    assert samples[0].n_bars == dummy_collection.n_bars
    assert samples[0].timestamps.equals(dummy_collection.timestamps)

    # Check independence (rough check)
    # In independent GBM, correlations should be close to 0 (or not necessarily preserved)
    # But we didn't implement correlation preservation for GBM yet in the code,
    # so this test just checks structure.


def test_uniform_bar_collection_sample_cbb_multivariate(dummy_collection):
    n_paths = 5
    # CBB preserves correlation
    samples = dummy_collection.sample(n_paths=n_paths, method="cbb", key=42)

    assert len(samples) == n_paths
    assert isinstance(samples[0], UniformBarCollection)
    assert len(samples[0].bar_map) == 2
    assert samples[0].n_bars == dummy_collection.n_bars

    # Check that we have data for both symbols
    assert "AAPL" in samples[0].bar_map
    assert "GOOG" in samples[0].bar_map


def test_sample_invalid_method(dummy_bars):
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, initial_bars=dummy_bars)
    with pytest.raises(ValueError, match="Unknown sampling method"):
        ubs.sample(n_paths=1, method="invalid_method")


def test_sample_empty_set():
    ubs = UniformBarSet(symbol="AAPL", horizon=Horizon.DAILY, initial_bars=[])
    with pytest.raises(ValueError, match="Cannot sample from empty UniformBarSet"):
        ubs.sample(n_paths=1)
