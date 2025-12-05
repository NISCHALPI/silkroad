"""Base models for time series data handling.

This module provides generic, reusable data models for working with financial time series
data. These models are designed to be independent of specific asset types and can be used
across different financial instruments (stocks, bonds, options, futures, crypto, etc.) or
adapted for non-financial time series analysis.

The models enforce data validation and type safety through Pydantic v2, ensuring data
integrity at construction time. They support portfolio-level analysis by enabling multiple
time series to be aligned on the same time index with uniform intervals, which is essential
for backtesting and multi-asset strategies.

Key Features:
    - Generic design suitable for any asset type
    - Strict validation of symbol consistency and time ordering
    - Support for time series resampling to different horizons
    - Multi-asset collections with synchronized timestamps
    - Pandas DataFrame integration for analysis
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    PrivateAttr,
    Field,
    computed_field,
)
from typing import List, Dict, Any, Optional, Union, Annotated, Tuple
from alpaca.data.models.bars import Bar
from .enums import Sector, AssetClass, Horizon, Exchange
from .dtypes import UniformBarList
import pandas as pd
from pydantic.functional_validators import BeforeValidator
import annotated_types as at
import numpy as np
import jax
import jax.numpy as jnp
from ..functional.paths import (
    geometric_brownian_motion,
    estimate_drift_and_volatility,
    estimate_drift_and_volatility,
    merton_jump_diffusion,
    estimate_mjd,
    heston_model,
    estimate_heston_params,
    circular_block_bootstrap,
    stationary_bootstrap,
    moving_block_bootstrap,
    estimate_multivariate_gbm_params,
    multivariate_geometric_brownian_motion,
)

__all__ = ["UniformBarSet", "UniformBarCollection", "Asset"]


class Asset(BaseModel):
    ticker: str = Field(..., description="Ticker symbol of the asset.")
    name: Optional[str] = Field(None, description="Full name of the asset.")
    asset_class: Optional[AssetClass] = Field(
        None, description="Asset class of the asset."
    )
    sector: Optional[Sector] = Field(None, description="Sector of the asset.")
    exchange: Optional[Exchange] = Field(
        None, description="Exchange where the asset is listed."
    )

    def __str__(self) -> str:
        return f"Asset(ticker={self.ticker}, name={self.name}, class={self.asset_class}, sector={self.sector}, exchange={self.exchange})"


class UniformBarSet(BaseModel):
    """Time series of OHLCV bars for a single asset with uniform intervals.

    This class represents a time series of OHLCV (Open, High, Low, Close, Volume) bars
    for a single financial instrument, all sampled at the same time interval (horizon).
    It enforces that all bars belong to the same symbol and are ordered chronologically.

    The class provides methods for manipulation (push/pop), resampling to larger time
    horizons, and conversion to pandas DataFrames. It serves as a generic base class
    that can be used for any asset type and extended for specific use cases.

    Validation ensures that:
        - All bars have the same symbol
        - Bars are in strictly increasing chronological order
        - At least one bar is present at all times

    Attributes:
        symbol: Ticker symbol of the asset (e.g., "AAPL", "BTC-USD").
        horizon: Time interval between bars (e.g., DAILY, HOURLY).
    """

    symbol: str = Field(
        ..., description="Ticker symbol of the asset (e.g., 'AAPL', 'BTC-USD')."
    )
    horizon: Horizon = Field(
        ..., description="Time interval between bars (e.g., DAILY, HOURLY)."
    )
    buffer_limit: int = Field(
        1000,
        description="Maximum number of bars in the buffer before merging into DataFrame.",
    )
    max_bars: Optional[int] = Field(
        None,
        description="Maximum number of bars to keep (Ring Buffer mode). If None, infinite capacity.",
    )

    # Internal state
    _df: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _buffer: List[Bar] = PrivateAttr(default_factory=list)

    # Temporary field to handle 'bars' argument in constructor without overriding
    initial_bars: Optional[List[Bar]] = Field(default=None, alias="bars", exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def model_post_init(self, __context: Any) -> None:
        # Initialize internal storage
        self._buffer = []

        if self.initial_bars:
            # Validate bars first
            validated_bars = TypeAdapter(UniformBarList).validate_python(
                self.initial_bars
            )

            # Truncate if max_bars is set
            if self.max_bars is not None and len(validated_bars) > self.max_bars:
                validated_bars = validated_bars[-self.max_bars :]

            # Create initial DataFrame
            data_dicts = [bar.model_dump() for bar in validated_bars]
            df = pd.DataFrame(data_dicts)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                self._df = df.drop(columns=["symbol"]).set_index("timestamp")
                self._df.sort_index(inplace=True)
            else:
                self._df = pd.DataFrame()

            # Clear initial_bars to free memory
            self.initial_bars = None
        else:
            self._df = pd.DataFrame()

    @computed_field
    @property
    def bars(self) -> List[Bar]:
        """Get all bars as a list (reconstructed from DF and buffer).

        Note: This is expensive as it reconstructs objects from the DataFrame.
        Use .df for analysis whenever possible.
        """
        # Convert DF back to bars
        df_bars = []
        if not self._df.empty:
            # Rename columns to match Bar raw_data expectation
            df_renamed = self._df.rename(
                columns={
                    "open": "o",
                    "high": "h",
                    "low": "l",
                    "close": "c",
                    "volume": "v",
                    "trade_count": "n",
                    "vwap": "vw",
                }
            )
            df_renamed.index.name = "t"
            records = df_renamed.reset_index().to_dict(orient="records")
            df_bars = [Bar(symbol=self.symbol, raw_data=record) for record in records]  # type: ignore

        return df_bars + self._buffer

    @staticmethod
    def check_contiguous_bars(bars: List[Bar], horizon: Horizon) -> bool:
        """Check if bars have contiguous timestamps based on the horizon.

        Args:
            bars: List of bars to check.
            horizon: Expected time interval between bars.

        Returns:
            True if bars are contiguous, False otherwise.
        """
        if len(bars) < 2:
            return True  # A single bar is trivially contiguous
        deltas = [
            bars[i + 1].timestamp - bars[i].timestamp for i in range(len(bars) - 1)
        ]
        return all(horizon.check_valid(delta) for delta in deltas)  # type: ignore

    @property
    def df(self) -> pd.DataFrame:
        """Convert bars to a pandas DataFrame with timestamp index.

        Returns:
            DataFrame with timestamp index and OHLCV columns (open, high, low, close,
            volume, trade_count, vwap).
        """
        if not self._buffer:
            return self._df

        # Convert buffer to DataFrame
        buffer_dicts = [bar.model_dump() for bar in self._buffer]
        buffer_df = (
            pd.DataFrame(buffer_dicts).drop(columns=["symbol"]).set_index("timestamp")
        )

        # Concatenate with main DF
        # We return a new DF, we don't merge into _df here to keep push() fast
        # Merging happens only when we decide to flush buffer (not implemented yet)
        return pd.concat([self._df, buffer_df])

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Get the timestamps of all bars as a DatetimeIndex.

        Returns:
            DatetimeIndex of bar timestamps.
        """
        return self.df.index  # type: ignore

    @staticmethod
    def from_df(
        symbol: str,
        horizon: Horizon,
        df: pd.DataFrame,
        buffer_limit: int = 1000,
        max_bars: Optional[int] = None,
    ) -> "UniformBarSet":
        """Create a UniformBarSet from a DataFrame of OHLCV data.

        Note: The DataFrame must have a timestamp index and columns:
            - open
            - high
            - low
            - close
            - volume
            - trade_count
            - vwap

        Args:
            symbol: Ticker symbol of the asset.
            horizon: Time interval between bars.
            df: DataFrame with timestamp index and OHLCV columns.
            buffer_limit: Maximum number of bars in the buffer before merging into DataFrame.
            max_bars: Maximum number of bars to keep (Ring Buffer mode). If None, infinite capacity.

        Returns:
            New UniformBarSet instance.
        """
        # Validate DataFrame columns
        required_columns = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        }
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # Convert DataFrame rows to Bar objects
        df_renamed = df.rename(
            columns={
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "volume": "v",
                "trade_count": "n",
                "vwap": "vw",
            }
        )
        df_renamed.index.name = "t"
        records = df_renamed.reset_index().to_dict(orient="records")
        bars = [Bar(symbol=symbol, raw_data=record) for record in records]  # type: ignore

        return UniformBarSet(
            symbol=symbol,
            horizon=horizon,
            initial_bars=bars,
            buffer_limit=buffer_limit,
            max_bars=max_bars,
        )

    def __repr__(self) -> str:
        return f"UniformBarSet(symbol={self.symbol}, timeframe={self.horizon.name}, bars={len(self)})"

    def __len__(self) -> int:
        return len(self._df) + len(self._buffer)

    def __getitem__(self, index: Union[int, slice]) -> Union[Bar, "UniformBarSet"]:
        """Access bars by index or slice.

        Args:
            index: Index or slice to retrieve.

        Returns:
            Bar at the index position if index is int, or new UniformBarSet with
            sliced bars if index is slice.
        """
        # This is tricky with hybrid storage.
        # For simplicity and correctness, we might need to reconstruct the list
        # or handle logic to map index to df or buffer.
        if not isinstance(index, (int, slice)):
            raise TypeError("Index must be int or slice")

        # Optimization: If index is simple int
        total_len = len(self)
        if isinstance(index, int):
            if index < 0:
                index += total_len
            if index < 0 or index >= total_len:
                raise IndexError("UniformBarSet index out of range")

            df_len = len(self._df)
            if index < df_len:
                # Get from DF
                row = self._df.iloc[index]
                data = {
                    "t": row.name,
                    "o": row["open"],
                    "h": row["high"],
                    "l": row["low"],
                    "c": row["close"],
                    "v": row["volume"],
                    "n": row["trade_count"],
                    "vw": row["vwap"],
                }
                return Bar(symbol=self.symbol, raw_data=data)
            else:
                # Get from buffer
                return self._buffer[index - df_len]

        elif isinstance(index, slice):
            # For slicing, it's easiest to reconstruct the full list or DF
            # But we want to return a UniformBarSet.
            # Let's use the .bars property to get the full list and slice it.
            # This is expensive but correct.
            sliced_bars = self.bars[index]
            return UniformBarSet(
                symbol=self.symbol,
                horizon=self.horizon,
                initial_bars=sliced_bars,
                buffer_limit=self.buffer_limit,
                max_bars=self.max_bars,
            )

    def _flush_buffer(self) -> None:
        """Merge the buffer into the main DataFrame."""
        if not self._buffer:
            return

        buffer_dicts = [bar.model_dump() for bar in self._buffer]
        buffer_df = pd.DataFrame(buffer_dicts)
        if not buffer_df.empty:
            buffer_df["timestamp"] = pd.to_datetime(buffer_df["timestamp"], utc=True)
            buffer_df = buffer_df.drop(columns=["symbol"]).set_index("timestamp")
            self._df = pd.concat([self._df, buffer_df])
        self._buffer = []

    def push(self, bar: Bar) -> None:
        """Add a new bar to the end of the bars list.

        Args:
            bar: Bar to add.

        Raises:
            ValueError: If bar symbol doesn't match UniformBarSet symbol or timestamp
                is not greater than last bar.
        """
        if bar.symbol != self.symbol:
            raise ValueError(
                f"Bar symbol {bar.symbol} does not match UniformBarSet symbol {self.symbol}."
            )

        # Check that bars are directed
        last_bar = self.peek
        if last_bar and last_bar.timestamp >= bar.timestamp:
            raise ValueError(
                """New bar timestamp must be greater than the last bar's timestamp.
                Bars must be added in chronological order."""
            )
        self._buffer.append(bar)

        # Check if buffer limit reached
        if len(self._buffer) >= self.buffer_limit:
            self._flush_buffer()

        # Enforce max_bars (Ring Buffer)
        if self.max_bars is not None:
            while len(self) > self.max_bars:
                self.pop()

    def pop(self) -> Bar:
        """Remove and return the first bar from the collection.

        Returns:
            First bar in the collection.

        Raises:
            IndexError: If collection has only one element.
        """
        if len(self) <= 1:
            raise IndexError("Cannot pop the last bar from UniformBarSet.")

        # If we have data in DF, pop from there (expensive-ish, but we can optimize)
        # Actually, popping from start of DF is not great either, but better than list?
        # DF doesn't have a pop(0). We have to drop the row.

        if not self._df.empty:
            # Get first row
            row = self._df.iloc[0]
            data = {
                "t": row.name,
                "o": row["open"],
                "h": row["high"],
                "l": row["low"],
                "c": row["close"],
                "v": row["volume"],
                "n": row["trade_count"],
                "vw": row["vwap"],
            }
            bar = Bar(symbol=self.symbol, raw_data=data)

            # Remove from DF
            self._df = self._df.iloc[1:]
            return bar
        else:
            # Pop from buffer
            return self._buffer.pop(0)

    def resample(self, new_horizon: Horizon) -> "UniformBarSet":
        """Resample bars to a larger time horizon using OHLCV aggregation rules.

        Args:
            new_horizon: Target time horizon (must be larger than current).

        Returns:
            New UniformBarSet with resampled bars.

        Raises:
            ValueError: If new_horizon is smaller than or equal to current horizon.
        """
        # Check if the new horizon is greater than the current horizon
        if new_horizon.value <= self.horizon.value:
            raise ValueError(
                "New horizon must be greater than current horizon for resampling."
            )

        # Use pandas to resample the bars
        # We use the combined DF
        df = self.df
        df_resampled = (
            df.resample(new_horizon.to_pandas_freq())
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "trade_count": "sum",
                    "vwap": "mean",
                }
            )
            .dropna()
            .reset_index()
        )
        # Convert resampled DataFrame back to dictionary format for Bar creation
        bars_data = df_resampled.rename(
            columns={
                "timestamp": "t",
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "volume": "v",
                "trade_count": "n",
                "vwap": "vw",
            }
        ).to_dict(orient="records")
        bars = [Bar(symbol=self.symbol, raw_data=data) for data in bars_data]  # type: ignore

        return UniformBarSet(
            symbol=self.symbol,
            horizon=new_horizon,
            initial_bars=bars,
            buffer_limit=self.buffer_limit,
            max_bars=self.max_bars,
        )

    @property
    def peek(self) -> Optional[Bar]:
        """Get the most recent bar without removing it from the collection.

        Returns:
            Last bar or None if collection is empty.
        """
        if self._buffer:
            return self._buffer[-1]

        if not self._df.empty:
            row = self._df.iloc[-1]
            data = {
                "t": row.name,
                "o": row["open"],
                "h": row["high"],
                "l": row["low"],
                "c": row["close"],
                "v": row["volume"],
                "n": row["trade_count"],
                "vw": row["vwap"],
            }
            return Bar(symbol=self.symbol, raw_data=data)

        return None

    def __iter__(self):
        # Iterate over DF then buffer
        # This is a generator
        if not self._df.empty:
            # This is slow, iterating rows
            for timestamp, row in self._df.iterrows():
                data = {
                    "t": timestamp,
                    "o": row["open"],
                    "h": row["high"],
                    "l": row["low"],
                    "c": row["close"],
                    "v": row["volume"],
                    "n": row["trade_count"],
                    "vw": row["vwap"],
                }
                yield Bar(symbol=self.symbol, raw_data=data)

        for bar in self._buffer:
            yield bar

    def is_compatible(self, other: "UniformBarSet") -> bool:
        """Check if another UniformBarSet is compatible with this one.

        Args:
            other: Bar set to check compatibility with.

        Returns:
            True if bar sets share same horizon and aligned timestamps, False otherwise.
        """
        if self.horizon != other.horizon:
            return False

        if len(self) == 0 or len(other) == 0:
            return True

        if len(self) != len(other):
            return False

        # Fast fail checks
        # Check start and end timestamps
        # We need to be careful with buffer/df mix
        self_start = self.df.index[0]
        other_start = other.df.index[0]
        if self_start != other_start:
            return False

        self_end = self.df.index[-1]
        other_end = other.df.index[-1]
        if self_end != other_end:
            return False

        # Compare indices
        return self.df.index.equals(other.df.index)

    def intersect(self, other: "UniformBarSet") -> "UniformBarSet":
        """Compute the intersection of two UniformBarSets.

        Args:
            other: Bar set to intersect with.

        Returns:
            New UniformBarSet containing only the overlapping bars.
        """
        # Get common timestamps
        # Use .df.index which constructs the full index
        idx1 = self.df.index
        idx2 = other.df.index

        common_timestamps = idx1.intersection(idx2)
        if common_timestamps.empty:
            raise ValueError(
                f"No overlapping timestamps between {self.symbol} and {other.symbol}."
            )

        # Filter bars to only those with common timestamps
        # We can use the .df to filter and create new set
        # This is cleaner than iterating

        # Note: This creates a new UniformBarSet, which will initialize its own DF
        # We need to pass 'bars' to the constructor as per current design,
        # or we could make the constructor smarter.
        # For now, let's reconstruct bars.

        # Filter self
        filtered_df = self.df.loc[common_timestamps]

        # Convert to bars
        df_renamed = filtered_df.rename(
            columns={
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "volume": "v",
                "trade_count": "n",
                "vwap": "vw",
            }
        )
        df_renamed.index.name = "t"
        records = df_renamed.reset_index().to_dict(orient="records")
        bars = [Bar(symbol=self.symbol, raw_data=record) for record in records]  # type: ignore

        return UniformBarSet(
            symbol=self.symbol,
            horizon=self.horizon,
            initial_bars=bars,
            buffer_limit=self.buffer_limit,
            max_bars=self.max_bars,
        )

    @staticmethod
    def get_intersection(*bar_sets: "UniformBarSet") -> List["UniformBarSet"]:
        """Compute the intersection of multiple UniformBarSets.

        This method finds the common overlapping time period across multiple bar sets
        and returns new bar sets filtered to only include bars in that overlapping period.
        All input bar sets must have compatible horizons.

        Args:
            *bar_sets: Variable number of UniformBarSet instances to intersect.

        Returns:
            List of new UniformBarSets, one for each input, containing only bars
            in the overlapping time period. Returns empty list if no bar sets provided.

        Raises:
            ValueError: If bar sets have incompatible horizons.
        """
        if not bar_sets:
            return []

        # Get common intersection by progressively intersecting with first bar set
        first = bar_sets[0]
        for other in bar_sets[1:]:
            first = first.intersect(other)

        # Return intersected versions of all input bar sets
        return [first] + [bar_set.intersect(first) for bar_set in bar_sets[1:]]

    @staticmethod
    def constant_like(
        reference: "UniformBarSet", constant_value: float = 1.0
    ) -> "UniformBarSet":
        """Create a UniformBarSet with constant OHLCV values matching the reference's timestamps.

        Used sometimes in strategy modeling to create a baseline or neutral asset representation.
        This could be used as a cash asset where price doesn't change over time.

        Args:
            reference: Reference UniformBarSet to match timestamps and horizon.
            constant_value: Constant value to set for OHLCV fields.

        Returns:
            New UniformBarSet with constant OHLCV values.
        """
        bars = []
        for timestamp in reference.df.index:
            data = {
                "t": timestamp,
                "o": constant_value,
                "h": constant_value,
                "l": constant_value,
                "c": constant_value,
                "v": 0,
                "n": 0,
                "vw": constant_value,
            }
            bars.append(Bar(symbol=reference.symbol, raw_data=data))  # type: ignore
        return UniformBarSet(
            symbol=reference.symbol,
            horizon=reference.horizon,
            initial_bars=bars,
            buffer_limit=reference.buffer_limit,
            max_bars=reference.max_bars,
        )

    def sample(
        self,
        n_paths: int,
        method: str = "gbm",
        key: Optional[Union[int, Any]] = None,
        **kwargs,
    ) -> List["UniformBarSet"]:
        """Generate synthetic price paths using Monte Carlo simulation.

        Args:
            n_paths: Number of independent paths to generate.
            n_steps: Number of time steps to simulate (including initial state).
            method: Sampling method. Options:
                - 'gbm': Geometric Brownian Motion
                - 'heston': Heston Stochastic Volatility
                - 'mjd': Merton Jump Diffusion
                - 'mbb': Moving Block Bootstrap
                - 'cbb': Circular Block Bootstrap
                - 'sb': Stationary Bootstrap
            key: Random seed (int) or JAX PRNG key. If None, uses seed 42.
            **kwargs: Model-specific parameters. If not provided, parameters
                are estimated from historical data.

        Returns:
            List of UniformBarSet objects, each representing a synthetic path.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        elif isinstance(key, int):
            key = jax.random.PRNGKey(key)

        # Prepare data for estimation/bootstrapping
        if self.df.empty:
            raise ValueError("Cannot sample from empty UniformBarSet.")

        prices = jnp.array(self.df["close"].values)
        if len(prices) < 2:
            raise ValueError("Insufficient history for sampling.")

        # Set n_steps to match current history length
        n_steps = len(prices)

        # Default dt to daily (1/252) if not provided
        dt = 1 / self.horizon.periods_annually()

        # Start from the beginning of the history for alternative paths
        init_price = prices[0]

        # Dispatch to sampling function
        if method == "gbm":
            if "drift" not in kwargs or "volatility" not in kwargs:
                est = estimate_drift_and_volatility(prices, dt)
                kwargs.update(est)

            paths = geometric_brownian_motion(
                n_paths=n_paths,
                n_steps=n_steps,
                init_price=init_price,  # type: ignore
                dt=dt,
                key=key,
                **kwargs,
            )

        elif method == "heston":
            if "mu" not in kwargs:
                est = estimate_heston_params(prices, dt)
                kwargs.update(est)

            # Remove init_price from kwargs if present to avoid multiple values error
            if "init_price" in kwargs:
                kwargs.pop("init_price")

            paths, _ = heston_model(
                n_paths=n_paths,
                n_steps=n_steps,
                init_price=init_price,  # type: ignore
                dt=dt,
                key=key,
                **kwargs,
            )

        elif method == "mjd":
            if "lambda_jump" not in kwargs:
                est = estimate_mjd(prices, dt)
                kwargs.update(est)

            # Rename 'sigma' to 'volatility' if present (mismatch between estimate_mjd and merton_jump_diffusion)
            if "sigma" in kwargs:
                kwargs["volatility"] = kwargs.pop("sigma")

            # Remove init_price from kwargs if present
            if "init_price" in kwargs:
                kwargs.pop("init_price")

            paths = merton_jump_diffusion(
                n_paths=n_paths,
                n_steps=n_steps,
                init_price=init_price,  # type: ignore
                dt=dt,
                key=key,
                **kwargs,
            )

        elif method in ["mbb", "cbb", "sb"]:
            log_returns = jnp.diff(jnp.log(prices))

            if method == "mbb":
                if "block_size" not in kwargs:
                    kwargs["block_size"] = int(jnp.sqrt(len(log_returns)))
                paths = moving_block_bootstrap(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    init_price=init_price,
                    log_returns=log_returns,
                    key=key,
                    **kwargs,
                )
            elif method == "cbb":
                if "block_size" not in kwargs:
                    kwargs["block_size"] = int(jnp.sqrt(len(log_returns)))
                paths = circular_block_bootstrap(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    init_price=init_price,
                    log_returns=log_returns,
                    key=key,
                    **kwargs,
                )
            elif method == "sb":
                paths = stationary_bootstrap(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    init_price=init_price,
                    log_returns=log_returns,
                    key=key,
                    **kwargs,
                )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Convert JAX paths to List[UniformBarSet]
        # Use existing timestamps
        timestamps = self.df.index

        paths_np = np.array(paths)
        result_sets = []

        for i in range(n_paths):
            path_prices = paths_np[i]
            bars = []
            for t, price in zip(timestamps, path_prices):
                # Create synthetic bar with O=H=L=C=price
                data = {
                    "t": t,
                    "o": price,
                    "h": price,
                    "l": price,
                    "c": price,
                    "v": 0,
                    "n": 0,
                    "vw": price,
                }
                bars.append(Bar(symbol=self.symbol, raw_data=data))

            result_sets.append(
                UniformBarSet(
                    symbol=self.symbol,
                    horizon=self.horizon,
                    initial_bars=bars,
                    buffer_limit=self.buffer_limit,
                    max_bars=self.max_bars,
                )
            )

        return result_sets


# Define CompatibleUniformBarMap type alias
def validate_mutually_compatible_uniform_bar_sets(
    bar_sets: Dict[str, UniformBarSet],
) -> Dict[str, UniformBarSet]:
    """Validator to ensure all UniformBarSets are mutually compatible.

    Args:
        bar_sets: Dictionary of bar sets to validate.

    Returns:
        Original dictionary if validation passes.

    Raises:
        ValueError: If bar sets are not mutually compatible.
    """
    if not bar_sets:
        return bar_sets

    # Optimization: Compare all against the first one (O(N))
    # instead of pairwise (O(N^2))
    iterator = iter(bar_sets.values())
    reference_set = next(iterator)

    for other_set in iterator:
        if not reference_set.is_compatible(other_set):
            raise ValueError("All UniformBarSets must be mutually compatible.")

    return bar_sets


# Type alias for a dictionary of mutually compatible UniformBarSets
# This is helpful for validation in UniformBarCollection
# and is useful in timeseries portfolio analysis
CompatibleUniformBarMap = Annotated[
    Dict[str, UniformBarSet],
    at.MinLen(1),
    BeforeValidator(validate_mutually_compatible_uniform_bar_sets),
]


class UniformBarCollection(BaseModel):
    """Collection of time-aligned price bars for multiple assets.

    This class manages multiple UniformBarSets, each representing a different financial
    instrument, all synchronized to the same time intervals and aligned timestamps. This
    synchronization is essential for portfolio-level analysis, backtesting, and multi-asset
    strategies where simultaneous data access across assets is required.

    The collection enforces strict compatibility constraints to ensure all bar sets:
        - Share the same horizon (time interval)
        - Have aligned timestamps across all assets
        - Contain the same number of bars

    This mutual compatibility validation makes it ideal for portfolio analysis where you
    need to access data for multiple assets at the same points in time. The class provides
    a multi-indexed DataFrame property for convenient analysis and supports operations like
    resampling all assets simultaneously.

    This is a generic base class that can be used for collections of any asset type.

    Attributes:
        bar_map: Dictionary mapping asset symbols to their UniformBarSets.
            Must contain at least one bar set and all bar sets must be mutually compatible.
    """

    bar_map: CompatibleUniformBarMap = Field(
        ..., description="Dictionary mapping asset symbols to their UniformBarSets."
    )
    _df_cache: Optional[pd.DataFrame] = PrivateAttr(default=None)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @property
    def horizon(self) -> Horizon:
        """Get the common time horizon shared by all bar sets.

        Returns:
            Time interval of all bar sets in the collection.
        """
        return next(iter(self.bar_map.values())).horizon

    @property
    def symbols(self) -> List[str]:
        """Get a list of all asset symbols in the collection.

        Returns:
            List of ticker symbols for all assets.
        """
        return list(self.bar_map.keys())

    @property
    def df(self) -> pd.DataFrame:
        """Get a combined DataFrame with multi-level index (symbol, timestamp).

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns.
        """
        if self._df_cache is not None:
            return self._df_cache

        dfs = []
        for symbol, bar_set in self.bar_map.items():
            asset_df = bar_set.df.copy()
            asset_df["symbol"] = symbol
            asset_df = asset_df.reset_index()  # timestamp becomes a column
            dfs.append(asset_df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.set_index(["symbol", "timestamp"])

        self._df_cache = combined_df
        return combined_df

    @property
    def close(self) -> pd.DataFrame:
        """Get a DataFrame of closing prices for all assets.

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and 'close' column.
        """
        df = self.df.reset_index()
        close_df = df.pivot(index="timestamp", columns="symbol", values="close")
        return close_df

    @property
    def log_returns(self) -> pd.DataFrame:
        """Get a DataFrame of log returns for all assets.

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and log return values.
        """
        close_df = self.close
        log_return_df = (close_df / close_df.shift(1)).dropna().apply(np.log)
        return log_return_df

    @property
    def arthimetic_returns(self) -> pd.DataFrame:
        """Get a DataFrame of arithmetic returns for all assets.

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and arithmetic return values.
        """
        return self.close.pct_change().dropna()

    def peek(self) -> Dict[str, Optional[Bar]]:
        """Get the most recent bar for each asset without removing it.

        Returns:
            Dictionary mapping symbols to their most recent Bar or None.
        """
        return {symbol: bar_set.peek for symbol, bar_set in self.bar_map.items()}

    def __len__(self) -> int:
        return len(self.bar_map)

    def __repr__(self) -> str:
        return f"UniformBarCollection(assets={len(self.bar_map)}, horizon={self.horizon.name})"

    def get_bar_set(self, symbol: str) -> UniformBarSet:
        """Retrieve the UniformBarSet for a specific asset symbol.

        Args:
            symbol: Ticker symbol of the desired asset.

        Returns:
            Bar set for the specified symbol.

        Raises:
            KeyError: If symbol is not found in the collection.
        """
        if symbol not in self.bar_map:
            raise KeyError(
                f"UniformBarSet with symbol {symbol} not found in the collection."
            )
        return self.bar_map[symbol]

    def get_subcollection(self, symbols: List[str]) -> "UniformBarCollection":
        """Create a sub-collection containing only specified asset symbols.

        Args:
            symbols: List of ticker symbols to include in the sub-collection.
        Returns:
            UniformBarCollection with only the specified assets.
        Raises:
            KeyError: If any symbol is not found in the collection.
        """
        if not all(symbol in self.bar_map for symbol in symbols):
            missing = [s for s in symbols if s not in self.bar_map]
            raise KeyError(f"Symbols not found in the collection: {missing}")

        sub_bar_map = {symbol: self.bar_map[symbol] for symbol in symbols}
        return UniformBarCollection(bar_map=sub_bar_map)

    def push(self, new_bars: Dict[str, UniformBarSet]) -> None:
        """Add new bars to existing bar sets in the collection.

        Args:
            new_bars: Dictionary mapping symbols to UniformBarSets containing
                bars to be added.

        Raises:
            ValueError: If new bar sets are not mutually compatible or have
                different horizon than existing collection.
            KeyError: If symbol is not found in the collection.
        """
        # Make a temporary collection to validate compatibility
        # This checks both mutual compatibility and horizon match
        # withing the new bars
        new_bar_collection = UniformBarCollection(bar_map=new_bars)

        # Check horizon compatibility with existing collection
        if new_bar_collection.horizon != self.horizon:
            raise ValueError(
                "New bar sets must have the same horizon as existing collection."
            )

        # Append bars to existing bar sets
        for symbol, new_bar_set in new_bars.items():
            if symbol not in self.bar_map:
                raise KeyError(
                    f"UniformBarSet with symbol {symbol} not found in the collection."
                )
            for bar in new_bar_set:
                self.bar_map[symbol].push(bar)

        # Invalidate cache
        self._df_cache = None

    def pop(self) -> Dict[str, Bar]:
        """Remove and return the first bar from each bar set in the collection.

        Returns:
            Dictionary mapping symbols to their removed first Bar.

        Raises:
            IndexError: If any bar set has only one element.
        """
        popped_bars = {}
        for symbol, bar_set in self.bar_map.items():
            popped_bars[symbol] = bar_set.pop()

        # Invalidate cache
        self._df_cache = None
        return popped_bars

    def resample(self, new_horizon: Horizon) -> "UniformBarCollection":
        """Resample all bar sets to a larger time horizon.

        Args:
            new_horizon: Target time horizon (must be larger than current).

        Returns:
            New collection with resampled bar sets.

        Raises:
            ValueError: If new_horizon is smaller than or equal to current horizon.
        """
        resampled_bar_sets = {
            symbol: bar_set.resample(new_horizon=new_horizon)
            for symbol, bar_set in self.bar_map.items()
        }
        return UniformBarCollection(bar_map=resampled_bar_sets)

    def __getitem__(
        self, index: Union[slice, int]
    ) -> Union[Dict[str, Bar], "UniformBarCollection"]:
        """Access bars at a specific index or slice across all assets.

        Args:
            index: Index or slice to retrieve.

        Returns:
            Dictionary mapping symbols to bars at index if index is int, or new
            UniformBarCollection with sliced data if index is slice.
        """
        # If index is an integer, return a dictionary of bars at that index
        if isinstance(index, int):
            return {symbol: bar_set[index] for symbol, bar_set in self.bar_map.items()}  # type: ignore

        # If index is a slice, return a new UniformBarCollection with sliced bars
        elif isinstance(index, slice):
            sliced_bar_sets = {
                symbol: UniformBarSet(
                    symbol=bar_set.symbol,
                    horizon=bar_set.horizon,
                    initial_bars=bar_set.bars[index],
                    buffer_limit=bar_set.buffer_limit,
                    max_bars=bar_set.max_bars,
                )
                for symbol, bar_set in self.bar_map.items()
            }
            return UniformBarCollection(bar_map=sliced_bar_sets)

    @property
    def n_bars(self) -> int:
        """Get the number of bars in each bar set (all should be equal).

        Returns:
            Number of bars in the bar sets.
        """
        return len(next(iter(self.bar_map.values())))

    @property
    def timestamps(self) -> pd.Index:
        """Get the list of timestamps for the bars (same for all assets).

        Returns:
            Index of timestamps for the bars.
        """
        return self.df.index.get_level_values("timestamp").unique()

    def sample(
        self,
        n_paths: int,
        method: str = "gbm",
        key: Optional[Union[int, Any]] = None,
        **kwargs,
    ) -> List["UniformBarCollection"]:
        """Generate synthetic portfolio paths (scenarios).

        Sampling method can be parametric (independent assets) or
        non-parametric (bootstrapping, preserves correlations).
        Here are the current options:
            - 'gbm': Geometric Brownian Motion (independent assets)
            - 'heston': Heston Stochastic Volatility (independent assets)
            - 'mjd': Merton Jump Diffusion (independent assets)
            - 'mbb': Moving Block Bootstrap (multivariate)
            - 'cbb': Circular Block Bootstrap (multivariate)
            - 'sb': Stationary Bootstrap (multivariate)

        Args:
            n_paths: Number of independent scenarios to generate.
            n_steps: Number of time steps to simulate.
            method: Sampling method.
            key: Random seed (int) or JAX PRNG key.
            **kwargs: Method-specific parameters.

        Returns:
            List of UniformBarCollection objects, each representing a scenario.
        """
        if key is None:
            key = jax.random.key(42)
        elif isinstance(key, int):
            key = jax.random.key(key)

        # Check if bootstrapping (supports multivariate inherently)
        if method in ["mbb", "cbb", "sb"]:
            # Prepare multivariate log returns (T, N_assets)
            close_df = self.close
            # Calculate log returns
            log_returns_df = (np.log(close_df) - np.log(close_df.shift(1))).dropna()  # type: ignore

            if log_returns_df.empty:
                raise ValueError("Insufficient history for sampling.")

            # Convert to JAX array
            log_returns = jnp.array(log_returns_df.values)

            # Determine n_steps from existing data
            n_steps = self.n_bars

            # Init prices (N_assets,) - Start from beginning
            init_prices = jnp.array(close_df.iloc[0].values)

            # Dispatch
            if method == "mbb":
                if "block_size" not in kwargs:
                    kwargs["block_size"] = int(jnp.sqrt(len(log_returns)))
                paths = moving_block_bootstrap(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    init_price=init_prices,
                    log_returns=log_returns,
                    key=key,
                    **kwargs,
                )
            elif method == "cbb":
                if "block_size" not in kwargs:
                    kwargs["block_size"] = int(jnp.sqrt(len(log_returns)))
                paths = circular_block_bootstrap(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    init_price=init_prices,
                    log_returns=log_returns,
                    key=key,
                    **kwargs,
                )
            elif method == "sb":
                paths = stationary_bootstrap(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    init_price=init_prices,
                    log_returns=log_returns,
                    key=key,
                    **kwargs,
                )

            # paths shape: (n_paths, n_steps, N_assets)
            # Reconstruct List[UniformBarCollection]

            # paths shape: (n_paths, n_steps, N_assets)
            # Reconstruct List[UniformBarCollection]

            timestamps = close_df.index

            paths_np = np.array(paths)
            symbols = list(close_df.columns)

            collections = []
            for i in range(n_paths):
                bar_map = {}
                for j, symbol in enumerate(symbols):
                    asset_prices = paths_np[i, :, j]
                    bars = []
                    for t, price in zip(timestamps, asset_prices):
                        data = {
                            "t": t,
                            "o": price,
                            "h": price,
                            "l": price,
                            "c": price,
                            "v": 0,
                            "n": 0,
                            "vw": price,
                        }
                        bars.append(Bar(symbol=symbol, raw_data=data))

                    bar_map[symbol] = UniformBarSet(
                        symbol=symbol, horizon=self.horizon, bars=bars  # type: ignore
                    )
                collections.append(UniformBarCollection(bar_map=bar_map))

            return collections

        elif method == "gbm":
            # Multivariate GBM (Preserves correlations)
            # Prepare multivariate log returns (T, N_assets)
            close_df = self.close

            # Check for sufficient history
            if len(close_df) < 2:
                raise ValueError("Insufficient history for sampling.")

            # Convert to JAX array
            # We need prices for estimation
            prices = jnp.array(close_df.values)

            # Determine n_steps from existing data
            n_steps = self.n_bars

            # Init prices (N_assets,) - Start from beginning
            init_prices = jnp.array(close_df.iloc[0].values)

            # Default dt
            dt = 1 / self.horizon.periods_annually()

            # Estimate parameters if not provided
            if "drift" not in kwargs or "cov_matrix" not in kwargs:
                est = estimate_multivariate_gbm_params(prices, dt)
                kwargs.update(est)

            # Generate paths
            # Shape: (n_paths, n_steps, n_assets)
            paths = multivariate_geometric_brownian_motion(
                n_paths=n_paths,
                n_steps=n_steps,
                init_prices=init_prices,
                dt=dt,
                key=key,
                **kwargs,
            )

            # Reconstruct List[UniformBarCollection]
            timestamps = close_df.index
            paths_np = np.array(paths)
            symbols = list(close_df.columns)

            collections = []
            for i in range(n_paths):
                bar_map = {}
                for j, symbol in enumerate(symbols):
                    asset_prices = paths_np[i, :, j]
                    bars = []
                    for t, price in zip(timestamps, asset_prices):
                        data = {
                            "t": t,
                            "o": price,
                            "h": price,
                            "l": price,
                            "c": price,
                            "v": 0,
                            "n": 0,
                            "vw": price,
                        }
                        bars.append(Bar(symbol=symbol, raw_data=data))

                    bar_map[symbol] = UniformBarSet(
                        symbol=symbol, horizon=self.horizon, bars=bars  # type: ignore
                    )
                collections.append(UniformBarCollection(bar_map=bar_map))

            return collections

        else:
            # Other Parametric methods (Heston, MJD): Sample each asset independently
            # Note: This does NOT preserve correlations for these models yet.

            asset_samples = {}  # symbol -> List[UniformBarSet]
            keys = jax.random.split(key, len(self.symbols))

            for i, symbol in enumerate(self.symbols):
                bar_set = self.bar_map[symbol]
                asset_samples[symbol] = bar_set.sample(
                    n_paths=n_paths,
                    method=method,
                    key=keys[i],
                    **kwargs,
                )

            # Reassemble
            collections = []
            for i in range(n_paths):
                bar_map = {
                    symbol: samples[i] for symbol, samples in asset_samples.items()
                }
                collections.append(UniformBarCollection(bar_map=bar_map))

            return collections
