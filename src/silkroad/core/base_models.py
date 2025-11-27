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
from .enums import Sector, AssetClass, Horizon
from .types import UniformBarList
import pandas as pd
from itertools import product
from pydantic.functional_validators import BeforeValidator
import annotated_types as at
import numpy as np


__all__ = [
    "UniformBarSet",
    "UniformBarCollection",
    "Asset",
]


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

    # Temporary field to handle 'bars' argument in constructor without overriding __init__
    initial_bars: Optional[List[Bar]] = Field(default=None, alias="bars", exclude=True)

    # model_config = ConfigDict(arbitrary_types_allowed=True)

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
            self._df = (
                pd.DataFrame(data_dicts).drop(columns=["symbol"]).set_index("timestamp")
            )
            self._df.sort_index(inplace=True)

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
                bars=sliced_bars,
                buffer_limit=self.buffer_limit,
                max_bars=self.max_bars,
            )

    def _flush_buffer(self) -> None:
        """Merge the buffer into the main DataFrame."""
        if not self._buffer:
            return

        buffer_dicts = [bar.model_dump() for bar in self._buffer]
        buffer_df = (
            pd.DataFrame(buffer_dicts).drop(columns=["symbol"]).set_index("timestamp")
        )
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
            bars=bars,
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
            bars=bars,
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
                    bars=bar_set.bars[index],
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


class Asset(BaseModel):
    """Base representation of a financial asset with metadata.

    This class provides the foundational structure for representing any financial asset,
    including their classification, sector assignment, and optional metadata. It is designed
    to be generic and extensible, serving as a base class for specific asset types (stocks,
    bonds, options, futures, cryptocurrencies, etc.).

    Assets are uniquely identified by their symbol and can be used in sets and dictionaries.
    Two assets are considered equal if they share the same symbol, regardless of other
    attributes. This design allows for flexible asset management in portfolio construction
    and analysis.

    The metadata field provides extensibility for asset-specific attributes without requiring
    subclassing, making it easy to add custom fields as needed for different use cases.

    Attributes:
        symbol: Ticker symbol of the asset (e.g., "AAPL", "GOOGL", "BTC-USD").
        asset_class: Classification of the asset (e.g., STOCK, BOND, CRYPTOCURRENCY).
        sector: Sector to which the asset belongs (e.g., TECHNOLOGY, HEALTHCARE).
        name: Full name of the asset (e.g., "Apple Inc."). Optional.
        exchange: Exchange where the asset is traded (e.g., "NASDAQ"). Optional.
        metadata: Additional custom metadata about the asset. Optional.
    """

    symbol: str
    asset_class: AssetClass
    sector: Sector
    name: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.symbol} ({self.asset_class.name}, {self.sector.name})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on symbol.

        Args:
            other: Object to compare with.

        Returns:
            True if both assets have the same symbol.

        Raises:
            NotImplementedError: If comparing with a non-Asset object.
        """
        if not isinstance(other, Asset):
            raise NotImplementedError(f"Cannot compare Asset with {type(other)}")
        return self.symbol == other.symbol

    def __hash__(self) -> int:
        """Generate hash based on symbol for use in sets and dicts.

        Returns:
            Hash value of the symbol.
        """
        return hash(self.symbol)
