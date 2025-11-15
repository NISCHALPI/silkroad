"""Generic base models for time series data handling.

This module provides reusable, generic data models for working with financial time series
data. These models are independent of specific asset types and can be used across different
financial instruments (stocks, bonds, options, etc.) or even adapted for non-financial
time series data.

The models enforce data validation and type safety through Pydantic v2, and support
portfolio-level analysis by ensuring multiple time series can be aligned on the same
time index with uniform intervals.

Classes:
    UniformBarSet: Generic time series of OHLCV bars at uniform intervals.
    UniformBarCollection: Collection of multiple UniformBarSets with aligned timestamps.
    Asset: Base model for generic asset metadata.

Typical usage example:
    >>> # Create bar sets for individual assets
    >>> asset_bars = UniformBarSet(
    ...     symbol="AAPL",
    ...     horizon=Horizon.ONE_DAY,
    ...     bars=bars_list
    ... )
    >>>
    >>> # Create a collection of aligned assets
    >>> collection = UniformBarCollection(bar_sets={
    ...     "AAPL": aapl_bars,
    ...     "GOOGL": googl_bars
    ... })
    >>>
    >>> # Access multi-indexed DataFrame for analysis
    >>> df = collection.df
    >>> df.loc[("AAPL", "2024-01-01"), "close"]

Notes:
    - These models are generic and can be extended for specific asset types.
    - All bars in a UniformBarSet must have the same symbol.
    - All bar sets in a UniformBarCollection must share the same horizon and timestamps.
    - Models use Pydantic validation to enforce data integrity at construction time.

See Also:
    silkroad.data.models: Domain-specific models (Stocks, etc.) that extend these base classes.
    silkroad.data.enums: Horizon, AssetClass, and Sector enumerations.
"""

from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo, Field
from typing import List, Dict, Any, Optional, Union, Annotated
from alpaca.data.models import Bar, BarSet
from .enums import Sector, AssetClass, Horizon
from .types import UniformBarList
import pandas as pd
from itertools import product
from pydantic.functional_validators import BeforeValidator
import annotated_types as at


__all__ = [
    "UniformBarSet",
    "UniformBarCollection",
    "Asset",
]


class UniformBarSet(BaseModel):
    """A generic collection of price bars for a single asset with uniform time intervals.

    This class represents a time series of OHLCV (Open, High, Low, Close, Volume) bars
    for a single financial instrument, all sampled at the same time interval (horizon).
    It enforces that all bars belong to the same symbol and provides methods for
    manipulation and resampling.

    This is a generic base class that can be used for any asset type (stocks, bonds,
    options, futures, crypto, etc.) and can be extended for specific use cases.

    Attributes:
        symbol (str): The ticker symbol of the asset (e.g., "AAPL", "GOOGL").
        horizon (Horizon): The time interval between bars (e.g., ONE_DAY, ONE_HOUR).
        bars (List[Bar]): List of price bars from Alpaca API, must contain at least one bar.

    Examples:
        >>> bars = [bar1, bar2, bar3]  # List of Alpaca Bar objects
        >>> bar_set = UniformBarSet(
        ...     symbol="AAPL",
        ...     horizon=Horizon.ONE_DAY,
        ...     bars=bars
        ... )
        >>> print(len(bar_set))  # Number of bars
        >>> df = bar_set.df  # Convert to pandas DataFrame
        >>> resampled = bar_set.resample(Horizon.ONE_WEEK)  # Resample to weekly

    Raises:
        ValueError: If bars contain different symbols than the UniformBarSet symbol.
        ValueError: If attempting to resample to a smaller or equal horizon.
    """

    symbol: str
    horizon: Horizon
    bars: UniformBarList

    model_config = ConfigDict(validate_assignment=True)

    @staticmethod
    def check_contiguous_bars(bars: List[Bar], horizon: Horizon) -> bool:
        """Check if bars have contiguous timestamps based on the horizon.
        Note that this method is not currently used in validation since
        there may be legitimate gaps in data (e.g., weekends, holidays).
        User is responsible for ensuring data continuity if required.

        Args:
            bars (List[Bar]): List of bars to check.
            horizon (Horizon): The expected time interval between bars.

        Returns:
            bool: True if bars are contiguous, False otherwise.
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
            pd.DataFrame: DataFrame with timestamp as index and OHLCV columns
                (open, high, low, close, volume, trade_count, vwap). The symbol
                column is excluded.

        Examples:
            >>> df = bar_set.df
            >>> print(df.columns)  # ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            >>> close_prices = df['close']
        """
        data_dicts = [bar.model_dump() for bar in self.bars]
        return pd.DataFrame(data_dicts).drop(columns=["symbol"]).set_index("timestamp")

    def __repr__(self) -> str:
        return f"UniformBarSet(symbol={self.symbol}, timeframe={self.horizon.name}, bars={len(self.bars)})"

    def __len__(self) -> int:
        return len(self.bars)

    def __getitem__(self, index: Union[int, slice]) -> Union[Bar, "UniformBarSet"]:
        """Access bars by index or slice.

        Args:
            index (Union[int, slice]): The index or slice to retrieve.
        Returns:
            Union[Bar, UniformBarSet]: If index is int, returns the Bar at that position.
                If index is slice, returns a new UniformBarSet with the sliced bars.
        Examples:
            >>> first_bar = bar_set[0]
            >>> subset = bar_set[10:20]
        """
        if isinstance(index, int):
            return self.bars[index]
        elif isinstance(index, slice):
            return UniformBarSet(
                symbol=self.symbol,
                horizon=self.horizon,
                bars=self.bars[index],
            )

    def push(self, bar: Bar) -> None:
        """Add a new bar to the end of the bars list.
        It is user's responsibility to ensure that bars are contiguous w.r.t. horizon.

        Args:
            bar (Bar): The bar to add.

        Raises:
            ValueError: If the bar's symbol doesn't match this UniformBarSet's symbol.
        """
        if bar.symbol != self.symbol:
            raise ValueError(
                f"Bar symbol {bar.symbol} does not match UniformBarSet symbol {self.symbol}."
            )
        # Check that bars are directed
        if self.bars[-1].timestamp >= bar.timestamp:
            raise ValueError(
                """New bar timestamp must be greater than the last bar's timestamp.
                Bars must be added in chronological order."""
            )
        self.bars.append(bar)

    def pop(self) -> Bar:
        """Remove and return the last bar from the bars list.

        Returns:
            Bar: The last bar in the collection.

        Raises:
            IndexError: If the bars list is empty.
        """
        # If bars has length 1, popping is not allowed to prevent empty bar sets
        if len(self.bars) == 1:
            raise IndexError("Cannot pop the last bar from UniformBarSet.")

        return self.bars.pop(0)  # Removes the first bar to maintain time order

    def resample(self, new_horizon: Horizon) -> "UniformBarSet":
        """Resample bars to a larger time horizon using OHLCV aggregation rules.

        This method aggregates bars into a larger time interval, applying appropriate
        aggregation functions: first for open, max for high, min for low, last for close,
        sum for volume and trade_count, and mean for vwap.

        Args:
            new_horizon (Horizon): The target time horizon (must be larger than current).

        Returns:
            UniformBarSet: A new UniformBarSet with resampled bars.

        Raises:
            ValueError: If new_horizon is smaller than or equal to the current horizon.

        Examples:
            >>> daily_bars = UniformBarSet(symbol="AAPL", horizon=Horizon.ONE_DAY, bars=bars)
            >>> weekly_bars = daily_bars.resample(Horizon.ONE_WEEK)
        """
        # Check if the new horizon is greater than the current horizon
        if new_horizon.value <= self.horizon.value:
            raise ValueError(
                "New horizon must be greater than current horizon for resampling."
            )

        # Use pandas to resample the bars
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
        )

    @property
    def peek(self) -> Optional[Bar]:
        """Get the most recent bar without removing it from the collection.

        Returns:
            Optional[Bar]: The last bar, or None if the collection is empty.

        Examples:
            >>> last_bar = bar_set.peek
            >>> if last_bar:
            ...     print(f"Last close: {last_bar.close}")
        """
        if not self.bars:
            return None
        return self.bars[-1]

    def __iter__(self):
        return iter(self.bars)

    def is_compatible(self, other: "UniformBarSet") -> bool:
        """Check if another UniformBarSet is compatible with this one.

        Two bar sets are compatible if they share the same horizon and have aligned
        timestamps. This is essential for combining multiple assets into collections
        for portfolio analysis.

        Args:
            other (UniformBarSet): The bar set to check compatibility with.

        Returns:
            bool: True if the bar sets are compatible, False otherwise.

        Examples:
            >>> aapl_bars.is_compatible(googl_bars)
            True
        """
        if self.horizon != other.horizon:
            return False

        if len(self.bars) == 0 or len(other.bars) == 0:
            return True

        if len(self.bars) != len(other.bars):
            return False
        return self.df.index.equals(other.df.index)


# Define CompatibleUniformBarMap type alias
def validate_mutually_compatible_uniform_bar_sets(
    bar_sets: Dict[str, UniformBarSet],
) -> Dict[str, UniformBarSet]:
    """Validator to ensure all UniformBarSets in the dictionary are mutually compatible.

    Args:
        bar_sets (Dict[str, UniformBarSet]): The dictionary of bar sets to validate.
    Returns:
        Dict[str, UniformBarSet]: The original dictionary if validation passes.
    Raises:
        ValueError: If bar sets are not mutually compatible.
    """
    for bar_set1, bar_set2 in product(bar_sets.values(), repeat=2):
        if not bar_set1.is_compatible(bar_set2):
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
    """A generic collection of time-aligned price bars for multiple assets.

    This class manages multiple UniformBarSets, each representing a different financial
    instrument, all synchronized to the same time intervals and timestamps. This alignment
    is essential for portfolio-level analysis and backtesting, as it ensures all assets
    share identical time indices.

    The collection enforces strict compatibility constraints: all bar sets must have the
    same horizon (time interval) and aligned timestamps. This makes it ideal for multi-asset
    strategies where simultaneous data access across assets is required.

    This is a generic base class that can be used for collections of any asset type.

    Attributes:
        bar_sets (Dict[str, UniformBarSet]): Dictionary mapping asset symbols to their
            respective UniformBarSets. Must contain at least one bar set.

    Examples:
        Create a collection from multiple bar sets:
            >>> aapl_bars = UniformBarSet(symbol="AAPL", horizon=Horizon.ONE_DAY, bars=bars1)
            >>> googl_bars = UniformBarSet(symbol="GOOGL", horizon=Horizon.ONE_DAY, bars=bars2)
            >>> collection = UniformBarCollection(bar_sets={
            ...     "AAPL": aapl_bars,
            ...     "GOOGL": googl_bars
            ... })
            >>> print(len(collection))  # Number of assets
            >>> df = collection.df  # Multi-indexed DataFrame

        Access specific bar set:
            >>> aapl_data = collection.get_bar_set("AAPL")

        Iterate over time periods:
            >>> for bars_dict in collection:
            ...     # bars_dict contains {"AAPL": Bar, "GOOGL": Bar} for each timestamp
            ...     print(bars_dict["AAPL"].close, bars_dict["GOOGL"].close)

    Raises:
        ValueError: If bar sets have different symbols.
        ValueError: If bar sets are not mutually compatible (different horizons or timestamps).

    Notes:
        - All bar sets must have unique symbols.
        - All bar sets must share the same horizon and aligned timestamps.
        - The df property returns a DataFrame with (symbol, timestamp) multi-index.
        - Iteration yields dictionaries of bars for each timestamp across all assets.
    """

    bar_map: CompatibleUniformBarMap

    model_config = ConfigDict(validate_assignment=True)

    @property
    def horizon(self) -> Horizon:
        """Get the common time horizon shared by all bar sets.

        Returns:
            Horizon: The time interval of all bar sets in the collection.
        """
        return next(iter(self.bar_map.values())).horizon

    @property
    def symbols(self) -> List[str]:
        """Get a list of all asset symbols in the collection.

        Returns:
            List[str]: List of ticker symbols for all assets.

        Examples:
            >>> collection.symbols
            ['AAPL', 'GOOGL', 'MSFT']
        """
        return list(self.bar_map.keys())

    @property
    def df(self) -> pd.DataFrame:
        """Get a combined DataFrame with multi-level index (symbol, timestamp).

        Returns:
            pd.DataFrame: DataFrame with MultiIndex (symbol, timestamp) as rows
                and OHLCV columns. All assets share the same timestamps due to
                validation in the constructor.

        Examples:
            >>> collection.df.loc[("AAPL", "2024-01-01"), "close"]  # Access specific symbol/date
            >>> collection.df.xs("AAPL", level=0)  # Get all data for AAPL
        """
        dfs = []
        for symbol, bar_set in self.bar_map.items():
            asset_df = bar_set.df.copy()
            asset_df["symbol"] = symbol
            asset_df = asset_df.reset_index()  # timestamp becomes a column
            dfs.append(asset_df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.set_index(["symbol", "timestamp"])
        return combined_df

    def peek(self) -> Dict[str, Optional[Bar]]:
        """Get the most recent bar for each asset without removing it.

        Returns:
            Dict[str, Optional[Bar]]: Dictionary mapping symbols to their most recent
                Bar object, or None if the bar set is empty.

        Examples:
            >>> last_bars = collection.peek()
            >>> print(last_bars["AAPL"].close)
        """
        return {symbol: bar_set.peek for symbol, bar_set in self.bar_map.items()}

    def __len__(self) -> int:
        return len(self.bar_map)

    def __repr__(self) -> str:
        return f"UniformBarCollection(assets={len(self.bar_map)}, horizon={self.horizon.name})"

    def get_bar_set(self, symbol: str) -> UniformBarSet:
        """Retrieve the UniformBarSet for a specific asset symbol.

        Args:
            symbol (str): The ticker symbol of the desired asset.

        Returns:
            UniformBarSet: The bar set for the specified symbol.

        Raises:
            KeyError: If the symbol is not found in the collection.

        Examples:
            >>> aapl_bars = collection.get_bar_set("AAPL")
            >>> print(len(aapl_bars))
        """
        if symbol not in self.bar_map:
            raise KeyError(
                f"UniformBarSet with symbol {symbol} not found in the collection."
            )
        return self.bar_map[symbol]

    def push(self, new_bars: Dict[str, UniformBarSet]) -> None:
        """Add new bars to existing bar sets in the collection.

        This method appends bars from the provided bar sets to the corresponding
        existing bar sets. The new bar sets must be mutually compatible and share
        the same horizon as the existing collection.

        Args:
            new_bars (Dict[str, UniformBarSet]): Dictionary mapping symbols to
                UniformBarSets containing bars to be added.

        Raises:
            ValueError: If new bar sets are not mutually compatible.
            ValueError: If new bar sets have different horizon than existing ones.

        Examples:
            >>> new_data = {"AAPL": new_aapl_bars, "GOOGL": new_googl_bars}
            >>> collection.push(new_data)
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

    def resample(self, new_horizon: Horizon) -> "UniformBarCollection":
        """Resample all bar sets to a larger time horizon.

        Creates a new collection where all bar sets have been resampled to the
        specified horizon using appropriate OHLCV aggregation rules.

        Args:
            new_horizon (Horizon): The target time horizon (must be larger than current).

        Returns:
            UniformBarCollection: A new collection with resampled bar sets.

        Raises:
            ValueError: If new_horizon is smaller than or equal to current horizon.

        Examples:
            >>> daily_collection = UniformBarCollection(bar_sets=daily_bars)
            >>> weekly_collection = daily_collection.resample(Horizon.ONE_WEEK)
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

        Supports both integer indexing (returns bars at that position) and slice
        indexing (returns a new collection with the sliced range).

        Args:
            index (Union[slice, int]): The index or slice to retrieve.

        Returns:
            Union[Dict[str, Bar], UniformBarCollection]: If index is int, returns
                a dictionary mapping symbols to bars at that index. If index is
                slice, returns a new UniformBarCollection with the sliced data.

        Examples:
            >>> # Get bars at index 0 (first timestamp) for all assets
            >>> first_bars = collection[0]
            >>> print(first_bars["AAPL"].close)
            >>>
            >>> # Get a slice from index 10 to 20
            >>> subset = collection[10:20]
            >>> print(len(subset.get_bar_set("AAPL")))  # 10
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
                )
                for symbol, bar_set in self.bar_map.items()
            }
            return UniformBarCollection(bar_map=sliced_bar_sets)


class Asset(BaseModel):
    """Base representation of a generic financial asset with metadata.

    This class provides the foundational structure for representing any financial asset,
    including their classification, sector, and optional metadata. It is designed to be
    generic and can be extended for specific asset types (stocks, bonds, options, etc.).

    Attributes:
        symbol (str): The ticker symbol of the asset (e.g., "AAPL", "GOOGL", "BTC-USD").
        asset_class (AssetClass): The classification of the asset (e.g., US_EQUITY, CRYPTO).
        sector (Sector): The sector to which the asset belongs (e.g., TECHNOLOGY, FINANCE).
        name (Optional[str]): The full name of the asset (e.g., "Apple Inc."). Defaults to None.
        exchange (Optional[str]): The exchange where the asset is traded. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata about the asset. Defaults to None.

    Examples:
        >>> asset = Asset(
        ...     symbol="AAPL",
        ...     asset_class=AssetClass.US_EQUITY,
        ...     sector=Sector.TECHNOLOGY,
        ...     name="Apple Inc.",
        ...     exchange="NASDAQ"
        ... )
        >>> print(asset)  # AAPL (US_EQUITY, TECHNOLOGY)

    Notes:
        Assets are considered equal if they have the same symbol.
        This class can be extended for specific asset types with additional attributes.
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
            other (object): Another object to compare with.

        Returns:
            bool: True if both assets have the same symbol.

        Raises:
            NotImplementedError: If comparing with a non-Asset object.
        """
        if not isinstance(other, Asset):
            raise NotImplementedError(f"Cannot compare Asset with {type(other)}")
        return self.symbol == other.symbol

    def __hash__(self) -> int:
        """Generate hash based on symbol for use in sets and dicts.

        Returns:
            int: Hash value of the symbol.
        """
        return hash(self.symbol)
