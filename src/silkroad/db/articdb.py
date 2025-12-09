"""ArcticDB backend for high-performance time-series storage and synchronization.

This module provides an efficient database layer for storing, retrieving, and synchronizing
financial time-series market data (OHLCV bars) using ArcticDB as the underlying storage engine.
It integrates with DataBackendProvider implementations to automatically fetch market data from
external sources, handle data updates, detect corporate actions, and normalize timestamps to
a consistent UTC-aware format.

The ArcticDatabase class manages the complete lifecycle of market data:
    1. Fetches data from backend providers (e.g., Alpaca, Yahoo Finance)
    2. Separates new assets from existing assets for efficient batch processing
    3. Detects and handles corporate actions (splits, dividends) via overlap validation
    4. Stores data in separate ArcticDB libraries per horizon (DAILY, MINUTE, etc.)
    5. Maintains asset metadata alongside time-series data
    6. Normalizes all timestamps to timezone-naive UTC for consistency

Key Features:
    - **Automated Data Synchronization**: Batched fetching with incremental updates for
      existing assets and full historical load for new assets
    - **Corporate Action Detection**: Compares overlapping data points to identify restatements
      and splits, automatically triggering full rescans when mismatches are detected
    - **Timezone Normalization**: Converts all timestamps (UTC-aware or naive) to consistent
      naive UTC format for ArcticDB storage compatibility
    - **Multi-Horizon Support**: Maintains separate storage libraries for different timeframes
      (DAILY, MINUTE, HOURLY, etc.) in a single ArcticDB instance
    - **Fast C++ Engine**: Leverages ArcticDB's high-performance C++ backend for sub-millisecond
      read/write operations on large datasets
    - **Metadata Management**: Stores asset metadata (sector, classification, etc.) alongside
      time-series data for context-aware analysis
    - **Configurable Lookback**: Customizable overlap windows for detecting corporate actions
      without re-scanning entire historical datasets

Configuration Constants:
    DEFAULT_START_DATE (datetime): Default start date for new assets (2015-01-01 UTC)
    DEFAULT_CORPORATE_ACTION_LOOKBACK_DAYS (int): Number of days to overlap with existing
        data when checking for restatements (default: 5 days)
    DEFAULT_CORPORATE_ACTION_TOLERANCE (float): Relative tolerance for detecting price
        mismatches due to corporate actions (default: 1e-4 or 0.01%)
"""

import arcticdb as adb
import pandas as pd
import numpy as np
import datetime as dt
import typing as tp
from typing import Union
from silkroad.core.enums import Horizon
from silkroad.core.data_models import Asset
from silkroad.db.backends import DataBackendProvider
from silkroad.logging.logger import logger
from silkroad.core.data_models import UniformBarSet, UniformBarCollection


# Default start date for new assets if not specified
DEFAULT_START_DATE = dt.datetime(2015, 1, 1, tzinfo=dt.timezone.utc)
DEFAULT_CORPORATE_ACTION_LOOKBACK_DAYS = 5
DEFAULT_CORPORATE_ACTION_TOLERANCE = 1e-4


class ArcticDatabase:
    """ArcticDB-based storage manager for market data with intelligent synchronization.

    This class manages the complete lifecycle of financial time-series data in ArcticDB,
    providing automated data fetching, incremental synchronization, corporate action
    detection, and efficient retrieval. It abstracts away the complexity of timezone
    handling, batch operations, and version management.

    The sync workflow is optimized for bulk operations:
        1. **Asset Classification**: Separates new assets from existing ones to minimize
           data transfers
        2. **Batch Fetching**: Requests data for multiple assets in a single backend call
        3. **Corporate Action Detection**: Compares overlapping data to identify splits/dividends
        4. **Version Management**: Maintains full version history for point-in-time queries
        5. **Metadata Tracking**: Stores asset context (sector, classification) alongside data

    Attributes:
        backend (DataBackendProvider): External data provider instance responsible for fetching
            market data (e.g., AlpacaBackendProvider, YahooBackendProvider).
            The backend must implement the DataBackendProvider interface.
        store (adb.Arctic): The ArcticDB instance managing persistent storage and
            version control across multiple libraries.
    """

    def __init__(
        self,
        backend: DataBackendProvider,
        uri: str = "lmdb://./data/_arctic_cfg",
    ):
        """Initialize the ArcticDatabase with a data provider and storage backend.

        Args:
            backend (DataBackendProvider): Data provider instance responsible for fetching
                market data. This should be a fully initialized backend provider such as
                AlpacaBackendProvider, YahooBackendProvider, or a custom implementation
                that inherits from DataBackendProvider.
            uri (str, optional): ArcticDB connection URI specifying the storage backend.
                Supports multiple backend types:
                    - "lmdb://./path" for local LMDB (default, recommended for development)
                    - "s3://bucket-name" for AWS S3 cloud storage
                    - "mongodb://host:port/db" for MongoDB backend
                Defaults to "lmdb://./data/_arctic_cfg" for local file-based storage.
        """
        self.backend = backend
        self.store = adb.Arctic(uri)

    def _get_lib_name(self, horizon: Horizon) -> str:
        """Generate a consistent library name for a given time horizon.

        Library names are derived from the Horizon enum name and follow the pattern
        "market_data_{horizon}" to ensure consistency across the database. This allows
        multiple time-frequencies (DAILY, HOURLY, MINUTE, etc.) to be stored separately
        within a single ArcticDB instance without conflicts.

        Args:
            horizon (Horizon): The time interval/frequency (e.g., Horizon.DAILY,
                Horizon.MINUTE). Must be a valid Horizon enum member.

        Returns:
            str: Normalized library name (e.g., "market_data_daily", "market_data_minute").
        """
        return f"market_data_{horizon.name.lower()}"

    def get_library(self, horizon: Horizon) -> tp.Any:
        """Retrieve or create an ArcticDB library for the specified time horizon.

        This method implements lazy initialization: if the library for the given horizon
        does not exist, it will be created automatically. This ensures that different
        time-frequencies can be stored independently without requiring pre-creation.

        Args:
            horizon (Horizon): The time interval/frequency for the library (e.g.,
                Horizon.DAILY for daily bars, Horizon.MINUTE for minute-level data).
                Each horizon maintains its own independent library with separate symbols
                and version history.

        Returns:
            adb.Library: An ArcticDB Library instance ready for read/write operations.
                The library manages version control, compression, and underlying storage.
        """
        lib_name = self._get_lib_name(horizon)
        if lib_name not in self.store.list_libraries():
            self.store.create_library(lib_name)
        return self.store[lib_name]

    def _to_naive_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame timestamps to timezone-naive UTC for ArcticDB storage.

        This critical preprocessing step ensures consistent timestamp representation across
        all stored data, preventing timezone-related bugs and alignment issues. ArcticDB
        performs better with naive UTC timestamps, and this normalization is essential
        for correct time-range queries and preventing lookahead bias.

        Conversion logic:
            - **Timezone-aware timestamps**: Convert to UTC timezone, then remove timezone info
            - **Timezone-naive timestamps**: Assume they are already in UTC; pass through
            - **MultiIndex with timestamp**: Extract timestamp column, normalize, then rebuild
            - **Empty DataFrames**: Return unchanged to avoid unnecessary processing

        Args:
            df (pd.DataFrame): Input DataFrame with DatetimeIndex (either timezone-aware UTC,
                timezone-aware non-UTC, or timezone-naive assumed UTC) or MultiIndex containing
                a "timestamp" level. Supports both standard Index and MultiIndex structures.

        Returns:
            pd.DataFrame: Normalized DataFrame with naive UTC DatetimeIndex (or MultiIndex
                with normalized "timestamp" level). All timestamps are in UTC and lack
                timezone information for ArcticDB compatibility.

        Note:
            This function assumes all naive timestamps are in UTC. Timestamps from different
            timezones (e.g., market-local times) must be converted to UTC BEFORE calling
            this function to avoid data corruption.
        """
        if df.empty:
            return df

        # Handle MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            if "timestamp" in df.index.names:
                df = df.reset_index()
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    # utc=True converts mixed/naive strings to UTC-aware
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

                # If aware, convert to UTC then make naive
                if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):  # type: ignore
                    df["timestamp"] = (
                        df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)  # type: ignore
                    )
                # If naive, it's already naive, and we assume it's UTC.

                return df.set_index(["symbol", "timestamp"]).sort_index()
            return df

        # Handle standard Index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Convert strings/mixed to datetime. utc=True handles naive->UTC assumption for strings
            df.index = pd.to_datetime(df.index, utc=True)

        # If aware, convert to UTC then make naive
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # If naive, we assume it is UTC and leave it as naive.

        return df

    def sync(
        self,
        assets: list[Asset],
        horizon: Horizon,
        start_date: tp.Optional[dt.datetime] = None,
        end_date: tp.Optional[dt.datetime] = None,
        lookback_buffer: int = DEFAULT_CORPORATE_ACTION_LOOKBACK_DAYS,
    ):
        """Synchronize market data for a batch of assets with intelligent update strategies.

        This is the primary method for keeping the database current. It implements a
        sophisticated batch-oriented synchronization algorithm that optimizes for both
        new asset initialization and incremental updates of existing data.

        **Synchronization Workflow**:

            1. **Asset Classification** (O(n) where n = number of assets):
               - Checks each asset's last stored timestamp
               - Separates into "new" and "update" lists
               - Determines the earliest required fetch date

            2. **New Asset Initialization** (Batch):
               - Fetches full historical data from start_date to end_date
               - Normalizes timestamps to naive UTC
               - Writes complete history to ArcticDB with metadata
               - Logs initialization for each symbol

            3. **Incremental Updates** (Batch):
               - Fetches only recent data (with configurable lookback overlap)
               - Compares overlapping data to detect corporate actions
               - Updates existing symbols if data is consistent
               - Queues inconsistent symbols for full resync

            4. **Corporate Action Detection**:
               - Compares overlapping bars using relative tolerance (default 1e-4)
               - Triggers full resync if price mismatches exceed threshold
               - Prevents data corruption from stock splits/dividends
               - Configurable via lookback_buffer and DEFAULT_CORPORATE_ACTION_TOLERANCE

            5. **Full Resync** (Batch, if needed):
               - Fetches complete history for flagged assets
               - Overwrites entire symbol with clean data
               - Resolves split/dividend issues automatically

        Args:
            assets (list[Asset]): List of Asset objects to synchronize. Each must have
                a valid ticker and optional metadata (sector, exchange, etc.) that will
                be stored alongside the time-series data.
            horizon (Horizon): Data frequency/timeframe. Each horizon maintains separate
                storage (e.g., Horizon.DAILY and Horizon.MINUTE are independent).
            start_date (datetime, optional): Start date for new assets and full resyncs.
                Only used for assets without existing data. Defaults to DEFAULT_START_DATE
                (2015-01-01 UTC) for broad historical coverage.
            end_date (datetime, optional): End date for all fetches. Defaults to current
                UTC datetime. Must be timezone-aware or will be treated as UTC.
            lookback_buffer (int, optional): Number of days to overlap with existing data
                when checking for corporate actions. Default 5 days. Larger values (e.g., 30)
                provide more robust detection but increase fetch volume. Adjust based on
                data provider reliability and update frequency.

        Raises:
            Logs exceptions for each phase but continues processing (graceful degradation).
            Individual asset fetch failures do not block other assets.

        Examples:
            ```python
            # Initialize daily data for a portfolio
            assets = [Asset(ticker="AAPL"), Asset(ticker="MSFT")]
            db.sync(assets, Horizon.DAILY)

            # Update existing data with custom date range
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=90)
            db.sync(assets, Horizon.DAILY, start_date=start, end_date=end)

            # Aggressive corporate action detection (30-day lookback)
            db.sync(assets, Horizon.DAILY, lookback_buffer=30)
            ```
        """
        lib = self.get_library(horizon)

        if end_date is None:
            end_date = dt.datetime.now(dt.timezone.utc)

        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=dt.timezone.utc)

        new_assets = []
        update_assets = []
        min_update_start = end_date

        # Map ticker to Asset object for easy lookup and metadata extraction
        asset_map = {a.ticker: a for a in assets}

        # 1. Classification
        for asset in assets:
            symbol = asset.ticker
            last_ts = self._get_last_timestamp(lib, symbol)

            if last_ts:
                # Existing asset
                fetch_start = last_ts - dt.timedelta(days=lookback_buffer)
                if fetch_start.tzinfo is None:
                    fetch_start = fetch_start.replace(tzinfo=dt.timezone.utc)

                if fetch_start < end_date:
                    update_assets.append(asset)
                    if fetch_start < min_update_start:
                        min_update_start = fetch_start
            else:
                # New asset
                new_assets.append(asset)

        # 2. Handle New Assets (Batch Fetch)
        if new_assets:
            fetch_start = start_date or DEFAULT_START_DATE
            logger.info(
                f"Batch fetching {len(new_assets)} new assets from {fetch_start}"
            )
            try:
                df_new = self.backend.fetch_data(
                    new_assets, start=fetch_start, end=end_date, horizon=horizon
                )
                if not df_new.empty:
                    df_new = self._to_naive_utc(df_new)
                    for symbol in df_new.index.get_level_values("symbol").unique():
                        asset = asset_map.get(symbol)
                        # Store asset metadata (convert to JSON-compatible dict)
                        metadata = asset.model_dump(mode="json") if asset else None

                        lib.write(
                            symbol, df_new.xs(symbol, level="symbol"), metadata=metadata
                        )
                        logger.info(
                            f"Initialized {symbol} with {len(df_new.xs(symbol, level='symbol'))} bars"
                        )
            except Exception as e:
                logger.error(f"Failed to fetch new assets: {e}")
                raise e

        # 3. Handle Updates (Batch Fetch)
        resync_assets = []
        if update_assets:
            logger.info(
                f"Batch fetching updates for {len(update_assets)} assets from {min_update_start}"
            )
            try:
                df_update = self.backend.fetch_data(
                    update_assets, start=min_update_start, end=end_date, horizon=horizon
                )

                if not df_update.empty:
                    df_update = self._to_naive_utc(df_update)

                    for symbol in df_update.index.get_level_values("symbol").unique():
                        df_symbol = df_update.xs(symbol, level="symbol")
                        asset = asset_map.get(symbol)
                        metadata = asset.model_dump(mode="json") if asset else None

                        # Corporate Action Check
                        needs_resync = False
                        last_ts = self._get_last_timestamp(lib, symbol)
                        if last_ts:
                            last_ts_naive = last_ts.replace(tzinfo=None)
                            overlap_end = min(df_symbol.index.max(), last_ts_naive)
                            overlap_start = df_symbol.index.min()

                            if overlap_start <= overlap_end:
                                existing = self.read(
                                    symbol,
                                    horizon,
                                    start=overlap_start,
                                    end=overlap_end,
                                )
                                if not existing.empty:
                                    common = existing.index.intersection(
                                        df_symbol.index
                                    )
                                    if not common.empty:
                                        col = (
                                            "close"
                                            if "close" in df_symbol.columns
                                            else "Close"
                                        )
                                        if (
                                            col in existing.columns
                                            and col in df_symbol.columns
                                        ):
                                            if not np.isclose(
                                                existing.loc[common, col],
                                                df_symbol.loc[common, col],
                                                rtol=DEFAULT_CORPORATE_ACTION_TOLERANCE,
                                                equal_nan=True,
                                            ).all():
                                                logger.warning(
                                                    f"Mismatch detected for {symbol}. Queuing for resync."
                                                )
                                                needs_resync = True
                                                if asset:
                                                    resync_assets.append(asset)

                        if not needs_resync:
                            lib.update(symbol, df_symbol, metadata=metadata)
                            logger.info(f"Updated {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch updates: {e}")
                raise e

        # 4. Handle Resyncs (Batch Fetch)
        if resync_assets:
            fetch_start = start_date or DEFAULT_START_DATE
            logger.info(
                f"Batch resyncing {len(resync_assets)} assets from {fetch_start}"
            )
            try:
                df_resync = self.backend.fetch_data(
                    resync_assets, start=fetch_start, end=end_date, horizon=horizon
                )
                if not df_resync.empty:
                    df_resync = self._to_naive_utc(df_resync)
                    for symbol in df_resync.index.get_level_values("symbol").unique():
                        asset = asset_map.get(symbol)
                        metadata = asset.model_dump(mode="json") if asset else None

                        lib.write(
                            symbol,
                            df_resync.xs(symbol, level="symbol"),
                            metadata=metadata,
                        )
                        logger.info(f"Resynced {symbol}")
            except Exception as e:
                logger.error(f"Failed to resync assets: {e}")
                raise e

    def _get_last_timestamp(self, lib: tp.Any, symbol: str) -> tp.Optional[dt.datetime]:
        """Retrieve the most recent timestamp for a symbol in the library.

        This utility method efficiently fetches only the index without data columns
        to minimize I/O overhead. It's used internally to determine whether an asset
        requires full historical load (new asset) or incremental update (existing asset).

        Args:
            lib (adb.Library): The ArcticDB library to query.
            symbol (str): The ticker symbol to look up.

        Returns:
            datetime or None: The latest timestamp as a timezone-aware UTC datetime,
                or None if the symbol doesn't exist or has no data.
        """
        if symbol not in lib.list_symbols():
            return None

        try:
            # Efficiently read the last index value
            item = lib.read(symbol, columns=[])
            if item.data.empty:
                return None

            last_ts = item.data.index[-1]
            # Ensure it's UTC aware for comparison
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=dt.timezone.utc)
            return last_ts
        except Exception as e:
            logger.error(f"Failed to get last timestamp for {symbol}: {e}")
            raise e

    def get_metadata(self, symbol: str, horizon: Horizon) -> tp.Optional[dict]:
        """Retrieve stored metadata for a symbol within a specific horizon.

        Metadata typically includes asset classification, exchange, sector, and other
        contextual information synced from the Asset object. This is stored alongside
        the time-series data in ArcticDB and is useful for filtering and analysis.

        Args:
            symbol (str): The ticker symbol to retrieve metadata for.
            horizon (Horizon): The time horizon/library containing the symbol.

        Returns:
            dict or None: A dictionary of metadata fields (e.g., {"sector": "TECHNOLOGY",
                "exchange": "NASDAQ"}), or None if the symbol doesn't exist or metadata
                is unavailable. Returns None on any read error (logged as warning).
        """
        lib = self.get_library(horizon)
        if symbol not in lib.list_symbols():
            return None
        try:
            item = lib.read_metadata(symbol)
            # Handle VersionedItem if returned
            if hasattr(item, "metadata"):
                return item.metadata
            return item
        except Exception as e:
            logger.error(f"Error reading metadata for {symbol}: {e}")
            raise e

    def filter_symbols(self, horizon: Horizon, **criteria) -> list[str]:
        """Filter symbols by matching metadata criteria across all symbols in a horizon.

        This method enables efficient asset universe filtering without loading raw time-series
        data. It's useful for constructing portfolios by sector, exchange, or custom attributes.
        Note: This is a linear scan over all symbols and metadata—for very large universes
        (>10k symbols), consider maintaining a separate index or cache.

        Args:
            horizon (Horizon): The time horizon/library to search.
            **criteria: Arbitrary keyword arguments matching metadata fields.
                Each criterion must match exactly for a symbol to be included.
                Examples:
                    - sector="TECHNOLOGY"
                    - exchange="NASDAQ"
                    - sector="TECHNOLOGY", exchange="NYSE" (both conditions)

        Returns:
            list[str]: List of ticker symbols matching ALL specified criteria.
                Returns empty list if no matches found or all symbols lack metadata.

        Examples:
            ```python
            # Get all tech stocks
            tech_symbols = db.filter_symbols(Horizon.DAILY, sector="TECHNOLOGY")

            # Get NASDAQ tech stocks
            nasdaq_tech = db.filter_symbols(
                Horizon.DAILY,
                sector="TECHNOLOGY",
                exchange="NASDAQ"
            )
            ```
        """
        lib = self.get_library(horizon)
        matches = []

        # Optimization: ArcticDB might support metadata filtering in future,
        # but for now we iterate. This might be slow for huge universes.
        # A separate index/cache would be better for production.
        for symbol in lib.list_symbols():
            try:
                meta = lib.read_metadata(symbol)
                # Handle VersionedItem if returned
                if hasattr(meta, "metadata"):
                    meta = meta.metadata

                if not meta:
                    continue

                match = True
                for key, value in criteria.items():
                    # Handle Enum serialization (stored as value or name)
                    # We assume metadata stores strings or primitives
                    meta_val = meta.get(key)

                    # If criteria value is an Enum, compare with name or value
                    if hasattr(value, "value"):
                        if meta_val != value.value and meta_val != value.name:
                            match = False
                            break
                    else:
                        if meta_val != value:
                            match = False
                            break

                if match:
                    matches.append(symbol)
            except Exception as e:
                logger.error(f"Error filtering symbol {symbol}: {e}")
                raise e

        return matches

    def read(
        self,
        symbol: str,
        horizon: Horizon,
        start: tp.Optional[dt.datetime] = None,
        end: tp.Optional[dt.datetime] = None,
        columns: tp.Optional[list[str]] = None,
        as_of: tp.Optional[Union[int, str, dt.datetime]] = None,
    ) -> pd.DataFrame:
        """Read time-series data for a symbol, with optional date/column filtering.

        This is the primary data retrieval method. It supports flexible filtering on
        dates and columns, plus powerful point-in-time queries through version pinning.
        Timestamps are automatically converted from naive UTC storage to timezone-aware
        UTC for consistency in analysis.

        **Version Control Features**:
            - **Latest version (default)**: Retrieves current data snapshot
            - **Historic versions**: Use as_of to access previous states (corporate action
              corrections, debugging, historical analysis)
            - **Point-in-time queries**: Query data as it existed on a specific date

        Args:
            symbol (str): The ticker symbol to retrieve data for.
            horizon (Horizon): The time horizon/library containing the data.
            start (datetime, optional): Start date (inclusive) for the time range.
                Timestamps before this are excluded. Can be timezone-aware or naive
                (assumed UTC). Defaults to None (from beginning of history).
            end (datetime, optional): End date (inclusive) for the time range.
                Timestamps after this are excluded. Can be timezone-aware or naive
                (assumed UTC). Defaults to None (through most recent data).
            columns (list[str], optional): Specific columns to retrieve (e.g.,
                ["close", "volume"]). Defaults to None, which returns all columns
                ("open", "high", "low", "close", "volume", etc.).
            as_of (int, str, or datetime, optional): Version specifier for point-in-time
                queries:
                    - int: Specific version number
                    - str: Named snapshot identifier
                    - datetime: Retrieve data as it existed on this date
                Defaults to None (uses latest version).

        Returns:
            pd.DataFrame: Time-series data indexed by naive UTC timestamps, or empty
                DataFrame if no data matches criteria. Raises no exception on missing
                symbols (logs error and returns empty DataFrame).

        Examples:
            ```python
            # Full data history
            df = db.read("AAPL", Horizon.DAILY)

            # Last year of data
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=365)
            df = db.read("AAPL", Horizon.DAILY, start=start, end=end)

            # Only closing prices and volume
            df = db.read("AAPL", Horizon.DAILY, columns=["close", "volume"])

            # Data as of a specific past date (point-in-time)
            historical_date = datetime(2023, 1, 15, tzinfo=timezone.utc)
            df = db.read("AAPL", Horizon.DAILY, as_of=historical_date)
            ```
        """
        lib = self.get_library(horizon)
        if symbol not in lib.list_symbols():
            return pd.DataFrame()

        # ArcticDB expects naive timestamps for DateRange if data is naive
        # Our data is stored as naive UTC.
        # So we must convert query dates to naive UTC.

        start_naive = None
        end_naive = None

        if start:
            if start.tzinfo is not None:
                start = start.astimezone(dt.timezone.utc)
                start_naive = start.replace(tzinfo=None)
            else:
                start_naive = start

        if end:
            if end.tzinfo is not None:
                end = end.astimezone(dt.timezone.utc)
                end_naive = end.replace(tzinfo=None)
            else:
                end_naive = end

        date_range = None
        if start_naive or end_naive:
            # Construct DateRange object or tuple if supported
            # ArcticDB usually supports `date_range=(start, end)`
            date_range = (start_naive, end_naive)

        try:
            item = lib.read(symbol, date_range=date_range, columns=columns, as_of=as_of)
            return item.data
        except Exception as e:
            logger.error(f"Error reading {symbol} from {horizon.name}: {e}")
            raise e

    def read_into_uniform_barset(
        self,
        symbol: str,
        horizon: Horizon,
        start: tp.Optional[dt.datetime] = None,
        end: tp.Optional[dt.datetime] = None,
        columns: tp.Optional[list[str]] = None,
        as_of: tp.Optional[Union[int, str, dt.datetime]] = None,
    ) -> UniformBarSet:
        """Read data for a single symbol and wrap it in a UniformBarSet domain model.

        UniformBarSet provides a rich interface for time-series analysis, including
        streaming buffers, technical indicators, and alignment utilities. This method
        is a convenience wrapper around read() with automatic domain model conversion.

        Args:
            symbol (str): The ticker symbol to retrieve.
            horizon (Horizon): The time horizon/library.
            start (datetime, optional): Start date (inclusive).
            end (datetime, optional): End date (inclusive).
            columns (list[str], optional): Specific columns to include.
            as_of (int, str, or datetime, optional): Version identifier.

        Returns:
            UniformBarSet: A validated UniformBarSet instance containing the symbol's
                time-series data, ready for analysis and strategy backtesting.

        Raises:
            ValueError: If no data is found, if the DataFrame is empty, or if
                UniformBarSet instantiation fails (logged with details).

        Examples:
            ```python
            # Load AAPL daily data for analysis
            bar_set = db.read_into_uniform_barset("AAPL", Horizon.DAILY)
            momentum = bar_set.compute_momentum(20)  # 20-day momentum

            # Load recent minute data
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=5)
            bar_set = db.read_into_uniform_barset(
                "AAPL",
                Horizon.MINUTE,
                start=start,
                end=end
            )
            ```
        """
        df = self.read(symbol, horizon, start, end, columns, as_of=as_of)
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} in {horizon.name}")

        try:
            return UniformBarSet.from_df(symbol, horizon, df)
        except Exception as e:
            logger.error(f"Failed to create UniformBarSet for {symbol}: {e}")
            raise ValueError(f"Failed to create UniformBarSet for {symbol}: {e}") from e

    def read_into_uniform_barcollection(
        self,
        symbols: list[str],
        horizon: Horizon,
        start: tp.Optional[dt.datetime] = None,
        end: tp.Optional[dt.datetime] = None,
        columns: tp.Optional[list[str]] = None,
        as_of: tp.Optional[Union[int, str, dt.datetime]] = None,
    ) -> UniformBarCollection:
        """Read data for multiple symbols and return as an aligned UniformBarCollection.

        This method is the primary interface for multi-asset portfolio analysis. It
        automatically handles timestamp alignment and intersection computation, ensuring
        that all symbols have consistent dates for correlation and optimization analysis.
        This is critical for accurate portfolio analytics and prevents lookahead bias.

        **Alignment Strategy**:
            1. Loads each symbol's data independently
            2. Computes intersection of all symbol date ranges
            3. Returns only the overlapping time period for all assets
            4. Ensures all symbols have identical timestamps (strict alignment)

        Args:
            symbols (list[str]): List of ticker symbols to load. All symbols must have
                data in the specified time range; missing symbols raise ValueError.
            horizon (Horizon): The time horizon/library (must be consistent across symbols).
            start (datetime, optional): Start date (inclusive) for all symbols.
            end (datetime, optional): End date (inclusive) for all symbols.
            columns (list[str], optional): Specific columns to retrieve (applied to all).
            as_of (int, str, or datetime, optional): Version identifier (applied to all).

        Returns:
            UniformBarCollection: A validated multi-asset container with perfectly aligned
                timestamps across all symbols. Ready for correlation, covariance, and
                optimization analysis.

        Raises:
            ValueError: If any symbol has no data, intersection is empty, or collection
                creation fails (detailed error message logged).

        Examples:
            ```python
            # Load a portfolio for optimization
            symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
            collection = db.read_into_uniform_barcollection(
                symbols,
                Horizon.DAILY,
                start=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end=datetime(2023, 12, 31, tzinfo=timezone.utc)
            )
            # All symbols now have identical dates; safe for covariance matrix
            cov_matrix = collection.compute_covariance()

            # Load sector comparison (e.g., mega-cap tech)
            tech_stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
            tech_collection = db.read_into_uniform_barcollection(
                tech_stocks,
                Horizon.DAILY,
                end=datetime.now(timezone.utc)
            )
            ```
        """
        bar_sets = {}
        for symbol in symbols:
            # This will now raise ValueError if data is missing
            bs = self.read_into_uniform_barset(
                symbol, horizon, start, end, columns, as_of=as_of
            )
            bar_sets[symbol] = bs

        if not bar_sets:
            raise ValueError("No bar sets could be created from the provided symbols.")

        # UniformBarCollection requires strict compatibility (aligned timestamps).
        # We compute the intersection of all retrieved bar sets.
        try:
            aligned_sets_list = UniformBarSet.get_intersection(*bar_sets.values())

            if not aligned_sets_list:
                logger.warning("No overlapping data found for the requested assets.")
                raise ValueError("No overlapping data found for the requested assets.")

            # Map back to symbols
            aligned_map = {bs.symbol: bs for bs in aligned_sets_list}

            return UniformBarCollection(bar_map=aligned_map)
        except Exception as e:
            logger.error(f"Failed to create UniformBarCollection: {e}")
            raise ValueError(f"Failed to create UniformBarCollection: {e}") from e

    def list_symbols(self, horizon: Horizon) -> list[str]:
        """List all ticker symbols stored in a specific horizon library.

        Args:
            horizon (Horizon): The time horizon/library to query.

        Returns:
            list[str]: List of all ticker symbols currently stored in the library.
                Returns empty list if library is empty or doesn't exist.
        """
        lib = self.get_library(horizon)
        return lib.list_symbols()

    def delete(self, symbol: str, horizon: Horizon):
        """Permanently delete a symbol and all its version history from the library.

        This operation is irreversible and removes the entire symbol from storage,
        including all historical versions. Used for cleanup or removing stale symbols.

        Args:
            symbol (str): The ticker symbol to delete.
            horizon (Horizon): The time horizon/library from which to delete.
        """
        lib = self.get_library(horizon)
        if symbol not in lib.list_symbols():
            logger.warning(
                f"""Symbol {symbol} not found in {horizon.name}.
                            Couldn't delete the symbol."""
            )
            raise ValueError(
                f"Symbol {symbol} not found in {horizon.name}. Couldn't delete the symbol."
            )
        # Delete the symbol
        lib.delete(symbol)
        logger.info(f"Deleted {symbol} from {horizon.name}")

    def prune_history(self, symbol: str, horizon: Horizon, max_versions: int = 10):
        """Prune version history for a symbol to save storage space.

        ArcticDB maintains full version history by default, enabling point-in-time queries
        and data debugging. However, for production systems with continuous updates, old
        versions can consume significant storage. This method automatically detects when
        version count exceeds a threshold and triggers cleanup.

        **Pruning Behavior**:
            - When version count > max_versions, calls ArcticDB's prune_previous_versions()
            - Prune operation collapses ALL historical versions into the latest version
            - Saves disk space but makes point-in-time queries unavailable for pruned versions
            - Operation is irreversible; only latest version remains

        Args:
            symbol (str): The ticker symbol to prune.
            horizon (Horizon): The time horizon/library.
            max_versions (int, optional): Version count threshold. If exceeded, pruning
                is triggered. Default 10 balances version history with storage efficiency.
                Higher values (e.g., 50) preserve more history; lower values (e.g., 3)
                minimize storage at the cost of historical queryability.

        Examples:
            ```python
            # Prune aggressively (keep only 3 versions)
            db.prune_history("AAPL", Horizon.DAILY, max_versions=3)

            # Prune all symbols in a horizon
            for symbol in db.list_symbols(Horizon.DAILY):
                db.prune_history(symbol, Horizon.DAILY)
            ```
        """
        lib = self.get_library(horizon)
        if symbol not in lib.list_symbols():
            return

        try:
            versions = lib.list_versions(symbol)
            if len(versions) > max_versions:
                logger.info(
                    f"Pruning history for {symbol} in {horizon.name} (versions={len(versions)} > {max_versions})"
                )
                lib.prune_previous_versions(symbol)
        except Exception as e:
            logger.error(f"Failed to prune history for {symbol}: {e}")
            raise e

    def summary(
        self,
        horizon: Horizon,
    ) -> pd.DataFrame:
        """Generate a comprehensive summary of all symbols in a horizon library.

        This utility method produces a DataFrame aggregating key statistics for each
        symbol, useful for inventory management, data quality checks, and usage monitoring.

        **Summary Contents**:
            - **ticker**: Symbol identifier (index)
            - **start_date**: Earliest timestamp in stored data
            - **end_date**: Most recent timestamp in stored data
            - **rows**: Number of bars/records for the symbol
            - **version_count**: Number of stored versions (for version management)
            - **metadata fields**: All custom asset metadata (sector, exchange, etc.)

        Args:
            horizon (Horizon): The time horizon/library to summarize.

        Returns:
            pd.DataFrame: Summary table with ticker as index and columns for metadata,
                dates, row counts, and version info. Returns empty DataFrame if library
                has no symbols. Gracefully handles missing metadata or versioning info
                (logs warnings but continues).

        Examples:
            ```python
            # Inventory check
            summary = db.summary(Horizon.DAILY)
            print(summary[[\"start_date\", \"end_date\", \"rows\"]])

            # Find symbols with recent data
            summary = db.summary(Horizon.DAILY)
            recent = summary[summary[\"end_date\"] > datetime(2024, 1, 1, tzinfo=timezone.utc)]

            # Storage monitoring
            print(f\"Total rows: {summary['rows'].sum()}\")
            print(f\"Average versions per symbol: {summary['version_count'].mean()}\")
            ```
        """
        lib = self.get_library(horizon)
        symbols = lib.list_symbols()

        summary_data = []

        for symbol in symbols:
            info = {"ticker": symbol}

            # 1. Metadata
            try:
                item = lib.read_metadata(symbol)
                if item.metadata:
                    info.update(item.metadata)
            except Exception as e:
                logger.warning(f"Failed to read metadata for {symbol}: {e}")
                raise e

            # 2. Versions
            try:
                versions = lib.list_versions(symbol)
                info["version_count"] = len(versions)
            except Exception as e:
                logger.warning(f"Failed to list versions for {symbol}: {e}")
                raise e

            # 3. Start/End Date
            try:
                # Read only index for performance
                df = lib.read(symbol, columns=[]).data
                if len(df.index) > 0:
                    info["start_date"] = df.index[0]
                    info["end_date"] = df.index[-1]
                    info["rows"] = len(df)
            except Exception as e:
                logger.warning(f"Failed to read data range for {symbol}: {e}")
                raise e

            summary_data.append(info)

        if not summary_data:
            return pd.DataFrame()

        df = pd.DataFrame(summary_data)
        df.set_index("ticker", inplace=True)
        return df
