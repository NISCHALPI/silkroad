"""Market data storage and retrieval module for Silkroad trading framework.

This module provides high-level interfaces for storing, retrieving, and updating
market data from multiple asset classes (stocks and cryptocurrencies) using ArcticDB
as the underlying storage engine. It includes intelligent handling of corporate actions
and multiple update strategies for maintaining data consistency.

Key Features:
    - ArcticDB-backed storage with LMDB backend for high-performance time-series data
    - Multi-asset support with automatic classification (stocks vs. crypto)
    - Corporate action detection and handling (splits, dividends)
    - Multiple update modes: full_refresh, smart_merge, append_only
    - Integration with Alpaca API for real-time data fetching
    - Timezone-aware data management with UTC normalization

Example:
    Basic usage of the MarketDataDB class:

    >>> from silkroad.app.db.market_data import MarketDataDB
    >>> from silkroad.core.data_models import Asset, Horizon
    >>> import datetime as dt
    >>>
    >>> # Initialize database with daily data frequency
    >>> db = MarketDataDB(horizon=Horizon.DAILY)
    >>>
    >>> # Update with market data from Alpaca
    >>> assets = [Asset(ticker="AAPL", name="Apple Inc.")]
    >>> db.update_from_alpaca(
    ...     api_key="YOUR_KEY",
    ...     api_secret="YOUR_SECRET",
    ...     assets=assets,
    ...     start=dt.datetime(2025, 1, 1),
    ...     end=dt.datetime(2025, 12, 1),
    ... )
    >>>
    >>> # Query market data
    >>> bar_set = db.get_ticker_data("AAPL", start, end)
"""

import arcticdb as adb
from pathlib import Path
import typing as tp
import datetime as dt
from silkroad.core.data_models import (
    UniformBarSet,
    Horizon,
    Asset,
    UniformBarCollection,
)
from silkroad.core.enums import AssetClass, Exchange
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from silkroad.logging.logger import logger
import pandas as pd

__all__ = ["MarketDataDB"]


# Mapping between Horizon and Alpaca TimeFrame
HORIZON_TO_TIMEFRAME = {
    Horizon.MINUTE: TimeFrame.Minute,
    Horizon.HOURLY: TimeFrame.Hour,
    Horizon.DAILY: TimeFrame.Day,
    Horizon.WEEKLY: TimeFrame.Week,
    Horizon.MONTHLY: TimeFrame.Month,
}

# Mapping between Horizon and library names
HORIZON_TO_LIBRARY_NAME = {
    Horizon.MINUTE: "minute",
    Horizon.HOURLY: "hourly",
    Horizon.DAILY: "daily",
    Horizon.WEEKLY: "weekly",
    Horizon.MONTHLY: "monthly",
}


class MarketDataDB:
    """ArcticDB-backed market data storage with corporate action handling.

    Provides a high-level interface for storing and retrieving market data from
    Alpaca for stocks and cryptocurrencies, with automatic detection and handling
    of corporate actions (splits, dividends) that retroactively adjust historical prices.

    Each database instance is tied to a specific data frequency (Horizon), ensuring
    consistency across all operations. Data is stored in ArcticDB using LMDB backend
    and accessed via a library named according to the horizon (e.g., 'daily', 'hourly').

    The class supports multiple update strategies:
    - **full_refresh**: Re-downloads all historical data from inception
    - **smart_merge**: Detects corporate actions at merge boundaries
    - **append_only**: Simple append without corporate action checks

    Attributes:
        db_path (Path): Database directory path.
        db_uri (str): ArcticDB connection URI.
        horizon (Horizon): Data frequency for this database instance.
        library_name (str): ArcticDB library name derived from horizon.
        timeframe (TimeFrame): Alpaca TimeFrame corresponding to horizon.
        ALPACA_START_DATE (datetime): Earliest available date in Alpaca (2015-01-01).
        DETECT_PERCENTAGE (float): Price change threshold (%) for corporate action detection.

    Example:
        >>> from silkroad.app.db.market_data import MarketDataDB
        >>> from silkroad.core.data_models import Asset, Horizon
        >>> import datetime as dt
        >>>
        >>> # Initialize database
        >>> db = MarketDataDB(horizon=Horizon.DAILY)
        >>>
        >>> # Update with market data
        >>> assets = [Asset(ticker="AAPL", name="Apple Inc.")]
        >>> db.update_from_alpaca(
        ...     api_key="YOUR_KEY",
        ...     api_secret="YOUR_SECRET",
        ...     assets=assets,
        ...     start=dt.datetime(2025, 1, 1),
        ...     end=dt.datetime(2025, 12, 1),
        ... )
        >>>
        >>> # Query data
        >>> barset = db.get_ticker_data("AAPL", start, end)
    """

    ALPACA_START_DATE = dt.datetime(2015, 1, 1, tzinfo=dt.timezone.utc)
    DETECT_PERCENTAGE = 1.0

    def __init__(
        self,
        db_path: tp.Optional[Path] = None,
        horizon: Horizon = Horizon.DAILY,
    ) -> None:
        """Initialize the market data database.

        Creates and connects to an ArcticDB database instance at the specified path.
        Validates that the provided horizon is supported by Alpaca data API and
        initializes the appropriate library for the given timeframe.

        Args:
            db_path (Path | None, optional): Path to database directory. If None,
                defaults to ~/stockdb. The directory is created if it doesn't exist.
                Defaults to None.
            horizon (Horizon, optional): Data frequency for all operations in this
                database instance. Must be one of: MINUTE, HOURLY, DAILY, WEEKLY, MONTHLY.
                Defaults to Horizon.DAILY.

        Raises:
            ValueError: If horizon is not in HORIZON_TO_TIMEFRAME mapping (unsupported
                frequency for Alpaca API).

        Note:
            All data stored in this database instance will be at the specified horizon.
            Changing horizon requires a separate database instance.
        """
        if horizon not in HORIZON_TO_TIMEFRAME:
            raise ValueError(
                f"Unsupported horizon: {horizon}. Supported horizons: {list(HORIZON_TO_TIMEFRAME.keys())}"
            )

        self.db_path = db_path or Path().home() / "stockdb"
        self.db_uri = "lmdb://" + str(self.db_path.resolve())
        self._db = adb.Arctic(uri=self.db_uri)

        # Store horizon and derive library name and timeframe
        self.horizon = horizon
        self.library_name = HORIZON_TO_LIBRARY_NAME[horizon]
        self.timeframe = HORIZON_TO_TIMEFRAME[horizon]

        self._library = self._db.get_library(self.library_name, create_if_missing=True)
        logger.info(
            f"Initialized MarketDataDB at {self.db_path} "
            f"(horizon: {horizon.name}, timeframe: {self.timeframe}, library: {self.library_name})"
        )

    def get_available_tickers(self) -> list[str]:
        """Retrieve list of all ticker symbols available in the database.

        Queries the ArcticDB library to get all symbols that have been written
        to this database instance.

        Returns:
            list[str]: List of ticker symbols (e.g., ['AAPL', 'GOOGL', 'BTC/USD']).
                Returns empty list if database is empty.
        """
        return self._library.list_symbols()

    def get_available_date_range(
        self, ticker: str
    ) -> tp.Tuple[dt.datetime, dt.datetime]:
        """Retrieve the available date range for a specific ticker.

        Queries the database to find the earliest and latest timestamps of data
        available for the given ticker. Ensures returned datetimes are timezone-aware
        (UTC).

        Args:
            ticker (str): Ticker symbol of the asset (e.g., 'AAPL', 'BTC/USD').

        Returns:
            tuple[datetime, datetime]: A tuple of (start_date, end_date) representing
                the range of available data. Both datetimes are timezone-aware (UTC).

        Raises:
            ValueError: If ticker is not found in the database.

        Note:
            Both returned datetimes are guaranteed to be timezone-aware with UTC
            timezone to prevent alignment and comparison errors.
        """
        # Check if ticker exists
        if ticker not in self.get_available_tickers():
            raise ValueError(f"Ticker {ticker} not found in database.")

        # Get the data for the ticker
        start = self._library.head(ticker, n=1).data
        end = self._library.tail(ticker, n=1).data

        # Grab the start and end date
        # ArcticDB stores as UTC but may return naive if we stripped it.
        # We assume stored data is always UTC.
        start_date = start.index[0]
        end_date = end.index[-1]

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=dt.timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=dt.timezone.utc)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def delete_ticker(self, ticker: str) -> None:
        """Delete all data for a specific ticker from the database.

        Removes all stored bars and metadata for the given ticker from the ArcticDB
        library. If the ticker doesn't exist, logs a warning but does not raise an error.

        Args:
            ticker (str): Ticker symbol to delete (e.g., 'AAPL', 'BTC/USD').

        Note:
            This operation is irreversible. To recover deleted data, you must
            re-download it using update_from_alpaca().
        """
        if ticker in self.get_available_tickers():
            self._library.delete(ticker)
            logger.info(f"Deleted ticker {ticker} from database.")
        else:
            logger.warning(f"Ticker {ticker} not found in database, cannot delete.")

    def get_latest_bar(self, ticker: str) -> tp.Optional[pd.Series]:
        """Retrieve the most recent bar (OHLCV data) for a ticker.

        Queries the database to get the last stored bar for the given ticker.
        Returns None if the ticker doesn't exist or has no data.

        Args:
            ticker (str): Ticker symbol (e.g., 'AAPL', 'BTC/USD').

        Returns:
            pd.Series | None: A pandas Series containing the latest bar data
                (open, high, low, close, volume). Returns None if ticker not found
                or database is empty for that ticker.

        Example:
            >>> latest = db.get_latest_bar("AAPL")
            >>> if latest is not None:
            ...     print(f"Close: {latest['close']}, Volume: {latest['volume']}")
        """
        if ticker not in self.get_available_tickers():
            return None

        df = self._library.tail(ticker, n=1).data
        if df.empty:
            return None

        return df.iloc[-1]

    @staticmethod
    def fetch_stock_data_from_alpaca(
        api_key: str,
        api_secret: str,
        symbols: list[str],
        start: dt.datetime,
        end: dt.datetime,
        adjustment: Adjustment = Adjustment.ALL,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """Fetch stock market data from Alpaca API.

        Retrieves OHLCV (Open, High, Low, Close, Volume) data for one or more
        stock tickers from the Alpaca Historical Data API for a specified date range
        and timeframe. Supports automatic adjustment for corporate actions (splits
        and dividends) via the adjustment parameter.

        Args:
            api_key (str): Alpaca API key for authentication.
            api_secret (str): Alpaca API secret for authentication.
            symbols (list[str]): List of stock ticker symbols to fetch
                (e.g., ['AAPL', 'GOOGL', 'MSFT']). At least one symbol required.
            start (dt.datetime): Start date (UTC) for the data range. Should be
                timezone-aware.
            end (dt.datetime): End date (UTC) for the data range. Should be
                timezone-aware. Must be after start date.
            adjustment (Adjustment, optional): Type of corporate action adjustment
                to apply. Defaults to Adjustment.ALL (adjust for both splits and
                dividends). Options: ALL, SPLIT, DIVIDEND, RAW.
            timeframe (TimeFrame, optional): Data granularity. Defaults to TimeFrame.Day.
                Options: Minute, Hour, Day, Week, Month.

        Returns:
            pd.DataFrame: DataFrame with MultiIndex (symbol, timestamp) containing
                columns: open, high, low, close, volume. Index is timezone-aware (UTC).
                Empty DataFrame if no data available for the date range.

        Raises:
            ValueError: If symbols list is empty or None.
            Exception: If Alpaca API authentication fails or API limit exceeded.

        Note:
            When using Adjustment.ALL, historical prices are retroactively adjusted
            for corporate actions. This means fetched data may differ from previous
            downloads if corporate actions occurred between requests.
        """
        if not symbols:
            raise ValueError("At least one symbol must be provided.")

        client = StockHistoricalDataClient(api_key, api_secret)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
        )
        bars = client.get_stock_bars(request_params)
        return bars.df  # type: ignore

    @staticmethod
    def fetch_crypto_data_from_alpaca(
        api_key: str,
        api_secret: str,
        symbols: list[str],
        start: dt.datetime,
        end: dt.datetime,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """Fetch cryptocurrency market data from Alpaca API.

        Retrieves OHLCV (Open, High, Low, Close, Volume) data for one or more
        cryptocurrency trading pairs from the Alpaca Crypto Data API for a specified
        date range and timeframe.

        Args:
            api_key (str): Alpaca API key for authentication.
            api_secret (str): Alpaca API secret for authentication.
            symbols (list[str]): List of crypto symbols in format 'BASE/QUOTE'
                (e.g., ['BTC/USD', 'ETH/USD', 'SOL/USD']). At least one symbol required.
            start (dt.datetime): Start date (UTC) for the data range. Should be
                timezone-aware.
            end (dt.datetime): End date (UTC) for the data range. Should be
                timezone-aware. Must be after start date.
            timeframe (TimeFrame, optional): Data granularity. Defaults to TimeFrame.Day.
                Options: Minute, Hour, Day, Week, Month.

        Returns:
            pd.DataFrame: DataFrame with MultiIndex (symbol, timestamp) containing
                columns: open, high, low, close, volume. Index is timezone-aware (UTC).
                Empty DataFrame if no data available for the date range.

        Raises:
            ValueError: If symbols list is empty or None.
            Exception: If Alpaca API authentication fails or API limit exceeded.

        Note:
            Cryptocurrency data from Alpaca does not support adjustment types
            like stocks. Prices are raw market prices without retroactive adjustments.
            Crypto symbols must use the format 'BASE/QUOTE' (e.g., 'BTC/USD').
        """
        if not symbols:
            raise ValueError("At least one symbol must be provided.")

        client = CryptoHistoricalDataClient(api_key, api_secret)
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        bars = client.get_crypto_bars(request_params)
        return bars.df  # type: ignore

    def update_from_alpaca(
        self,
        api_key: str,
        api_secret: str,
        assets: list[Asset],
        start: dt.datetime,
        end: dt.datetime,
        adjustment: Adjustment = Adjustment.ALL,
        update_mode: str = "smart_merge",
    ) -> None:
        """Update the database with market data fetched from Alpaca API.

        Fetches data from Alpaca for specified assets and stores it in the database
        using one of three strategies. Automatically classifies assets as stocks or
        cryptocurrencies and uses the appropriate Alpaca endpoint.

        **Corporate Actions Warning**:
            When using `Adjustment.ALL`, historical prices are retroactively adjusted
            for splits and dividends. This means new downloads may NOT match previously
            stored data, which is why smart_merge is recommended.

        **Update Modes**:
            - ``full_refresh``: Re-downloads ALL data from ALPACA_START_DATE to `end`.
              Overwrites existing data. Recommended after corporate actions.
            - ``smart_merge`` (Default): Downloads `[start, end]`, detects price
              discontinuities at merge boundary, triggers full_refresh if detected.
              **Recommended for most use cases.**
            - ``append_only``: Only downloads `[start, end]` and appends without checks.
              **DANGEROUS** if corporate actions occurred - will create discontinuities.

        Args:
            api_key (str): Alpaca API key for authentication.
            api_secret (str): Alpaca API secret for authentication.
            assets (list[Asset]): List of Asset objects containing ticker symbols
                and metadata. Asset.asset_class is used to route to appropriate endpoint.
            start (dt.datetime): Start date (UTC) for the data update. Must be
                timezone-aware. Ignored in full_refresh mode.
            end (dt.datetime): End date (UTC) for the data update. Must be
                timezone-aware. Should be after start date.
            adjustment (Adjustment, optional): Type of corporate action adjustment
                for stocks. Defaults to Adjustment.ALL. Ignored for crypto assets.
                Options: ALL, SPLIT, DIVIDEND, RAW.
            update_mode (str, optional): Update strategy to use. Defaults to
                'smart_merge'. Options: 'full_refresh', 'smart_merge', 'append_only'.

        Raises:
            ValueError: If update_mode is not one of the valid options.

        Note:
            - All data is stored at the frequency specified by self.horizon.
            - Crypto assets ignore the adjustment parameter (set to None).
            - Data is normalized to UTC-naive format for storage consistency.
            - Metadata is created for each asset tracking source and adjustment check time.
        """
        # Seperate assets by type and process accordingly
        logger.debug(f"Starting update with mode: {update_mode}")
        stock_assets = [
            a
            for a in assets
            if a.asset_class
            in [AssetClass.STOCK, AssetClass.COMMODITY, AssetClass.REAL_ESTATE]
            or a.asset_class is None
        ]
        crypto_assets = [a for a in assets if a.asset_class == AssetClass.CRYPTO]
        logger.debug(
            f"Found {len(stock_assets)} stock/commodity/real estate assets for update."
        )
        logger.debug(f"Found {len(crypto_assets)} crypto assets for update.")

        if update_mode == "full_refresh":
            # Download from inception (or earliest available date)
            # Alpaca's max history is ~2015 for most stocks
            historical_start = self.ALPACA_START_DATE

            # Fetch stock data
            if stock_assets:
                stock_tickers = [a.ticker for a in stock_assets]
                df_stocks = self.fetch_stock_data_from_alpaca(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbols=stock_tickers,
                    start=historical_start,
                    end=end,
                    adjustment=adjustment,
                    timeframe=self.timeframe,
                )
                logger.info(
                    f"Full refresh (stocks): Downloaded {len(df_stocks)} bars for {len(stock_assets)} assets from {historical_start}"
                )
                self._process_and_store(
                    df_stocks,
                    stock_assets,
                    start,
                    end,
                    api_key,
                    api_secret,
                    adjustment,
                    update_mode,
                )

            # Fetch crypto data
            if crypto_assets:
                crypto_tickers = [a.ticker for a in crypto_assets]
                df_crypto = self.fetch_crypto_data_from_alpaca(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbols=crypto_tickers,
                    start=historical_start,
                    end=end,
                    timeframe=self.timeframe,
                )
                logger.info(
                    f"Full refresh (crypto): Downloaded {len(df_crypto)} bars for {len(crypto_assets)} assets from {historical_start}"
                )
                self._process_and_store(
                    df_crypto,
                    crypto_assets,
                    start,
                    end,
                    api_key,
                    api_secret,
                    None,
                    update_mode,
                )

        elif update_mode == "smart_merge":

            # Merge stock data
            if stock_assets:
                stock_tickers = [a.ticker for a in stock_assets]
                df_stocks = self.fetch_stock_data_from_alpaca(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbols=stock_tickers,
                    start=start,
                    end=end,
                    adjustment=adjustment,
                    timeframe=self.timeframe,
                )
                logger.info(
                    f"Smart merge (stocks): Downloaded {len(df_stocks)} bars for update period"
                )
                self._process_and_store(
                    df_stocks,
                    stock_assets,
                    start,
                    end,
                    api_key,
                    api_secret,
                    adjustment,
                    update_mode,
                )

            # Merge crypto data
            if crypto_assets:
                crypto_tickers = [a.ticker for a in crypto_assets]
                df_crypto = self.fetch_crypto_data_from_alpaca(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbols=crypto_tickers,
                    start=start,
                    end=end,
                    timeframe=self.timeframe,
                )
                logger.info(
                    f"Smart merge (crypto): Downloaded {len(df_crypto)} bars for update period"
                )
                self._process_and_store(
                    df_crypto,
                    crypto_assets,
                    start,
                    end,
                    api_key,
                    api_secret,
                    None,
                    update_mode,
                )

        elif update_mode == "append_only":
            logger.warning(
                "⚠️  Using append_only mode. This may create data inconsistencies if corporate actions occurred!"
            )

            # Append stock data
            if stock_assets:
                stock_tickers = [a.ticker for a in stock_assets]
                df_stocks = self.fetch_stock_data_from_alpaca(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbols=stock_tickers,
                    start=start,
                    end=end,
                    adjustment=adjustment,
                    timeframe=self.timeframe,
                )
                self._process_and_store(
                    df_stocks,
                    stock_assets,
                    start,
                    end,
                    api_key,
                    api_secret,
                    adjustment,
                    update_mode,
                )

            # Append crypto data
            if crypto_assets:
                crypto_tickers = [a.ticker for a in crypto_assets]
                df_crypto = self.fetch_crypto_data_from_alpaca(
                    api_key=api_key,
                    api_secret=api_secret,
                    symbols=crypto_tickers,
                    start=start,
                    end=end,
                    timeframe=self.timeframe,
                )
                self._process_and_store(
                    df_crypto,
                    crypto_assets,
                    start,
                    end,
                    api_key,
                    api_secret,
                    None,
                    update_mode,
                )

        else:
            raise ValueError(
                f"Invalid update_mode: {update_mode}. Use 'full_refresh', 'smart_merge', or 'append_only'."
            )

        logger.info(f"Database update complete using {update_mode} mode.")

    def _create_metadata(self, asset: Asset, end: dt.datetime) -> dict[str, tp.Any]:
        """Create metadata dictionary for an asset.

        Generates a metadata dictionary containing asset information and database
        operation metadata. This metadata is stored with the asset data in ArcticDB
        for later retrieval and auditing.

        Args:
            asset (Asset): Asset object containing ticker, name, and classification.
            end (dt.datetime): End datetime of the data fetch operation (UTC).

        Returns:
            dict[str, Any]: Metadata dictionary with keys:
                - 'name' (str): Asset name
                - 'asset_class' (str | None): Asset class enum value (e.g., 'STOCK')
                - 'sector' (str | None): Industry sector enum value
                - 'exchange' (str | None): Trading exchange enum value
                - 'source' (str): Data source identifier (always 'Alpaca')
                - 'last_adjustment_check' (str): ISO format timestamp of operation

        Note:
            This metadata is written to ArcticDB and can be retrieved during read operations.
        """
        return {
            "name": asset.name,
            "asset_class": asset.asset_class.value if asset.asset_class else None,
            "sector": asset.sector.value if asset.sector else None,
            "exchange": asset.exchange.value if asset.exchange else None,
            "source": "Alpaca",
            "last_adjustment_check": end.isoformat(),
        }

    def _write_asset_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        asset: Asset,
        end: dt.datetime,
    ) -> None:
        """Write asset data to the database, overwriting any existing data.

        Normalizes the index to UTC-naive format for storage consistency and writes
        the data with metadata to the ArcticDB library. Used in full_refresh mode
        and for initial writes of new tickers.

        Args:
            ticker (str): Ticker symbol to write data for.
            data (pd.DataFrame): OHLCV data with datetime index. Index should be
                timezone-aware (will be converted to UTC-naive for storage).
            asset (Asset): Asset object for metadata generation.
            end (dt.datetime): End datetime of operation for metadata.

        Note:
            This method completely overwrites any existing data for the ticker.
            For incremental updates, use _library.update() via smart_merge instead.
        """
        # Ensure UTC and remove timezone info for storage compatibility
        if data.index.tz is not None:  # type: ignore
            data.index = data.index.tz_convert("UTC").tz_localize(None)  # type: ignore

        self._library.write(
            symbol=ticker,
            data=data,
            metadata=self._create_metadata(asset, end),
        )

    def _process_and_store(
        self,
        df: pd.DataFrame,
        assets: list[Asset],
        start: dt.datetime,
        end: dt.datetime,
        api_key: str,
        api_secret: str,
        adjustment: tp.Optional[Adjustment],
        update_mode: str,
    ) -> None:
        """Process fetched data and store it in database based on update mode.

        Iterates through assets, extracts their data from the MultiIndex DataFrame,
        and stores using the specified update mode. Handles corporate action detection
        in smart_merge mode by comparing prices at merge boundaries.

        Args:
            df (pd.DataFrame): MultiIndex DataFrame with (symbol, timestamp) index
                and OHLCV columns. From fetch_stock_data_from_alpaca or fetch_crypto_data_from_alpaca.
            assets (list[Asset]): Asset objects whose data is in df.
            start (dt.datetime): Start date of the update window (UTC).
            end (dt.datetime): End date of the update window (UTC).
            api_key (str): Alpaca API key for potential recursive full_refresh calls.
            api_secret (str): Alpaca API secret for potential recursive full_refresh calls.
            adjustment (Adjustment | None): Adjustment type for recursive calls.
                None for crypto assets.
            update_mode (str): Strategy: 'full_refresh', 'smart_merge', or 'append_only'.

        Raises:
            KeyError: Caught and logged if ticker not found in fetched data.

        Note:
            In smart_merge mode, if price change > DETECT_PERCENTAGE (default 1%)
            is detected at merge boundary, this method recursively calls update_from_alpaca
            with full_refresh mode to correct potential adjustment discontinuities.
        """
        for asset in assets:
            try:
                ticker_df = df.xs(asset.ticker, level="symbol")
                if ticker_df.empty:
                    logger.warning(f"No data found for ticker {asset.ticker}.")
                    continue

                # Normalize index to UTC-naive
                if ticker_df.index.tz is not None:  # type: ignore
                    ticker_df.index = ticker_df.index.tz_convert("UTC").tz_localize(  # type: ignore
                        None
                    )

                # Dispatch based on update_mode
                if update_mode == "full_refresh":
                    self._write_asset_data(asset.ticker, ticker_df, asset, end)
                    logger.info(
                        f"✓ {asset.ticker}: Wrote {len(ticker_df)} bars (full refresh)"
                    )

                elif update_mode == "append_only":
                    if asset.ticker in self.get_available_tickers():
                        self._library.update(symbol=asset.ticker, data=ticker_df)
                        logger.info(
                            f"✓ {asset.ticker}: Appended {len(ticker_df)} bars (append_only)"
                        )
                    else:
                        self._write_asset_data(asset.ticker, ticker_df, asset, end)
                        logger.info(
                            f"✓ {asset.ticker}: Initial write {len(ticker_df)} bars (append_only)"
                        )

                elif update_mode == "smart_merge":
                    if asset.ticker not in self.get_available_tickers():
                        self._write_asset_data(asset.ticker, ticker_df, asset, end)
                        logger.info(
                            f"✓ {asset.ticker}: Initial write {len(ticker_df)} bars"
                        )
                        continue

                    # Check overlap/conflict
                    existing_data = self._library.tail(asset.ticker, n=5).data
                    if existing_data.index.tz is not None:  # type: ignore
                        existing_data.index = existing_data.index.tz_convert(
                            "UTC"
                        ).tz_localize(
                            None
                        )  # type: ignore

                    overlap_date = ticker_df.index[0]
                    if overlap_date in existing_data.index:
                        stored_close = existing_data.loc[overlap_date, "close"]
                        new_close = ticker_df.iloc[0]["close"]

                        # Detect >1% price mismatch (likely adjustment)
                        price_diff_pct = (
                            abs(new_close - stored_close) / stored_close * 100
                        )

                        if price_diff_pct > self.DETECT_PERCENTAGE:
                            logger.warning(
                                f"⚠️  {asset.ticker}: Detected {price_diff_pct:.2f}% price change at {overlap_date}. "
                                f"Likely corporate action. Triggering full refresh..."
                            )
                            # Recursively call with full_refresh mode
                            self.update_from_alpaca(
                                api_key=api_key,
                                api_secret=api_secret,
                                assets=[asset],
                                start=start,
                                end=end,
                                adjustment=adjustment or Adjustment.ALL,
                                update_mode="full_refresh",
                            )
                            continue

                    # Safe to append
                    self._library.update(
                        symbol=asset.ticker,
                        data=ticker_df,
                        metadata=self._create_metadata(asset, end),
                    )
                    logger.info(f"✓ {asset.ticker}: Appended {len(ticker_df)} bars")

            except KeyError:
                logger.warning(f"Ticker {asset.ticker} not found in fetched data.")

    def refresh(
        self,
        api_key: str,
        api_secret: str,
        assets: list[Asset],
        adjustment: Adjustment = Adjustment.ALL,
        end: tp.Optional[dt.datetime] = None,
    ) -> None:
        """Refresh database by updating all assets to the latest available date.

        Convenience method that updates each asset from its last stored date to
        the specified end date (or yesterday if not specified). Uses smart_merge
        mode for efficient updates with corporate action detection.

        Args:
            api_key (str): Alpaca API key for authentication.
            api_secret (str): Alpaca API secret for authentication.
            assets (list[Asset]): Asset objects to refresh. For new assets, download
                starts from ALPACA_START_DATE.
            adjustment (Adjustment, optional): Corporate action adjustment for stocks.
                Defaults to Adjustment.ALL. Ignored for crypto assets.
            end (dt.datetime | None, optional): End date for refresh (UTC).
                Defaults to yesterday (now() - 1 day). Must be timezone-aware if provided.

        Note:
            - Uses smart_merge mode for all updates (detects corporate actions).
            - For new tickers, downloads from ALPACA_START_DATE (2015-01-01).
            - For existing tickers, starts from last_date + 1 day.
            - Skips assets that are already up-to-date.
            - Data frequency determined by database's horizon property.

        Example:
            >>> assets = [Asset(ticker="AAPL"), Asset(ticker="GOOGL")]
            >>> db.refresh(api_key, api_secret, assets)
        """
        if end is None:
            end = (
                Exchange.NYSE.previous_market_close()
            )  # Default to yesterday's market close

        for asset in assets:
            if asset.ticker in self.get_available_tickers():
                _, last_date = self.get_available_date_range(asset.ticker)
                # Ensure last_date is timezone-aware for comparison
                if last_date.tzinfo is None:
                    last_date = last_date.replace(tzinfo=dt.timezone.utc)

                start = last_date + dt.timedelta(days=1)
            else:
                start = self.ALPACA_START_DATE

            if start >= end:
                logger.info(f"{asset.ticker} is already up-to-date.")
                continue

            self.update_from_alpaca(
                api_key=api_key,
                api_secret=api_secret,
                assets=[asset],
                start=start,
                end=end,
                adjustment=adjustment,
                update_mode="smart_merge",
            )

    def reset(self) -> None:
        """Reset the database by deleting all stored data.

        Removes all tickers and associated data from the ArcticDB library.
        This operation is irreversible and cannot be undone.

        Warning:
            This completely clears the database. To recover data, you must
            re-download using update_from_alpaca() or refresh().
        """
        for ticker in self.get_available_tickers():
            self._library.delete(ticker)
        logger.info("Database has been reset. All data deleted.")

    def get_ticker_data(
        self,
        ticker: str,
        start: dt.datetime,
        end: dt.datetime,
        localize: str = "UTC",
    ) -> UniformBarSet:
        """Retrieve OHLCV data for a ticker and date range as a UniformBarSet.

        Queries the database for the specified ticker and date range, returns
        a UniformBarSet object that wraps both historical data and provides
        streaming buffer capabilities for live data integration.

        Args:
            ticker (str): Ticker symbol of the asset (e.g., 'AAPL', 'BTC/USD').
            start (dt.datetime): Start date (UTC) for the query range. Must be
                timezone-aware.
            end (dt.datetime): End date (UTC) for the query range. Must be
                timezone-aware. Must be after start date.
            localize (str, optional): Target timezone for returned data. Defaults to
                'UTC'. Accepts timezone strings (e.g., 'US/Eastern', 'Europe/London').

        Returns:
            UniformBarSet: Wrapper object containing the OHLCV data with metadata.
                Includes symbol, horizon, and historical DataFrame. Ready for use
                in portfolio analysis and strategy backtesting.

        Raises:
            ValueError: If ticker not found in database or if start >= end.

        Note:
            - Data is stored internally as UTC-naive but returned localized
              to the requested timezone.
            - All timestamps are timezone-aware in the returned UniformBarSet.
            - UniformBarSet provides both historical data access and streaming
              buffer for live updates.

        Example:
            >>> import datetime as dt
            >>> bar_set = db.get_ticker_data(
            ...     'AAPL',
            ...     dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ...     dt.datetime(2025, 12, 31, tzinfo=dt.timezone.utc),
            ...     localize='US/Eastern'
            ... )
            >>> print(bar_set.symbol, bar_set.horizon)
        """
        # Check if ticker exists
        if ticker not in self.get_available_tickers():
            raise ValueError(
                f"""Ticker {ticker} not found in database. Use update_from_alpaca to add it first to the database."""
            )

        # Check if start date and end date are valid
        if start >= end:
            raise ValueError("Start date must be before end date.")

        # Read data from ArcticDB
        # ArcticDB reads as naive if we wrote as naive.
        data = self._library.read(ticker, date_range=(start, end)).data

        # Localize data to the specified timezone
        # Assume stored data is UTC (naive)
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")

        data.index = data.index.tz_convert(localize)

        return UniformBarSet.from_df(
            symbol=ticker,
            horizon=self.horizon,
            df=data,  # type: ignore
        )

    def get_uniform_bar_collection(
        self,
        ticker: list[str],
        start: dt.datetime,
        end: dt.datetime,
        localize: str = "UTC",
    ) -> UniformBarCollection:
        """Retrieve OHLCV data for multiple tickers as a UniformBarCollection.

        Queries the database for multiple tickers, returns a UniformBarCollection
        that manages a unified time index across all assets. Automatically intersects
        date ranges to ensure all assets have data on the same dates.

        Args:
            ticker (list[str]): List of ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT']).
                All tickers must exist in the database.
            start (dt.datetime): Start date (UTC) for the query range. Must be
                timezone-aware.
            end (dt.datetime): End date (UTC) for the query range. Must be
                timezone-aware. Must be after start date.
            localize (str, optional): Target timezone for returned data. Defaults to
                'UTC'. Accepts timezone strings (e.g., 'US/Eastern', 'Europe/London').

        Returns:
            UniformBarCollection: Collection object managing multiple UniformBarSet
                instances with a unified, intersected time index. Contains bar_map
                dictionary keyed by ticker symbol.

        Raises:
            ValueError: If any ticker not found in database or if start >= end.

        Note:
            - Data is automatically aligned to a common time index via intersection.
            - This ensures all assets have data on exactly the same dates.
            - All timestamps are timezone-aware in the returned collection.
            - Useful for portfolio analysis requiring consistent time alignment.

        Example:
            >>> import datetime as dt
            >>> collection = db.get_uniform_bar_collection(
            ...     ['AAPL', 'GOOGL', 'MSFT'],
            ...     dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ...     dt.datetime(2025, 12, 31, tzinfo=dt.timezone.utc)
            ... )
            >>> print(f"Assets: {list(collection.bar_map.keys())}")
            >>> aapl_bars = collection.bar_map['AAPL']
        """
        bar_sets = []
        for t in ticker:
            bar_set = self.get_ticker_data(t, start, end, localize)
            bar_sets.append(bar_set)

        # Intersect the date indices to ensure uniformity
        intersected_bars = UniformBarSet.get_intersection(*bar_sets)
        return UniformBarCollection(bar_map={bs.symbol: bs for bs in intersected_bars})

    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of all tickers in the database.

        Compiles metadata and date range information for each ticker stored
        in the ArcticDB library into a single summary DataFrame for easy
        inspection and auditing.

        Returns:
            pd.DataFrame: Summary DataFrame with columns and index (ticker):
                - 'name' (str | None): Asset name from metadata
                - 'asset_class' (str | None): Asset class from metadata
                - 'sector' (str | None): Industry sector from metadata
                - 'exchange' (str | None): Trading exchange from metadata
                - 'source' (str | None): Data source from metadata
                - 'last_adjustment_check' (str | None): Timestamp of last adjustment check
                - 'start_date' (dt.datetime): Earliest date of stored data
                - 'end_date' (dt.datetime): Latest date of stored data
                - 'num_bars' (int): Total number of bars stored

        Note:
            - Metadata fields may be None if not set during data writes.
            - Date range and bar count are derived from the stored data.
        """
        summary_rows = []
        for ticker in self.get_available_tickers():
            metadata = self._library.read_metadata(ticker).metadata
            data = self._library.read(ticker).data
            start_date = data.index.min()
            end_date = data.index.max()
            num_bars = len(data)
            summary_rows.append(
                {
                    "ticker": ticker,
                    "name": metadata.get("name"),
                    "asset_class": metadata.get("asset_class"),
                    "sector": metadata.get("sector"),
                    "exchange": metadata.get("exchange"),
                    "source": metadata.get("source"),
                    "last_adjustment_check": metadata.get("last_adjustment_check"),
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_bars": num_bars,
                }
            )

        return pd.DataFrame(summary_rows).set_index("ticker")
