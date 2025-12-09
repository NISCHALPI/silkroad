"""Data backend providers for fetching OHLCV market data from various sources.

This module provides abstract and concrete implementations for fetching historical
market data (OHLCV bars) from different data providers. All backend providers
return data in a standardized format suitable for use with Silkroad's core
data models (UniformBarSet, UniformBarCollection).

The module includes:
    - DataBackendProvider: Abstract base class defining the interface
    - YahooFinanceBackendProvider: Fetches data from Yahoo Finance
    - AlpacaBackendProvider: Fetches data from Alpaca Markets (stocks and crypto)

All providers return DataFrames with:
    - MultiIndex: (symbol, timestamp)
    - Columns: ['open', 'high', 'low', 'close', 'volume']
    - Timezone-aware timestamps in UTC
"""

import abc
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.enums import DataFeed
from silkroad.core.enums import Horizon, AssetClass
from alpaca.data.enums import Adjustment
from silkroad.core.data_models import Asset
from silkroad.logging.logger import logger
import datetime as dt
import typing as tp
import pandas as pd
import yfinance as yf

REQUIRED = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


class DataBackendProvider(abc.ABC):
    """Abstract base class for market data backend providers.

    All data backend providers must inherit from this class and implement the
    fetch_data method. This ensures a consistent interface for retrieving
    historical OHLCV data across different data sources.

    The output DataFrame must adhere to the following format:
        - MultiIndex with levels: ['symbol', 'timestamp']
        - Required columns: ['open', 'high', 'low', 'close', 'volume']
        - All timestamps must be timezone-aware (UTC)
        - Data must be sorted by symbol and timestamp

    Example:
        >>> class CustomBackend(DataBackendProvider):
        ...     def fetch_data(self, asset, start, end, horizon, **kwargs):
        ...         # Implementation here
        ...         return df
    """

    @abc.abstractmethod
    def fetch_data(
        self,
        asset: list[Asset],
        start: dt.datetime,
        end: dt.datetime,
        horizon: Horizon,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for given assets and time range.

        Args:
            asset: List of Asset objects to fetch data for.
            start: Start datetime for the data request (timezone-aware recommended).
            end: End datetime for the data request (timezone-aware recommended).
            horizon: Time frequency for the bars (e.g., DAILY, HOURLY).
            **kwargs: Additional provider-specific parameters.

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and columns
            ['open', 'high', 'low', 'close', 'volume']. Empty DataFrame
            if no data is available.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass


class AlpacaBackendProvider(DataBackendProvider):
    """Alpaca Markets data backend provider.

    Fetches historical market data from Alpaca Markets, supporting both
    stocks and cryptocurrencies. Uses separate clients for each asset class
    and applies corporate action adjustments by default.

    Features:
        - Dual support for stocks and cryptocurrencies
        - Corporate action adjustments (splits, dividends) applied by default
        - Multi-asset batch requests
        - Date range validation with warnings
        - Automatic asset class detection and routing

    Supported Horizons:
        - MINUTE: 1-minute bars
        - HOURLY: 1-hour bars
        - DAILY: Daily bars
        - WEEKLY: Weekly bars
        - MONTHLY: Monthly bars

    Note:
        Requires valid Alpaca API credentials. Free tier has rate limits and
        limited historical data access. See Alpaca's documentation for details.

    Attributes:
        stock_client: StockHistoricalDataClient for equity data.
        crypto_client: CryptoHistoricalDataClient for cryptocurrency data.
        feed: DataFeed enum specifying the data feed for stocks.

    Example:
        >>> provider = AlpacaBackendProvider(api_key="KEY", api_secret="SECRET")
        >>> assets = [
        ...     Asset(ticker="AAPL", asset_class=AssetClass.STOCK),
        ...     Asset(ticker="BTC/USD", asset_class=AssetClass.CRYPTO)
        ... ]
        >>> start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
        >>> end = dt.datetime(2023, 12, 31, tzinfo=dt.timezone.utc)
        >>> df = provider.fetch_data(assets, start, end, Horizon.DAILY)
    """

    def __init__(
        self, api_key: str, api_secret: str, feed: tp.Optional[DataFeed] = None
    ):
        """Initialize the Alpaca data backend provider.

        Args:
            api_key: Alpaca API key for authentication.
            api_secret: Alpaca API secret for authentication.
            feed: Data feed to use for stock data (default is None).

        Note:
            - Stocks and cryptos use separate clients internally.
            - Ensure API credentials have necessary permissions.
        """
        self.stock_client = StockHistoricalDataClient(
            api_key=api_key, secret_key=api_secret
        )
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=api_key, secret_key=api_secret
        )

        # Data feed for receiving data
        self.feed = feed

    def fetch_data(
        self,
        asset: list[Asset],
        start: dt.datetime,
        end: dt.datetime,
        horizon: Horizon,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Alpaca Markets.

        Automatically routes assets to appropriate clients based on asset_class.
        Stocks use StockHistoricalDataClient, cryptos use CryptoHistoricalDataClient.
        Corporate action adjustments (Adjustment.ALL) are applied by default to stocks.

        Args:
            asset: List of Asset objects to fetch data for.
            start: Start datetime for the data request (timezone-aware recommended).
            end: End datetime for the data request (timezone-aware recommended).
            horizon: Time frequency for the bars (MINUTE, HOURLY, DAILY, WEEKLY, MONTHLY).
            **kwargs: Additional keyword arguments:
                - adjustment (Adjustment): Override default adjustment setting (stocks only).
                - limit (int, optional): Max number of bars to fetch per asset.
                - currency (str, optional): Currency for stock data (default is "USD").

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) containing OHLCV data.
            Returns empty DataFrame if no data is available or all requests fail.

        Warnings:
            Logs warnings if fetched data start/end dates don't match requested dates.
            Prints error messages if API requests fail.

        Note:
            The method continues execution even if one asset class fails, returning
            data for successful requests.
        """
        stocks = [a.ticker for a in asset if a.asset_class == AssetClass.STOCK]
        cryptos = [a.ticker for a in asset if a.asset_class == AssetClass.CRYPTO]

        # Check if there are other type of assets
        other_assets = [
            a
            for a in asset
            if a.asset_class
            not in (
                AssetClass.STOCK,
                AssetClass.CRYPTO,
                AssetClass.COMMODITY,
                AssetClass.REAL_ESTATE,
            )
        ]
        if other_assets:
            raise NotImplementedError(
                f"AlpacaBackendProvider does not support asset classes: "
                f"{', '.join(set(a.asset_class.value for a in other_assets))}"
            )

        timeframe = horizon.to_alpaca_timeframe_unit()

        dfs = []

        if stocks:
            req = StockBarsRequest(
                symbol_or_symbols=stocks,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment=kwargs.get("adjustment", Adjustment.ALL),
                feed=self.feed,
                limit=kwargs.get("limit", None),
                currency=kwargs.get("currency", "USD"),
            )
            try:
                stock_bars = self.stock_client.get_stock_bars(req)
                if not stock_bars.df.empty:  # type: ignore
                    dfs.append(stock_bars.df)  # type: ignore
            except Exception as e:
                print(f"Error fetching stock data: {e}")

        if cryptos:
            req = CryptoBarsRequest(
                symbol_or_symbols=cryptos,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=kwargs.get("limit", None),
            )
            try:
                crypto_bars = self.crypto_client.get_crypto_bars(req)
                if not crypto_bars.df.empty:  # type: ignore
                    dfs.append(crypto_bars.df)  # type: ignore
            except Exception as e:
                print(f"Error fetching crypto data: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs)
        # Filter columns
        df = df[REQUIRED]

        # Check if data covers the requested range
        fetched_start = df.index.get_level_values("timestamp").min().date()
        fetched_end = df.index.get_level_values("timestamp").max().date()
        requested_start = start.date()
        requested_end = end.date()

        if fetched_start > requested_start:
            logger.warning(
                f"Fetched data start date ({fetched_start}) is later than requested start date ({requested_start})."
            )
        if fetched_end < requested_end:
            logger.warning(
                f"Fetched data end date ({fetched_end}) is earlier than requested end date ({requested_end})."
            )

        return df
