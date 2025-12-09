import pytest
import pandas as pd
import datetime as dt
from unittest.mock import MagicMock, patch
from silkroad.db.backends import AlpacaBackendProvider
from silkroad.core.data_models import Asset
from silkroad.core.enums import Horizon, AssetClass, Exchange, Sector


# Fixtures
@pytest.fixture
def sample_assets():
    return [
        Asset(  # type: ignore
            ticker="AAPL",
            asset_class=AssetClass.STOCK,
            exchange=Exchange.NASDAQ,
            sector=Sector.TECHNOLOGY,
        ),
        Asset(  # type: ignore
            ticker="BTC/USD",
            asset_class=AssetClass.CRYPTO,
            exchange=Exchange.OTHER,
            sector=Sector.OTHERS,
        ),
    ]


@pytest.fixture
def sample_dates():
    start = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 1, 10, tzinfo=dt.timezone.utc)
    return start, end


class TestAlpacaBackendProvider:

    @patch("silkroad.db.backends.StockHistoricalDataClient")
    @patch("silkroad.db.backends.CryptoHistoricalDataClient")
    def test_fetch_data_mixed_assets(
        self, MockCryptoClient, MockStockClient, sample_assets, sample_dates
    ):
        # Setup mocks
        mock_stock_client = MockStockClient.return_value
        mock_crypto_client = MockCryptoClient.return_value

        provider = AlpacaBackendProvider(api_key="test", api_secret="test")
        start, end = sample_dates

        # Mock Stock Response
        dates = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        stock_df = pd.DataFrame(
            {
                "open": [150.0] * len(dates),
                "high": [155.0] * len(dates),
                "low": [145.0] * len(dates),
                "close": [152.0] * len(dates),
                "volume": [10000] * len(dates),
                "trade_count": [500] * len(dates),
                "vwap": [151.0] * len(dates),
                "symbol": ["AAPL"] * len(dates),
            },
            index=dates,
        )
        stock_df.index.name = "timestamp"
        stock_df = stock_df.reset_index().set_index(["symbol", "timestamp"])

        mock_stock_response = MagicMock()
        mock_stock_response.df = stock_df
        mock_stock_client.get_stock_bars.return_value = mock_stock_response

        # Mock Crypto Response
        crypto_df = pd.DataFrame(
            {
                "open": [30000.0] * len(dates),
                "high": [31000.0] * len(dates),
                "low": [29000.0] * len(dates),
                "close": [30500.0] * len(dates),
                "volume": [100] * len(dates),
                "trade_count": [1000] * len(dates),
                "vwap": [30200.0] * len(dates),
                "symbol": ["BTC/USD"] * len(dates),
            },
            index=dates,
        )
        crypto_df.index.name = "timestamp"
        crypto_df = crypto_df.reset_index().set_index(["symbol", "timestamp"])

        mock_crypto_response = MagicMock()
        mock_crypto_response.df = crypto_df
        mock_crypto_client.get_crypto_bars.return_value = mock_crypto_response

        # Execute
        result = provider.fetch_data(sample_assets, start, end, Horizon.DAILY)

        # Verify
        assert not result.empty
        assert len(result) == len(stock_df) + len(crypto_df)
        assert "AAPL" in result.index.get_level_values("symbol")
        assert "BTC/USD" in result.index.get_level_values("symbol")

        # Verify columns
        expected_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
        for col in expected_cols:
            assert col in result.columns

        # Verify calls
        mock_stock_client.get_stock_bars.assert_called_once()
        mock_crypto_client.get_crypto_bars.assert_called_once()

    @patch("silkroad.db.backends.StockHistoricalDataClient")
    @patch("silkroad.db.backends.CryptoHistoricalDataClient")
    @patch("silkroad.db.backends.logger")
    def test_date_mismatch_warning(
        self,
        mock_logger,
        MockCryptoClient,
        MockStockClient,
        sample_assets,
        sample_dates,
    ):
        provider = AlpacaBackendProvider(api_key="test", api_secret="test")
        start, end = sample_dates

        # Mock Stock Response with shorter range
        short_start = start + dt.timedelta(days=2)
        short_end = end - dt.timedelta(days=2)
        dates = pd.date_range(start=short_start, end=short_end, freq="D", tz="UTC")

        stock_df = pd.DataFrame(
            {
                "open": [150.0] * len(dates),
                "high": [155.0] * len(dates),
                "low": [145.0] * len(dates),
                "close": [152.0] * len(dates),
                "volume": [10000] * len(dates),
                "trade_count": [500] * len(dates),
                "vwap": [151.0] * len(dates),
                "symbol": ["AAPL"] * len(dates),
            },
            index=dates,
        )
        stock_df.index.name = "timestamp"
        stock_df = stock_df.reset_index().set_index(["symbol", "timestamp"])

        mock_stock_response = MagicMock()
        mock_stock_response.df = stock_df
        MockStockClient.return_value.get_stock_bars.return_value = mock_stock_response

        # Only test with stock asset
        stock_asset = [a for a in sample_assets if a.asset_class == AssetClass.STOCK]

        provider.fetch_data(stock_asset, start, end, Horizon.DAILY)

        assert mock_logger.warning.call_count == 2

    @patch("silkroad.db.backends.StockHistoricalDataClient")
    @patch("silkroad.db.backends.CryptoHistoricalDataClient")
    def test_fetch_data_partial_failure(
        self, MockCryptoClient, MockStockClient, sample_assets, sample_dates
    ):
        # Setup mocks
        mock_stock_client = MockStockClient.return_value
        mock_crypto_client = MockCryptoClient.return_value

        provider = AlpacaBackendProvider(api_key="test", api_secret="test")
        start, end = sample_dates

        # Mock Stock Response (Success)
        dates = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        stock_df = pd.DataFrame(
            {
                "open": [150.0] * len(dates),
                "high": [155.0] * len(dates),
                "low": [145.0] * len(dates),
                "close": [152.0] * len(dates),
                "volume": [10000] * len(dates),
                "trade_count": [500] * len(dates),
                "vwap": [151.0] * len(dates),
                "symbol": ["AAPL"] * len(dates),
            },
            index=dates,
        )
        stock_df.index.name = "timestamp"
        stock_df = stock_df.reset_index().set_index(["symbol", "timestamp"])

        mock_stock_response = MagicMock()
        mock_stock_response.df = stock_df
        mock_stock_client.get_stock_bars.return_value = mock_stock_response

        # Mock Crypto Response (Failure)
        mock_crypto_client.get_crypto_bars.side_effect = Exception("API Error")

        # Execute
        result = provider.fetch_data(sample_assets, start, end, Horizon.DAILY)

        # Verify
        assert not result.empty
        assert "AAPL" in result.index.get_level_values("symbol")
        assert "BTC/USD" not in result.index.get_level_values("symbol")

        # Verify calls
        mock_stock_client.get_stock_bars.assert_called_once()
        mock_crypto_client.get_crypto_bars.assert_called_once()

    @patch("silkroad.db.backends.StockHistoricalDataClient")
    @patch("silkroad.db.backends.CryptoHistoricalDataClient")
    def test_fetch_data_empty(
        self, MockCryptoClient, MockStockClient, sample_assets, sample_dates
    ):
        # Setup mocks
        mock_stock_client = MockStockClient.return_value
        mock_crypto_client = MockCryptoClient.return_value

        provider = AlpacaBackendProvider(api_key="test", api_secret="test")
        start, end = sample_dates

        # Mock Empty Responses
        mock_stock_response = MagicMock()
        mock_stock_response.df = pd.DataFrame()
        mock_stock_client.get_stock_bars.return_value = mock_stock_response

        mock_crypto_response = MagicMock()
        mock_crypto_response.df = pd.DataFrame()
        mock_crypto_client.get_crypto_bars.return_value = mock_crypto_response

        # Execute
        result = provider.fetch_data(sample_assets, start, end, Horizon.DAILY)

        # Verify
        assert result.empty
