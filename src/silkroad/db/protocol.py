"""Database protocol definition for Silkroad storage backends."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from datetime import datetime
import pandas as pd

from silkroad.core.enums import Horizon
from silkroad.core.news_models import NewsArticle


class DatabaseProvider(ABC):
    """Abstract base class definition for database providers."""

    # --- Market Data Operations ---
    @abstractmethod
    def save_market_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        horizon: Horizon,
        mode: str = "append",
        metadata: Optional[dict] = None,
    ):
        """Save market data for a standardized symbol.

        Args:
            symbol: Ticker symbol.
            data: DataFrame with OHLCV columns and timestamp index/column.
            horizon: Time horizon (DAILY, MINUTE, etc.).
            mode: 'append' (default) or 'replace'.
            metadata: Optional dictionary of asset metadata (sector, exchange, etc.).
        """
        pass

    @abstractmethod
    def get_market_data(
        self, symbols: List[str], horizon: Horizon, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Retrieve market data for multiple symbols.

        Returns:
            DataFrame with MultiIndex (symbol, timestamp).
        """
        pass

    @abstractmethod
    def get_last_timestamp(self, symbol: str, horizon: Horizon) -> Optional[datetime]:
        """Get the last stored timestamp for a symbol."""
        pass

    @abstractmethod
    def list_symbols(self, horizon: Horizon) -> List[str]:
        """List all symbols available for a horizon."""
        pass

    @abstractmethod
    def filter_symbols(self, horizon: Horizon, **criteria) -> List[str]:
        """Filter symbols based on stored metadata values.

        Args:
            horizon: Time horizon.
            **criteria: Meta keys and values to filter by (e.g. sector="Technology").
        """
        pass

    @abstractmethod
    def get_symbol_metadata(self, horizon: Horizon) -> pd.DataFrame:
        """Get aggregate metadata (start, end, etc) for all symbols."""
        pass

    @abstractmethod
    def delete_range(
        self, symbol: str, horizon: Horizon, start: datetime, end: datetime
    ):
        """Delete data for a symbol within a specific time range."""
        pass

    # --- News Operations ---
    @abstractmethod
    def save_news(self, news: List[NewsArticle]):
        """Save a batch of news articles."""
        pass

    @abstractmethod
    def get_news(
        self, tickers: Optional[List[str]], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Retrieve news articles.

        Args:
            tickers: Optional list of tickers to filter by. If None, returns all news.
            start: Start datetime.
            end: End datetime.

        Returns:
            DataFrame suitable for analysis.
        """
        pass

    # --- Raw SQL Safety Hatch ---
    @abstractmethod
    def execute_query(self, query: str, params: tuple = ()) -> Any:
        """Execute a raw SQL query (use sparingly)."""
        pass
