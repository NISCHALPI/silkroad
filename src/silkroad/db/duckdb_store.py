"""DuckDB-based storage backend implementation."""

import duckdb
import pandas as pd
import json
from datetime import datetime, timezone
from typing import List, Optional, Any
from pathlib import Path

from silkroad.core.enums import Horizon
from silkroad.core.news_models import NewsArticle
from silkroad.db.protocol import DatabaseProvider
from silkroad.logging.logger import logger


class DuckDBStore(DatabaseProvider):
    """DuckDB implementation of the DatabaseProvider."""

    DEFAULT_DB_PATH = Path.home() / ".cache" / "silkroad" / "silkroad.duckdb"

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    # --- Schema ---
    def _init_schema(self):
        """Initialize database schema if not exists."""
        # News Tables
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP,
                source VARCHAR,
                headline VARCHAR,
                content VARCHAR,
                url VARCHAR,
                sentiment DOUBLE,
                metadata JSON
            );
            
            CREATE TABLE IF NOT EXISTS news_tickers (
                article_id VARCHAR,
                ticker VARCHAR,
                PRIMARY KEY (article_id, ticker),
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            );

            CREATE TABLE IF NOT EXISTS symbol_attributes (
                symbol VARCHAR PRIMARY KEY,
                metadata VARCHAR
            );
        """
        )

    def _get_table_name(self, horizon: Horizon) -> str:
        return f"market_data_{horizon.name.lower()}"

    def _ensure_market_table(self, horizon: Horizon):
        table_name = self._get_table_name(horizon)
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                vwap DOUBLE,
                trade_count DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            );
        """
        )

    # --- Market Data ---

    def save_market_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        horizon: Horizon,
        mode: str = "append",
        metadata: Optional[dict] = None,
    ):
        """Save market data for a specific symbol and horizon.

        Args:
            symbol (str): The asset ticker.
            data (pd.DataFrame): Market data with simple column names (open, high, low, close, volume).
                Must contain 'timestamp' or be indexed by timestamp.
            horizon (Horizon): Time horizon of the data.
            mode (str, optional): 'append' (INSERT OR IGNORE) or 'replace' (INSERT OR REPLACE). Defaults to "append".
            metadata (Optional[dict], optional): Metadata to associate with the symbol. Defaults to None.
        """
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        # Upsert metadata if provided
        if metadata:
            meta_json = json.dumps(metadata)
            self.conn.execute(
                "INSERT OR REPLACE INTO symbol_attributes (symbol, metadata) VALUES (?, ?)",
                [symbol, meta_json],
            )

        if data.empty:
            return

        # Ensure index is not a column yet if we resetting
        df_to_write = data.copy()

        # Normalize: DuckDB expects flat table
        # Handle MultiIndex or Index if they contain required info
        if isinstance(df_to_write.index, pd.MultiIndex):
            # Resetting index usually moves levels to columns
            df_to_write.reset_index(inplace=True)
        elif isinstance(df_to_write.index, pd.DatetimeIndex):
            if "timestamp" not in df_to_write.columns:
                df_to_write.reset_index(inplace=True)
                if (
                    "timestamp" not in df_to_write.columns
                    and "index" in df_to_write.columns
                ):
                    df_to_write.rename(columns={"index": "timestamp"}, inplace=True)

        # If MultiIndex with symbol
        if "symbol" not in df_to_write.columns:
            df_to_write["symbol"] = symbol

        # Rename columns if needed
        if "timestamp" not in df_to_write.columns and "Date" in df_to_write.columns:
            df_to_write.rename(columns={"Date": "timestamp"}, inplace=True)

        # Force Naive UTC
        if "timestamp" in df_to_write.columns:
            # If aware, convert to UTC then remove tz
            if pd.api.types.is_datetime64tz_dtype(df_to_write["timestamp"]):
                df_to_write["timestamp"] = (
                    df_to_write["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
                )

        # Ensure required columns exist
        required_cols = [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
        ]

        # Fill missing optional cols with NaN/0
        for col in required_cols:
            if col not in df_to_write.columns:
                df_to_write[col] = None

        df_to_write = df_to_write[required_cols]

        # Deduplicate to avoid PK constraint errors from source duplicates
        df_to_write.drop_duplicates(
            subset=["symbol", "timestamp"], keep="last", inplace=True
        )

        # Register properly
        self.conn.register("temp_market_data", df_to_write)

        if mode == "replace":
            self.conn.execute(
                f"INSERT OR REPLACE INTO {table_name} SELECT * FROM temp_market_data"
            )
        else:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {table_name} SELECT * FROM temp_market_data"
            )

        self.conn.unregister("temp_market_data")

    def get_market_data(
        self, symbols: List[str], horizon: Horizon, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Retrieve market data for a list of symbols within a time range.

        Args:
            symbols (List[str]): List of asset tickers.
            horizon (Horizon): Time horizon.
            start (datetime): Start datetime (inclusive).
            end (datetime): End datetime (inclusive).

        Returns:
            pd.DataFrame: DataFrame indexed by (symbol, timestamp).
        """
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        # Ensure query params are naive UTC to match storage
        if start.tzinfo is not None:
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
        if end.tzinfo is not None:
            end = end.astimezone(timezone.utc).replace(tzinfo=None)

        # Use parameters for list? DuckDB python API supports list as param usually.
        # But safeguards:
        placeholders = ",".join(["?"] * len(symbols))
        query = f"""
            SELECT * FROM {table_name}
            WHERE symbol IN ({placeholders})
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY symbol, timestamp
        """
        params = symbols + [start, end]

        df = self.conn.execute(query, params).df()

        if not df.empty:
            # Ensure proper types
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)

            df.set_index(["symbol", "timestamp"], inplace=True)

        return df

    def get_last_timestamp(self, symbol: str, horizon: Horizon) -> Optional[datetime]:
        """Get the last stored timestamp for a symbol.

        Args:
            symbol (str): Asset ticker.
            horizon (Horizon): Time horizon.

        Returns:
            Optional[datetime]: The last timestamp (Naive UTC) or None if no data exists.
        """
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        res = self.conn.execute(
            f"""
            SELECT MAX(timestamp) as last_ts FROM {table_name} WHERE symbol = ?
        """,
            [symbol],
        ).fetchone()

        if res and res[0]:
            # DuckDB returns python datetime, usually naive or aware depending on version/Pandas
            ts = pd.to_datetime(res[0])
            if ts.tz is None:
                ts = ts.tz_localize(timezone.utc)
            return ts
        return None

    def list_symbols(self, horizon: Horizon) -> List[str]:
        """List all unique symbols available for a specific horizon.

        Args:
            horizon (Horizon): Time horizon.

        Returns:
            List[str]: Alphabetically sorted list of tickers.
        """
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        res = self.conn.execute(
            f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in res]

    def filter_symbols(self, horizon: Horizon, **criteria) -> List[str]:
        """Filter symbols based on stored metadata values.

        Args:
            horizon: Time horizon (used to ensure symbol existence in that horizon).
            **criteria: Meta keys and values to filter by.
        """
        if not criteria:
            return self.list_symbols(horizon)

        # Build JSON extract query
        # We join symbol_attributes with the market data table to ensure symbol exists in horizon
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        query = f"""
            SELECT DISTINCT t.symbol 
            FROM {table_name} t
            JOIN symbol_attributes m ON t.symbol = m.symbol
            WHERE 1=1
        """
        params = []
        for key, value in criteria.items():
            # Use json_extract_string to get text value for comparison
            query += f" AND json_extract_string(m.metadata, '$.{key}') = ?"
            params.append(str(value))

        query += " ORDER BY t.symbol"

        res = self.conn.execute(query, params).fetchall()
        return [r[0] for r in res]

    def get_symbol_metadata(self, horizon: Horizon) -> pd.DataFrame:
        """Get aggregate metadata for all symbols in a horizon.

        Args:
            horizon (Horizon): Time horizon.

        Returns:
            pd.DataFrame: DataFrame with index 'symbol' and columns [start_date, end_date, count].
                Dates are timezone-aware UTC.
        """
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        # Aggregate start and end dates per symbol
        df = self.conn.execute(
            f"""
            SELECT 
                symbol, 
                MIN(timestamp) as start_date, 
                MAX(timestamp) as end_date,
                COUNT(*) as count
            FROM {table_name}
            GROUP BY symbol
        """
        ).df()

        if not df.empty:
            # Ensure UTC
            for col in ["start_date", "end_date"]:
                if pd.api.types.is_datetime64_ns_dtype(
                    df[col]
                ) or pd.api.types.is_datetime64_dtype(df[col]):
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize(timezone.utc)

            df.set_index("symbol", inplace=True)

        return df

    def delete_range(
        self, symbol: str, horizon: Horizon, start: datetime, end: datetime
    ):
        """Delete data for a symbol within a specific time range.

        Args:
            symbol (str): Asset ticker.
            horizon (Horizon): Time horizon.
            start (datetime): Start datetime (inclusive).
            end (datetime): End datetime (inclusive).
        """
        self._ensure_market_table(horizon)
        table_name = self._get_table_name(horizon)

        if start.tzinfo is not None:
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
        if end.tzinfo is not None:
            end = end.astimezone(timezone.utc).replace(tzinfo=None)

        self.conn.execute(
            f"""
            DELETE FROM {table_name}
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        """,
            [symbol, start, end],
        )

    # --- News Data ---
    def save_news(self, news: List[NewsArticle]):
        """Save a list of news articles.

        Args:
            news (List[NewsArticle]): List of NewsArticle objects.
                - Articles are saved to 'news_articles' (INSERT OR REPLACE).
                - Ticker associations are saved to 'news_tickers' (INSERT OR IGNORE).
                - Timestamps are converted to Naive UTC.
        """
        if not news:
            return

        # Prepare DataFrames
        articles_data = []
        tickers_data = []

        for n in news:
            # Ensure timestamp is naive UTC for storage
            ts = n.timestamp
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

            articles_data.append(
                {
                    "id": n.id,
                    "timestamp": ts,
                    "source": n.source,
                    "headline": n.headline,
                    "content": n.content,
                    "url": n.url,
                    "sentiment": n.sentiment,
                    "metadata": json.dumps(n.metadata),
                }
            )
            for t in n.tickers:
                tickers_data.append({"article_id": n.id, "ticker": t})

        df_articles = pd.DataFrame(articles_data)
        df_tickers = pd.DataFrame(tickers_data)

        # Batch Insert
        if not df_articles.empty:
            self.conn.register("temp_news", df_articles)
            self.conn.execute(
                "INSERT OR REPLACE INTO news_articles SELECT * FROM temp_news"
            )
            self.conn.unregister("temp_news")

        if not df_tickers.empty:
            self.conn.register("temp_news_tickers", df_tickers)
            self.conn.execute(
                "INSERT OR IGNORE INTO news_tickers SELECT * FROM temp_news_tickers"
            )
            self.conn.unregister("temp_news_tickers")

    def get_news(
        self, tickers: Optional[List[str]], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Retrieve news articles.

        Args:
            tickers (Optional[List[str]]): List of tickers to filter by.
                If None/Empty, returns articles for all tickers (subject to time range).
            start (datetime): Start datetime (inclusive).
            end (datetime): End datetime (inclusive).

        Returns:
            pd.DataFrame: DataFrame containing article details (id, timestamp, headline, etc.).
        """
        query = """
            SELECT a.* 
            FROM news_articles a
        """

        if start.tzinfo is not None:
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
        if end.tzinfo is not None:
            end = end.astimezone(timezone.utc).replace(tzinfo=None)

        params = [start, end]

        if tickers:
            placeholders = ",".join(["?"] * len(tickers))
            query += f"""
                JOIN news_tickers t ON a.id = t.article_id
                WHERE t.ticker IN ({placeholders})
                AND a.timestamp >= ? AND a.timestamp <= ?
            """
            params = tickers + params
        else:
            query += " WHERE a.timestamp >= ? AND a.timestamp <= ?"

        df = self.conn.execute(query, params).df()

        # Parse JSON metadata back if needed, or leave as string
        # Typically pandas usage prefers flat, but user can parse later.
        return df

    def execute_query(self, query: str, params: tuple = ()) -> Any:
        """Execute a raw SQL query.

        Args:
            query (str): The SQL query string.
            params (tuple, optional): Parameters to bind to the query. Defaults to ().

        Returns:
            Any: The result of the query execution (e.g., DuckDBPyConnection or result descriptor).
        """
        return self.conn.execute(query, params)
