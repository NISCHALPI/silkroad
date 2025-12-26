"""Market Data Manager for handling high-level synchronization logic."""

import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Optional, Tuple, Dict
import logging

from silkroad.core.enums import Horizon, Exchange
from silkroad.core.data_models import Asset, UniformBarSet, UniformBarCollection
from silkroad.core.news_models import NewsArticle
from silkroad.db.protocol import DatabaseProvider
from silkroad.db.backends import DataBackendProvider
from typing import Any
from silkroad.db.backends import DataBackendProvider

# Constants default from previous implementation
DEFAULT_START_DATE = dt.datetime(2015, 1, 1, tzinfo=dt.timezone.utc)
DEFAULT_LOOKBACK_DAYS = 5
DEFAULT_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data synchronization, integrity, and corporate actions.

    This class orchestrates the flow between the DataBackend (Alpaca/Yahoo) and
    the Storage Provider (DuckDB/SQL). It does NOT handle SQL directly.
    """

    def __init__(self, db: DatabaseProvider, backend: DataBackendProvider):
        self.db = db
        self.backend = backend

    def sync(
        self,
        assets: List[Asset],
        horizon: Horizon,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        lookback_buffer: int = DEFAULT_LOOKBACK_DAYS,
    ):
        """Synchronize market data for a list of assets.

        Orchestrates the sync process:
        1. Classifies assets into NEW (full sync) and UPDATE (incremental).
        2. Detects corporate actions (splits) for existing assets by checking tails.
        3. Performs full sync for new/split assets.
        4. Performs incremental updates for safe existing assets.

        Args:
            assets (List[Asset]): List of assets to sync.
            horizon (Horizon): Time horizon.
            start_date (Optional[dt.datetime], optional): Start date for FULL syncs.
                Defaults to DEFAULT_START_DATE.
            end_date (Optional[dt.datetime], optional): End date for sync. Defaults to last market close.
            lookback_buffer (int, optional): Days to look back for overlap/split detection. Defaults to 5.
        """
        if end_date is None:
            end_date = Exchange.NYSE.previous_market_close()

        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=dt.timezone.utc)

        # 1. Classify Assets
        new_assets, update_assets, update_meta = self._classify_assets(
            assets, horizon, user_start_date=start_date
        )

        logger.info(
            f"Sync Request for {len(assets)} assets. Classification: "
            f"{len(new_assets)} NEW (Full Sync), {len(update_assets)} UPDATE (Incremental)."
        )

        # 2. Tail Check (Corporate Action Detection) for Update Assets
        safe_updates = []
        resync_candidates = []

        if update_assets:
            safe_updates, resync_candidates = self._detect_corporate_actions(
                update_assets, update_meta, horizon, end_date, lookback_buffer
            )

        # 3. Process Full Resyncs (New + corrupted/split assets)
        full_sync_group = new_assets + resync_candidates
        if full_sync_group:
            self._process_full_sync(
                full_sync_group, horizon, start_date or DEFAULT_START_DATE, end_date
            )

        # 4. Process Safe Updates (Incremental Append)
        if safe_updates:
            self._process_incremental_updates(
                safe_updates, update_meta, horizon, end_date, lookback_buffer
            )

    def _classify_assets(
        self,
        assets: List[Asset],
        horizon: Horizon,
        user_start_date: Optional[dt.datetime],
    ) -> Tuple[List[Asset], List[Asset], Dict[str, dt.datetime]]:
        """Separate assets into New (no history) and Update (has history).

        Args:
            assets (List[Asset]): Input assets.
            horizon (Horizon): Time horizon.
            user_start_date (Optional[dt.datetime]): User requested start date.

        Returns:
            Tuple[List[Asset], List[Asset], Dict[str, dt.datetime]]:
                - New assets (to be fully synced)
                - Update assets (to be incrementally updated)
                - Metadata dictionary {ticker: last_stored_timestamp_naive}
        """
        new_assets = []
        update_assets = []
        update_meta = {}

        for asset in assets:
            last_ts = self.db.get_last_timestamp(asset.ticker, horizon)
            if last_ts:
                # Store naive if backend returns naive, but we standardizing on UTC aware for internal logic
                update_assets.append(asset)
                update_meta[asset.ticker] = last_ts
            else:
                new_assets.append(asset)

        return new_assets, update_assets, update_meta

    def _detect_corporate_actions(
        self,
        assets: List[Asset],
        last_timestamps: Dict[str, dt.datetime],
        horizon: Horizon,
        end_date: dt.datetime,
        lookback_days: int,
    ) -> Tuple[List[Asset], List[Asset]]:
        """Fetch tails and compare with stored data to detect splits/restatements."""

        tail_start = end_date - dt.timedelta(days=lookback_days * 2)

        try:
            logger.info("Fetching tails for corporate action detection...")
            incoming_tails = self.backend.fetch_data(
                assets, tail_start, end_date, horizon
            )
        except Exception as e:
            logger.error(f"Tail fetch failed: {e}", exc_info=True)
            return (
                assets,
                [],
            )  # Assume safe if check fails? Or Assume unsafe? Safest is to not update.
            # actually if fetch fails, we can't update anyway.

        safe = []
        resync = []

        if incoming_tails.empty:
            return assets, []

        for asset in assets:
            symbol = asset.ticker
            if symbol not in incoming_tails.index.get_level_values(0):
                # No new data? Safe (nothing to do)
                safe.append(asset)
                continue

            incoming_df = (
                incoming_tails.xs(symbol, level="symbol")
                if "symbol" in incoming_tails.index.names
                else incoming_tails
            )
            last_ts = last_timestamps[symbol]

            # Get overlap from DB
            overlap_start = incoming_df.index.min()
            overlap_end = min(
                incoming_df.index.max(), last_ts
            )  # We only compare up to what we have

            if overlap_start > overlap_end:
                # No overlap, it's a gap forward. Safe to append.
                safe.append(asset)
                continue

            # Load stored data for this overlap
            stored_df = self.db.get_market_data(
                [symbol], horizon, overlap_start, overlap_end
            )
            if stored_df.empty:
                # Should not happen if last_ts exists, but maybe sparse data
                safe.append(asset)
                continue

            # Align and compare
            # Since stored_df acts as having multiindex (symbol, ts), we need to extract
            stored_close = stored_df.xs(symbol, level="symbol")["close"]
            incoming_close = incoming_df["close"]

            # Intersection of indices
            params_idx = stored_close.index.intersection(incoming_close.index)

            if params_idx.empty:
                safe.append(asset)
                continue

            s_c = stored_close.loc[params_idx]
            i_c = incoming_close.loc[params_idx]

            # Comparison
            is_close = np.isclose(
                s_c.values, i_c.values, rtol=DEFAULT_TOLERANCE, equal_nan=True
            )
            if not is_close.all():
                logger.warning(
                    f"SPLIT/MISMATCH detected for {symbol}. "
                    f"Stored tail differs from Backend. Triggering Full Resync."
                )
                resync.append(asset)
            else:
                safe.append(asset)

        return safe, resync

    def _process_full_sync(
        self,
        assets: List[Asset],
        horizon: Horizon,
        start: dt.datetime,
        end: dt.datetime,
    ):
        """Fetch everything and Replace in DB."""
        if not assets:
            return

        logger.info(f"Starting Full Sync for {len(assets)} assets...")
        try:
            # Fetch in bulk
            data = self.backend.fetch_data(assets, start, end, horizon)
            if data.empty:
                logger.warning(
                    "Full Sync: Backend returned NO data for any of the requested assets."
                )
                return

            # Check which symbols we actually got data for
            fetched_symbols = set(data.index.get_level_values("symbol").unique())
            requested_symbols = {a.ticker for a in assets}
            missing = requested_symbols - fetched_symbols
            if missing:
                logger.warning(
                    f"Full Sync: No data found for {len(missing)} symbols: {list(missing)}"
                )

            # Write per symbol
            success_count = 0
            for symbol in fetched_symbols:
                df = data.xs(symbol, level="symbol")

                # Get asset metadata
                asset = next((a for a in assets if a.ticker == symbol), None)
                metadata = asset.model_dump(mode="json") if asset else None

                # For full sync/resync, we use 'replace' mode
                self.db.save_market_data(
                    symbol, df, horizon, mode="replace", metadata=metadata
                )
                success_count += 1

            logger.info(
                f"Full Sync Completed: Successfully synced {success_count}/{len(assets)} assets."
            )

        except Exception as e:
            logger.error(f"Full sync failed with critical error: {e}", exc_info=True)

    def _process_incremental_updates(
        self,
        assets: List[Asset],
        meta: Dict[str, dt.datetime],
        horizon: Horizon,
        end: dt.datetime,
        lookback: int,
    ):
        """Fetch only new data + small overlap (handled by Insert Ignore/Replace logic)."""
        # Determine earliest start date needed
        # To be safe, we pick the oldest 'last_ts' - lookback

        min_ts = min([meta[a.ticker] for a in assets if a.ticker in meta], default=None)
        if not min_ts:
            return

        fetch_start = min_ts - dt.timedelta(days=lookback)

        logger.info(
            f"Starting Incremental Update for {len(assets)} assets from {fetch_start} to {end}..."
        )
        try:
            data = self.backend.fetch_data(assets, fetch_start, end, horizon)
            if data.empty:
                logger.info("Incremental Update: No new data available from backend.")
                return

            # Check coverage
            fetched_symbols = set(data.index.get_level_values("symbol").unique())

            # Write
            count = 0
            for symbol in fetched_symbols:
                df = data.xs(symbol, level="symbol")
                # Get asset metadata
                asset = next((a for a in assets if a.ticker == symbol), None)
                metadata = asset.model_dump(mode="json") if asset else None

                # Use replace to handle the overlap gracefully (Upsert)
                # Since we checked tails and they matched, overwriting the overlap is safe/idempotent
                self.db.save_market_data(
                    symbol, df, horizon, mode="replace", metadata=metadata
                )
                count += 1

            logger.info(
                f"Incremental Update Completed: Updated {count} symbols with new data."
            )

        except Exception as e:
            logger.error(f"Incremental update failed: {e}", exc_info=True)

    # --- Utility Methods ---

    def read_barset(
        self,
        symbol: str,
        horizon: Horizon,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
    ) -> UniformBarSet:
        """Read data for a symbol and return as a UniformBarSet."""
        if start and start.tzinfo is None:
            start = start.replace(tzinfo=dt.timezone.utc)
        if end and end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)

        df = self.db.get_market_data(
            symbols=[symbol],
            horizon=horizon,
            start=start or dt.datetime.min,
            end=end or dt.datetime.max,
        )
        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # Handle MultiIndex handling from DuckDB return
        if isinstance(df.index, pd.MultiIndex):
            # We expect (symbol, timestamp)
            # Droplevel symbol to get DatetimeIndex
            if "symbol" in df.index.names:
                df = df.droplevel("symbol")

        return UniformBarSet.from_df(symbol, horizon, df)

    def read_collection(
        self,
        symbols: List[str],
        horizon: Horizon,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
    ) -> UniformBarCollection:
        """Read data for multiple symbols and return as a UniformBarCollection.

        This guarantees that all bar sets in the collection are aligned (intersection of timestamps).
        """
        bar_sets = {}
        for symbol in symbols:
            try:
                bs = self.read_barset(symbol, horizon, start, end)
                bar_sets[symbol] = bs
            except ValueError:
                logger.warning(f"Skipping {symbol} (no data)")

        if not bar_sets:
            raise ValueError("No data found for any of the requested symbols.")

        # Align
        aligned_sets_list = UniformBarSet.get_intersection(*bar_sets.values())

        if not aligned_sets_list:
            raise ValueError("No overlapping data found for the requested assets.")

        aligned_map = {bs.symbol: bs for bs in aligned_sets_list}
        return UniformBarCollection(bar_map=aligned_map)

    def filter_symbols(self, horizon: Horizon, **criteria) -> List[str]:
        """Filter symbols based on metadata criteria."""
        return self.db.filter_symbols(horizon, **criteria)

    def get_summary(self, horizon: Horizon) -> pd.DataFrame:
        """Get summary statistics for the library."""
        return self.db.get_symbol_metadata(horizon)

    # --- News Utilities ---

    def add_news_from_dataframe(self, df: pd.DataFrame):
        """Add news articles from a DataFrame.

        This function is a wrapper around add_news.
        It is provided for convenience.

        Args:
            df: DataFrame with columns:
                - headline (req)
                - content (req)
                - source (req)
                - timestamp (req)
                - url (opt)
                - sentiment (opt)
                - tickers (opt, list or string)
                - metadata (opt, dict)
        """
        if df.empty:
            return

        news_items = []
        for _, row in df.iterrows():
            # Handle tickers which might be a string "AAPL,MSFT" or list ["AAPL", "MSFT"]
            tickers = row.get("tickers", [])
            if isinstance(tickers, str):
                tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            # Normalize list just in case it's numpy array or something
            elif isinstance(tickers, (np.ndarray, list)):
                tickers = list(tickers)
            else:
                tickers = []

            article = NewsArticle(
                timestamp=pd.to_datetime(row["timestamp"]),
                headline=row["headline"],
                content=row["content"],
                source=row["source"],
                url=row.get("url"),
                sentiment=float(row.get("sentiment", 0.0)),
                tickers=tickers,
                metadata=row.get("metadata", {}),
            )
            news_items.append(article)

        self.db.save_news(news_items)
        logger.info(f"Saved {len(news_items)} news articles.")

    def get_news_history(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Get news history for tickers.

        Args:
            tickers: List of tickers.
            start_date: Start date.
            end_date: End date.
            include_global: If True, includes general market news (ticker='MARKET').
        """
        if start_date is None:
            start_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        if end_date is None:
            end_date = dt.datetime.now(dt.timezone.utc)

        # Ensure UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=dt.timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=dt.timezone.utc)

        query_tickers = list(tickers) if tickers else None

        # If we have specific tickers and want global news, add MARKET ticker
        if include_global and query_tickers is not None:
            from silkroad.core.news_models import MARKET_TICKER

            if MARKET_TICKER not in query_tickers:
                query_tickers.append(MARKET_TICKER)

        return self.db.get_news(query_tickers, start_date, end_date)
