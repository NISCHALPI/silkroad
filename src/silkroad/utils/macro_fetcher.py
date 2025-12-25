# type: ignore
"""Module for fetching and processing macroeconomic data for asset return prediction.

This module provides tools to fetch data from FRED and Yahoo Finance, transform
non-stationary data (CPI, GDP) into stationary features (YoY change), and
handle publication lags to prevent look-ahead bias in machine learning models.
"""

import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from fredapi import Fred
from silkroad.logging.logger import logger


__all__ = [
    "MacroDataLoader",
    "fetch_macroeconomic_data",
    "fetch_macroeconomic_data_by_datetime_index",
]


@dataclass
class MacroFeature:
    """Configuration for a specific macroeconomic indicator.

    Attributes:
        name (str): The column name for the output DataFrame (e.g., 'infl_cpi_yoy').
        series_id (str): The ticker symbol (FRED series ID or Yahoo Ticker).
        source (str): Data source ('fred' or 'yahoo').
        frequency (str): Native frequency of the data ('D', 'W', 'M', 'Q').
        lag_days (int): Approximate days between period end and data release.
                        Used to prevent look-ahead bias.
        transformation (str): Transformation to apply for stationarity.
                              Options: 'level', 'yoy' (Year-over-Year %), 'diff' (Difference).
    """

    name: str
    series_id: str
    source: str
    frequency: str = "D"
    lag_days: int = 1
    transformation: str = "level"


class MacroDataLoader:
    """Fetching and processing engine for macroeconomic time-series data.

    This class handles the retrieval of disparate economic indicators, applies
    mathematical transformations (stationarity), shifts data to account for
    publication lags (point-in-time correctness), and interpolates lower-frequency
    data into a daily resolution suitable for ML models.

    Attributes:
        fred (Fred): Authenticated FRED API client.
        features (List[MacroFeature]): Registry of features to fetch.
    """

    def __init__(self, fred_client: Fred):
        """Initializes the loader with a FRED client and default feature set.

        Args:
            fred_client (Fred): An initialized fredapi.Fred instance.
        """
        self.fred = fred_client
        self.features = self._get_default_features()

    def _get_default_features(self) -> tp.List[MacroFeature]:
        """Returns the standard list of macroeconomic features for asset allocation.

        Includes Yields, Inflation, Growth, Liquidity, and Risk metrics.

        Returns:
            List[MacroFeature]: Configuration objects for each series.
        """
        return [
            # --- Rates (Daily) ---
            # 10Y Treasury Yield
            MacroFeature("rate_us10y", "DGS10", "fred", "D", 1, "level"),
            # 3M Treasury Yield (Risk Free Rate proxy)
            MacroFeature("rate_risk_free", "DGS3MO", "fred", "D", 1, "level"),
            # 10Y-2Y Spread (Recession Indicator)
            MacroFeature("spread_10y2y", "T10Y2Y", "fred", "D", 1, "level"),
            # --- Inflation (Monthly/Daily) ---
            # CPI All Urban Consumers (Monthly) -> YoY %
            # Lag: released ~15th of following month (~45 days from period start)
            MacroFeature("infl_cpi_yoy", "CPIAUCSL", "fred", "M", 45, "yoy"),
            # 5Y Breakeven Inflation (Market Expectations)
            MacroFeature("infl_breakeven_5y", "T5YIE", "fred", "D", 1, "level"),
            # WTI Crude Oil Price
            MacroFeature("price_oil", "DCOILWTICO", "fred", "D", 1, "level"),
            # --- Growth (Monthly) ---
            # Unemployment Rate (Monthly)
            # Lag: released 1st Friday of following month (~35-40 days from period start)
            MacroFeature("econ_unemployment", "UNRATE", "fred", "M", 40, "level"),
            # Industrial Production -> YoY %
            MacroFeature("econ_ind_prod", "INDPRO", "fred", "M", 45, "yoy"),
            # Retail Sales -> YoY %
            MacroFeature("econ_retail_sales", "RSAFS", "fred", "M", 45, "yoy"),
            # Housing Starts -> YoY %
            MacroFeature("econ_housing_starts", "HOUST", "fred", "M", 45, "yoy"),
            # --- Liquidity & Forex (Monthly/Daily) ---
            # M2 Money Supply -> YoY %
            MacroFeature("liq_m2_money", "M2SL", "fred", "M", 45, "yoy"),
            # USD Index (Yahoo)
            MacroFeature("fx_dxy", "DX-Y.NYB", "yahoo", "D", 0, "level"),
            # --- Risk & Sentiment (Daily/Monthly) ---
            # VIX Volatility Index (Yahoo)
            MacroFeature("risk_vix", "^VIX", "yahoo", "D", 0, "level"),
            # US High Yield Option-Adjusted Spread
            MacroFeature("risk_hy_spread", "BAMLH0A0HYM2", "fred", "D", 1, "level"),
            # UMich Consumer Sentiment
            MacroFeature("sent_michigan", "UMCSENT", "fred", "M", 45, "level"),
        ]

    def fetch_data(
        self,
        start_date: tp.Union[str, datetime],
        end_date: tp.Union[str, datetime],
        transform_stationary: bool = True,
        interpolation_method: str = "ffill",
    ) -> pd.DataFrame:
        """Fetches, aligns, and processes macro data into a daily DataFrame.

        The pipeline follows these steps:
        1. Expand start_date backwards (buffer) to allow for lag and transformations.
        2. Fetch raw data concurrently.
        3. Apply transformations (YoY, Diff) on raw time series.
        4. Shift data forward by `lag_days` to simulate publication availability.
        5. Reindex to target daily frequency and interpolate gaps.

        Args:
            start_date (Union[str, datetime]): Start of the output date range.
            end_date (Union[str, datetime]): End of the output date range.
            transform_stationary (bool): If True, applies 'yoy' or 'diff' transformations defined in features.
            interpolation_method (str): Method for upsampling ('ffill', 'linear', 'cubic', 'time').
                                        'ffill' creates a step function (safest).
                                        'cubic'/'time' creates smooth curves (better for ML gradients).

        Returns:
            pd.DataFrame: A daily resolution DataFrame (UTC index) with all features aligned.
        """
        # Ensure start/end dates are UTC and normalized
        target_start = pd.to_datetime(start_date)
        if target_start.tz is None:
            target_start = target_start.tz_localize("UTC")
        else:
            target_start = target_start.tz_convert("UTC")
        target_start = target_start.normalize()

        target_end = pd.to_datetime(end_date)
        if target_end.tz is None:
            target_end = target_end.tz_localize("UTC")
        else:
            target_end = target_end.tz_convert("UTC")
        target_end = target_end.normalize()

        # Buffer calculation:
        # We need ~400 days prior to start_date because:
        # 1. YoY transformation requires T-365 days.
        # 2. Publication lags shifts data forward by ~45 days.
        fetch_start = target_start - timedelta(days=450)

        logger.info(
            f"Initializing macro fetch. Target: {target_start.date()} to {target_end.date()}. "
            f"Fetching raw data from: {fetch_start.date()}"
        )

        # 1. Concurrent Fetching
        raw_data: tp.Dict[str, pd.Series] = {}
        with ThreadPoolExecutor() as executor:
            # Map future to feature config
            future_to_feature = {
                executor.submit(
                    self._fetch_single_series, feat, fetch_start, target_end
                ): feat
                for feat in self.features
            }

            for future in as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    series = future.result()
                    if not series.empty:
                        raw_data[feature.name] = series
                    else:
                        logger.warning(f"Feature '{feature.name}' returned empty.")
                except Exception as exc:
                    logger.error(f"Error fetching '{feature.name}': {exc}")

        if not raw_data:
            logger.error("No macroeconomic data could be fetched.")
            return pd.DataFrame()

        # 2. Processing Pipeline
        processed_series = []
        # Target daily index covering the requested range
        daily_idx = pd.date_range(target_start, target_end, freq="D", tz="UTC")

        for feature in self.features:
            if feature.name not in raw_data:
                continue

            s = raw_data[feature.name]

            # A. Transformation (Calculate YoY/Diff on the raw timeline)
            if transform_stationary:
                s = self._apply_transformation(s, feature)

            # B. Lagging (Point-in-Time Adjustment)
            # Shift the data forward. E.g., Jan 1 data becomes available Feb 15.
            # We use tshift (or shift of index) to move the timestamp forward.
            s.index = s.index + timedelta(days=feature.lag_days)

            # C. Alignment & Interpolation
            # Reindex to the specific daily range requested.
            # This creates NaNs for days between releases.
            # We must handle duplicates if the lag caused overlaps (rare).
            s = s[~s.index.duplicated(keep="last")]

            # Combine the available sparse data with the target daily index
            # We use 'union' first to ensure the known data points exist in the index
            # before interpolation, then trim later.
            combined_idx = daily_idx.union(s.index).sort_values()
            s_aligned = s.reindex(combined_idx)

            # Interpolate
            if interpolation_method == "ffill":
                # Step function (holds value until new release)
                s_filled = s_aligned.ffill()
            else:
                # Smooth interpolation (Linear, Time, Cubic)
                # 'time' is robust for irregular intervals.
                try:
                    s_filled = s_aligned.interpolate(
                        method=interpolation_method, limit_direction="forward"
                    )
                except ValueError:
                    # Fallback if specific method fails (e.g. requires strict numeric index)
                    logger.warning(
                        f"Interpolation '{interpolation_method}' failed for {feature.name}. "
                        "Falling back to 'time'."
                    )
                    s_filled = s_aligned.interpolate(
                        method="time", limit_direction="forward"
                    )

                # Interpolate typically leaves the leading edge NaN if no data exists before start
                # We ffill any remaining gaps if the spline didn't catch them
                s_filled = s_filled.ffill()

            # Crop to the requested target range and name the series
            s_final = s_filled.loc[target_start:target_end]
            s_final.name = feature.name
            processed_series.append(s_final)

        if not processed_series:
            return pd.DataFrame(index=daily_idx)

        # Merge all features
        macro_df = pd.concat(processed_series, axis=1)

        # Final check: forward fill any tiny gaps remaining from alignment issues
        macro_df = macro_df.ffill().bfill()

        logger.info(f"Macro data processed. Shape: {macro_df.shape}")
        return macro_df

    def _fetch_single_series(
        self, feature: MacroFeature, start: datetime, end: datetime
    ) -> pd.Series:
        """Worker method to fetch a single series from its source."""
        try:
            if feature.source == "fred":
                # FRED API fetch
                s = self.fred.get_series(
                    feature.series_id, observation_start=start, observation_end=end
                )
                # Cast and Timezone normalize
                if isinstance(s, pd.Series):
                    if s.index.tz is None:
                        s.index = s.index.tz_localize("UTC")
                    else:
                        s.index = s.index.tz_convert("UTC")
                    return s

            elif feature.source == "yahoo":
                # YFinance fetch
                # Note: YF end_date is exclusive, add buffer
                yf_end = end + timedelta(days=2)
                df = yf.download(
                    feature.series_id,
                    start=start,
                    end=yf_end,
                    progress=False,
                    interval="1d",
                    auto_adjust=True,
                )

                if df.empty:
                    return pd.Series(dtype=float)

                # Extract 'Close' column handling MultiIndex columns (new yfinance behavior)
                # Logic: If MultiIndex, the ticker is usually level 1, price type level 0
                target_col = None

                # Check for standard columns
                if "Close" in df.columns:
                    target_col = df["Close"]
                elif "Adj Close" in df.columns:
                    target_col = df["Adj Close"]

                # Handle MultiIndex scenario (e.g. columns are ('Close', 'DX-Y.NYB'))
                if isinstance(df.columns, pd.MultiIndex):
                    # If we failed to find it directly, try accessing via XS or iloc
                    # Usually yfinance returns a DataFrame if multiple tickers,
                    # but for single ticker it might still be MultiIndex in recent versions
                    target_col = df.iloc[:, 0]  # Take first column (usually Close)

                if target_col is None:
                    return pd.Series(dtype=float)

                # Ensure it's a Series (sometimes yfinance returns 1-col DF)
                if isinstance(target_col, pd.DataFrame):
                    s = target_col.iloc[:, 0]
                else:
                    s = target_col

                # Timezone normalize
                if s.index.tz is None:
                    s.index = s.index.tz_localize("UTC")
                else:
                    s.index = s.index.tz_convert("UTC")

                return s

        except Exception as e:
            logger.warning(f"Exception fetching {feature.series_id}: {e}")
            return pd.Series(dtype=float)

        return pd.Series(dtype=float)

    def _apply_transformation(
        self, series: pd.Series, feature: MacroFeature
    ) -> pd.Series:
        """Applies mathematical transformations to ensure stationarity."""
        if feature.transformation == "level":
            return series

        # Determine step size based on frequency
        # Note: FRED data is indexed by period start.
        # Monthly data freq is inferred.
        periods = 1
        if feature.frequency == "M":
            periods = 12  # YoY for monthly
        elif feature.frequency == "W":
            periods = 52  # YoY for weekly
        elif feature.frequency == "D":
            periods = 252  # YoY for daily (trading days)

        if feature.transformation == "diff":
            return series.diff()

        if feature.transformation == "yoy":
            return series.pct_change(periods)

        return series

    def __repr__(self) -> str:
        return f"<MacroDataLoader features={len(self.features)}>"

    def fetch_feature_by_datetime_index(
        self,
        feature: MacroFeature,
        datetime_index: pd.DatetimeIndex,
        transform_stationary: bool = True,
        interpolation_method: str = "ffill",
    ) -> pd.Series:
        """Fetches a single macro feature aligned to a specific index.

        This method fetches a single macroeconomic feature over the range defined by
        the provided datetime index and reindexes the result to match that index.
        This is useful for ensuring that macro data aligns perfectly with
        other datasets that share the same timestamps.

        Args:
            feature_name (str): The name of the feature to fetch (as defined in MacroFeature).
            datetime_index (pd.DatetimeIndex): The target index to align data to.
            transform_stationary (bool): If True, applies 'yoy' or 'diff' transformations defined in features.
            interpolation_method (str): Method for upsampling ('ffill', 'linear', 'cubic', 'time').

        Returns:
            pd.Series: Macro feature reindexed to the provided index.
        """
        if datetime_index.empty:
            return pd.Series(dtype=float)

        # Ensure we cover the full range
        start_date = datetime_index.min()
        end_date = datetime_index.max()

        # Fetch broadly on daily resolution (UTC midnight)
        df = self.fetch_data(
            start_date, end_date, transform_stationary, interpolation_method
        )

        if feature.name not in df.columns:
            return pd.Series(dtype=float)

        # Reindex to the target index.
        # We use 'ffill' to propagate the daily (midnight) value to any intraday timestamps.
        # Note: df is sorted by definition from fetch_data, which is required for ffill.

        # Handle Timezone matching for reindex if necessary
        # df.index is UTC. If datetime_index is naive, we might need to localize.
        # Assuming datetime_index is consistent with the project (UTC-aware).

        return df[feature.name].reindex(datetime_index, method="ffill")

    def fetch_data_by_datetime_index(
        self,
        datetime_index: pd.DatetimeIndex,
        transform_stationary: bool = True,
        interpolation_method: str = "ffill",
    ) -> pd.DataFrame:
        """Fetches macro data aligned to a specific index.

        This method fetches macroeconomic data over the range defined by
        the provided datetime index and reindexes the result to match that index.
        This is useful for ensuring that macro data aligns perfectly with
        other datasets that share the same timestamps.

        Args:
            datetime_index (pd.DatetimeIndex): The target index to align data to.
            transform_stationary (bool): If True, applies 'yoy' or 'diff' transformations defined in features.
            interpolation_method (str): Method for upsampling ('ffill', 'linear', 'cubic', 'time').

        Returns:
            pd.DataFrame: Macro data reindexed to the provided index.
        """
        if datetime_index.empty:
            return pd.DataFrame()

        # Ensure we cover the full range
        start_date = datetime_index.min()
        end_date = datetime_index.max()

        # Fetch broadly on daily resolution (UTC midnight)
        df = self.fetch_data(
            start_date, end_date, transform_stationary, interpolation_method
        )

        # Reindex to the target index.
        # We use 'ffill' to propagate the daily (midnight) value to any intraday timestamps.
        # Note: df is sorted by definition from fetch_data, which is required for ffill.

        # Handle Timezone matching for reindex if necessary
        # df.index is UTC. If datetime_index is naive, we might need to localize.
        # Assuming datetime_index is consistent with the project (UTC-aware).

        return df.reindex(datetime_index, method="ffill")

    def fetch_risk_free_rate_by_datetime_index(
        self,
        datetime_index: pd.DatetimeIndex,
        transform_stationary: bool = False,
        interpolation_method: str = "ffill",
    ) -> pd.Series:
        """Fetches the risk-free rate aligned to a specific index.

        This is a convenience method to fetch the standard risk-free rate feature
        defined in the default features list.

        Args:
            datetime_index (pd.DatetimeIndex): The target index to align data to.
            transform_stationary (bool): If True, applies 'yoy' or 'diff' transformations defined in features.
            interpolation_method (str): Method for upsampling ('ffill', 'linear', 'cubic', 'time').

        Returns:
            pd.Series: Risk-free rate feature reindexed to the provided index.

        """
        # Find the risk-free feature
        risk_free_feature = next(
            (feat for feat in self.features if feat.name == "rate_risk_free"), None
        )

        if risk_free_feature is None:
            logger.error("Risk-free feature 'rate_risk_free' not found in features.")
            return pd.Series(dtype=float)

        return self.fetch_feature_by_datetime_index(
            risk_free_feature,
            datetime_index,
            transform_stationary,
            interpolation_method,
        )
