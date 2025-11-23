"""Module for data fetching utilities."""

import pandas as pd
import yfinance as yf
from fredapi import Fred
import typing as tp
from silkroad.logger.logger import logger
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime, timezone
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment

__all__ = ["fetch_macroeconomic_data"]


def _process_fred_data(
    fred_data: tp.Dict[str, pd.Series],
    start_date: tp.Union[str, datetime],
    end_date: tp.Union[str, datetime],
) -> pd.DataFrame:
    """Processes FRED data into a daily DataFrame that covers the full date range.
    For missing data, we fill with forward-fill and back-fill which ensures
    that fed data is available for all days in the range.

    Args:
        fred_data (tp.Dict[str, pd.Series]): Dictionary of FRED series data.
        start_date (tp.Union[str, datetime]): Start date for the data range.
        end_date (tp.Union[str, datetime]): End date for the data range.

    Returns:
        pd.DataFrame: Processed DataFrame with daily frequency.
    """
    # Create DataFrame from FRED data
    fred_df = pd.DataFrame(fred_data)

    # Ensure index is datetime
    fred_df.index = pd.to_datetime(fred_df.index, utc=True).normalize()

    # Extend the date range to cover all days
    full_date_range = pd.date_range(
        start=start_date, end=end_date, freq="D", tz=timezone.utc
    ).normalize()

    # Create an empty DataFrame with the full date range
    full_fred_df = pd.DataFrame(index=full_date_range)

    # concat the original fred_df to the full_fred_df
    # on left to ensure all dates are included
    fred_df = pd.concat(
        [full_fred_df, fred_df.loc[full_fred_df.index.intersection(fred_df.index)]],
        axis=1,
    )

    return fred_df.ffill().bfill()


def fetch_macroeconomic_data(
    fred_client: Fred,
    start_date: tp.Union[str, datetime],
    end_date: tp.Union[str, datetime],
) -> pd.DataFrame:
    """
    Fetches key daily macro features for a given date range.

    This function sources data from FRED (Federal Reserve Economic Data) and
    Yahoo Finance to build a daily time-series DataFrame of essential
    macroeconomic indicators.

    Note on Point-in-Time Correctness:
    Monthly data (CPI, Unemployment) is forward-filled. This means the value for a
    given month is propagated to all subsequent days until a new value is

    available. For rigorous backtesting, you should simulate the actual release
    lag (e.g., CPI for October is released in mid-November). This function
    provides the "as-of" data, not the lagged "point-in-time" data.

    Args:
        fred_client (Fred): An initialized Fred client instance.
        start_date (tp.Union[str, datetime]): The start of the date range in 'YYYY-MM-DD' format.
        end_date (tp.Union[str, datetime]): The end of the date range in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with a daily DatetimeIndex and columns for
                      'vix', 'cpi', 'unemployment_rate', and 'ted_spread'.
                      Returns an empty DataFrame if no data can be fetched.
    """
    logger.info("Starting to fetch macroeconomic data from FRED and Yahoo Finance")

    # --- 1. Fetch VIX Data (Daily) ---
    logger.debug(
        f"Fetching VIX data from Yahoo Finance for period {start_date} to {end_date}"
    )
    vix_df: pd.DataFrame = yf.download("^VIX", start=start_date, end=end_date, progress=False, interval="1d")  # type: ignore
    vix_df.columns = [f"vix_{col[0].lower()}" for col in vix_df.columns]
    vix_df.drop(columns="vix_volume", inplace=True)
    # Localize timezone to UTC
    vix_df.index = vix_df.index.tz_localize(timezone.utc)  # type: ignore

    # Check if VIX data is empty
    if vix_df.empty:
        logger.error("No VIX data fetched from Yahoo Finance. VIX data is required.")
        raise ValueError("VIX data is required but could not be fetched.")

    logger.info(f"Successfully fetched VIX data: {len(vix_df)} rows")

    try:
        # --- 2. Fetch FRED Data ---
        logger.info("Initializing FRED API connection")
        fred = fred_client
        fred_series = {
            "cpi": "CPIAUCSL",  # Monthly
            "unemployment_rate": "UNRATE",  # Monthly
            "ted_spread": "TEDRATE",  # Daily
        }

        fred_data = {}
        for key, series_id in fred_series.items():
            logger.debug(f"Fetching {key} data from FRED with series ID {series_id}")
            series_data = fred.get_series(
                series_id, observation_start=start_date, observation_end=end_date
            )
            if series_data.empty:
                logger.warning(f"No data fetched for {key} (series: {series_id})")
            else:
                logger.debug(f"Successfully fetched {len(series_data)} rows for {key}")
            fred_data[key] = series_data

    except Exception as e:
        logger.error(f"Error fetching data from FRED: {e}", exc_info=True)
        raise e

    # Process FRED data to daily frequency
    fred_df = _process_fred_data(fred_data, start_date, end_date)

    return pd.merge(vix_df, fred_df, left_index=True, right_index=True)


def fetch_universe_daily(
    alpaca_client: StockHistoricalDataClient,
    tickers: tp.List[str],
    start_date: datetime,
    end_date: datetime,
    adj: Adjustment = Adjustment.ALL,
) -> pd.DataFrame:
    """Fetches daily data for a universe of tickers from Alpaca.

    Args:
        tickers (tp.List[str]): List of ticker symbols to fetch.
        start_date (datetime): Start date for data fetching.
        end_date (datetime): End date for data fetching.
        adj (Adjustment): Adjustment type for the data.

    Returns:
        pd.DataFrame: DataFrame containing the daily data for the tickers.
    """
    logger.info(f"Fetching daily data for tickers: {tickers}")

    try:
        request = StockBarsRequest(symbol_or_symbols=tickers, start=start_date, end=end_date, timeframe=TimeFrame.Day, adjustment=adj)  # type: ignore
        # Get the stock bars
        bars = alpaca_client.get_stock_bars(request)

    except Exception as e:
        logger.error(f"Error fetching daily data from Alpaca: {e}", exc_info=True)

    return bars.df  # type: ignore
