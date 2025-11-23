"""
Feature extraction module for quantitative trading strategies.

This module provides comprehensive feature engineering for short-term equity return prediction,
designed for use in systematic portfolio rebalancing and alpha generation pipelines.

The feature extraction approach combines multiple signal categories:
    - Momentum dynamics (first and second-order statistics)
    - Technical indicators (trend, oscillators, volatility)
    - Market-relative metrics (beta, idiosyncratic risk, relative strength)
    - Macroeconomic context (VIX, rates, economic indicators)
    - Calendar effects (day-of-week, seasonality)

Key Functions
-------------
ohlcv_to_features : pd.Series
    Transforms raw OHLCV price data into a rich feature vector (~60 features) suitable
    for machine learning models predicting 1-week forward returns. Emphasizes momentum
    regime detection and mean-reversion signals with second-order dynamics.

generate_dataset : pd.DataFrame
    Creates a complete training/backtesting dataset from historical data using a
    walk-forward process. For each date, computes features using only past data
    (no look-ahead bias) and pairs with forward returns over the next N days.
    Returns a DataFrame with feature columns (X) and multiple target columns (Y)
    representing returns at each forward horizon.

Design Philosophy
-----------------
- **No look-ahead bias**: All features use only information available at computation time
- **Second-order emphasis**: Acceleration and convexity capture regime shifts early
- **Robustness**: Non-parametric rank features and z-scores work across market conditions
- **Alpha isolation**: Market-relative features separate stock-specific returns from beta
- **Production-ready**: Designed for daily batch processing in live trading systems

Typical Usage
-------------
>>> import pandas as pd
>>> from feature_extrator import generate_dataset
>>>
>>> # Load your historical data
>>> stock_ohlcv = pd.read_csv('stock_prices.csv', index_col='date', parse_dates=True)
>>> spy_ohlcv = pd.read_csv('spy_prices.csv', index_col='date', parse_dates=True)
>>> macro = pd.read_csv('macro_data.csv', index_col='date', parse_dates=True)
>>>
>>> # Generate dataset with 5-day forward returns
>>> dataset = generate_dataset(stock_ohlcv, spy_ohlcv, macro, forward_days=5)
>>>
>>> # Split features and targets
>>> feature_cols = [col for col in dataset.columns if not col.startswith('next_return_')]
>>> X = dataset[feature_cols]
>>> y = dataset[['next_return_1d', 'next_return_2d', 'next_return_3d',
>>>              'next_return_4d', 'next_return_5d']]

Requirements
------------
- pandas >= 1.3.0
- numpy >= 1.20.0
- talib >= 0.4.0 (TA-Lib wrapper for Python)
- statsmodels >= 0.12.0

Notes
-----
- Requires TA-Lib C library to be installed on the system
- All dataframes must have DatetimeIndex for proper time-series operations
- Minimum 252 trading days (~1 year) of history required for robust feature computation
- Features may contain NaNs; consider imputation in your ML pipeline
"""

from statsmodels.regression.linear_model import OLS
import numpy as np
import statsmodels.api as sm
from typing import Dict
import pandas as pd
import talib as ta


FEATURE_NAMES = [
    # First-Order Momentum
    "return_1d",
    "return_3d",
    "return_5d",
    "return_2w",
    "return_1m",
    # Second-Order Momentum
    "mom_accel_1w",
    "mom_accel_2w",
    "mom_convexity",
    "return_5d_zscore_60d",
    "return_5d_zscore_120d",
    "return_5d_rank_1y",
    "return_5d_from_20d_high",
    "return_5d_from_20d_low",
    # Technical Indicators
    "sma_20",
    "sma_50",
    "sma_200",
    "sma_20_over_50",
    "sma_50_over_200",
    "price_over_sma_200",
    "rsi_14",
    "rsi_6",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_position",
    "bb_width",
    "bb_width_trend",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "atr_14",
    # Market-Relative Features
    "capm_beta_1y",
    "idiosyncratic_risk_1y",
    "spy_return_5d",
    "relative_return_5d",
    # Macroeconomic Features
    "vix_1w_avg",
    "vix_1w_std",
    "vix_close",
    "vix_high",
    "vix_low",
    "vix_open",
    "unemployment_rate",
    "cpi",
    "term_spread_1m_avg",
    "vix_change_1w",
    # Calendar Features
    "is_friday",
    "month_of_year",
    "day_of_month",
]

TARGET_NAMES = [
    "next_return_1d",
    "next_return_2d",
    "next_return_3d",
    "next_return_4d",
    "next_return_5d",
]


def ohlcv_to_features(
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    macro_data: pd.DataFrame,
) -> pd.Series:
    """
    Generate a rich set of predictive features from OHLCV data for 1-week ahead return forecasting.

    This function creates a comprehensive feature vector combining:
      - First- and second-order momentum signals (acceleration, convexity, z-scores, ranks)
      - Classic technical indicators via TA-Lib (SMA, RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR)
      - Market-relative metrics (CAPM beta, idiosyncratic risk, relative strength vs SPY)
      - Macroeconomic context (VIX dynamics, CPI, unemployment, term spread)
      - Calendar effects

    The design emphasizes momentum regime detection and mean-reversion signals, with a particular
    focus on second-order dynamics (changes in momentum) which have been shown to significantly
    improve short-term equity return prediction.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Daily OHLCV data for the target asset, indexed by date.
        Must contain columns: ['open', 'high', 'low', 'close', 'volume'].
        Requires at least 252 trading days (~1 year) of history.

    market_ohlcv : pd.DataFrame
        Daily OHLCV for the broad market benchmark (typically SPY).
        Must contain at least a 'close' column and 252 days of history.

    macro_data : pd.DataFrame
        Daily or mixed-frequency macroeconomic indicators, indexed by date.
        Expected columns (optional, gracefully handled if missing):
            - vix_close, vix_open, vix_high, vix_low
            - unemployment_rate
            - cpi
            - ted_spread (or term_spread)
        Requires at least 21 days of recent data.

    Returns
    -------
    pd.Series
        A float-typed Series containing ~60 engineered features suitable for machine learning
        models predicting 1-week forward returns. Missing values are represented as NaN.

    Raises
    ------
    ValueError
        If required columns are missing, insufficient history, or invalid price data.

    Notes
    -----
    - All features are computed using only information available at the latest date
      (no look-ahead bias).
    - Heavy emphasis on second-order momentum statistics: acceleration and convexity
      capture regime shifts early.
    - Non-parametric rank features and z-scores provide robustness across market conditions.
    - Market-relative features help isolate alpha from beta exposure.
    - Designed for daily batch processing in a walk-forward backtesting/ML pipeline.
    """
    # ------------------------------------------------------------------ #
    # Input validation (unchanged)
    # ------------------------------------------------------------------ #
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in ohlcv.columns]
    if missing_cols:
        raise ValueError(f"ohlcv missing required columns: {missing_cols}")

    if len(ohlcv) < 252:
        raise ValueError(f"Not enough data: {len(ohlcv)} rows, minimum 252 required.")

    if len(market_ohlcv) < 252 or len(macro_data) < 21:
        raise ValueError("market_ohlcv and macro_data must have sufficient history.")

    if (ohlcv["close"] <= 0).any():
        raise ValueError("Close prices contain non-positive values.")

    features: dict = {}
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]
    opn = ohlcv["open"]

    close_np = close.to_numpy(dtype=float)
    high_np = high.to_numpy(dtype=float)
    low_np = low.to_numpy(dtype=float)
    vol_np = volume.to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 1. First-Order Momentum (already strong)
    # ------------------------------------------------------------------ #
    features["return_1d"] = (
        close.iloc[-1] / close.iloc[-2] - 1 if len(close) >= 2 else np.nan
    )
    features["return_3d"] = (
        close.iloc[-1] / close.iloc[-4] - 1 if len(close) >= 4 else np.nan
    )
    features["return_5d"] = (
        close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else np.nan
    )
    features["return_2w"] = (
        close.iloc[-1] / close.iloc[-11] - 1 if len(close) >= 11 else np.nan
    )
    features["return_1m"] = (
        close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else np.nan
    )

    # ------------------------------------------------------------------ #
    # 2. SECOND-ORDER MOMENTUM FEATURES
    # ------------------------------------------------------------------ #
    # A. Momentum Acceleration (2nd derivative)
    prev_1w_return = (
        close.iloc[-6] / close.iloc[-11] - 1 if len(close) >= 11 else np.nan
    )
    features["mom_accel_1w"] = features["return_5d"] - prev_1w_return  # Speeding up?

    prev_2w_return = (
        close.iloc[-11] / close.iloc[-16] - 1 if len(close) >= 16 else np.nan
    )
    features["mom_accel_2w"] = (features["return_5d"] - features["return_2w"]) / 2

    # B. Momentum Convexity (3rd derivative — changes in acceleration)
    prev_prev_1w = close.iloc[-11] / close.iloc[-16] - 1 if len(close) >= 16 else np.nan
    features["mom_convexity"] = (
        (features["return_5d"] - prev_1w_return) - (prev_1w_return - prev_prev_1w)
        if not np.isnan(prev_prev_1w)
        else np.nan
    )

    # C. Z-score of short-term returns (how extreme is current momentum?)
    features["return_5d_zscore_60d"] = (
        (
            (features["return_5d"] - close.pct_change(5).iloc[-60:-1].mean())
            / close.pct_change(5).iloc[-60:-1].std()
            if close.pct_change(5).iloc[-60:-1].std() != 0
            else np.nan
        )
        if len(close) >= 65
        else np.nan
    )

    features["return_5d_zscore_120d"] = (
        (features["return_5d"] - close.pct_change(5).iloc[-120:-1].mean())
        / close.pct_change(5).iloc[-120:-1].std()
        if len(close) >= 125
        else np.nan
    )

    # D. Momentum Rank vs History (non-parametric, very robust)
    past_1w_returns = [
        close.iloc[-1 - i * 5] / close.iloc[-6 - i * 5] - 1
        for i in range(1, 51)
        if len(close) >= 6 + i * 5
    ]
    if past_1w_returns:
        rank = sum(features["return_5d"] > r for r in past_1w_returns) / len(
            past_1w_returns
        )
        features["return_5d_rank_1y"] = (
            rank  # 0 to 1 (1 = strongest momentum in past year)
        )

    # E. Momentum Reversal Signal (is it overextended?)
    features["return_5d_from_20d_high"] = close.iloc[-1] / close.iloc[-21:-1].max() - 1
    features["return_5d_from_20d_low"] = close.iloc[-1] / close.iloc[-21:-1].min() - 1

    # ------------------------------------------------------------------ #
    # 2. Technical Indicators (using TA-Lib)
    # ------------------------------------------------------------------ #
    close_np = close.to_numpy(dtype=float)
    high_np = high.to_numpy(dtype=float)
    low_np = low.to_numpy(dtype=float)

    features["sma_20"] = ta.SMA(close_np, timeperiod=20)[-1]
    features["sma_50"] = ta.SMA(close_np, timeperiod=50)[-1]
    features["sma_200"] = ta.SMA(close_np, timeperiod=200)[-1]

    features["sma_20_over_50"] = features["sma_20"] - features["sma_50"]
    features["sma_50_over_200"] = features["sma_50"] - features["sma_200"]
    features["price_over_sma_200"] = (
        close.iloc[-1] / features["sma_200"] if features["sma_200"] else np.nan
    )

    features["rsi_14"] = ta.RSI(close_np, timeperiod=14)[-1]
    features["rsi_6"] = (
        ta.RSI(close_np, timeperiod=6)[-1] if len(close_np) >= 6 else np.nan
    )

    macd_line, macd_signal, macd_hist = ta.MACD(
        close_np, fastperiod=12, slowperiod=26, signalperiod=9
    )
    features["macd_line"] = macd_line[-1]
    features["macd_signal"] = macd_signal[-1]
    features["macd_histogram"] = macd_hist[-1]

    # Compute full BBANDS series for trend
    upper = ta.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2)[0]
    middle = ta.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2)[1]
    lower = ta.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2)[2]
    features["bb_position"] = (
        (close.iloc[-1] - lower[-1]) / (upper[-1] - lower[-1])
        if upper[-1] != lower[-1]
        else 0.5
    )
    features["bb_width"] = (
        (upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else np.nan
    )
    bb_width_1w_ago = (
        (upper[-6] - lower[-6]) / middle[-6]
        if len(upper) >= 6 and middle[-6] != 0
        else np.nan
    )
    features["bb_width_trend"] = (
        features["bb_width"] - bb_width_1w_ago
        if not np.isnan(bb_width_1w_ago)
        else np.nan
    )

    features["adx_14"] = ta.ADX(high_np, low_np, close_np, timeperiod=14)[-1]

    # Stochastic Oscillator
    slowk, slowd = ta.STOCH(
        high_np, low_np, close_np, fastk_period=14, slowk_period=3, slowd_period=3
    )
    features["stoch_k"] = slowk[-1] if len(slowk) > 0 else np.nan
    features["stoch_d"] = slowd[-1] if len(slowd) > 0 else np.nan

    # ATR
    features["atr_14"] = (
        ta.ATR(high_np, low_np, close_np, timeperiod=14)[-1]
        if len(close_np) >= 14
        else np.nan
    )

    # ------------------------------------------------------------------ #
    # 3. Market-Relative Features (vs SPY)
    # ------------------------------------------------------------------ #
    asset_ret = close.pct_change().dropna()
    market_ret = market_ohlcv["close"].pct_change().dropna()

    # Align returns on shared dates
    aligned = pd.concat([asset_ret, market_ret], axis=1, join="inner")
    aligned.columns = ["asset", "market"]

    if len(aligned) < 60:
        features["capm_beta_1y"] = np.nan
        features["idiosyncratic_risk_1y"] = np.nan
    else:
        X = sm.add_constant(aligned["market"])
        y = aligned["asset"]
        model = OLS(y, X).fit()
        features["capm_beta_1y"] = model.params.iloc[1]
        features["idiosyncratic_risk_1y"] = model.resid.std() * np.sqrt(252)

    # SPY returns and relative
    spy_close = market_ohlcv["close"]
    features["spy_return_5d"] = (
        spy_close.iloc[-1] / spy_close.iloc[-6] - 1 if len(spy_close) >= 6 else np.nan
    )
    features["relative_return_5d"] = (
        features["return_5d"] - features["spy_return_5d"]
        if not np.isnan(features["spy_return_5d"])
        else np.nan
    )

    # ------------------------------------------------------------------ #
    # 4. Macroeconomic Features
    # ------------------------------------------------------------------ #
    # Defensive access with fallback to NaN
    def safe_mean(s, n):
        return s.iloc[-n:].mean() if len(s) >= n else np.nan

    def safe_std(s, n):
        return s.iloc[-n:].std() if len(s) >= n else np.nan

    def safe_last(s):
        return s.iloc[-1] if len(s) > 0 else np.nan

    features["vix_1w_avg"] = safe_mean(macro_data.get("vix_close", pd.Series()), 5)
    features["vix_1w_std"] = safe_std(macro_data.get("vix_close", pd.Series()), 5)

    # Today vix high, low, close as last columns
    features["vix_close"] = safe_last(macro_data.get("vix_close", pd.Series()))
    features["vix_high"] = safe_last(macro_data.get("vix_high", pd.Series()))
    features["vix_low"] = safe_last(macro_data.get("vix_low", pd.Series()))
    features["vix_open"] = safe_last(macro_data.get("vix_open", pd.Series()))

    features["unemployment_rate"] = safe_mean(
        macro_data.get("unemployment_rate", pd.Series()), 21
    )
    features["cpi"] = safe_mean(macro_data.get("cpi", pd.Series()), 21)
    features["term_spread_1m_avg"] = safe_mean(
        macro_data.get("ted_spread", pd.Series()), 21
    )

    # VIX change
    vix_close_series = macro_data.get("vix_close", pd.Series())
    features["vix_change_1w"] = (
        vix_close_series.iloc[-1] / vix_close_series.iloc[-6] - 1
        if len(vix_close_series) >= 6
        else np.nan
    )

    # ------------------------------------------------------------------ #
    # 5. Calendar Features
    # ------------------------------------------------------------------ #
    latest_date = ohlcv.index[-1]
    features["is_friday"] = 1.0 if latest_date.dayofweek == 4 else 0.0
    features["month_of_year"] = (
        latest_date.month / 12.0
    )  # Normalize to [0,1] # Seasonality
    features["day_of_month"] = (
        latest_date.day / 31.0
    )  # Normalize to [0,1] # Month-end effects

    # ------------------------------------------------------------------ #
    # Return final feature vector
    # ------------------------------------------------------------------ #
    return pd.Series(features, dtype=float)


def generate_dataset(
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    macro_data: pd.DataFrame,
    min_history_days: int = 252,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Generate a dataset of features (X) and forward returns (Y) from historical data.

    This function simulates a walk-forward process: for each eligible date in the history,
    it computes the features using data up to that date (no look-ahead) and pairs it with
    the forward returns over the next `forward_days` trading days.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Full historical daily OHLCV for the target asset, indexed by date.
    market_ohlcv : pd.DataFrame
        Full historical daily OHLCV for the market benchmark (e.g., SPY).
    macro_data : pd.DataFrame
        Full historical macroeconomic data, indexed by date.
    min_history_days : int, optional
        Minimum days of history required to start computing features (default 252).
    forward_days : int, optional
        Number of trading days ahead for the return labels Y (default 5, approx. 1 week).

    Returns
    -------
    pd.DataFrame
        Dataset with features as columns, dates as index, and additional columns 'next_return_1d',
        'next_return_2d', ..., 'next_return_Nd' for each forward day's return.
        Rows with NaN in any Y column are dropped.

    Notes
    -----
    - Assumes dataframes are sorted by date and have datetime indices.
    - Handles cases where future data is insufficient (sets Y to NaN).
    - Features may contain NaNs; consider imputing in your ML pipeline.
    - Y columns represent returns from the current close to the close at each future day.
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv must have a DatetimeIndex.")

    # Raise error if insufficient data
    if len(ohlcv) < min_history_days + forward_days:
        raise ValueError(
            "Not enough data to generate dataset with the specified min_history_days and forward_days."
        )

    # Get eligible dates: those with at least min_history_days prior
    all_dates = ohlcv.index
    start_idx = min_history_days
    eligible_dates = all_dates[start_idx:-forward_days]  # Ensure room for Y

    features_list = []
    y_dict = {f"next_return_{i}d": [] for i in range(1, forward_days + 1)}
    valid_dates = []

    for date in eligible_dates:
        # Slice data up to this date (inclusive)
        idx = (
            all_dates.get_loc(date) + 1
        )  # Up to and including this date # type: ignore
        ohlcv_up_to = ohlcv.iloc[:idx]
        market_up_to = market_ohlcv.loc[:date]  # Use loc for date slicing
        macro_up_to = macro_data.loc[:date]

        # Compute features
        try:
            features = ohlcv_to_features(ohlcv_up_to, market_up_to, macro_up_to)
        except ValueError as e:
            raise ValueError(f"Error computing features for date {date}: {e}")

        # Compute Y: returns from close[date] to close[date + 1], ..., close[date + forward_days]
        current_close = ohlcv["close"].loc[date]
        returns = []
        all_valid = True

        for day_offset in range(1, forward_days + 1):
            future_idx = all_dates.get_loc(date) + day_offset  # type: ignore
            if future_idx < len(all_dates):
                future_close = ohlcv["close"].iloc[future_idx]
                y = future_close / current_close - 1
                returns.append(y)
            else:
                all_valid = False
                break

        # Only add if all forward returns are valid
        if all_valid and len(returns) == forward_days:
            features_list.append(features)
            for i, ret in enumerate(returns, start=1):
                y_dict[f"next_return_{i}d"].append(ret)
            valid_dates.append(date)

    # Compile into DataFrame
    if not features_list:
        raise ValueError("No valid features could be computed from the data.")

    X = pd.DataFrame(features_list, index=valid_dates)
    y_df = pd.DataFrame(y_dict, index=valid_dates)
    dataset = pd.concat([X, y_df], axis=1)

    # Drop rows with any NaN in Y columns
    y_columns = [f"next_return_{i}d" for i in range(1, forward_days + 1)]
    dataset = dataset.dropna(subset=y_columns)

    return dataset
