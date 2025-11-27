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
"""

import warnings
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Robust import for TA-Lib with clear error messaging
try:
    import talib as ta
except ImportError:
    raise ImportError(
        "TA-Lib is not installed. Please install the C library and the Python wrapper "
        "to use this module (e.g., `pip install ta-lib`). "
        "See: https://github.com/mrjbq7/ta-lib"
    )

# Suppress specific pandas settings warnings for cleaner output
pd.options.mode.chained_assignment = None

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

FEATURE_NAMES: List[str] = [
    # First-Order Momentum
    "log_return_1d",
    "log_return_3d",
    "log_return_5d",
    "log_return_2w",
    "log_return_1m",
    # Second-Order Momentum
    "mom_accel_1w",
    "mom_accel_2w",
    "mom_convexity",
    "log_return_5d_zscore_60d",
    "log_return_5d_zscore_120d",
    "log_return_5d_rank_1y",
    "log_return_5d_from_20d_high",
    "log_return_5d_from_20d_low",
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
    "spy_log_return_5d",
    "relative_log_return_5d",
    # Macroeconomic Features
    "macro_risk_vix",
    "macro_vix_1w_change",
    "macro_rate_us10y",
    "macro_rate_risk_free",
    "macro_spread_10y2y",
    "macro_infl_cpi_yoy",
    "macro_infl_breakeven_5y",
    "macro_price_oil",
    "macro_econ_unemployment",
    "macro_econ_ind_prod",
    "macro_econ_retail_sales",
    "macro_econ_housing_starts",
    "macro_liq_m2_money",
    "macro_fx_dxy",
    "macro_risk_hy_spread",
    "macro_sent_michigan",
    # Calendar Features
    "is_friday",
    "month_of_year",
    "day_of_month",
    # Volume and Liquidity Features
    "volume_5d_avg",
    "volume_20d_avg",
    "volume_60d_avg",
    "volume_ratio_20d",
    "volume_acceleration",
    "obv_5d_change",
    "obv_20d_change",
    "amihud_illiquidity_20d",
    "spread_proxy",
    "volume_volatility_20d",
    # Microstructure and OHLC Pattern Features
    "overnight_gap",
    "gap_persistence",
    "intraday_range",
    "range_expansion_5d",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_size_ratio",
    "close_position_in_range",
    # Higher-Order Moments and Tail Risk
    "log_return_skewness_20d",
    "log_return_skewness_60d",
    "log_return_kurtosis_20d",
    "log_return_kurtosis_60d",
    "downside_deviation_20d",
    "max_drawdown_60d",
    # Volatility Regime Features
    "realized_vol_5d",
    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_short_long",
    "parkinson_vol_20d",
    "vol_trend_1w",
    "relative_vol_vs_spy",
    # Mean Reversion Signals
    "price_vs_sma_10d",
    "price_vs_sma_30d",
    "price_vs_sma_60d",
    "log_return_autocorr_1d",
    "log_return_autocorr_5d",
    "hurst_exponent_60d",
    # Cross-Sectional Features
    "momentum_rank_5d",
    "momentum_rank_20d",
    "volatility_rank",
    "volume_rank",
    "rsi_rank",
    "relative_strength_rank",
    # Interaction Features
    "momentum_x_volatility",
    "volume_x_return",
    "rsi_x_trend",
    "beta_x_market",
    "vix_x_momentum",
]

TARGET_NAMES: List[str] = [
    "next_log_return_1d",
    "next_log_return_2d",
    "next_log_return_3d",
    "next_log_return_4d",
    "next_log_return_5d",
]


# ------------------------------------------------------------------ #
# Helper Functions
# ------------------------------------------------------------------ #


def _compute_realized_volatility(close: pd.Series, window: int) -> float:
    """Computes annualized realized volatility based on close-to-close returns.

    Args:
        close (pd.Series): Series of close prices.
        window (int): Lookback window size.

    Returns:
        float: Annualized volatility (std dev * sqrt(252)). NaN if insufficient data.
    """
    if len(close) < window + 1:
        return np.nan
    # ddof=1 is standard for sample standard deviation
    return close.pct_change().iloc[-window:].std(ddof=1) * np.sqrt(252)


def _compute_parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int
) -> float:
    """Computes Parkinson volatility using High-Low range.

    Parkinson volatility is more efficient than close-to-close volatility as it
    incorporates intraday range information.

    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        window (int): Lookback window size.

    Returns:
        float: Annualized Parkinson volatility. NaN if insufficient data.
    """
    if len(high) < window or len(low) < window:
        return np.nan

    # Handle potential division by zero or log of zero
    valid_idx = (low > 0) & (high > 0)
    if not valid_idx.all():
        return np.nan

    hl_ratio = np.log(high.iloc[-window:] / low.iloc[-window:])
    const = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(const * hl_ratio.pow(2).mean()) * np.sqrt(252)  # type: ignore


def _compute_percentile_rank(value: float, series: pd.Series) -> float:
    """Computes the percentile rank of a value within a historical series.

    Args:
        value (float): The current value to rank.
        series (pd.Series): The historical distribution.

    Returns:
        float: Rank between 0.0 and 1.0. NaN if inputs are invalid.
    """
    if pd.isna(value) or len(series) == 0:
        return np.nan
    # strict inequality count + fraction of ties / total
    return (series < value).mean()


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Computes On-Balance Volume (OBV).

    Args:
        close (pd.Series): Close prices.
        volume (pd.Series): Volume data.

    Returns:
        pd.Series: Cumulative OBV series.
    """
    # Vectorized calculation
    direction = np.sign(close.diff())
    direction.iloc[0] = 0  # type: ignore
    return (direction * volume).cumsum()  # type: ignore


def _compute_autocorrelation(returns: pd.Series, lag: int) -> float:
    """Computes autocorrelation of returns at a specific lag.

    Args:
        returns (pd.Series): Return series.
        lag (int): Lag period.

    Returns:
        float: Correlation coefficient. NaN if insufficient data or zero variance.
    """
    if len(returns) < lag + 10:
        return np.nan
    try:
        return returns.autocorr(lag=lag)
    except Exception:
        return np.nan


def _compute_hurst_exponent(price: pd.Series, window: int) -> float:
    """Computes a simplified Hurst exponent to estimate trend persistence.

    H < 0.5: Mean reverting
    H ~ 0.5: Random Walk
    H > 0.5: Trending

    Args:
        price (pd.Series): Price series.
        window (int): Lookback window.

    Returns:
        float: Estimated Hurst exponent. NaN on calculation failure.
    """
    if len(price) < window:
        return np.nan

    try:
        # Use the last 'window' prices
        prices = price.iloc[-window:].values
        lags = range(2, min(20, window // 2))

        # Calculate standard deviation of differences for various lags
        tau = []
        valid_lags = []

        for lag in lags:
            # diff array length: N - lag
            diffs = np.subtract(prices[lag:], prices[:-lag])  # type: ignore
            std_val = np.std(diffs)
            if std_val > 0:
                tau.append(std_val)
                valid_lags.append(lag)

        if len(valid_lags) < 3:
            return np.nan

        # Linear regression on log-log scale
        # equation: log(std) = Hurst * log(lag) + C
        x = np.log(valid_lags)
        y = np.log(tau)

        poly = np.polyfit(x, y, 1)
        return poly[0]  # The slope is the Hurst exponent
    except Exception:
        return np.nan


def _safe_division(n: float, d: float, default: float = np.nan) -> float:
    """Helper for safe division."""
    return n / d if d != 0 and not np.isnan(d) else default


# ------------------------------------------------------------------ #
# Main Logic
# ------------------------------------------------------------------ #


def ohlcv_to_features(
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    macro_data: pd.DataFrame,
    universe_ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.Series:
    """Transforms raw OHLCV price data into a rich feature vector.

    Calculates ~95 features emphasizing second-order momentum, technicals,
    and market-relative metrics. All features are calculated using only
    data up to the last index of the input dataframe (no look-ahead bias).

    Args:
        ohlcv (pd.DataFrame): DataFrame with columns [open, high, low, close, volume].
            Must contain at least 252 rows of history.
        market_ohlcv (pd.DataFrame): Benchmark DataFrame (e.g., SPY) with 'close'.
        macro_data (pd.DataFrame): DataFrame with macro indicators (vix_close, etc.).
        universe_ohlcv (Dict[str, pd.DataFrame], optional): Dictionary of DataFrames
            for the entire stock universe. Used for cross-sectional ranking.

    Returns:
        pd.Series: A Series indexed by FEATURE_NAMES containing float values.

    Raises:
        ValueError: If required columns are missing or history is too short.
    """
    # 1. Input Validation
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in ohlcv.columns for col in required_cols):
        missing = [c for c in required_cols if c not in ohlcv.columns]
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    if len(ohlcv) < 252:
        raise ValueError(
            f"Insufficient data history: {len(ohlcv)} rows, minimum 252 required."
        )

    # 2. Data Preparation (Extract NumPy arrays for TA-Lib efficiency)
    # Using .astype(float) ensures compatibility with TA-Lib's double requirements
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]
    opn = ohlcv["open"]

    c_np = close.values.astype(float)
    h_np = high.values.astype(float)
    l_np = low.values.astype(float)

    # Pre-calculate returns (Log Returns)
    # Use log returns for additivity and better statistical properties
    log_close = close.apply(np.log).astype(float)
    returns = log_close.diff()

    features = {}

    # ------------------------------------------------------------------ #
    # A. Momentum (First & Second Order)
    # ------------------------------------------------------------------ #
    curr_price = close.iloc[-1]
    curr_log_price = log_close.iloc[-1]

    features["log_return_1d"] = returns.iloc[-1]
    features["log_return_3d"] = curr_log_price - log_close.iloc[-4]
    features["log_return_5d"] = curr_log_price - log_close.iloc[-6]
    features["log_return_2w"] = curr_log_price - log_close.iloc[-11]
    features["log_return_1m"] = curr_log_price - log_close.iloc[-21]

    # Momentum Acceleration
    ret_5d = features["log_return_5d"]
    prev_1w_return = log_close.iloc[-6] - log_close.iloc[-11]
    prev_2w_return = log_close.iloc[-11] - log_close.iloc[-16]

    features["mom_accel_1w"] = ret_5d - prev_1w_return
    features["mom_accel_2w"] = (ret_5d - features["log_return_2w"]) / 2.0

    # Momentum Convexity (Jerk)
    prev_prev_1w = log_close.iloc[-11] - log_close.iloc[-16]
    features["mom_convexity"] = (ret_5d - prev_1w_return) - (
        prev_1w_return - prev_prev_1w
    )

    # Z-Scores (Regime Detection)
    # Calculate rolling 5d returns (log returns)
    roll_ret_5d = log_close.diff(5)

    # 60-day window z-score
    mean_60 = roll_ret_5d.iloc[-61:-1].mean()
    std_60 = roll_ret_5d.iloc[-61:-1].std()
    features["log_return_5d_zscore_60d"] = _safe_division(ret_5d - mean_60, std_60)

    # 120-day window z-score
    mean_120 = roll_ret_5d.iloc[-121:-1].mean()
    std_120 = roll_ret_5d.iloc[-121:-1].std()
    features["log_return_5d_zscore_120d"] = _safe_division(ret_5d - mean_120, std_120)

    # Rank vs History (Non-parametric)
    # Extract last 50 weekly returns
    past_weekly_returns = [
        log_close.iloc[-1 - i * 5] - log_close.iloc[-6 - i * 5] for i in range(1, 51)
    ]
    features["log_return_5d_rank_1y"] = _compute_percentile_rank(
        ret_5d, pd.Series(past_weekly_returns)
    )

    # Reversal Signals (Distance from recent extremes)
    high_20d = high.iloc[-21:-1].max()
    low_20d = low.iloc[-21:-1].min()
    features["log_return_5d_from_20d_high"] = (
        curr_price / high_20d - 1 if high_20d > 0 else np.nan
    )
    features["log_return_5d_from_20d_low"] = (
        curr_price / low_20d - 1 if low_20d > 0 else np.nan
    )

    # ------------------------------------------------------------------ #
    # B. Technical Indicators (TA-Lib)
    # ------------------------------------------------------------------ #
    # Moving Averages
    features["sma_20"] = ta.SMA(c_np, timeperiod=20)[-1]
    features["sma_50"] = ta.SMA(c_np, timeperiod=50)[-1]
    features["sma_200"] = ta.SMA(c_np, timeperiod=200)[-1]

    features["sma_20_over_50"] = (
        _safe_division(features["sma_20"], features["sma_50"]) - 1
    )
    features["sma_50_over_200"] = (
        _safe_division(features["sma_50"], features["sma_200"]) - 1
    )
    features["price_over_sma_200"] = _safe_division(curr_price, features["sma_200"]) - 1

    # Oscillators
    features["rsi_14"] = ta.RSI(c_np, timeperiod=14)[-1]
    features["rsi_6"] = ta.RSI(c_np, timeperiod=6)[-1]

    macd, macd_signal, macd_hist = ta.MACD(
        c_np, fastperiod=12, slowperiod=26, signalperiod=9
    )
    features["macd_line"] = macd[-1]
    features["macd_signal"] = macd_signal[-1]
    features["macd_histogram"] = macd_hist[-1]

    # Bollinger Bands
    bb_up, bb_mid, bb_low = ta.BBANDS(c_np, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_range = bb_up[-1] - bb_low[-1]
    features["bb_position"] = _safe_division(
        curr_price - bb_low[-1], bb_range, default=0.5
    )
    features["bb_width"] = _safe_division(bb_range, bb_mid[-1])

    # BB Width Trend
    prev_bb_width = _safe_division(bb_up[-6] - bb_low[-6], bb_mid[-6])
    features["bb_width_trend"] = features["bb_width"] - prev_bb_width

    # Volatility & Trend Strength
    features["adx_14"] = ta.ADX(h_np, l_np, c_np, timeperiod=14)[-1]
    features["atr_14"] = ta.ATR(h_np, l_np, c_np, timeperiod=14)[-1]

    # Stochastic
    stoch_k, stoch_d = ta.STOCH(
        h_np, l_np, c_np, fastk_period=14, slowk_period=3, slowd_period=3
    )
    features["stoch_k"] = stoch_k[-1]
    features["stoch_d"] = stoch_d[-1]

    # ------------------------------------------------------------------ #
    # C. Market-Relative Features (Alpha/Beta)
    # ------------------------------------------------------------------ #
    # Align asset returns with market returns on index
    mkt_close = market_ohlcv["close"]
    mkt_ret = mkt_close.pct_change()

    # Combine and drop NaNs to ensure alignment
    aligned = pd.concat([returns, mkt_ret], axis=1, join="inner").dropna().iloc[-252:]
    aligned.columns = ["asset", "market"]

    if len(aligned) >= 60:
        # We use covariance method for beta as it's faster than OLS for single variables
        cov_mat = np.cov(aligned["asset"], aligned["market"])
        beta = cov_mat[0, 1] / cov_mat[1, 1]
        alpha = aligned["asset"].mean() - beta * aligned["market"].mean()

        # Calculate residuals manually for speed
        residuals = aligned["asset"] - (alpha + beta * aligned["market"])

        features["capm_beta_1y"] = beta
        features["idiosyncratic_risk_1y"] = residuals.std(ddof=1) * np.sqrt(252)
    else:
        features["capm_beta_1y"] = np.nan
        features["idiosyncratic_risk_1y"] = np.nan

    # Simple relative return
    spy_curr = mkt_close.iloc[-1]
    spy_ret_5d = np.log(spy_curr / mkt_close.iloc[-6]) if len(mkt_close) > 6 else np.nan

    features["spy_log_return_5d"] = spy_ret_5d
    features["relative_log_return_5d"] = ret_5d - spy_ret_5d

    # ------------------------------------------------------------------ #
    # D. Macroeconomic Features
    # ------------------------------------------------------------------ #
    def _get_safe_macro(col_name: str) -> float:
        if col_name in macro_data.columns and not macro_data[col_name].empty:
            return macro_data[col_name].iloc[-1]
        return np.nan

    # Direct mappings
    features["macro_risk_vix"] = _get_safe_macro("risk_vix")
    features["macro_rate_us10y"] = _get_safe_macro("rate_us10y")
    features["macro_rate_risk_free"] = _get_safe_macro("rate_risk_free")
    features["macro_spread_10y2y"] = _get_safe_macro("spread_10y2y")
    features["macro_infl_cpi_yoy"] = _get_safe_macro("infl_cpi_yoy")
    features["macro_infl_breakeven_5y"] = _get_safe_macro("infl_breakeven_5y")
    features["macro_price_oil"] = _get_safe_macro("price_oil")
    features["macro_econ_unemployment"] = _get_safe_macro("econ_unemployment")
    features["macro_econ_ind_prod"] = _get_safe_macro("econ_ind_prod")
    features["macro_econ_retail_sales"] = _get_safe_macro("econ_retail_sales")
    features["macro_econ_housing_starts"] = _get_safe_macro("econ_housing_starts")
    features["macro_liq_m2_money"] = _get_safe_macro("liq_m2_money")
    features["macro_fx_dxy"] = _get_safe_macro("fx_dxy")
    features["macro_risk_hy_spread"] = _get_safe_macro("risk_hy_spread")
    features["macro_sent_michigan"] = _get_safe_macro("sent_michigan")

    # Derived Macro Features
    if "risk_vix" in macro_data.columns:
        vix_s = macro_data["risk_vix"]
        if len(vix_s) > 6:
            features["macro_vix_1w_change"] = vix_s.iloc[-1] / vix_s.iloc[-6] - 1
        else:
            features["macro_vix_1w_change"] = np.nan
    else:
        features["macro_vix_1w_change"] = np.nan

    # ------------------------------------------------------------------ #
    # E. Calendar & Volume Features
    # ------------------------------------------------------------------ #
    curr_date = ohlcv.index[-1]
    features["is_friday"] = 1.0 if curr_date.dayofweek == 4 else 0.0
    features["month_of_year"] = curr_date.month / 12.0
    features["day_of_month"] = curr_date.day / 31.0

    features["volume_5d_avg"] = volume.iloc[-5:].mean()
    features["volume_20d_avg"] = volume.iloc[-20:].mean()
    features["volume_60d_avg"] = volume.iloc[-60:].mean()
    features["volume_ratio_20d"] = _safe_division(
        volume.iloc[-1], features["volume_20d_avg"]
    )

    # Volume Acceleration
    vol_cur = volume.iloc[-5:].mean()
    vol_prev = volume.iloc[-10:-5].mean()
    features["volume_acceleration"] = _safe_division(vol_cur - vol_prev, vol_prev)

    # OBV
    obv = _compute_obv(close, volume)
    obv_val = obv.iloc[-1]
    features["obv_5d_change"] = _safe_division(
        obv_val - obv.iloc[-6], abs(obv.iloc[-6])
    )
    features["obv_20d_change"] = _safe_division(
        obv_val - obv.iloc[-21], abs(obv.iloc[-21])
    )

    # Liquidity (Amihud)
    # |Return| / (Price * Vol)
    dollar_vol = close * volume
    illiquidity = returns.abs().iloc[-20:] / dollar_vol.iloc[-20:]
    features["amihud_illiquidity_20d"] = illiquidity.mean() * 1e6  # scaled

    features["spread_proxy"] = _safe_division(high.iloc[-1] - low.iloc[-1], curr_price)

    vol_std = volume.iloc[-20:].std()
    vol_mean = volume.iloc[-20:].mean()
    features["volume_volatility_20d"] = _safe_division(vol_std, vol_mean)

    # ------------------------------------------------------------------ #
    # F. Microstructure & Higher Moments
    # ------------------------------------------------------------------ #
    features["overnight_gap"] = _safe_division(
        opn.iloc[-1] - close.iloc[-2], close.iloc[-2]
    )

    # Gap Persistence
    gap = opn.iloc[-1] - close.iloc[-2]
    intraday_move = curr_price - opn.iloc[-1]
    # 1.0 if price continued in direction of gap, 0.0 if it reversed (filled)
    features["gap_persistence"] = (
        1.0 if (np.sign(gap) == np.sign(intraday_move) and gap != 0) else 0.0
    )

    features["intraday_range"] = _safe_division(
        high.iloc[-1] - low.iloc[-1], curr_price
    )

    avg_range_5d = ((high.iloc[-6:-1] - low.iloc[-6:-1]) / close.iloc[-6:-1]).mean()
    features["range_expansion_5d"] = _safe_division(
        features["intraday_range"], avg_range_5d
    )

    # Candle Shadows
    day_range = high.iloc[-1] - low.iloc[-1]
    if day_range > 0:
        features["upper_shadow_ratio"] = (
            high.iloc[-1] - max(opn.iloc[-1], curr_price)
        ) / day_range
        features["lower_shadow_ratio"] = (
            min(opn.iloc[-1], curr_price) - low.iloc[-1]
        ) / day_range
        features["body_size_ratio"] = abs(curr_price - opn.iloc[-1]) / day_range
        features["close_position_in_range"] = (curr_price - low.iloc[-1]) / day_range
    else:
        features["upper_shadow_ratio"] = 0.0
        features["lower_shadow_ratio"] = 0.0
        features["body_size_ratio"] = 0.0
        features["close_position_in_range"] = 0.5

    # Skew/Kurtosis
    features["log_return_skewness_20d"] = returns.iloc[-20:].skew()
    features["log_return_skewness_60d"] = returns.iloc[-60:].skew()
    features["log_return_kurtosis_20d"] = returns.iloc[-20:].kurtosis()
    features["log_return_kurtosis_60d"] = returns.iloc[-60:].kurtosis()

    # Downside Deviation & Drawdown
    neg_ret = returns.iloc[-20:][returns.iloc[-20:] < 0]
    features["downside_deviation_20d"] = (
        neg_ret.std(ddof=1) * np.sqrt(252) if len(neg_ret) > 0 else 0.0
    )

    roll_max = close.iloc[-60:].cummax()
    drawdown = (close.iloc[-60:] - roll_max) / roll_max
    features["max_drawdown_60d"] = drawdown.min()

    # ------------------------------------------------------------------ #
    # G. Volatility Regime & Mean Reversion
    # ------------------------------------------------------------------ #
    features["realized_vol_5d"] = _compute_realized_volatility(close, 5)
    features["realized_vol_20d"] = _compute_realized_volatility(close, 20)
    features["realized_vol_60d"] = _compute_realized_volatility(close, 60)
    features["vol_ratio_short_long"] = _safe_division(
        features["realized_vol_5d"], features["realized_vol_60d"]
    )

    features["parkinson_vol_20d"] = _compute_parkinson_volatility(high, low, 20)

    vol_prev_week = _compute_realized_volatility(close.iloc[:-5], 20)
    features["vol_trend_1w"] = features["realized_vol_20d"] - vol_prev_week

    spy_vol_20 = _compute_realized_volatility(mkt_close, 20)
    features["relative_vol_vs_spy"] = _safe_division(
        features["realized_vol_20d"], spy_vol_20
    )

    # Mean Reversion
    for w in [10, 30, 60]:
        sma_val = ta.SMA(c_np, timeperiod=w)[-1]
        features[f"price_vs_sma_{w}d"] = _safe_division(curr_price - sma_val, sma_val)

    features["log_return_autocorr_1d"] = _compute_autocorrelation(returns, 1)
    features["log_return_autocorr_5d"] = _compute_autocorrelation(returns, 5)
    features["hurst_exponent_60d"] = _compute_hurst_exponent(close, 60)

    # ------------------------------------------------------------------ #
    # H. Cross-Sectional (Universe) Features
    # ------------------------------------------------------------------ #
    # Initialize with NaN
    cs_keys = [
        "momentum_rank_5d",
        "momentum_rank_20d",
        "volatility_rank",
        "volume_rank",
        "rsi_rank",
        "relative_strength_rank",
    ]
    for k in cs_keys:
        features[k] = np.nan

    if universe_ohlcv and len(universe_ohlcv) > 1:
        # Minimal loop to extract comparison stats
        stats_ret5 = []
        stats_ret20 = []
        stats_vol = []
        stats_vol_avg = []
        stats_rsi = []

        for _, df in universe_ohlcv.items():
            if len(df) < 21:
                continue
            c = df["close"]
            stats_ret5.append(c.iloc[-1] / c.iloc[-6] - 1)
            stats_ret20.append(c.iloc[-1] / c.iloc[-21] - 1)
            stats_vol.append(_compute_realized_volatility(c, 20))
            stats_vol_avg.append(df["volume"].iloc[-20:].mean())
            stats_rsi.append(ta.RSI(c.values.astype(float), 14)[-1])

        # Compute Ranks
        features["momentum_rank_5d"] = _compute_percentile_rank(
            ret_5d, pd.Series(stats_ret5)
        )
        features["momentum_rank_20d"] = _compute_percentile_rank(
            features["log_return_2w"], pd.Series(stats_ret20)
        )
        features["volatility_rank"] = _compute_percentile_rank(
            features["realized_vol_20d"], pd.Series(stats_vol)
        )
        features["volume_rank"] = _compute_percentile_rank(
            features["volume_20d_avg"], pd.Series(stats_vol_avg)
        )
        features["rsi_rank"] = _compute_percentile_rank(
            features["rsi_14"], pd.Series(stats_rsi)
        )

        # Composite Relative Strength
        if not (
            np.isnan(features["momentum_rank_5d"])
            or np.isnan(features["momentum_rank_20d"])
        ):
            features["relative_strength_rank"] = (
                features["momentum_rank_5d"] + features["momentum_rank_20d"]
            ) / 2.0

    # ------------------------------------------------------------------ #
    # I. Interaction Features
    # ------------------------------------------------------------------ #
    features["momentum_x_volatility"] = ret_5d * features["realized_vol_20d"]
    features["volume_x_return"] = features["volume_ratio_20d"] * ret_5d
    features["rsi_x_trend"] = features["rsi_14"] * features["sma_20_over_50"]
    features["beta_x_market"] = features["capm_beta_1y"] * spy_ret_5d
    features["vix_x_momentum"] = features["macro_vix_1w_change"] * ret_5d

    # Ensure strict alignment with FEATURE_NAMES and float type
    return pd.Series({k: features.get(k, np.nan) for k in FEATURE_NAMES}, dtype=float)


def generate_dataset(
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    macro_data: pd.DataFrame,
    universe_ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
    min_history_days: int = 252,
    forward_days: int = 5,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Creates a complete training/backtesting dataset using a walk-forward process.

    Iterates through history, computing features based only on past data
    (preserving temporal causality) and aligning them with future returns.

    Args:
        ohlcv (pd.DataFrame): Full history for target asset.
        market_ohlcv (pd.DataFrame): Full history for benchmark.
        macro_data (pd.DataFrame): Full history for macro indicators.
        universe_ohlcv (Dict[str, pd.DataFrame], optional): Full universe history.
        min_history_days (int): Startup period before first feature calculation.
        forward_days (int): Number of days for target return calculation.
        show_progress (bool): If True, displays tqdm progress bar.

    Returns:
        pd.DataFrame: Combined DataFrame of X (features) and Y (next_return_Xd).

    Raises:
        ValueError: If indices are not DatetimeIndex or data is insufficient.
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv must have a DatetimeIndex.")

    # Common index alignment
    common_idx = ohlcv.index.intersection(market_ohlcv.index).intersection(  # type: ignore
        macro_data.index  # type: ignore
    )
    common_idx = common_idx.sort_values()

    ohlcv = ohlcv.loc[common_idx]
    market_ohlcv = market_ohlcv.loc[common_idx]
    macro_data = macro_data.loc[common_idx]

    if len(ohlcv) < min_history_days + forward_days:
        raise ValueError(
            "Not enough aligned data for the requested history and forward window."
        )

    # Dates we can actually compute on
    valid_dates = common_idx[min_history_days:-forward_days]
    n_samples = len(valid_dates)

    # Pre-allocate feature matrix (X) and target matrix (Y)
    X_data = np.full((n_samples, len(FEATURE_NAMES)), np.nan)
    Y_data = np.full((n_samples, forward_days), np.nan)

    # Keep track of which rows successfully computed
    computed_indices: List[int] = []

    # Progress bar setup
    iterator = range(n_samples)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Generating Features", unit="day")
        except ImportError:
            pass

    # Main Walk-Forward Loop
    # Note: While a loop is slower than vectorization, it is required here because
    # `ohlcv_to_features` is designed to simulate point-in-time arrival of data,
    # preventing look-ahead bias by strictly slicing inputs.
    for i in iterator:
        date = valid_dates[i]

        # Slicing index (integer location)
        # +1 because .iloc slice is exclusive at the end
        loc_idx = ohlcv.index.get_loc(date) + 1  # type: ignore

        # 1. Feature Calculation (Past Data)
        try:
            # Optimization: Slice Universe only if needed
            sliced_universe = None
            if universe_ohlcv:
                sliced_universe = {t: d.loc[:date] for t, d in universe_ohlcv.items()}

            features = ohlcv_to_features(
                ohlcv.iloc[:loc_idx],
                market_ohlcv.iloc[:loc_idx],
                macro_data.iloc[:loc_idx],
                universe_ohlcv=sliced_universe,
            )
            X_data[i, :] = features.values

            # 2. Target Calculation (Future Data)
            current_close = ohlcv["close"].iloc[loc_idx - 1]
            for d in range(1, forward_days + 1):
                future_close = ohlcv["close"].iloc[loc_idx - 1 + d]
                Y_data[i, d - 1] = np.log(future_close / current_close)

            computed_indices.append(i)

        except Exception:
            # In production, you might log this error.
            # Here we skip the date if data is malformed.
            continue

    if not computed_indices:
        raise ValueError("No valid features could be generated from the provided data.")

    # Filter valid rows
    valid_idx_arr = np.array(computed_indices)
    X_final = X_data[valid_idx_arr]
    Y_final = Y_data[valid_idx_arr]
    final_dates = valid_dates[valid_idx_arr]

    # Construct Final DataFrame
    df_X = pd.DataFrame(X_final, columns=FEATURE_NAMES, index=final_dates)

    y_cols = [f"next_log_return_{d}d" for d in range(1, forward_days + 1)]
    df_Y = pd.DataFrame(Y_final, columns=y_cols, index=final_dates)

    return pd.concat([df_X, df_Y], axis=1)
