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

# TODO: Add feature extractor for rank based modelling approaches
