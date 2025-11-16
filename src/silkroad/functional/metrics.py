"""Portfolio performance and risk metrics for quantitative finance analysis.

This module provides a collection of statistical functions for evaluating trading
strategies and investment portfolios. All metrics operate on return series and
support various time frequencies (daily, hourly, monthly, etc.).

The module implements industry-standard risk-adjusted performance measures:
    - **Sharpe Ratio**: Measures excess return per unit of total risk (volatility)
    - **Sortino Ratio**: Measures excess return per unit of downside risk only
    - **Maximum Drawdown**: Captures the largest peak-to-trough decline

Key Features:
    - Automatic conversion between log returns and simple returns
    - Support for annualization across different time periods
    - Configurable risk-free rate and minimum acceptable return thresholds
    - Numerical stability with edge case handling

Note:
    All functions expect log returns as input and automatically convert to simple
    returns internally for accurate metric calculations. This design choice ensures
    compatibility with portfolio return computation while maintaining mathematical
    correctness for risk metrics.

Examples:
    >>> import numpy as np
    >>> from silkroad.functional.metrics import sharpe_ratio, max_drawdown
    >>>
    >>> # Daily log returns from a trading strategy
    >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
    >>>
    >>> # Calculate annualized Sharpe ratio
    >>> sr = sharpe_ratio(daily_returns, risk_free_rate=0.02, periods=252)
    >>> print(f"Sharpe Ratio: {sr:.2f}")
    >>>
    >>> # Calculate maximum drawdown
    >>> mdd = max_drawdown(daily_returns)
    >>> print(f"Max Drawdown: {mdd:.2%}")
"""

import numpy as np
from typing import Union

__all__ = [
    "sharpe_ratio",
    "sortin_ratio",
    "max_drawdown",
]


def sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods: Union[int, float] = 252
) -> float:
    """Calculate the annualized Sharpe Ratio of a series of returns.

    The Sharpe Ratio measures risk-adjusted return by comparing the excess return
    of an investment to its volatility. A higher Sharpe Ratio indicates better
    risk-adjusted performance.

    The calculation assumes simple periodic log-returns (not simple returns) and adjusts
    the annualized risk-free rate to match the return period frequency.

    The Sharpe Ratio is computed as:

    .. math::

        SR = \\frac{N \\cdot \\mu - r_f}{\\sigma \\cdot \\sqrt{N}}

    where:
        - :math:`\\mu` is the mean of periodic returns
        - :math:`r_f` is the annualized risk-free rate
        - :math:`\\sigma` is the standard deviation of returns
        - :math:`N` is the number of periods per year

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).
        risk_free_rate: Annualized risk-free rate of return. Default is 0.0.
        periods: Number of periods in a year. Default is 252 for daily returns.
            Use 252 for daily, 12 for monthly, 4 for quarterly returns.

    Returns:
        The annualized Sharpe Ratio. Higher values indicate better risk-adjusted
        performance. Typical interpretations: <1 (poor), 1-2 (good), 2-3 (very good),
        >3 (excellent).

    Examples:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> sharpe_ratio(daily_returns, risk_free_rate=0.02, periods=252)
        2.547

        >>> monthly_returns = np.array([0.05, -0.02, 0.08, 0.03])
        >>> sharpe_ratio(monthly_returns, risk_free_rate=0.03, periods=12)
        1.234
    """
    returns = np.exp(returns) - 1  # Convert log-returns to simple returns
    mu, std = np.mean(returns), np.std(returns)
    return (periods * mu - risk_free_rate) / (std * np.sqrt(periods))


def sortin_ratio(
    returns: np.ndarray,
    minimum_acceptable_return: float = 0.0,
    periods: Union[int, float] = 252,
) -> float:
    """Calculate the annualized Sortino Ratio of a series of returns.

    The Sortino Ratio is a variation of the Sharpe Ratio that focuses on downside risk
    by considering only negative returns (downside deviation) in its calculation.
    This makes it a more appropriate measure for investors who are primarily concerned
    with downside risk.

    The calculation assumes simple periodic log-returns (not simple returns) and adjusts
    the annualized minimum acceptable return to match the return period frequency.

    The Sortino Ratio is computed as:

    .. math::

        SoR = \\frac{N \\cdot (\\mu - r_m)}{\\sigma_d \\cdot \\sqrt{N}}

    where:
        - :math:`\\mu` is the mean of periodic returns
        - :math:`r_m` is the annualized minimum acceptable return
        - :math:`\\sigma_d` is the standard deviation of negative returns (downside deviation)
        - :math:`N` is the number of periods per year

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).
        minimum_acceptable_return: Annualized minimum acceptable return. Default is 0.0.
        periods: Number of periods in a year. Default is 252 for daily returns.

    Returns:
        The annualized Sortino Ratio. Higher values indicate better risk-adjusted
        performance considering downside risk.

    Examples:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> sortin_ratio(daily_returns, minimum_acceptable_return=0.02, periods=252)
        3.123
    """
    returns = np.exp(returns) - 1  # Convert log-returns to simple returns
    mu = np.mean(returns)
    downside_returns = returns[returns <= minimum_acceptable_return / periods]
    if len(downside_returns) == 0:
        return float("inf")  # No downside risk
    sigma_d = np.std(downside_returns)
    return (periods * mu - minimum_acceptable_return) / (sigma_d * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    """Calculate the maximum drawdown from a series of returns.

    Maximum drawdown is the largest peak-to-trough decline in the value of an
    investment portfolio, representing the worst loss an investor could have
    experienced during a specific period.

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).

    Returns:
        The maximum drawdown as a positive float value.
    """
    returns = np.exp(returns) - 1  # Convert log-returns to simple returns
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak
    return np.max(drawdowns)
