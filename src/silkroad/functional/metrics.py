"""Portfolio performance and risk metrics for quantitative finance analysis.

This module provides a collection of statistical functions for evaluating trading
strategies and investment portfolios. All metrics operate on return series and
support various time frequencies (daily, hourly, monthly, etc.).

The module implements industry-standard risk-adjusted performance measures:
    - **Sharpe Ratio**: Measures excess return per unit of total risk (volatility)
    - **Sortino Ratio**: Measures excess return per unit of downside risk only
    - **Maximum Drawdown**: Captures the largest peak-to-trough decline
    - **Value at Risk (VaR)**: Estimates potential loss at a given confidence level
    - **Conditional Value at Risk (CVaR)**: Estimates expected loss beyond the Va
    - **Calmar Ratio**: Ratio of annualized return to maximum drawdown
    - **Omega Ratio**: Ratio of gains to losses relative to a threshold return
    - **Ulcer Index**: Measures depth and duration of drawdowns
    - **Information Ratio**: Measures active return relative to a benchmark
    - **Tail Ratio**: Ratio of extreme positive to extreme negative returns

Key Features:
    - Automatic conversion between log returns and simple returns
    - Support for annualization across different time periods
    - Configurable risk-free rate and minimum acceptable return thresholds
    - Numerical stability with edge case handling
    - JAX JIT compilation for performance optimization
    - Differentiable implementations suitable for gradient-based optimization

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

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "to_numpy",  # Helper to convert JAX functions to NumPy functions (Sometimes Needed)
    "expected_annualized_return",
    "annulized_volatility",
    "CVaR",
    "VaR",
    "calmar_ratio",
    "omega_ratio",
    "ulcer_index",
    "information_ratio",
    "tail_ratio",
]


@jax.jit
def sharpe_ratio(
    returns: jax.Array, risk_free_rate: float = 0.0, periods: tp.Union[int, float] = 252
) -> jax.Array:
    """Calculate the annualized Sharpe Ratio of a series of returns.

    This implementation converts log-returns to arithmetic returns, calculates the
    excess returns over a geometrically de-annualized risk-free rate, and uses
    an unbiased estimator (sample standard deviation) for volatility.

    The formula used is:

    .. math::

        SR_{ann} = \\sqrt{N} \\times \\frac{E[R_p - R_f]}{\\sigma_{(R_p - R_f)}}

    where:
        - :math:`R_p` is the periodic arithmetic return
        - :math:`R_f` is the periodic risk-free rate (derived geometrically)
        - :math:`\\sigma` is the sample standard deviation (ddof=1)
        - :math:`N` is the number of periods per year

    Args:
        returns: Array of periodic log-returns (e.g., daily).
        risk_free_rate: Annualized risk-free rate (e.g., 0.04 for 4%). Default is 0.0.
        periods: Number of periods in a year (e.g., 252 for daily).

    Returns:
        The annualized Sharpe Ratio.
    """
    # 1. Convert log-returns to simple arithmetic returns
    #    (Sharpe is defined on investable wealth, not log-wealth)
    returns = jnp.exp(returns) - 1

    # 2. Convert annualized risk-free rate to periodic rate
    #    Uses geometric conversion for precision: (1+r)^(1/N) - 1
    rf_periodic = (1 + risk_free_rate) ** (1 / periods) - 1

    # 3. Calculate Differential (Excess) Returns
    excess_returns = returns - rf_periodic

    # 4. Calculate Moments
    #    We use ddof=1 for the standard deviation to provide an
    #    unbiased estimator of the population variance.
    mu = jnp.mean(excess_returns)
    std = jnp.std(excess_returns, ddof=1)

    # 5. Annualize
    #    Standard deviation scales with sqrt(T), Mean scales with T.
    #    The ratio scales with sqrt(T).
    return jnp.sqrt(periods) * mu / (std + 1e-9)


@jax.jit
def sortino_ratio(
    returns: jax.Array,
    minimum_acceptable_return: float = 0.0,
    periods: tp.Union[int, float] = 252,
) -> jax.Array:
    """Calculate the annualized Sortino Ratio of a series of returns.

    The Sortino Ratio penalizes only downside volatility. This implementation
    uses the Lower Partial Moment (LPM) of order 2 for the denominator.

    The formula used is:

    .. math::

        SoR_{ann} = \\sqrt{N} \\times \\frac{E[R_p - MAR_p]}{DD}

    where:
        - :math:`MAR_p` is the periodic Minimum Acceptable Return
        - :math:`DD` is the Downside Deviation (LPM order 2)
        - :math:`N` is the number of periods per year

    Args:
        returns: Array of periodic log-returns.
        minimum_acceptable_return: Annualized minimum target return (MAR).
                                   Default is 0.0.
        periods: Number of periods in a year. Default is 252.

    Returns:
        The annualized Sortino Ratio.
    """
    # 1. Convert log-returns to simple arithmetic returns
    returns = jnp.exp(returns) - 1

    # 2. Convert annualized MAR to periodic MAR (Geometric)
    mar_periodic = (1 + minimum_acceptable_return) ** (1 / periods) - 1

    # 3. Calculate Excess Return over MAR
    #    Note: Sortino numerator uses the mean of returns minus MAR
    mu_excess = jnp.mean(returns) - mar_periodic

    # 4. Calculate Downside Deviation (LPM Order 2)
    #    We isolate returns below MAR. Returns above MAR become 0.
    downside_diff = jnp.minimum(0.0, returns - mar_periodic)

    #    We use jnp.mean (ddof=0) for Downside Deviation as this is the
    #    standard definition for Lower Partial Moments.
    downside_deviation = jnp.sqrt(jnp.mean(jnp.square(downside_diff)))

    # 5. Annualize
    return jnp.sqrt(periods) * mu_excess / (downside_deviation + 1e-9)


@jax.jit
def max_drawdown(returns: jax.Array) -> jax.Array:
    """Calculate the maximum drawdown from a series of returns.

    Maximum drawdown is the largest peak-to-trough decline in the value of an
    investment portfolio, representing the worst loss an investor could have
    experienced during a specific period.

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).

    Returns:
        The maximum drawdown as a positive float value.
    """
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    cumulative = jnp.cumprod(1 + returns)
    peak = jax.lax.associative_scan(jnp.maximum, cumulative)
    drawdowns = (peak - cumulative) / peak
    return jnp.max(drawdowns)


@jax.jit
def expected_annualized_return(
    returns: jax.Array, periods: tp.Union[int, float] = 252
) -> jax.Array:
    """Expected annualized return from periodic returns.

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).
        periods: Number of periods in a year. Default is 252 for daily returns.
    Returns:
        The annualized return as a float value.
    """
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    avg_return = jnp.mean(returns)
    return (1 + avg_return) ** periods - 1


@jax.jit
def annulized_volatility(
    returns: jax.Array, periods: tp.Union[int, float] = 252
) -> jax.Array:
    """Annualized volatility from periodic returns.

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).
        periods: Number of periods in a year. Default is 252 for daily returns.

    Returns:
        The annualized volatility as a float value.
    """
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    std_dev = jnp.std(returns, ddof=1)
    return std_dev * jnp.sqrt(periods)


@jax.jit
def CVaR(
    returns: jax.Array,
    alpha: float = 0.95,
) -> jax.Array:
    """Calculate the Conditional Value at Risk (CVaR) at a specified confidence level.

    CVaR, also known as Expected Shortfall, measures the expected loss in the worst
    (1 - alpha) % of cases. It provides insight into the tail risk of
    an investment portfolio.

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).
        alpha: Confidence level for CVaR calculation. Default is 0.95.

    Returns:
        The CVaR value as a float.
    """
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    var_threshold = jnp.percentile(returns, (1 - alpha) * 100)
    # Use masking instead of boolean indexing for JIT compatibility
    mask = returns <= var_threshold
    # Calculate mean of masked values: sum(masked) / count(masked)
    # Add epsilon to denominator to avoid division by zero if mask is empty (though unlikely)
    cvar = jnp.sum(returns * mask) / (jnp.sum(mask) + 1e-9)
    return cvar


@jax.jit
def VaR(
    returns: jax.Array,
    alpha: float = 0.95,
) -> jax.Array:
    """Calculate the Value at Risk (VaR) at a specified confidence level.

    VaR measures the maximum expected loss over a given time period at a certain
    confidence level. It is widely used in risk management to assess potential losses.

    Args:
        returns: Array of periodic log-returns (e.g., daily, hourly).
        alpha: Confidence level for VaR calculation. Default is 0.95.

    Returns:
        The VaR value as a float.
    """
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    var = jnp.percentile(returns, (1 - alpha) * 100)
    return var


@jax.jit
def information_ratio(
    returns: jax.Array,
    benchmark_returns: jax.Array,
    periods: tp.Union[int, float] = 252,
) -> jax.Array:
    """Calculate the Annualized Information Ratio.

    The Information Ratio measures the portfolio returns beyond the returns of a
    benchmark, compared to the volatility of those returns (tracking error).

    Args:
        returns: Array of periodic log-returns.
        benchmark_returns: Array of periodic log-returns of the benchmark.
        periods: Number of periods in a year. Default 252.

    Returns:
        The Annualized Information Ratio.
    """
    returns_simple = jnp.exp(returns) - 1
    benchmark_simple = jnp.exp(benchmark_returns) - 1

    active_return = returns_simple - benchmark_simple

    # Use ddof=1 for unbiased estimator of tracking error
    tracking_error = jnp.std(active_return, ddof=1)

    # Annualize by sqrt(N)
    return jnp.sqrt(periods) * jnp.mean(active_return) / (tracking_error + 1e-9)


@jax.jit
def calmar_ratio(returns: jax.Array, periods: tp.Union[int, float] = 252) -> jax.Array:
    """Calculate the Calmar Ratio using CAGR.

    The Calmar Ratio is the ratio of Compound Annual Growth Rate (CAGR) to
    the absolute maximum drawdown.

    Args:
        returns: Array of periodic log-returns.
        periods: Number of periods in a year. Default is 252.

    Returns:
        The Calmar Ratio.
    """
    # Calculate CAGR (Geometric Mean)
    # Since inputs are log returns, CAGR is simply exp(mean(log_rets) * N) - 1
    cagr = jnp.exp(jnp.mean(returns) * periods) - 1

    mdd = max_drawdown(returns)

    return cagr / (jnp.abs(mdd) + 1e-9)


@jax.jit
def omega_ratio(
    returns: jax.Array, threshold: float = 0.0, periods: tp.Union[int, float] = 252
) -> jax.Array:
    """Calculate the Omega Ratio.

    Args:
        returns: Array of periodic log-returns.
        threshold: The target return threshold (annualized). Default is 0.0.
        periods: Number of periods in a year. Default is 252.

    Returns:
        The Omega Ratio.
    """
    returns = jnp.exp(returns) - 1

    # Geometric de-annualization for consistency with Sharpe/Sortino
    threshold_periodic = (1 + threshold) ** (1 / periods) - 1

    excess = returns - threshold_periodic

    upside = jnp.sum(jnp.maximum(excess, 0.0))
    downside = jnp.sum(jnp.maximum(-excess, 0.0))

    return upside / (downside + 1e-9)


@jax.jit
def ulcer_index(returns: jax.Array) -> jax.Array:
    """Calculate the Ulcer Index.

    The Ulcer Index measures the depth and duration of drawdowns in prices.
    It is the square root of the mean of the squared percentage drawdowns.

    Args:
        returns: Array of periodic log-returns.

    Returns:
        The Ulcer Index.
    """
    returns = jnp.exp(returns) - 1
    cumulative = jnp.cumprod(1 + returns)
    peak = jax.lax.associative_scan(jnp.maximum, cumulative)
    drawdowns = (peak - cumulative) / peak

    return jnp.sqrt(jnp.mean(jnp.square(drawdowns)))


@jax.jit
def tail_ratio(returns: jax.Array) -> jax.Array:
    """Calculate the Tail Ratio.

    The Tail Ratio is the ratio of the 95th percentile of returns to the absolute
    value of the 5th percentile. It measures the skewness of the return distribution.

    Args:
        returns: Array of periodic log-returns.

    Returns:
        The Tail Ratio.
    """
    returns = jnp.exp(returns) - 1
    p95 = jnp.percentile(returns, 95)
    p5 = jnp.percentile(returns, 5)

    return p95 / (jnp.abs(p5) + 1e-9)
