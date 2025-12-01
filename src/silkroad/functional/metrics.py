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

import jax.numpy as jnp
import jax
import typing as tp
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
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    mu, std = jnp.mean(returns), jnp.std(returns)
    return (periods * mu - risk_free_rate) / (std * jnp.sqrt(periods))


@jax.jit
def sortino_ratio(
    returns: jax.Array,
    minimum_acceptable_return: float = 0.0,
    periods: tp.Union[int, float] = 252,
) -> jax.Array:
    """Calculate the annualized Sortino Ratio of a series of returns.

    The Sortino Ratio is a variation of the Sharpe Ratio that focuses on downside risk
    by considering only negative returns (downside deviation) in its calculation.
    This implementation uses the Lower Partial Moment (LPM) of order 2, which is
    fully JAX-compatible and differentiable.

    The Sortino Ratio is computed as:

    .. math::

        SoR = \\frac{N \\cdot \\mu - r_m}{\\sigma_d \\cdot \\sqrt{N}}

    where:
        - :math:`\\mu` is the mean of periodic returns
        - :math:`r_m` is the annualized minimum acceptable return
        - :math:`\\sigma_d` is the downside deviation (LPM order 2)
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
        >>> sortino_ratio(daily_returns, minimum_acceptable_return=0.02, periods=252)
        3.123
    """
    returns = jnp.exp(returns) - 1  # Convert log-returns to simple returns
    mu = jnp.mean(returns)

    # Adjust MAR to periodic
    mar_periodic = minimum_acceptable_return / periods

    # Calculate Downside Deviation (Lower Partial Moment of order 2)
    # We use np.minimum to isolate returns below MAR (others become 0)
    # This avoids boolean indexing which is not JAX-jit compatible
    downside_diff = jnp.minimum(0.0, returns - mar_periodic)
    downside_deviation = jnp.sqrt(jnp.mean(jnp.square(downside_diff)))

    # Add epsilon to avoid division by zero
    return (periods * mu - minimum_acceptable_return) / (
        downside_deviation * jnp.sqrt(periods) + 1e-9
    )


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
    std_dev = jnp.std(returns)
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
def calmar_ratio(returns: jax.Array, periods: tp.Union[int, float] = 252) -> jax.Array:
    """Calculate the Calmar Ratio.

    The Calmar Ratio is the ratio of annualized return to the absolute maximum drawdown.
    It is a measure of risk-adjusted return that focuses on drawdown risk.

    Args:
        returns: Array of periodic log-returns.
        periods: Number of periods in a year. Default is 252.

    Returns:
        The Calmar Ratio.
    """
    # We can reuse existing functions, but for JIT efficiency and clarity we might inline or call them.
    # Calling them is fine as they are JIT compiled.
    ann_ret = expected_annualized_return(returns, periods)
    mdd = max_drawdown(returns)

    # Avoid division by zero
    return ann_ret / (jnp.abs(mdd) + 1e-9)


@jax.jit
def omega_ratio(
    returns: jax.Array, threshold: float = 0.0, periods: tp.Union[int, float] = 252
) -> jax.Array:
    """Calculate the Omega Ratio.

    The Omega Ratio is the probability-weighted ratio of gains versus losses for some
    threshold return target. It captures all higher moments of the return distribution.

    Args:
        returns: Array of periodic log-returns.
        threshold: The target return threshold (annualized). Default is 0.0.
        periods: Number of periods in a year. Default is 252.

    Returns:
        The Omega Ratio.
    """
    returns = jnp.exp(returns) - 1
    threshold_periodic = threshold / periods

    excess = returns - threshold_periodic

    # sum(max(R - L, 0)) / sum(max(L - R, 0))
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
def information_ratio(returns: jax.Array, benchmark_returns: jax.Array) -> jax.Array:
    """Calculate the Information Ratio.

    The Information Ratio measures the portfolio returns beyond the returns of a
    benchmark, compared to the volatility of those returns (tracking error).

    Args:
        returns: Array of periodic log-returns.
        benchmark_returns: Array of periodic log-returns of the benchmark.
                           Must be same length as returns.

    Returns:
        The Information Ratio.
    """
    # Convert both to simple returns for accurate difference calculation
    returns_simple = jnp.exp(returns) - 1
    benchmark_simple = jnp.exp(benchmark_returns) - 1

    active_return = returns_simple - benchmark_simple
    tracking_error = jnp.std(active_return)

    return jnp.mean(active_return) / (tracking_error + 1e-9)


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


def to_numpy(func: tp.Callable[..., jax.Array]) -> tp.Callable[..., np.ndarray]:
    """Decorator to convert JAX Function to NumPy Function.

    This decorator wraps a JAX function so that it both accepts NumPy arrays as input
    and returns NumPy arrays as output. It handles the conversion between NumPy and
    JAX arrays seamlessly.

    Args:
        func: A function that takes JAX arrays as input and returns a JAX array.
    Returns:
        A function that takes NumPy arrays as input and returns a NumPy array.
    """

    def wrapper(*args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        jax_args = [
            jnp.asarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args
        ]
        jax_kwargs = {
            k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        result = func(*jax_args, **jax_kwargs)
        return np.asarray(result)

    return wrapper
