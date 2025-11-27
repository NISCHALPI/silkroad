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

import jax.numpy as jnp
import jax
import typing as tp
import numpy as np

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "to_numpy",  # Helper to convert JAX functions to NumPy functions (Sometimes Needed)
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
