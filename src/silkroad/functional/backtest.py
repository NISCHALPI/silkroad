"""Vectorized backtesting engine with transaction costs using JAX.

This module provides a high-performance, vectorized backtesting framework for
portfolio strategies using JAX's functional programming paradigm. The implementation
leverages ``jax.lax.scan`` for efficient sequential state transitions, making it
particularly well-suited for processing long time series data and enabling
hyperparameter optimization through ``jax.vmap`` for parallel execution across
multiple parameter configurations.

The backtesting engine simulates the evolution of a portfolio over discrete time
steps. At each time step, the engine evaluates whether a rebalancing event should
occur based on a user-provided binary mask. When rebalancing is triggered, the
portfolio is adjusted to match target weights while accounting for proportional
transaction costs. Portfolio valuation is performed using close prices to provide
accurate end-of-day portfolio values.

Assumptions:
    The backtesting framework operates under the following assumptions:

    1. **Rebalancing Timing**: All rebalancing trades are executed at the open
       price of each trading day. This approach eliminates lookahead bias by
       ensuring that trading decisions are made using information available at
       the time of execution.

    2. **Transaction Costs**: Transaction costs are modeled as proportional to
       the absolute dollar value of turnover, expressed in basis points (bps).
       One basis point equals 0.01% or 1/10000 of the traded value.

    3. **Portfolio Composition**: The portfolio consists of N assets. If a cash
       position is required, it must be explicitly included as one of the N
       assets with appropriate pricing (typically unit price of 1.0).

    4. **Portfolio Valuation**: Portfolio values are marked-to-market using
       close prices at the end of each trading day.

    5. **Weight Normalization**: Target portfolio weights must sum to exactly
       1.0 at every time step. This constraint ensures that the entire portfolio
       value is allocated across the available assets.

    6. **Price Handling**: Zero prices are treated as missing data or delisted
       securities. The engine applies safe division to prevent numerical issues
       when encountering zero prices.

Mathematical Framework:
    The backtesting engine implements the following mathematical operations at
    each rebalancing step t:

    1. **Gross Portfolio Value Calculation**:
       The gross portfolio value is computed as the sum of each asset's holdings
       multiplied by its current open price:

       .. math::
           V_t^{gross} = \\sum_{i=1}^{N} N_{t-1,i} \\cdot P_{t,i}^{open}

    2. **Theoretical Target Shares Estimation**:
       Before accounting for transaction costs, the theoretical number of shares
       required to achieve target weights is calculated:

       .. math::
           N_t^{theo} = \\frac{V_t^{gross} \\cdot w_t}{P_t^{open}}

    3. **Transaction Cost Computation**:
       Transaction costs are computed based on the absolute change in share
       holdings from the theoretical rebalance:

       .. math::
           TC_t = \\sum_{i=1}^{N} |N_t^{theo} - N_{t-1,i}| \\cdot P_{t,i}^{open} \\cdot \\frac{T_{bps}}{10000}

    4. **Net Portfolio Value Determination**:
       The net portfolio value available for investment is the gross value minus
       transaction costs:

       .. math::
           V_t^{net} = V_t^{gross} - TC_t

    5. **Actual Target Shares Calculation**:
       The final share holdings are computed using the net portfolio value:

       .. math::
           N_t = \\frac{V_t^{net} \\cdot w_t}{P_t^{open}}

Attributes:
    backtest: The main backtesting function that executes the vectorized
        portfolio simulation. This is the primary public interface exposed
        by this module.

Example:
    The following example demonstrates how to run a simple backtest with a
    three-asset portfolio consisting of cash and two risky assets:

    >>> import jax.numpy as jnp
    >>> from silkroad.functional.backtest import backtest
    >>>
    >>> # Initialize with 1000 shares of asset 0 (e.g., Cash with Price=1)
    >>> N_0 = jnp.array([1000.0, 0.0, 0.0])
    >>>
    >>> # Create price data for 252 trading days and 3 assets
    >>> open_prices = jnp.ones((252, 3)) * 100.0
    >>> open_prices = open_prices.at[:, 0].set(1.0)  # Asset 0 is Cash
    >>> close_prices = open_prices  # Simplified: close equals open
    >>>
    >>> # Define equal weight portfolio allocation (33% each asset)
    >>> weights = jnp.ones((252, 3)) / 3
    >>>
    >>> # Rebalance every 5 trading days
    >>> rebal_mask = jnp.zeros(252).at[::5].set(1.0)
    >>>
    >>> # Apply 10 basis points transaction cost
    >>> costs_bps = jnp.zeros((252, 3))
    >>> costs_bps = costs_bps.at[:, 0].set(0.0) # Cash has no transaction cost
    >>> costs_bps = costs_bps.at[:, 1].set(10.0) # Stock 1 has 10 bps transaction cost
    >>> costs_bps = costs_bps.at[:, 2].set(10.0) # Stock 2 has 10 bps transaction cost
    >>>
    >>> # Execute the backtest
    >>> pv, tc, holdings = backtest(
    ...     N_0, open_prices, close_prices,
    ...     weights, rebal_mask, costs_bps
    ... )

See Also:
    - :class:`BacktestConfig`: Pydantic model for validating backtest inputs.
    - :func:`state_transition`: Core rebalancing logic for a single step.
    - ``jax.lax.scan``: JAX primitive used for efficient sequential iteration.
"""

import typing as tp

import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["backtest"]


class BacktestConfig(BaseModel):
    """Configuration and validation for backtest parameters.

    This class serves as a validator for the inputs to the backtest function using Pydantic.
    It ensures that all inputs are JAX arrays of consistent shapes and that logical constraints
    (like weights summing to 1.0) are met.

    It is recommended to use this class to validate inputs to the backtest function and use the
    following pattern:

    >>> config = BacktestConfig(
    ...     N_0=N_0,
    ...     open=open_prices,
    ...     close=close_prices,
    ...     target_weights=weights,
    ...     rebalancing_mask=rebal_mask,
    ...     transaction_costs_bps=costs_bps,
    ... )
    >>> pv, tc, holdings = backtest(**config.model_dump())
    """

    N_0: jax.Array = Field(
        ..., description="Initial share holdings for each asset. Shape: (N,)."
    )
    open: jax.Array = Field(
        ..., description="Open prices over the backtest period. Shape: (T, N)."
    )
    close: jax.Array = Field(
        ..., description="Close prices over the backtest period. Shape: (T, N)."
    )
    target_weights: jax.Array = Field(
        ..., description="Target portfolio weights over time. Shape: (T, N)."
    )
    rebalancing_mask: jax.Array = Field(
        ..., description="Binary indicator for rebalancing days. Shape: (T,)."
    )
    transaction_costs_bps: jax.Array = Field(
        ..., description="Transaction costs in basis points over time. Shape: (T, N)."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def validate_inputs(cls, data):
        """Validates input types, shapes, and constraints."""
        params = [
            data.get("N_0"),
            data.get("open"),
            data.get("close"),
            data.get("target_weights"),
            data.get("rebalancing_mask"),
            data.get("transaction_costs_bps"),
        ]

        if any(p is None for p in params):
            # Let Pydantic handle missing fields standard error
            return data

        if not all(isinstance(param, jax.Array) for param in params):
            raise TypeError("All inputs must be JAX arrays.")

        # Check shapes
        two_dim_params = [
            data["open"],
            data["close"],
            data["target_weights"],
        ]
        one_dim_params = [data["rebalancing_mask"]]

        if not all(param.ndim == 2 for param in two_dim_params):
            raise ValueError("Open, close, and target_weights must be 2D arrays.")

        # Check transaction_costs_bps shape
        if data["transaction_costs_bps"].ndim != 2:
            raise ValueError(
                "Transaction_costs_bps must be a 2D array of shape (T, N)."
            )

        if not all(param.ndim == 1 for param in one_dim_params):
            raise ValueError("Rebalancing_mask must be a 1D array.")

        # Check consistent time dimension
        T = data["open"].shape[0]
        if not all(
            param.shape[0] == T
            for param in two_dim_params
            + one_dim_params
            + [data["transaction_costs_bps"]]
        ):
            raise ValueError(
                "All time-dependent inputs must have the same number of time steps."
            )

        # Check weights sum to 1.0
        # Use a tolerance for floating point comparisons
        weights = data["target_weights"]
        sum_weights = jnp.sum(weights, axis=-1)
        if not jnp.allclose(sum_weights, 1.0, atol=1e-5):
            raise ValueError("Target weights must sum to 1.0 for every time step.")

        return data


@jax.jit
def state_transition(
    N_prev: jax.Array,
    P_trade: jax.Array,
    W_target: jax.Array,
    mask: jax.Array,
    T_bps: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Execute a single rebalancing step with transaction costs.

    This function implements the core portfolio rebalancing logic. It calculates
    the new asset holdings after rebalancing to target weights, accounting for
    proportional transaction costs.

    The function enforces the constraint that `W_target` should sum to 1.0. If
    it does not, the behavior is mathematically consistent (allocating that % of
    value), but this is generally considered a configuration error in this framework.

    Args:
        N_prev (jax.Array): Previous number of shares held for each asset.
            Shape: (N,). Units: shares.
        P_trade (jax.Array): Current trade prices for rebalancing.
            Shape: (N,). Units: currency per share. Zero values indicate missing data.
        W_target (jax.Array): Target portfolio weights for each asset.
            Shape: (N,). Must satisfy $\\sum w_i = 1$.
        mask (jax.Array): Rebalancing indicator.
            Shape: (). Values: 0 (hold) or 1 (rebalance).
        transaction_costs_bps (jax.Array): Transaction costs in basis points for each asset.
            Shape: (N,). Example: 10.0 means 0.1% cost for that asset.

    Returns:
        tuple[jax.Array, jax.Array]:
            **N_new** (jax.Array): New share holdings after rebalancing. Shape: (N,).
            **realized_cost** (jax.Array): Transaction costs incurred. Shape: ().
    """
    # 1. Calculate Total Portfolio Value using current holdings and prices
    val_gross = jnp.sum(N_prev * P_trade)

    # 2. Safe division: handle zero prices (missing data / delisted)
    P_safe = jnp.where(P_trade == 0.0, 1.0, P_trade)
    val_gross_safe = jnp.maximum(val_gross, 1e-9)

    # 3. Calculate Theoretical Target Shares (before costs)
    N_target_theoretical = (val_gross_safe * W_target) / P_safe

    # 4. Calculate Approximate Transaction Costs
    delta_N_approx = N_target_theoretical - N_prev
    cost_approx = jnp.sum(jnp.abs(delta_N_approx) * P_trade * (T_bps / 10000.0))

    # 5. Calculate Net Portfolio Value (after paying fees)
    val_net = jnp.maximum(val_gross - cost_approx, 0.0)

    # 6. Calculate Real Target Shares based on Net Value
    N_rebal = (val_net * W_target) / P_safe

    # 7. Apply mask: If mask=0, hold previous state
    N_new = jnp.where(mask == 1, N_rebal, N_prev)
    realized_cost = jnp.where(mask == 1, cost_approx, 0.0)

    return N_new, realized_cost


@jax.jit
def backtest_scan_fn(
    carry: jax.Array,
    x: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Scan function for a single backtest step.

    Used internally by `jax.lax.scan`.

    Args:
        carry (jax.Array): State carried forward (N_current).
        x (tuple): Inputs for step (open, close, weights, mask, costs).

    Returns:
        tuple: (new_carry, (val_close, cost, N_new))
    """
    N_prev = carry
    price_open, price_close, w_target, mask, t_bps = x

    # Rebalance at Open
    N_new, cost = state_transition(N_prev, price_open, w_target, mask, t_bps)

    # Portfolio Value at Close = Shares * Close Price
    val_close = jnp.sum(N_new * price_close)

    return N_new, (val_close, cost, N_new)


@jax.jit
def backtest(
    N_0: jax.Array,
    open: jax.Array,
    close: jax.Array,
    target_weights: jax.Array,
    rebalancing_mask: jax.Array,
    transaction_costs_bps: jax.Array,
) -> tp.Tuple[jax.Array, jax.Array, jax.Array]:
    """Execute a vectorized backtest with transaction costs using JAX.

    This function simulates the evolution of a portfolio over a specified time
    horizon. The simulation iterates through the provided time series data,
    evaluating at each time step whether a rebalancing event should occur based
    on the binary rebalancing mask. When rebalancing is triggered (mask value
    equals 1), the portfolio holdings are adjusted to match the target weights
    while accounting for proportional transaction costs deducted from the
    portfolio value.

    The function is JIT-compiled using ``@jax.jit`` for optimal performance.
    The internal implementation leverages ``jax.lax.scan`` for efficient
    sequential state transitions, making it highly suitable for:

    - Processing long time series (thousands of trading days)
    - Hyperparameter optimization via ``jax.vmap`` for parallel backtests
    - Gradient-based optimization of portfolio strategies via ``jax.grad``

    The function enforces that ``target_weights`` must sum to 1.0 at every time
    step. This constraint is validated by the ``BacktestConfig`` class when used
    with the configuration pattern. Direct calls to this function bypass Pydantic
    validation, so users should ensure input validity when calling directly.

    Args:
        N_0 (jax.Array): Initial share holdings for each asset at the start of
            the backtest. This represents the number of shares held in each
            asset before any trading occurs.

            - Shape: ``(N,)`` where ``N`` is the number of assets in the
              portfolio.
            - Units: Number of shares (can be fractional).
            - Example: ``jnp.array([1000.0, 0.0, 0.0])`` represents 1000 shares
              of asset 0 and no shares of assets 1 and 2.

        open (jax.Array): Open prices for each asset at each time step. These
            prices are used for executing rebalancing trades. Rebalancing at
            open prices eliminates lookahead bias since the open price is known
            at the start of the trading day.

            - Shape: ``(T, N)`` where ``T`` is the number of time steps and
              ``N`` is the number of assets.
            - Units: Currency per share (e.g., dollars per share).
            - Note: Zero values are treated as missing data or delisted
              securities. The function applies safe division to prevent
              numerical errors.

        close (jax.Array): Close prices for each asset at each time step. These
            prices are used for marking-to-market the portfolio and computing
            end-of-day portfolio values.

            - Shape: ``(T, N)`` where ``T`` is the number of time steps and
              ``N`` is the number of assets.
            - Units: Currency per share (e.g., dollars per share).

        target_weights (jax.Array): Target portfolio weights specifying the
            desired allocation to each asset at each time step. The weights
            represent the fraction of total portfolio value to allocate to
            each asset.

            - Shape: ``(T, N)`` where ``T`` is the number of time steps and
              ``N`` is the number of assets.
            - Constraint: Must sum to exactly 1.0 along the asset axis (axis=1)
              for every time step. Violation of this constraint may lead to
              unexpected behavior.
            - Example: ``jnp.array([[0.5, 0.3, 0.2]])`` allocates 50% to asset
              0, 30% to asset 1, and 20% to asset 2.

        rebalancing_mask (jax.Array): Binary indicator specifying whether to
            rebalance the portfolio at each time step. A value of 1.0 triggers
            rebalancing to target weights, while 0.0 maintains current holdings.

            - Shape: ``(T,)`` where ``T`` is the number of time steps.
            - Values: 0.0 (hold current positions) or 1.0 (rebalance to target
              weights).
            - Example: ``jnp.zeros(252).at[::5].set(1.0)`` rebalances every
              5 trading days over a 252-day period.

        transaction_costs_bps (jax.Array): Transaction costs for each asset at each time step
            expressed in basis points. One basis point equals 0.01% or 1/10000
            of the traded value. Costs are applied proportionally to the
            absolute dollar value of turnover.

            - Shape: ``(T, N)`` where ``T`` is the number of time steps and
              ``N`` is the number of assets.
            - Units: Basis points (bps). For example, 10.0 represents 0.1%
              transaction cost.
            - Note: Costs are only incurred on rebalancing days (when
              ``rebalancing_mask == 1``).

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: A tuple containing three JAX
            arrays with the backtest results:

            - **portfolio_values** (jax.Array): The total portfolio value at
              the close of each trading day, computed as the sum of share
              holdings multiplied by close prices.

              - Shape: ``(T,)``
              - Units: Currency (e.g., dollars)

            - **transaction_costs** (jax.Array): The transaction costs incurred
              at each time step. Non-zero values occur only on rebalancing days.

              - Shape: ``(T,)``
              - Units: Currency (e.g., dollars)

            - **N_holdings** (jax.Array): The number of shares held in each
              asset at the end of each time step, after any rebalancing.

              - Shape: ``(T, N)``
              - Units: Number of shares

    Raises:
        TypeError: If any input is not a JAX array (when using BacktestConfig
            validation).
        ValueError: If input shapes are inconsistent or if target_weights do
            not sum to 1.0 (when using BacktestConfig validation).

    Note:
        **Cash Position Simulation**: If you need to simulate a cash position,
        include it as an explicit asset in the portfolio. Set the cash asset's
        open and close prices to 1.0 (unit value) throughout the backtest.
        Adjust the cash weight to 0.0 when fully invested in other assets and
        to higher values when reducing exposure. Additionally, ensure the
        transaction cost column for the cash asset is set to 0.0, as cash
        changes typically do not incur transaction fees.

    Example:
        Basic usage with a three-asset portfolio:

        >>> import jax.numpy as jnp
        >>> from silkroad.functional.backtest import backtest
        >>>
        >>> # Initial holdings: 1000 shares of cash (asset 0)
        >>> N_0 = jnp.array([1000.0, 0.0, 0.0])
        >>>
        >>> # Price data for 252 trading days
        >>> open_prices = jnp.ones((252, 3)) * 100.0
        >>> open_prices = open_prices.at[:, 0].set(1.0)  # Cash at $1
        >>> close_prices = open_prices
        >>>
        >>> # Equal weight allocation
        >>> weights = jnp.ones((252, 3)) / 3
        >>>
        >>> # Weekly rebalancing with 10 bps cost
        >>> rebal_mask = jnp.zeros(252).at[::5].set(1.0)
        >>> costs_bps = jnp.ones(252) * 10.0
        >>>
        >>> portfolio_values, transaction_costs, holdings = backtest(
        ...     N_0, open_prices, close_prices,
        ...     weights, rebal_mask, costs_bps
        ... )

    See Also:
        BacktestConfig: Pydantic model for input validation.
        state_transition: Core single-step rebalancing logic.
    """
    xs = (open, close, target_weights, rebalancing_mask, transaction_costs_bps)

    _, outputs = jax.lax.scan(backtest_scan_fn, N_0, xs)

    return outputs
