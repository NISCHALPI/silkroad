"""Functional Interface For Backtesting.

This module contains functional implementations of backtesting utilities using JAX.
It provides a differentiable backtesting engine that supports end-of-period rebalancing
and transaction costs.
"""

from typing import Tuple, Union, Dict
import jax
import jax.numpy as jnp
from functools import partial
from silkroad.functional.metrics import (
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    CVaR,
    VaR,
    annulized_volatility,
    expected_annualized_return,
    ulcer_index,
    tail_ratio,
)

__all__ = [
    "backtest",
    "backtest_step",
]


@jax.jit
def backtest_step(
    state: Tuple[jax.Array, jax.Array],
    inputs: Tuple[jax.Array, jax.Array, jax.Array],
    transaction_cost_bp: float,
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
    """Runs a single step of the backtest simulation.

    This function calculates the portfolio return, updates the portfolio value,
    computes the drifted weights due to price movements, and then rebalances
    to the target weights at the end of the period, deducting transaction costs.
    Rebalancing is conditional on `rebalance_mask`.

    Args:
        state: A tuple representing the current state of the backtest:
            - current_value: A scalar jax.Array representing the current portfolio value.
            - current_weights: A jax.Array of shape (N,) representing current asset weights.
        inputs: A tuple containing input data for the current step:
            - log_returns: A jax.Array of shape (N,) representing the log return for each
              asset over the period (ln(p_{t+1} / p_t)).
            - target_weights: A jax.Array of shape (N,) representing the target weights
              to rebalance to at the END of the period.
            - rebalance_mask: A boolean scalar jax.Array. If True, rebalance to
              target_weights. If False, do not rebalance (keep drifted weights).
        transaction_cost_bp: The transaction cost in basis points (bps) to apply
            to the rebalancing turnover.

    Returns:
        A tuple containing the updated state and metrics for the step:
            - New State: A tuple (new_value, new_weights).
            - Metrics: A tuple (step_log_return, transaction_costs, turnover).
    """
    current_value, current_weights = state
    log_returns, target_weights, rebalance_mask = inputs

    # 1. Portfolio Return
    # We hold 'current_weights' throughout the period.
    # Portfolio return is weighted sum of asset simple returns.
    # r_p = sum(w_i * (exp(r_i) - 1))
    asset_simple_returns = jnp.expm1(log_returns)
    portfolio_return = jnp.sum(current_weights * asset_simple_returns)

    # 2. Update Value (Pre-Cost)
    # V_{grown} = V_t * (1.0 + r_p)
    grown_value = current_value * (1.0 + portfolio_return)

    # 3. Calculate Drifted Weights
    # Weights drift because assets grow at different rates.
    # w_{drift} = (w_{t} * exp(r_{i, t})) / (1 + r_{p, t})
    drifted_weights = (current_weights * jnp.exp(log_returns)) / (
        1.0 + portfolio_return
    )

    # 4. Rebalance
    # We rebalance from drifted_weights to target_weights at the END of the period.
    # If rebalance_mask is False, we keep drifted_weights (effective target is drifted_weights).

    # Broadcast mask to shape (N,)
    effective_target_weights = jnp.where(
        rebalance_mask, target_weights, drifted_weights
    )

    # Turnover = sum(|effective_target_weights - drifted_weights|)
    turnover = jnp.sum(jnp.abs(effective_target_weights - drifted_weights))

    # Cost is calculated on the GROWN value (the capital we have available to trade).
    transaction_costs = grown_value * turnover * (transaction_cost_bp / 10000.0)

    # 5. Update Final Value
    new_value = grown_value - transaction_costs

    # 6. Update State
    # We end the period with effective_target_weights
    new_state = (new_value, effective_target_weights)

    # Metrics
    # Log return relative to V_t
    step_log_return = jnp.log(new_value / current_value)

    metrics = (step_log_return, transaction_costs, turnover)

    return new_state, metrics


@jax.jit
def _backtest_jit(
    init_value: float,
    init_weights: jax.Array,
    log_returns: jax.Array,
    target_weights: jax.Array,
    mask: jax.Array,
    transaction_cost_bp: float,
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
    """JIT-compiled core for backtest."""
    init_state = (jnp.array(init_value, dtype=jnp.float32), init_weights)

    # Slice target weights to match returns length
    t_weights = target_weights[: log_returns.shape[0]]
    # Slice mask to match returns length
    t_mask = mask[: log_returns.shape[0]]

    step_fn = partial(backtest_step, transaction_cost_bp=transaction_cost_bp)

    final_state, metrics = jax.lax.scan(
        step_fn, init_state, (log_returns, t_weights, t_mask)
    )

    return final_state, metrics


def backtest(
    init_value: float,
    init_weights: jax.Array,
    log_returns: jax.Array,
    target_weights: jax.Array,
    transaction_cost_bp: float = 0.0,
    rebalance_mask: Union[None, jax.Array] = None,
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
    """Runs the backtest simulation over a period of time.

    This function simulates the performance of a portfolio by scanning the `backtest_step`
    function over the provided asset returns and target weights. It tracks the portfolio
    value, weights, and various metrics (returns, transaction costs, turnover) at each step.

    The simulation proceeds as follows for each time step t:
    1.  **Portfolio Return**: Calculates the return of the current portfolio based on
        holdings at the start of the period and asset returns during the period.
    2.  **Growth**: Updates the portfolio value based on the calculated return.
    3.  **Drift**: Updates the asset weights to reflect price movements (drift).
    4.  **Rebalancing**: Rebalances the portfolio from the drifted weights to the
        specified `target_weights` for the *next* period.
        *If `rebalance_mask[t]` is False, rebalancing is skipped.*
    5.  **Costs**: Calculates and deducts transaction costs associated with the rebalancing
        from the portfolio value.

    Args:
        init_value: A float representing the initial monetary value of the portfolio.
        init_weights: A jax.Array of shape (N,) representing the initial asset weights.
            Sum of weights should ideally be 1.0.
        log_returns: A jax.Array of shape (T, N) representing the log returns of N assets
            over T time steps. `log_returns[t]` is the log return of the assets during
            the period t (from t to t+1).
        target_weights: A jax.Array of shape (T, N) representing the target weights
            for each period. `target_weights[t]` is the allocation we want to hold
            *after* rebalancing at the end of period t (to be held during period t+1).
        transaction_cost_bp: A float representing the transaction cost in basis points (bps).
            Defaults to 0.0.
        rebalance_mask: An optional jax.Array of shape (T,) of booleans.
            If `rebalance_mask[t]` is True, the portfolio rebalances to `target_weights[t]`
            at the end of step t. If False, it holds the drifted weights.
            Defaults to None (rebalance every step).

    Returns:
        A tuple containing the final state and the history of metrics:
            - **Final State**: A tuple `(final_value, final_weights)`.
                - `final_value`: A scalar jax.Array representing the portfolio value after T steps.
                - `final_weights`: A jax.Array of shape (N,) representing the weights after T steps.
            - **Metrics History**: A tuple `(returns_history, costs_history, turnover_history)`.
                - `returns_history`: A jax.Array of shape (T,) containing portfolio log returns for each step.
                - `costs_history`: A jax.Array of shape (T,) containing transaction costs for each step.
                - `turnover_history`: A jax.Array of shape (T,) containing turnover for each step.
    """
    # Handle mask
    if rebalance_mask is None:
        mask = jnp.ones(log_returns.shape[0], dtype=bool)
    else:
        mask = rebalance_mask

    return _backtest_jit(
        init_value,
        init_weights,
        log_returns,
        target_weights,
        mask,
        transaction_cost_bp,
    )


def backtest_with_metrics(
    init_value: float,
    init_weights: jax.Array,
    log_returns: jax.Array,
    target_weights: jax.Array,
    transaction_cost_bp: float = 0.0,
    rebalance_mask: Union[None, jax.Array] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: Union[int, float] = 252,
    alpha: float = 0.05,
) -> Dict[str, Union[float, jax.Array]]:
    """Runs backtest and computes performance metrics.

    This function runs the backtest simulation and computes key performance metrics
    such as Sharpe ratio, Sortino ratio, and maximum drawdown based on the portfolio
    returns obtained from the backtest.

    Args:
        init_value: A float representing the initial monetary value of the portfolio.
        init_weights: A jax.Array of shape (N,) representing the initial asset weights.
        log_returns: A jax.Array of shape (T, N) representing the log returns of N assets
            over T time steps.
        target_weights: A jax.Array of shape (T, N) representing the target weights
            for each period.
        transaction_cost_bp: A float representing the transaction cost in basis points (bps).
            Defaults to 0.0.
        rebalance_mask: An optional jax.Array of shape (T,) of booleans for rebalancing.
        risk_free_rate: A float representing the risk-free rate for Sharpe ratio calculation.
            Defaults to 0.0.
        periods_per_year: An integer representing the number of periods per year for
            annualizing metrics. Defaults to 252 (trading days).
        alpha: A float representing the significance level for VaR and CVaR calculations.
            Defaults to 0.05.

    Returns:
        A dictionary containing the final portfolio value, weights, and performance metrics:
            - 'final_value': Final portfolio value after T steps.
            - 'final_weights': Final asset weights after T steps.
            - 'returns_history': Portfolio log returns for each step.
            - 'log_returns_history': Portfolio log returns for each step (same as returns_history).
            - 'transaction_costs_history': Transaction costs for each step.
            - 'turnover_history': Turnover for each step.
            - 'sharpe_ratio': Sharpe ratio of the portfolio returns.
            - 'sortino_ratio': Sortino ratio of the portfolio returns.
            - 'max_drawdown': Maximum drawdown of the portfolio returns.
    """
    (r_final_value, r_final_weights), (
        returns_history,
        costs_history,
        turnover_history,
    ) = backtest(
        init_value=init_value,
        init_weights=init_weights,
        log_returns=log_returns,
        target_weights=target_weights,
        transaction_cost_bp=transaction_cost_bp,
        rebalance_mask=rebalance_mask,
    )
    # Compute performance metrics from returns history
    sharpe = sharpe_ratio(
        returns=returns_history,
        risk_free_rate=risk_free_rate,
        periods=periods_per_year,
    )
    sortino = sortino_ratio(
        returns=returns_history,
        minimum_acceptable_return=risk_free_rate,
        periods=periods_per_year,
    )
    mdd = max_drawdown(returns=returns_history)
    calmar = calmar_ratio(
        returns=returns_history,
        periods=periods_per_year,
    )
    omega = omega_ratio(
        returns=returns_history, threshold=risk_free_rate, periods=periods_per_year
    )
    cvar = CVaR(returns=returns_history, alpha=alpha)
    var = VaR(returns=returns_history, alpha=alpha)
    ann_vol = annulized_volatility(returns=returns_history, periods=periods_per_year)
    exp_ret = expected_annualized_return(
        returns=returns_history, periods=periods_per_year
    )
    ulcer = ulcer_index(returns=returns_history)
    tail = tail_ratio(returns=returns_history)

    return {
        "final_value": r_final_value,
        "final_weights": r_final_weights,
        "returns_history": returns_history,
        "log_returns_history": returns_history,
        "transaction_costs_history": costs_history,
        "turnover_history": turnover_history,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "CVaR": cvar,
        "VaR": var,
        "annualized_volatility": ann_vol,
        "expected_annualized_return": exp_ret,
        "ulcer_index": ulcer,
        "tail_ratio": tail,
    }
