import pytest
import jax
import jax.numpy as jnp
from silkroad.functional.backtest import backtest


def test_backtest_shapes():
    T, N = 10, 3
    init_value = 1000.0
    init_weights = jnp.array([0.3, 0.3, 0.4])
    # Returns of shape (T, N)
    log_returns = jnp.zeros((T, N))
    target_weights = jnp.ones((T, N)) / N

    final_state, metrics = backtest(
        init_value, init_weights, log_returns, target_weights
    )

    final_value, final_weights = final_state
    returns, transaction_costs, turnover = metrics

    assert final_value.shape == ()
    assert final_weights.shape == (N,)
    assert returns.shape == (T,)
    assert transaction_costs.shape == (T,)
    assert turnover.shape == (T,)


def test_no_movement():
    T, N = 5, 2
    init_value = 100.0
    init_weights = jnp.array([0.5, 0.5])
    # Zero returns
    log_returns = jnp.zeros((T, N))
    # Target weights same as init
    target_weights = jnp.tile(init_weights, (T, 1))

    final_state, metrics = backtest(
        init_value, init_weights, log_returns, target_weights
    )

    final_value, final_weights = final_state
    returns, transaction_costs, turnover = metrics

    assert jnp.allclose(final_value, init_value)
    assert jnp.allclose(returns, 0.0)
    assert jnp.allclose(transaction_costs, 0.0)


def test_simple_growth():
    # Asset 0 doubles in price, Asset 1 stays same
    # We hold 100% Asset 0 initially.
    # Target weights are also 100% Asset 0, so no rebalancing needed.
    T, N = 1, 2  # One period
    init_value = 100.0
    init_weights = jnp.array([1.0, 0.0])
    # Asset 0 returns 100% (doubles), Asset 1 returns 0%
    asset_simple_returns = jnp.array([[1.0, 0.0]])
    log_returns = jnp.log1p(asset_simple_returns)
    target_weights = jnp.array([[1.0, 0.0]])

    final_state, metrics = backtest(
        init_value, init_weights, log_returns, target_weights
    )

    final_value, final_weights = final_state
    returns, transaction_costs, turnover = metrics

    # Value should double: 100 -> 200
    assert jnp.allclose(final_value, 200.0)
    # Returns are now log returns. 100% simple return -> ln(2) log return
    assert jnp.allclose(returns[0], jnp.log(2.0))


def test_transaction_costs():
    # Switch from [1, 0] to [0, 1]
    # Zero returns
    # Drifted weights = initial weights = [1, 0]
    # Target weights = [0, 1]
    # Turnover = |0-1| + |1-0| = 2
    # Cost = V_grown * 2 * 10bps/10000
    # V_grown = 100 (no return)
    # Cost = 100 * 2 * 0.001 = 0.2
    T, N = 1, 2
    init_value = 100.0
    init_weights = jnp.array([1.0, 0.0])
    log_returns = jnp.zeros((T, N))
    target_weights = jnp.array([[0.0, 1.0]])
    cost_bp = 10.0

    final_state, metrics = backtest(
        init_value,
        init_weights,
        log_returns,
        target_weights,
        transaction_cost_bp=cost_bp,
    )

    final_value, final_weights = final_state
    returns, transaction_costs, turnover = metrics

    expected_cost = 100.0 * 2.0 * (10.0 / 10000.0)
    assert jnp.allclose(transaction_costs[0], expected_cost)
    assert jnp.allclose(final_value, 100.0 - expected_cost)


def test_differentiability():
    T, N = 5, 3
    init_value = 1000.0
    init_weights = jnp.ones(N) / N
    # Create dummy returns
    asset_simple_returns = jnp.linspace(0.01, 0.05, T * N).reshape(T, N)
    log_returns = jnp.log1p(asset_simple_returns)

    # We want to optimize target_weights to maximize final value
    target_weights = jnp.ones((T, N)) / N

    def loss_fn(weights):
        final_state, _ = backtest(init_value, init_weights, log_returns, weights)
        final_value, _ = final_state
        return -final_value

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(target_weights)

    assert grads.shape == target_weights.shape
    assert not jnp.all(jnp.isnan(grads))


def test_conditional_rebalancing():
    # 2 periods.
    # Period 0: Mask = False (No rebalance).
    # Period 1: Mask = True (Rebalance).
    T, N = 2, 2
    init_value = 100.0
    init_weights = jnp.array([0.5, 0.5])

    # Asset 0 returns 100% in period 0, 0% in period 1.
    # Asset 1 returns 0% in period 0, 0% in period 1.
    asset_simple_returns = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    log_returns = jnp.log1p(asset_simple_returns)

    # Target weights are always [0.5, 0.5]
    target_weights = jnp.array([[0.5, 0.5], [0.5, 0.5]])

    # Mask: [False, True]
    rebalance_mask = jnp.array([False, True])

    # Expected behavior:
    # Period 0:
    #   Start weights: [0.5, 0.5]
    #   Returns: [1.0, 0.0]
    #   Grown Value: 100 * (0.5*2 + 0.5*1) = 150
    #   Drifted Weights:
    #       w0 = 0.5 * 2 / 1.5 = 1/1.5 = 2/3
    #       w1 = 0.5 * 1 / 1.5 = 1/3
    #   Mask is False -> No rebalance.
    #   End weights = [2/3, 1/3]
    #   Turnover = 0
    #   Cost = 0

    # Period 1:
    #   Start weights: [2/3, 1/3]
    #   Returns: [0.0, 0.0]
    #   Grown Value: 150 * 1 = 150
    #   Drifted Weights: [2/3, 1/3]
    #   Mask is True -> Rebalance to [0.5, 0.5]
    #   Turnover = |0.5 - 2/3| + |0.5 - 1/3| = |-1/6| + |1/6| = 1/3
    #   Cost = 150 * (1/3) * bps

    cost_bp = 10.0

    final_state, metrics = backtest(
        init_value,
        init_weights,
        log_returns,
        target_weights,
        transaction_cost_bp=cost_bp,
        rebalance_mask=rebalance_mask,
    )

    final_value, final_weights = final_state
    returns, transaction_costs, turnover = metrics

    # Check Period 0
    assert jnp.allclose(turnover[0], 0.0)
    assert jnp.allclose(transaction_costs[0], 0.0)

    # Check Period 1
    expected_turnover = 1.0 / 3.0
    expected_cost = 150.0 * expected_turnover * (10.0 / 10000.0)
    assert jnp.allclose(turnover[1], expected_turnover)
    assert jnp.allclose(transaction_costs[1], expected_cost)


def test_buy_and_hold_equivalence():
    # If mask is all False, it should be equivalent to buy and hold.
    T, N = 5, 2
    init_value = 100.0
    init_weights = jnp.array([0.5, 0.5])

    # Random returns
    key = jax.random.PRNGKey(0)
    asset_simple_returns = jax.random.uniform(key, (T, N), minval=-0.1, maxval=0.1)
    log_returns = jnp.log1p(asset_simple_returns)

    # Target weights (irrelevant because mask is False, but needed for shape)
    target_weights = jnp.ones((T, N)) / N

    # Mask: All False
    rebalance_mask = jnp.zeros(T, dtype=bool)

    final_state, metrics = backtest(
        init_value,
        init_weights,
        log_returns,
        target_weights,
        rebalance_mask=rebalance_mask,
    )

    final_value, final_weights = final_state
    returns, transaction_costs, turnover = metrics

    # Verify no trading
    assert jnp.allclose(turnover, 0.0)
    assert jnp.allclose(transaction_costs, 0.0)

    # Calculate Buy and Hold Value manually
    # Value = InitValue * sum(InitWeight_i * Product(1 + r_i,t))
    # Cumulative return for each asset
    cum_asset_growth = jnp.prod(1.0 + asset_simple_returns, axis=0)
    expected_final_value = init_value * jnp.sum(init_weights * cum_asset_growth)

    assert jnp.allclose(final_value, expected_final_value)


def test_vmap_backtest():
    # Test running multiple backtests in parallel using vmap
    from silkroad.functional.backtest import _backtest_jit

    B = 10  # Batch size
    T, N = 5, 3
    init_value = 1000.0
    init_weights = jnp.ones(N) / N

    # Create a batch of returns: shape (B, T, N)
    key = jax.random.PRNGKey(42)
    asset_simple_returns = jax.random.uniform(key, (B, T, N), minval=-0.05, maxval=0.05)
    log_returns = jnp.log1p(asset_simple_returns)

    # Target weights: shape (B, T, N)
    target_weights = jnp.ones((B, T, N)) / N

    # Mask: shape (B, T)
    mask = jnp.ones((B, T), dtype=bool)

    # Transaction cost: scalar (broadcasted) or we can map over it too.
    # Here we treat it as static for vmap, or we can pass it.
    # _backtest_jit signature: (init_value, init_weights, log_returns, target_weights, mask, transaction_cost_bp)
    # We want to map over log_returns (2), target_weights (3), mask (4).
    # init_value, init_weights, transaction_cost_bp are shared.

    vmap_backtest = jax.vmap(_backtest_jit, in_axes=(None, None, 0, 0, 0, None))

    final_states, metrics = vmap_backtest(
        init_value, init_weights, log_returns, target_weights, mask, 10.0  # bps
    )

    final_values, final_weights = final_states
    returns, costs, turnover = metrics

    assert final_values.shape == (B,)
    assert final_weights.shape == (B, N)
    assert returns.shape == (B, T)
    assert costs.shape == (B, T)
    assert turnover.shape == (B, T)

    # Verify that the first element matches a single run
    single_state, single_metrics = _backtest_jit(
        init_value, init_weights, log_returns[0], target_weights[0], mask[0], 10.0
    )

    assert jnp.allclose(final_values[0], single_state[0])
    assert jnp.allclose(returns[0], single_metrics[0])
