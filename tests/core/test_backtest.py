import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError
from silkroad.functional.backtest import backtest, state_transition, BacktestConfig


@pytest.fixture
def sample_data():
    T, N = 10, 2
    N_0 = jnp.array([1000.0, 1000.0])  # Fully invested start

    # Constant prices
    open_prices = jnp.ones((T, N)) * 100.0
    close_prices = jnp.ones((T, N)) * 101.0

    # Equal weights
    weights = jnp.ones((T, N)) * 0.5

    # Rebalance every step
    mask = jnp.ones(T)

    # No costs
    costs_bps = jnp.zeros((T, N))

    return {
        "N_0": N_0,
        "open": open_prices,
        "close": close_prices,
        "target_weights": weights,
        "rebalancing_mask": mask,
        "transaction_costs_bps": costs_bps,
        "T": T,
        "N": N,
    }


def test_config_validation(sample_data):
    # Valid config should pass
    BacktestConfig(**sample_data)

    # Invalid weights (sum != 1) should fail
    bad_data = sample_data.copy()
    bad_data["target_weights"] = jnp.ones((10, 2)) * 0.4  # Sums to 0.8

    with pytest.raises(ValidationError) as excinfo:
        BacktestConfig(**bad_data)
    assert "Target weights must sum to 1.0" in str(excinfo.value)


def test_state_transition_basic():
    # Initial state: 100 shares each
    N_prev = jnp.array([100.0, 100.0])

    # Prices: $100 each
    prices = jnp.array([100.0, 100.0])

    # Value = 200 * 100 = 20,000

    # Target: 50/50
    weights = jnp.array([0.5, 0.5])  # Sums to 1.0

    # Rebalance, no costs
    mask = jnp.array(1.0)
    costs = jnp.zeros(2)

    N_new, realized_cost = state_transition(N_prev, prices, weights, mask, costs)

    # Should hold 100 shares of each (already balanced)
    expected_shares = jnp.array([100.0, 100.0])

    assert jnp.allclose(N_new, expected_shares)
    assert jnp.allclose(realized_cost, 0.0)


def test_state_transition_rebalance_effect():
    # Initial state: 200 shares A, 0 shares B
    N_prev = jnp.array([200.0, 0.0])
    prices = jnp.array([100.0, 100.0])
    # Total Value = 20,000

    # Target: 50/50 -> 10,000 each -> 100 shares each
    weights = jnp.array([0.5, 0.5])

    mask = jnp.array(1.0)
    mask = jnp.array(1.0)
    costs = jnp.zeros(2)

    N_new, realized_cost = state_transition(N_prev, prices, weights, mask, costs)

    expected = jnp.array([100.0, 100.0])
    assert jnp.allclose(N_new, expected)


def test_state_transition_no_rebalance():
    N_prev = jnp.array([10.0, 10.0])
    prices = jnp.array([100.0, 100.0])
    weights = jnp.array([0.8, 0.2])  # Different target, sum=1.0

    # Mask = 0 (Hold)
    mask = jnp.array(0.0)
    costs = jnp.ones(2) * 10.0  # Costs shouldn't matter

    N_new, realized_cost = state_transition(N_prev, prices, weights, mask, costs)

    assert jnp.allclose(N_new, N_prev)
    assert jnp.allclose(realized_cost, 0.0)


def test_state_transition_costs():
    # 100 shares @ $100 = $10,000
    N_prev = jnp.array([100.0, 0.0])  # $10k in A
    prices = jnp.array([100.0, 100.0])

    # Switch completely to B
    weights = jnp.array([0.0, 1.0])  # Sums to 1.0

    mask = jnp.array(1.0)
    mask = jnp.array(1.0)
    costs_bps = jnp.ones(2) * 100.0  # 1% for both assets

    # Sell 100 A ($10k), Buy B.
    # Trade Volume approx: Sell $10k, Buy $10k -> $20k volume?
    # N_theo: 0 A, 100 B.
    # Delta: -100 A, +100 B.
    # Cost: |-100|*100*1bp + |100|*100*1bp
    # Cost = 10000 * 1% + 10000 * 1% = 100 + 100 = $200.

    # Net Val = 10000 - 200 = 9800.
    # N_rebal = 9800 * [0, 1] / 100 = [0, 98].

    N_new, realized_cost = state_transition(N_prev, prices, weights, mask, costs_bps)

    assert jnp.allclose(N_new[0], 0.0)
    assert jnp.allclose(N_new[1], 98.0, atol=0.2)  # Approx
    assert realized_cost > 100.0


def test_backtest_jit(sample_data):
    # Ensure it runs under JIT
    jit_backtest = jax.jit(backtest)

    pv, tc, holdings = jit_backtest(
        sample_data["N_0"],
        sample_data["open"],
        sample_data["close"],
        sample_data["target_weights"],
        sample_data["rebalancing_mask"],
        sample_data["transaction_costs_bps"],
    )

    assert pv.shape == (sample_data["T"],)
    assert tc.shape == (sample_data["T"],)
    assert holdings.shape == (sample_data["T"], sample_data["N"])


def test_backtest_cash_as_asset():
    # Emulate cash using an asset with constant price 1.0
    T, N = 5, 2
    # Asset 0 = Cash, Asset 1 = Stock
    prices = jnp.ones((T, N))
    prices = prices.at[:, 0].set(1.0)  # Cash stays 1.0
    prices = prices.at[:, 1].set(100.0)  # Stock stays 100.0

    open_prices = prices
    close_prices = prices

    # Start with 100k Cash, 0 Stock
    N_0 = jnp.array([100_000.0, 0.0])

    # Target: 50% Cash, 50% Stock
    weights = jnp.ones((T, N)) * 0.5  # Sums to 1.0

    mask = jnp.ones(T)
    costs_bps = jnp.zeros((T, N))

    pv, tc, holdings = backtest(
        N_0, open_prices, close_prices, weights, mask, costs_bps
    )

    # Value should remain 100k
    assert jnp.allclose(pv, 100_000.0)

    # Holdings: 50k cash, 500 stock
    # Cash shares = 50000
    # Stock shares = 500
    expected_cash = 50_000.0
    expected_stock = 500.0

    assert jnp.allclose(holdings[:, 0], expected_cash)
    assert jnp.allclose(holdings[:, 1], expected_stock)
