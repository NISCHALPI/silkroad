import jax
import jax.numpy as jnp
import pytest
from silkroad.functional.backtest import backtest, state_transition


@pytest.fixture
def sample_data():
    T, N = 10, 2
    N_0 = jnp.zeros(N)
    cash_0 = jnp.array(100_000.0)

    # Constant prices for simplicity in some tests
    open_prices = jnp.ones((T, N)) * 100.0
    close_prices = jnp.ones((T, N)) * 101.0

    # Equal weights
    weights = jnp.ones((T, N)) * 0.5

    # Rebalance every step
    mask = jnp.ones(T)

    # No costs
    costs_bps = jnp.zeros(T)

    return {
        "N_0": N_0,
        "cash_0": cash_0,
        "open": open_prices,
        "close": close_prices,
        "weights": weights,
        "mask": mask,
        "costs_bps": costs_bps,
        "T": T,
        "N": N,
    }


def test_state_transition_basic():
    # Initial state: $10k cash, no shares
    N_prev = jnp.array([0.0, 0.0])
    cash_prev = jnp.array(10_000.0)

    # Prices: $100 each
    prices = jnp.array([100.0, 100.0])

    # Target: 50/50
    weights = jnp.array([0.5, 0.5])

    # Rebalance, no costs
    mask = jnp.array(1.0)
    costs = jnp.array(0.0)

    N_new, cash_new, realized_cost = state_transition(
        N_prev, cash_prev, prices, weights, mask, costs
    )

    # Should buy 50 shares of each ($5000 each)
    expected_shares = jnp.array([50.0, 50.0])

    assert jnp.allclose(N_new, expected_shares)
    assert jnp.allclose(cash_new, 0.0, atol=1e-5)
    assert jnp.allclose(realized_cost, 0.0)


def test_state_transition_no_rebalance():
    N_prev = jnp.array([10.0, 10.0])
    cash_prev = jnp.array(1000.0)
    prices = jnp.array([100.0, 100.0])
    weights = jnp.array([0.8, 0.2])  # Different target

    # Mask = 0 (Hold)
    mask = jnp.array(0.0)
    costs = jnp.array(10.0)  # Costs shouldn't matter

    N_new, cash_new, realized_cost = state_transition(
        N_prev, cash_prev, prices, weights, mask, costs
    )

    assert jnp.allclose(N_new, N_prev)
    assert jnp.allclose(cash_new, cash_prev)
    assert jnp.allclose(realized_cost, 0.0)


def test_state_transition_costs():
    # $10k cash
    N_prev = jnp.array([0.0])
    cash_prev = jnp.array(10_000.0)
    prices = jnp.array([100.0])
    weights = jnp.array([1.0])
    mask = jnp.array(1.0)

    # 100 bps = 1% cost
    costs_bps = jnp.array(100.0)

    # Theoretical buy: $10k / $100 = 100 shares
    # Cost approx: 100 shares * $100 * 1% = $100
    # Net value: $9900
    # Actual buy: $9900 / $100 = 99 shares

    N_new, cash_new, realized_cost = state_transition(
        N_prev, cash_prev, prices, weights, mask, costs_bps
    )

    assert jnp.allclose(N_new, 99.0, atol=0.1)
    assert realized_cost > 0.0


def test_backtest_jit(sample_data):
    # Ensure it runs under JIT
    jit_backtest = jax.jit(backtest)

    pv, tc, holdings, cash = jit_backtest(
        sample_data["N_0"],
        sample_data["cash_0"],
        sample_data["open"],
        sample_data["close"],
        sample_data["weights"],
        sample_data["mask"],
        sample_data["costs_bps"],
    )

    assert pv.shape == (sample_data["T"],)
    assert tc.shape == (sample_data["T"],)
    assert holdings.shape == (sample_data["T"], sample_data["N"])
    assert cash.shape == (sample_data["T"],)


def test_backtest_logic(sample_data):
    # Simple case: Buy and Hold 50/50
    # Prices increase by 1% every day (open 100, close 101)
    # Rebalance only on day 0

    T, N = sample_data["T"], sample_data["N"]
    mask = jnp.zeros(T)
    mask = mask.at[0].set(1.0)

    pv, tc, holdings, cash = backtest(
        sample_data["N_0"],
        sample_data["cash_0"],
        sample_data["open"],
        sample_data["close"],
        sample_data["weights"],
        mask,
        sample_data["costs_bps"],
    )

    # Day 0: Buy 500 shares of each (Total $100k, $50k each asset, price $100)
    # Holdings should be constant after day 0
    expected_holdings = 500.0
    assert jnp.allclose(holdings[0], expected_holdings)
    assert jnp.allclose(holdings[-1], expected_holdings)

    # Day 0 Close Value: 1000 shares * $101 = $101,000
    assert jnp.allclose(pv[0], 101_000.0)

    # Cash should be 0 (fully invested)
    assert jnp.allclose(cash[-1], 0.0, atol=1e-5)


def test_backtest_costs(sample_data):
    # High turnover strategy with costs
    T, N = sample_data["T"], sample_data["N"]

    # Flip weights every day: [1, 0] -> [0, 1]
    weights = jnp.zeros((T, N))
    weights = weights.at[::2, 0].set(1.0)
    weights = weights.at[1::2, 1].set(1.0)

    # High costs: 100 bps (1%)
    costs_bps = jnp.ones(T) * 100.0

    pv, tc, holdings, cash = backtest(
        sample_data["N_0"],
        sample_data["cash_0"],
        sample_data["open"],
        sample_data["close"],
        weights,
        sample_data["mask"],  # Rebalance every day
        costs_bps,
    )

    # Costs should be positive every day
    assert jnp.all(tc > 0.0)

    # Portfolio value should decrease due to costs (prices are constant-ish)
    # Open 100, Close 101. Gain 1%. Cost 1% on turnover.
    # Turnover is 100% every day (sell all A, buy all B).
    # So roughly flat or slight loss depending on exact math.

    # Just check that costs are accumulating
    assert jnp.sum(tc) > 1000.0


def test_backtest_cash_handling():
    # Target weights sum to 0.5 -> 50% cash
    T, N = 5, 2
    N_0 = jnp.zeros(N)
    cash_0 = jnp.array(100_000.0)
    open_prices = jnp.ones((T, N)) * 100.0
    close_prices = jnp.ones((T, N)) * 100.0
    weights = jnp.ones((T, N)) * 0.25  # Sum = 0.5
    mask = jnp.ones(T)
    costs_bps = jnp.zeros(T)

    pv, tc, holdings, cash = backtest(
        N_0, cash_0, open_prices, close_prices, weights, mask, costs_bps
    )

    # Should have ~50k cash
    assert jnp.allclose(cash, 50_000.0, atol=1e-3)
    # Portfolio value should be stable at 100k
    assert jnp.allclose(pv, 100_000.0, atol=1e-3)
