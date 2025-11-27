import pytest
import jax
import jax.numpy as jnp
from silkroad.functional.paths import geometric_brownian_motion, heston_model


def test_gbm_shapes():
    n_paths = 10
    n_steps = 100
    init_price = 100.0
    drift = 0.05
    volatility = 0.2
    dt = 1.0 / 252.0
    key = jax.random.PRNGKey(0)

    prices = geometric_brownian_motion(
        n_paths, n_steps, init_price, drift, volatility, dt, key
    )

    assert prices.shape == (n_paths, n_steps)
    assert jnp.allclose(prices[:, 0], init_price)
    assert jnp.all(prices > 0)


def test_gbm_properties():
    # Verify statistical properties of GBM
    # Log returns should be normally distributed with:
    # Mean = (mu - 0.5 * sigma^2) * dt
    # Variance = sigma^2 * dt

    n_paths = 10000
    n_steps = 2  # Just one step needed for distribution check
    init_price = 100.0
    drift = 0.05
    volatility = 0.2
    dt = 1.0
    key = jax.random.PRNGKey(42)

    prices = geometric_brownian_motion(
        n_paths, n_steps, init_price, drift, volatility, dt, key
    )

    log_returns = jnp.log(prices[:, 1] / prices[:, 0])

    expected_mean = (drift - 0.5 * volatility**2) * dt
    expected_std = volatility * jnp.sqrt(dt)

    sample_mean = jnp.mean(log_returns)
    sample_std = jnp.std(log_returns)

    # Allow some tolerance for Monte Carlo error
    assert jnp.allclose(sample_mean, expected_mean, atol=0.01)
    assert jnp.allclose(sample_std, expected_std, atol=0.01)


def test_heston_shapes():
    n_paths = 10
    n_steps = 100
    init_price = 100.0
    mu = 0.05
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.7
    dt = 1.0 / 252.0
    key = jax.random.PRNGKey(0)

    prices, variances = heston_model(
        n_paths, n_steps, init_price, mu, v0, kappa, theta, sigma_v, rho, dt, key
    )

    assert prices.shape == (n_paths, n_steps)
    assert variances.shape == (n_paths, n_steps)
    assert jnp.allclose(prices[:, 0], init_price)
    assert jnp.allclose(variances[:, 0], v0)
    assert jnp.all(prices > 0)
    # Variance should be non-negative (truncation scheme ensures this)
    assert jnp.all(variances >= 0)


def test_heston_correlation():
    # Verify negative correlation between price returns and variance changes
    n_paths = 10000
    n_steps = 100
    init_price = 100.0
    mu = 0.0
    v0 = 0.04
    kappa = 1.0
    theta = 0.04
    sigma_v = 0.1
    rho = -0.9  # Strong negative correlation
    dt = 1.0 / 252.0
    key = jax.random.PRNGKey(1)

    prices, variances = heston_model(
        n_paths, n_steps, init_price, mu, v0, kappa, theta, sigma_v, rho, dt, key
    )

    # Calculate log returns and variance changes
    log_returns = jnp.diff(jnp.log(prices), axis=1)
    var_changes = jnp.diff(variances, axis=1)

    # Flatten to compute overall correlation
    corr = jnp.corrcoef(log_returns.flatten(), var_changes.flatten())[0, 1]

    # Should be close to rho, but not exact due to discretization and stochasticity
    # The correlation of the Brownian motions is rho.
    # The realized correlation of returns and variance changes should be negative.
    assert corr < -0.5


def test_mjd_shapes():
    from silkroad.functional.paths import merton_jump_diffusion

    n_paths = 10
    n_steps = 100
    init_price = 100.0
    drift = 0.05
    volatility = 0.2
    lambda_jump = 1.0
    mean_jump_size = -0.1
    std_jump_size = 0.1
    dt = 1.0 / 252.0
    key = jax.random.PRNGKey(0)

    prices = merton_jump_diffusion(
        n_paths,
        n_steps,
        init_price,
        drift,
        volatility,
        lambda_jump,
        mean_jump_size,
        std_jump_size,
        dt,
        key,
    )

    assert prices.shape == (n_paths, n_steps)
    assert jnp.allclose(prices[:, 0], init_price)
    assert jnp.all(prices > 0)


def test_ou_reversion():
    from silkroad.functional.paths import ornstein_uhlenbeck

    # Test that OU process stays near the mean theta
    n_paths = 1000
    n_steps = 200
    init_value = 0.05
    kappa = 5.0  # Strong mean reversion
    theta = 0.1  # Long run mean
    sigma = 0.05
    dt = 1.0 / 252.0
    key = jax.random.PRNGKey(0)

    paths = ornstein_uhlenbeck(
        n_paths, n_steps, init_value, kappa, theta, sigma, dt, key
    )

    assert paths.shape == (n_paths, n_steps)
    assert jnp.allclose(paths[:, 0], init_value)

    # Check that the final mean is closer to theta than init_value
    final_mean = jnp.mean(paths[:, -1])

    # Distance to theta should decrease
    init_dist = abs(init_value - theta)
    final_dist = abs(final_mean - theta)

    assert final_dist < init_dist
