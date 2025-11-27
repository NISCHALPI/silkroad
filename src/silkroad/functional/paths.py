"""Market Data Generators.

This module provides functions to generate synthetic asset price paths using
stochastic processes such as Geometric Brownian Motion (GBM) and the Heston
Stochastic Volatility model. These are useful for Monte Carlo simulations
and testing trading strategies.
"""

from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import linregress
from arch import arch_model


__all__ = [
    "estimate_drift_and_volatility",
    "geometric_brownian_motion",
    "estimate_heston_params",
    "heston_model",
    "estimate_mjd",
    "merton_jump_diffusion",
    "estimate_ou_params",
    "ornstein_uhlenbeck",
]


def estimate_drift_and_volatility(
    price_paths: jax.Array, dt: float
) -> Dict[str, jax.Array]:
    """Estimates drift and volatility from given price paths.

    Args:
        price_paths: A jax.Array of shape (n_steps,) containing asset prices.
        dt: Inverse of the number of time steps in a year (e.g., 1/252 for daily data).
    Returns:
        A dictionary with keys "drift" and "volatility" estimated from the price paths.
    """
    # Calculate log returns
    log_returns = jnp.diff(jnp.log(price_paths), axis=0)

    # Calcuate Annualized drift and volatility
    mean_log_return = jnp.mean(log_returns)
    var_log_return = jnp.var(log_returns)
    drift = mean_log_return / dt + 0.5 * var_log_return / dt
    volatility = jnp.sqrt(var_log_return / dt)
    return {"drift": drift, "volatility": volatility}


def geometric_brownian_motion(
    n_paths: int,
    n_steps: int,
    init_price: float,
    drift: float,
    volatility: float,
    dt: float,
    key: jax.Array,
) -> jax.Array:
    """Generates asset price paths using Geometric Brownian Motion (GBM).

    The GBM model is defined by the SDE:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    The analytical solution is:
        S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)

    Args:
        n_paths: Number of paths to simulate.
        n_steps: Number of time steps.
        init_price: Initial price S_0.
        drift: Annualized drift parameter (mu).
        volatility: Annualized volatility parameter (sigma).
        dt: Time step size (e.g., 1/252 for daily steps).
        key: JAX PRNG key.

    Returns:
        A jax.Array of shape (n_paths, n_steps) containing the price paths.
    """
    # Generate random Brownian increments
    # Shape: (n_paths, n_steps - 1)
    # We generate n_steps - 1 increments to go from t=0 to t=T
    # The first price is fixed at init_price
    dW = jax.random.normal(key, shape=(n_paths, n_steps - 1)) * jnp.sqrt(dt)

    # Calculate log returns
    # log(S_{t+1} / S_t) = (mu - 0.5 * sigma^2) * dt + sigma * dW_t
    log_returns = (drift - 0.5 * volatility**2) * dt + volatility * dW

    # Cumulative sum to get log price path relative to S_0
    # Shape: (n_paths, n_steps - 1)
    log_price_path = jnp.cumsum(log_returns, axis=1)

    # Prepend 0 to log_price_path to represent S_0 (log(S_0/S_0) = 0)
    # Shape: (n_paths, n_steps)
    zeros = jnp.zeros((n_paths, 1))
    log_price_path = jnp.concatenate([zeros, log_price_path], axis=1)

    # Convert back to prices
    prices = init_price * jnp.exp(log_price_path)

    return prices


def estimate_heston_params(prices: jax.Array, dt: float) -> Dict[str, jax.Array]:
    """Estimates Heston model parameters from given price paths.

    Args:
        prices: A jax.Array of shape (n_steps,) containing asset prices.
        dt: Time step size.
    Returns:
        A tuple (mu, v0, kappa, theta, sigma_v) estimated from
        the price paths.
    """
    returns = 100 * jnp.diff(jnp.log(prices))  # Arch prefers percentage inputs

    # Fit GARCH(1,1) with asymmetry (GJR-GARCH) to capture leverage
    garch = arch_model(np.array(returns), vol="Garch", p=1, o=1, q=1)  # type: ignore
    res = garch.fit(disp="off")

    # Extract GARCH parameters to map to Heston (Heuristic Mapping)
    # omega (constant), alpha (shock), beta (persistence), gamma (asymmetry)
    omega = res.params["omega"] / 10000  # Scaling back down
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    gamma = res.params["gamma[1]"]

    # Persistence for GJR-GARCH = alpha + beta + gamma/2
    persistence = alpha + beta + 0.5 * gamma

    # Heston Theta (Long Run Variance) = omega / (1 - persistence)
    # This gives variance per step. We need annualized variance.
    long_run_var = omega / (1 - persistence)
    theta = long_run_var / dt

    # Heston Kappa (Mean Reversion Speed)
    # Kappa ~ (1 - persistence) / dt
    kappa = (1 - persistence) / dt

    # Heston Vol of Vol (sigma_v)
    # This is rough, often set mathematically relative to Kurtosis,
    # but for simple sims, 0.2 to 0.5 is standard. Let's calculate standard dev of conditional vol.
    cond_vol = res.conditional_volatility / 100
    sigma_v = jnp.std(cond_vol) * jnp.sqrt(1 / dt)  # type: ignore

    # Heston Correlation (rho)
    # Correlation between price returns and volatility changes
    vol_changes = jnp.diff(cond_vol)  # type: ignore
    price_returns = jnp.diff(jnp.log(prices))[-len(vol_changes) :]
    rho = jnp.corrcoef(price_returns, vol_changes)[0, 1]

    # Current Variance (v0)
    v0 = cond_vol[-1] ** 2

    # Drift (mu)
    mu = jnp.mean(price_returns) / dt

    return {
        "mu": mu,
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho,
        "init_price": prices[-1],
    }


def heston_model(
    n_paths: int,
    n_steps: int,
    init_price: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    dt: float,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Generates asset price paths using the Heston Stochastic Volatility model.

    The Heston model is defined by the system of SDEs:
        dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW^S_t
        dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW^v_t

    where dW^S_t and dW^v_t are correlated Brownian motions with correlation rho.

    Args:
        n_paths: Number of paths to simulate.
        n_steps: Number of time steps.
        init_price: Initial price S_0.
        mu: Drift of the asset price.
        v0: Initial variance.
        kappa: Rate of mean reversion for variance.
        theta: Long-run average variance.
        sigma_v: Volatility of variance (vol of vol).
        rho: Correlation between asset and variance Brownian motions.
        dt: Time step size.
        key: JAX PRNG key.

    Returns:
        A tuple (prices, variances):
            - prices: jax.Array of shape (n_paths, n_steps).
            - variances: jax.Array of shape (n_paths, n_steps).
    """
    # 1. Generate correlated Brownian motions
    key_z1, key_z2 = jax.random.split(key)
    z1 = jax.random.normal(key_z1, shape=(n_paths, n_steps - 1))
    z2 = jax.random.normal(key_z2, shape=(n_paths, n_steps - 1))

    # Construct correlated shocks
    # dW_S = z1 * sqrt(dt)
    # dW_v = (rho * z1 + sqrt(1 - rho^2) * z2) * sqrt(dt)
    dW_S = z1 * jnp.sqrt(dt)
    dW_v = (rho * z1 + jnp.sqrt(1 - rho**2) * z2) * jnp.sqrt(dt)

    # 2. Simulation Loop (Euler-Maruyama with truncation for variance)
    # We use scan for the time loop because variance depends on previous variance

    def step_fn(carry, inputs):
        # carry: (current_log_price, current_variance)
        # inputs: (dW_S_t, dW_v_t)
        log_S, v = carry
        dW_S_t, dW_v_t = inputs

        # Ensure variance is positive (Full Truncation Scheme)
        v_pos = jnp.maximum(v, 0.0)
        sqrt_v = jnp.sqrt(v_pos)

        # Update Log Price
        # d(log S) = (mu - 0.5 * v) * dt + sqrt(v) * dW_S
        new_log_S = log_S + (mu - 0.5 * v_pos) * dt + sqrt_v * dW_S_t

        # Update Variance
        # dv = kappa * (theta - v) * dt + sigma_v * sqrt(v) * dW_v
        new_v = v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * dW_v_t

        return (new_log_S, new_v), (new_log_S, new_v)

    init_log_price = jnp.log(init_price) * jnp.ones(n_paths)
    init_variance = v0 * jnp.ones(n_paths)
    init_carry = (init_log_price, init_variance)

    # Scan over time steps
    # Transpose inputs to (n_steps-1, n_paths) for scan to iterate over time
    inputs = (dW_S.T, dW_v.T)

    _, (log_prices, variances) = jax.lax.scan(step_fn, init_carry, inputs)

    # Transpose back to (n_paths, n_steps-1)
    log_prices = log_prices.T
    variances = variances.T

    # Prepend initial values
    log_prices = jnp.concatenate([init_log_price[:, None], log_prices], axis=1)
    variances = jnp.concatenate([init_variance[:, None], variances], axis=1)

    prices = jnp.exp(log_prices)

    return prices, variances


def estimate_mjd(prices: jax.Array, dt: float) -> Dict[str, jax.Array]:
    """Estimates Merton Jump Diffusion (MJD) parameters from given price paths.

    Args:
        prices: A jax.Array of shape (n_steps,) containing asset prices.
        dt: Time step size.
    Returns:
        A dictionary with keys "drift", "sigma", "lambda_jump", "mean_jump_size", and "std_jump_size"
        estimated from the price paths.
    """
    log_returns = jnp.diff(jnp.log(prices))

    # Initial estimate of vol
    std_dev = jnp.std(log_returns)
    mean_ret = jnp.mean(log_returns)

    # Threshold for jumps (e.g., 3 standard deviations)
    jump_threshold = 3 * std_dev

    # Separate Jumps from Normal Returns
    jumps = log_returns[jnp.abs(log_returns - mean_ret) > jump_threshold]
    normal_returns = log_returns[jnp.abs(log_returns - mean_ret) <= jump_threshold]

    # 1. Diffusion Parameters (from normal returns)
    sigma = jnp.std(normal_returns) / jnp.sqrt(dt)
    # Drift adjustment (complex, but simple approximation is annual mean)
    drift = jnp.mean(normal_returns) / dt + 0.5 * sigma**2

    # 2. Jump Parameters
    lambda_jump = len(jumps) / (len(log_returns) * dt)  # Jumps per year
    lambda_jump = jnp.array(lambda_jump)

    if len(jumps) > 0:
        mean_jump_size = jnp.mean(jumps)
        std_jump_size = jnp.std(jumps)
    else:
        # Default if no jumps found in history
        mean_jump_size = jnp.array(0.0)
        std_jump_size = jnp.array(0.0)

    return {
        "drift": drift,
        "sigma": sigma,
        "lambda_jump": lambda_jump,
        "mean_jump_size": mean_jump_size,
        "std_jump_size": std_jump_size,
    }


def merton_jump_diffusion(
    n_paths: int,
    n_steps: int,
    init_price: float,
    drift: float,
    volatility: float,
    lambda_jump: float,
    mean_jump_size: float,
    std_jump_size: float,
    dt: float,
    key: jax.Array,
) -> jax.Array:
    """Generates asset price paths using Merton Jump Diffusion (MJD) model.

    The MJD model extends GBM with a Poisson jump process:
        dS_t/S_t = (mu - lambda * k) * dt + sigma * dW_t + (Y - 1) * dN_t

    where:
        - dN_t is a Poisson process with intensity lambda.
        - Y is the jump size, log(Y) ~ N(mean_jump_size, std_jump_size^2).
        - k = E[Y - 1] = exp(mean_jump_size + 0.5 * std_jump_size^2) - 1.

    Args:
        n_paths: Number of paths to simulate.
        n_steps: Number of time steps.
        init_price: Initial price S_0.
        drift: Annualized drift parameter (mu).
        volatility: Annualized volatility parameter (sigma).
        lambda_jump: Expected number of jumps per year.
        mean_jump_size: Mean of log jump size.
        std_jump_size: Std dev of log jump size.
        dt: Time step size.
        key: JAX PRNG key.

    Returns:
        A jax.Array of shape (n_paths, n_steps) containing the price paths.
    """
    key_diff, key_jump_num, key_jump_size = jax.random.split(key, 3)

    # 1. Diffusion Component (GBM)
    # dW ~ N(0, dt)
    dW = jax.random.normal(key_diff, shape=(n_paths, n_steps - 1)) * jnp.sqrt(dt)

    # 2. Jump Component
    # Number of jumps in each step ~ Poisson(lambda * dt)
    n_jumps = jax.random.poisson(
        key_jump_num, lambda_jump * dt, shape=(n_paths, n_steps - 1)
    )

    # Total jump size in log terms for each step
    # We need to sum 'n_jumps' random variables for each step.
    # Since n_jumps is small and random, it's easier to approximate or use a trick.
    # However, a common approximation for small dt is that there is at most 1 jump.
    # But for correctness, we can simulate the sum of normals.
    # Sum of N normals N(m, s^2) is N(N*m, N*s^2).
    # So given n_jumps = k, the log jump size is N(k * mean_jump, k * std_jump^2).

    # We can generate a standard normal Z_J and scale it.
    Z_J = jax.random.normal(key_jump_size, shape=(n_paths, n_steps - 1))

    # Total log jump size = n_jumps * mean_jump_size + sqrt(n_jumps) * std_jump_size * Z_J
    log_jump_factor = n_jumps * mean_jump_size + jnp.sqrt(n_jumps) * std_jump_size * Z_J

    # 3. Combine
    # k = E[Y] - 1
    k = jnp.exp(mean_jump_size + 0.5 * std_jump_size**2) - 1.0

    # Drift adjustment for jumps
    # log(S_{t+1}/S_t) = (mu - lambda * k - 0.5 * sigma^2) * dt + sigma * dW + log_jump_factor
    drift_correction = lambda_jump * k

    log_returns = (
        (drift - drift_correction - 0.5 * volatility**2) * dt
        + volatility * dW
        + log_jump_factor
    )

    # Cumulative sum
    log_price_path = jnp.cumsum(log_returns, axis=1)

    # Prepend 0
    zeros = jnp.zeros((n_paths, 1))
    log_price_path = jnp.concatenate([zeros, log_price_path], axis=1)

    prices = init_price * jnp.exp(log_price_path)

    return prices


def estimate_ou_params(paths: jax.Array, dt: float) -> Dict[str, jax.Array]:
    """Estimates Ornstein-Uhlenbeck (OU) process parameters from given paths.

    The OU process is defined by the SDE:
        dX_t = kappa * (theta - X_t) * dt + sigma * dW_t

    Args:
        paths: A jax.Array of shape (n_steps,) containing the OU process paths.
        dt: Time step size.
    Returns:
        A dictionary with keys "kappa", "theta", and "sigma" estimated from the paths.
    """
    x_t = paths[:-1]
    x_t1 = paths[1:]

    slope, intercept, r_value, p_value, std_err = linregress(x_t, x_t1)
    # Convert to jax arrays
    slope = jnp.array(slope)
    intercept = jnp.array(intercept)
    std_err = jnp.array(std_err)

    # Solve for OU parameters
    # slope = exp(-kappa * dt)  => kappa = -ln(slope) / dt
    kappa = -jnp.log(slope) / dt

    # theta = intercept / (1 - slope)
    theta = intercept / (1 - slope)

    # Residuals to estimate sigma
    residuals = x_t1 - (slope * x_t + intercept)
    variance = jnp.var(residuals)
    sigma = jnp.sqrt(variance * 2 * kappa / (1 - jnp.exp(-2 * kappa * dt)))

    return {"kappa": kappa, "theta": theta, "sigma": sigma}


def ornstein_uhlenbeck(
    n_paths: int,
    n_steps: int,
    init_value: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    key: jax.Array,
) -> jax.Array:
    """Generates paths using the Ornstein-Uhlenbeck (OU) process.

    The OU process is defined by the SDE:
        dX_t = kappa * (theta - X_t) * dt + sigma * dW_t

    This is a mean-reverting process.

    Args:
        n_paths: Number of paths to simulate.
        n_steps: Number of time steps.
        init_value: Initial value X_0.
        kappa: Rate of mean reversion.
        theta: Long-run mean.
        sigma: Volatility parameter.
        dt: Time step size.
        key: JAX PRNG key.

    Returns:
        A jax.Array of shape (n_paths, n_steps) containing the paths.
    """
    # Exact solution discretization:
    # X_{t+1} = X_t * exp(-kappa * dt) + theta * (1 - exp(-kappa * dt)) + sigma * sqrt((1 - exp(-2*kappa*dt)) / (2*kappa)) * Z
    # This is exact for any dt.

    decay = jnp.exp(-kappa * dt)
    mean_reversion = theta * (1.0 - decay)
    std_dev = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    # Generate random shocks
    Z = jax.random.normal(key, shape=(n_paths, n_steps - 1))

    def step_fn(current_x, z):
        next_x = current_x * decay + mean_reversion + std_dev * z
        return next_x, next_x

    init_x = init_value * jnp.ones(n_paths)

    # Scan
    # Transpose Z to iterate over time
    _, path = jax.lax.scan(step_fn, init_x, Z.T)

    # Transpose back
    path = path.T

    # Prepend initial value
    path = jnp.concatenate([init_x[:, None], path], axis=1)

    return path
