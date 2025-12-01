"""Market Data Generators.

This module provides functions for Monte Carlo simulation of asset price paths
using various stochastic models commonly used in quantitative finance. It includes
both parameter estimation from historical data and path generation capabilities.

Stochastic Models Supported:
    - **Geometric Brownian Motion (GBM)**: The classical model for equity prices
      assuming constant drift and volatility.
    - **Heston Stochastic Volatility**: Captures volatility clustering and
      mean-reversion in variance, with correlation between price and volatility.
    - **Merton Jump Diffusion (MJD)**: Extends GBM with Poisson-driven jumps
      to model sudden price movements (e.g., earnings surprises, crashes).
    - **Ornstein-Uhlenbeck (OU)**: Mean-reverting process suitable for
      interest rates, spreads, or volatility modeling.

Bootstrapping Methods:
    - **Moving Block Bootstrap (MBB)**: Resamples contiguous blocks of returns
      to preserve short-term autocorrelation structure.
    - **Circular Block Bootstrap (CBB)**: Similar to MBB but wraps around
      the data to eliminate boundary effects.
    - **Stationary Bootstrap (SB)**: Uses geometrically distributed block sizes
      for stationarity-preserving resampling.

Example:
    Simulate GBM paths from historical data::

        import jax
        import jax.numpy as jnp

        # Estimate parameters from historical prices
        params = estimate_drift_and_volatility(prices, dt=1/252)

        # Generate 1000 paths of 252 steps (1 year daily)
        key = jax.random.PRNGKey(42)
        paths = geometric_brownian_motion(
            n_paths=1000,
            n_steps=252,
            init_price=100.0,
            drift=params["drift"],
            volatility=params["volatility"],
            dt=1/252,
            key=key,
        )

Note:
    All functions are designed to be JIT-compilable with JAX for GPU/TPU
    acceleration. Parameter estimation functions may use NumPy/SciPy internally
    but return JAX arrays for seamless integration with path generators.

See Also:
    - :mod:`silkroad.functional.metrics`: Risk and performance metrics.
    - :mod:`silkroad.functional.backtest`: Backtesting utilities.
"""

from typing import Tuple, Dict, Union
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import linregress
from arch import arch_model


__all__ = [
    "estimate_drift_and_volatility",
    "geometric_brownian_motion",
    "estimate_multivariate_gbm_params",
    "multivariate_geometric_brownian_motion",
    "estimate_heston_params",
    "heston_model",
    "estimate_mjd",
    "merton_jump_diffusion",
    "estimate_ou_params",
    "ornstein_uhlenbeck",
    "moving_block_bootstrap",
    "circular_block_bootstrap",
    "stationary_bootstrap",
]

# =============================================================================
# Statistical Estimators And Path Generators
# =============================================================================
# These functions estimate parameters from historical data and generate
# synthetic paths based on stochastic models.


def estimate_drift_and_volatility(
    price_paths: jax.Array, dt: float
) -> Dict[str, jax.Array]:
    """Estimate annualized drift and volatility from historical price data.

    Computes maximum likelihood estimates of the drift ($\\mu$) and volatility
    ($\\sigma$) parameters for a Geometric Brownian Motion model from observed
    price data.

    The estimation is based on log returns:

    $$\\hat{\\sigma} = \\sqrt{\\frac{\\text{Var}(r)}{\\Delta t}}$$

    $$\\hat{\\mu} = \\frac{\\mathbb{E}[r]}{\\Delta t} + \\frac{1}{2}\\hat{\\sigma}^2$$

    where $r = \\log(S_{t+1}/S_t)$ are the log returns.

    Args:
        price_paths: A JAX array of shape ``(n_steps,)`` containing consecutive
            asset prices. Must have at least 2 elements to compute returns.
        dt: Time step size as a fraction of a year. For daily data with 252
            trading days, use ``dt=1/252``. For monthly data, use ``dt=1/12``.

    Returns:
        A dictionary containing:
            - ``"drift"``: Annualized drift parameter (float).
            - ``"volatility"``: Annualized volatility parameter (float).

    Example:
        >>> prices = jnp.array([100.0, 101.5, 99.8, 102.3, 103.1])
        >>> params = estimate_drift_and_volatility(prices, dt=1/252)
        >>> print(f"Drift: {params['drift']:.4f}, Vol: {params['volatility']:.4f}")

    Note:
        The drift estimate includes the Itô correction term ($+0.5\\sigma^2$)
        to account for the difference between arithmetic and geometric means
        in continuous-time finance.
    """
    # Calculate log returns
    log_returns = jnp.diff(jnp.log(price_paths), axis=0)

    # Calcuate Annualized drift and volatility
    mean_log_return = jnp.mean(log_returns)
    var_log_return = jnp.var(log_returns)
    drift = mean_log_return / dt + 0.5 * var_log_return / dt
    volatility = jnp.sqrt(var_log_return / dt)
    return {"drift": drift, "volatility": volatility}


@partial(jax.jit, static_argnames=["n_paths", "n_steps"])
def geometric_brownian_motion(
    n_paths: int,
    n_steps: int,
    init_price: float,
    drift: float,
    volatility: float,
    dt: float,
    key: jax.Array,
) -> jax.Array:
    """Generate asset price paths using Geometric Brownian Motion (GBM).

    Simulates price paths following the classical GBM stochastic differential
    equation widely used for equity price modeling:

    $$dS_t = \\mu S_t \\, dt + \\sigma S_t \\, dW_t$$

    The analytical solution used for simulation is:

    $$S_t = S_0 \\exp\\left((\\mu - \\frac{1}{2}\\sigma^2)t + \\sigma W_t\\right)$$

    This implementation uses exact discretization (not Euler-Maruyama) so there
    is no discretization error regardless of the time step size.

    Args:
        n_paths: Number of independent Monte Carlo paths to simulate.
        n_steps: Number of time steps per path (including the initial price).
        init_price: Initial asset price $S_0$ at time $t=0$.
        drift: Annualized drift parameter $\\mu$ (expected return).
        volatility: Annualized volatility parameter $\\sigma$ (standard deviation
            of returns).
        dt: Time step size as a fraction of a year. For daily steps with 252
            trading days, use ``dt=1/252``.
        key: JAX PRNG key for reproducible random number generation.
            Use ``jax.random.PRNGKey(seed)`` to create.

    Returns:
        A JAX array of shape ``(n_paths, n_steps)`` containing simulated price
        paths. Each row is an independent path, and columns represent time steps.

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> paths = geometric_brownian_motion(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     drift=0.08,
        ...     volatility=0.20,
        ...     dt=1/252,
        ...     key=key,
        ... )
        >>> print(paths.shape)  # (1000, 252)

    See Also:
        - :func:`estimate_drift_and_volatility`: Estimate GBM parameters from data.
        - :func:`heston_model`: For stochastic volatility modeling.
        - :func:`merton_jump_diffusion`: For jump-augmented GBM.
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


@partial(jax.jit, static_argnames=["n_paths", "n_steps"])
def multivariate_geometric_brownian_motion(
    n_paths: int,
    n_steps: int,
    init_prices: jax.Array,
    drift: jax.Array,
    cov_matrix: jax.Array,
    dt: float,
    key: jax.Array,
) -> jax.Array:
    """Generate correlated asset price paths using Multivariate Geometric Brownian Motion.

    Simulates multiple asset price paths that are correlated according to a
    covariance matrix. This is essential for portfolio modeling where
    inter-asset correlations significantly impact risk and return.

    The model for $N$ assets is:

    $$dS_t^i = \\mu_i S_t^i \\, dt + S_t^i \\sum_{j=1}^N L_{ij} \\, dW_t^j$$

    where $L$ is the Cholesky factor of the covariance matrix $\\Sigma = LL^T$.

    Args:
        n_paths: Number of independent scenarios (Monte Carlo paths) to generate.
        n_steps: Number of time steps per path (including initial prices).
        init_prices: JAX array of shape ``(n_assets,)`` with initial prices $S_0$.
        drift: JAX array of shape ``(n_assets,)`` with annualized drifts $\\mu$.
        cov_matrix: JAX array of shape ``(n_assets, n_assets)`` representing
            the annualized covariance matrix of log returns.
        dt: Time step size as a fraction of a year.
        key: JAX PRNG key for random number generation.

    Returns:
        A JAX array of shape ``(n_paths, n_steps, n_assets)`` containing
        simulated price paths for all assets in each scenario.
    """
    n_assets = len(init_prices)

    # Cholesky decomposition of covariance matrix: Sigma = L @ L.T
    # We need L to correlate the standard normal shocks
    L = jnp.linalg.cholesky(cov_matrix)

    # Generate independent standard normal random variables
    # Shape: (n_paths, n_steps - 1, n_assets)
    Z = jax.random.normal(key, shape=(n_paths, n_steps - 1, n_assets))

    # Correlate the shocks: dW_correlated = Z @ L.T
    # Shape: (n_paths, n_steps - 1, n_assets)
    dW = jnp.matmul(Z, L.T) * jnp.sqrt(dt)

    # Drift term: (mu - 0.5 * sigma^2) * dt
    # sigma^2 is the diagonal of the covariance matrix
    variances = jnp.diag(cov_matrix)
    drift_term = (drift - 0.5 * variances) * dt

    # Calculate log returns
    # Shape: (n_paths, n_steps - 1, n_assets)
    log_returns = drift_term + dW

    # Accumulate log returns to get log prices
    # Shape: (n_paths, n_steps, n_assets)
    # Prepend zeros for the initial step
    zeros = jnp.zeros((n_paths, 1, n_assets))
    accumulated_log_returns = jnp.cumsum(
        jnp.concatenate([zeros, log_returns], axis=1), axis=1
    )

    # Convert to prices: S_t = S_0 * exp(sum(log_returns))
    prices = init_prices * jnp.exp(accumulated_log_returns)

    return prices


def estimate_multivariate_gbm_params(
    price_paths: jax.Array, dt: float
) -> Dict[str, jax.Array]:
    """Estimate parameters for Multivariate GBM from historical price data.

    Computes the drift vector and covariance matrix of log returns.

    Args:
        price_paths: JAX array of shape ``(n_steps, n_assets)`` containing
            historical prices.
        dt: Time step size.

    Returns:
        Dictionary containing:
            - ``"drift"``: JAX array of shape ``(n_assets,)``.
            - ``"cov_matrix"``: JAX array of shape ``(n_assets, n_assets)``.
    """
    # Calculate log returns: (n_steps-1, n_assets)
    log_returns = jnp.diff(jnp.log(price_paths), axis=0)

    # Mean and Covariance of log returns
    mean_log_returns = jnp.mean(log_returns, axis=0)
    cov_log_returns = jnp.cov(log_returns, rowvar=False)

    # Annualize
    # Drift = (mean + 0.5 * var) / dt
    # Note: cov_log_returns diagonal is variance
    variances = jnp.diag(cov_log_returns)
    drift = mean_log_returns / dt + 0.5 * variances / dt

    cov_matrix = cov_log_returns / dt

    return {"drift": drift, "cov_matrix": cov_matrix}


def estimate_heston_params(prices: jax.Array, dt: float) -> Dict[str, jax.Array]:
    """Estimate Heston stochastic volatility model parameters from price data.

    Uses a GARCH(1,1) model with leverage effects (GJR-GARCH) as a proxy to
    estimate the continuous-time Heston model parameters. This is a heuristic
    mapping that captures the key dynamics of volatility clustering and
    mean reversion.

    The Heston model parameters estimated are:
        - $\\mu$: Drift of the asset price
        - $v_0$: Initial variance (from conditional volatility)
        - $\\kappa$: Speed of mean reversion
        - $\\theta$: Long-run variance
        - $\\sigma_v$: Volatility of variance (vol-of-vol)
        - $\\rho$: Correlation between price and variance innovations

    Args:
        prices: A JAX array of shape ``(n_steps,)`` containing consecutive
            asset prices. Should have sufficient history (typically 250+ days)
            for reliable GARCH estimation.
        dt: Time step size as a fraction of a year. For daily data with 252
            trading days, use ``dt=1/252``.

    Returns:
        A dictionary containing:
            - ``"mu"``: Annualized drift estimate.
            - ``"v0"``: Initial variance (squared conditional volatility).
            - ``"kappa"``: Mean reversion speed.
            - ``"theta"``: Long-run average variance.
            - ``"sigma_v"``: Volatility of variance.
            - ``"rho"``: Price-volatility correlation (leverage effect).
            - ``"init_price"``: Last observed price (for simulation start).

    Example:
        >>> params = estimate_heston_params(historical_prices, dt=1/252)
        >>> prices, variances = heston_model(
        ...     n_paths=1000, n_steps=252, **params, dt=1/252, key=key
        ... )

    Note:
        The mapping from GARCH to Heston parameters is approximate. The
        Feller condition ($2\\kappa\\theta > \\sigma_v^2$) is not enforced,
        so variance paths may become negative. The simulation uses a
        truncation scheme to handle this.

    See Also:
        - :func:`heston_model`: Generate paths using estimated parameters.
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


@partial(jax.jit, static_argnames=["n_paths", "n_steps"])
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
    """Generate asset price paths using the Heston stochastic volatility model.

    Simulates paths from the Heston (1993) model, which extends GBM with
    stochastic variance that exhibits mean-reversion and correlation with
    the asset price (leverage effect).

    The model is defined by the coupled SDEs:

    $$dS_t = \\mu S_t \\, dt + \\sqrt{v_t} S_t \\, dW^S_t$$

    $$dv_t = \\kappa(\\theta - v_t) \\, dt + \\sigma_v \\sqrt{v_t} \\, dW^v_t$$

    where $\\text{Corr}(dW^S_t, dW^v_t) = \\rho$.

    The simulation uses Euler-Maruyama discretization with full truncation
    to ensure non-negative variance (variance is floored at zero when computing
    the square root).

    Args:
        n_paths: Number of independent Monte Carlo paths to simulate.
        n_steps: Number of time steps per path (including initial values).
        init_price: Initial asset price $S_0$ at time $t=0$.
        mu: Annualized drift of the asset price.
        v0: Initial variance $v_0$ at time $t=0$.
        kappa: Speed of mean reversion for the variance process. Higher values
            mean faster reversion to $\\theta$.
        theta: Long-run average variance (mean-reversion level).
        sigma_v: Volatility of variance (vol-of-vol). Controls the randomness
            in the variance process.
        rho: Correlation between the asset and variance Brownian motions.
            Typically negative for equities (leverage effect).
        dt: Time step size as a fraction of a year.
        key: JAX PRNG key for reproducible random number generation.

    Returns:
        A tuple ``(prices, variances)`` where:
            - ``prices``: JAX array of shape ``(n_paths, n_steps)`` with
              simulated asset prices.
            - ``variances``: JAX array of shape ``(n_paths, n_steps)`` with
              simulated variance paths (can be used for VIX-like analysis).

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> prices, variances = heston_model(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     mu=0.05,
        ...     v0=0.04,
        ...     kappa=2.0,
        ...     theta=0.04,
        ...     sigma_v=0.3,
        ...     rho=-0.7,
        ...     dt=1/252,
        ...     key=key,
        ... )

    Note:
        The Feller condition $2\\kappa\\theta > \\sigma_v^2$ ensures variance
        stays strictly positive in continuous time. With discrete simulation,
        variance may still go negative; this implementation uses truncation
        (flooring at zero) to handle such cases.

    See Also:
        - :func:`estimate_heston_params`: Estimate parameters from data.
        - :func:`geometric_brownian_motion`: Simpler constant-volatility model.
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
    """Estimate Merton Jump Diffusion model parameters from price data.

    Separates historical returns into a continuous diffusion component and
    discrete jumps using a threshold-based approach. Returns exceeding 3
    standard deviations are classified as jumps.

    The MJD model extends GBM with compound Poisson jumps:

    $$\\frac{dS_t}{S_t} = (\\mu - \\lambda k) \\, dt + \\sigma \\, dW_t + (Y-1) \\, dN_t$$

    where $N_t$ is a Poisson process and $\\log(Y) \\sim N(m_J, \\sigma_J^2)$.

    Args:
        prices: A JAX array of shape ``(n_steps,)`` containing consecutive
            asset prices. Should have sufficient history to capture jump events.
        dt: Time step size as a fraction of a year. For daily data with 252
            trading days, use ``dt=1/252``.

    Returns:
        A dictionary containing:
            - ``"drift"``: Annualized diffusion drift $\\mu$.
            - ``"sigma"``: Annualized diffusion volatility $\\sigma$.
            - ``"lambda_jump"``: Expected number of jumps per year $\\lambda$.
            - ``"mean_jump_size"``: Mean of log jump size $m_J$.
            - ``"std_jump_size"``: Standard deviation of log jump size $\\sigma_J$.

    Example:
        >>> params = estimate_mjd(historical_prices, dt=1/252)
        >>> paths = merton_jump_diffusion(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     **params,
        ...     dt=1/252,
        ...     key=key,
        ... )

    Note:
        The threshold-based separation is a heuristic. In practice, more
        sophisticated methods (e.g., maximum likelihood, filtering) may
        provide better parameter estimates for jump-diffusion models.

    See Also:
        - :func:`merton_jump_diffusion`: Generate paths with estimated parameters.
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
    """Generate asset price paths using the Merton Jump Diffusion model.

    Simulates paths from Merton's (1976) jump-diffusion model, which augments
    Geometric Brownian Motion with a compound Poisson process to capture
    sudden, discontinuous price movements.

    The model is defined by:

    $$\\frac{dS_t}{S_t} = (\\mu - \\lambda k) \\, dt + \\sigma \\, dW_t + (Y-1) \\, dN_t$$

    where:
        - $N_t$ is a Poisson process with intensity $\\lambda$
        - $\\log(Y) \\sim N(m_J, \\sigma_J^2)$ is the log jump size
        - $k = \\mathbb{E}[Y-1] = e^{m_J + \\sigma_J^2/2} - 1$ is the drift adjustment

    Args:
        n_paths: Number of independent Monte Carlo paths to simulate.
        n_steps: Number of time steps per path (including initial price).
        init_price: Initial asset price $S_0$ at time $t=0$.
        drift: Annualized drift parameter $\\mu$ (before jump adjustment).
        volatility: Annualized diffusion volatility $\\sigma$.
        lambda_jump: Expected number of jumps per year $\\lambda$ (Poisson intensity).
        mean_jump_size: Mean of log jump size $m_J$. Negative values model
            downward jumps (crashes), positive values model upward jumps.
        std_jump_size: Standard deviation of log jump size $\\sigma_J$.
            Controls jump size dispersion.
        dt: Time step size as a fraction of a year.
        key: JAX PRNG key for reproducible random number generation.

    Returns:
        A JAX array of shape ``(n_paths, n_steps)`` containing simulated
        price paths with both continuous and jump components.

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> paths = merton_jump_diffusion(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     drift=0.08,
        ...     volatility=0.20,
        ...     lambda_jump=2.0,  # ~2 jumps per year
        ...     mean_jump_size=-0.05,  # 5% average downward jump
        ...     std_jump_size=0.10,
        ...     dt=1/252,
        ...     key=key,
        ... )

    Note:
        When ``lambda_jump * dt`` is small (typical for daily steps), the
        probability of multiple jumps per step is negligible. The implementation
        correctly handles the case of multiple jumps using a sum of normals
        approximation.

    See Also:
        - :func:`estimate_mjd`: Estimate parameters from historical data.
        - :func:`geometric_brownian_motion`: Jump-free version.
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
    """Estimate Ornstein-Uhlenbeck process parameters from observed paths.

    Uses ordinary least squares regression on the discretized OU process to
    estimate the mean-reversion parameters. The discrete-time AR(1) model
    is fitted and parameters are converted to continuous-time equivalents.

    The OU process is defined by:

    $$dX_t = \\kappa(\\theta - X_t) \\, dt + \\sigma \\, dW_t$$

    The discrete relationship is:

    $$X_{t+1} = e^{-\\kappa \\Delta t} X_t + \\theta(1 - e^{-\\kappa \\Delta t}) + \\epsilon_t$$

    Args:
        paths: A JAX array of shape ``(n_steps,)`` containing observed values
            of the mean-reverting process.
        dt: Time step size as a fraction of a year.

    Returns:
        A dictionary containing:
            - ``"kappa"``: Speed of mean reversion. Higher values mean faster
              reversion to the mean.
            - ``"theta"``: Long-run mean (equilibrium level).
            - ``"sigma"``: Volatility parameter.

    Example:
        >>> # Estimate parameters from historical spread data
        >>> params = estimate_ou_params(spread_series, dt=1/252)
        >>> # Half-life of mean reversion (in years)
        >>> half_life = jnp.log(2) / params["kappa"]

    Note:
        This estimation assumes the process is stationary and observed without
        noise. For short time series or non-stationary data, consider using
        maximum likelihood estimation or Kalman filtering.

    See Also:
        - :func:`ornstein_uhlenbeck`: Generate paths with estimated parameters.
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
    """Generate paths using the Ornstein-Uhlenbeck mean-reverting process.

    Simulates paths from the Ornstein-Uhlenbeck (OU) process, which exhibits
    mean-reversion—the tendency to drift toward a long-run equilibrium level.
    Commonly used for modeling interest rates (Vasicek model), volatility,
    or pairs trading spreads.

    The OU process is defined by the SDE:

    $$dX_t = \\kappa(\\theta - X_t) \\, dt + \\sigma \\, dW_t$$

    This implementation uses the exact discretization formula (not Euler):

    $$X_{t+\\Delta t} = X_t e^{-\\kappa \\Delta t} + \\theta(1 - e^{-\\kappa \\Delta t})
    + \\sigma\\sqrt{\\frac{1 - e^{-2\\kappa \\Delta t}}{2\\kappa}} Z$$

    where $Z \\sim N(0,1)$.

    Args:
        n_paths: Number of independent Monte Carlo paths to simulate.
        n_steps: Number of time steps per path (including initial value).
        init_value: Initial value $X_0$ at time $t=0$.
        kappa: Speed of mean reversion. Higher values mean faster reversion.
            The half-life is $\\ln(2)/\\kappa$.
        theta: Long-run mean (equilibrium level) that the process reverts to.
        sigma: Volatility parameter controlling the noise amplitude.
        dt: Time step size as a fraction of a year.
        key: JAX PRNG key for reproducible random number generation.

    Returns:
        A JAX array of shape ``(n_paths, n_steps)`` containing simulated paths.

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Simulate a mean-reverting spread for pairs trading
        >>> spread_paths = ornstein_uhlenbeck(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_value=0.0,
        ...     kappa=5.0,  # Half-life ~ 50 trading days
        ...     theta=0.0,  # Mean spread
        ...     sigma=0.10,
        ...     dt=1/252,
        ...     key=key,
        ... )

    Note:
        Unlike GBM, the OU process can take negative values, which is
        appropriate for spreads or rate differentials but not for prices.
        For mean-reverting prices, consider exponentiating the OU process.

    See Also:
        - :func:`estimate_ou_params`: Estimate parameters from observed data.
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


# =============================================================================
# Bootstrapping Path Generators
# =============================================================================
# These functions generate paths by resampling historical data with replacement.
# Bootstrap methods are model-free and preserve empirical distribution features.


def moving_block_bootstrap(
    n_paths: int,
    n_steps: int,
    init_price: Union[float, jax.Array],
    log_returns: jax.Array,
    block_size: int,
    key: jax.Array,
) -> jax.Array:
    """Generate price paths using Moving Block Bootstrap from historical data.

    Resamples contiguous blocks of historical returns with replacement to
    construct synthetic price paths. This non-parametric method preserves
    short-term autocorrelation structure in returns while generating new
    scenarios.

    The algorithm:
        1. Randomly select block starting indices from ``[0, T - block_size]``
        2. Extract contiguous blocks of ``block_size`` returns
        3. Concatenate blocks to form new return sequences
        4. Convert returns to price paths

    Args:
        n_paths: Number of independent bootstrap paths to generate.
        n_steps: Number of time steps per path (including initial price).
        init_price: Initial asset price $S_0$.
        log_returns: A JAX array of shape ``(T,)`` containing historical
            log returns. Should have sufficient length for the chosen block size.
        block_size: Number of consecutive returns in each resampled block.
            Larger blocks preserve more autocorrelation but reduce diversity.
            Typical choices: 5-20 for daily data, sqrt(T) as a rule of thumb.
        key: JAX PRNG key for reproducible random sampling.

    Returns:
        A JAX array of shape ``(n_paths, n_steps)`` containing bootstrapped
        price paths.

    Example:
        >>> # Bootstrap 1-year paths from 5 years of daily data
        >>> historical_returns = jnp.diff(jnp.log(historical_prices))
        >>> paths = moving_block_bootstrap(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     log_returns=historical_returns,
        ...     block_size=10,  # 2-week blocks
        ...     key=key,
        ... )

    Note:
        MBB may underrepresent returns near the end of the historical sample
        since blocks cannot extend past the data boundary. Use
        :func:`circular_block_bootstrap` to avoid this edge effect.

    See Also:
        - :func:`circular_block_bootstrap`: Handles boundary effects via wrapping.
        - :func:`stationary_bootstrap`: Random block sizes for stationarity.
    """
    T = log_returns.shape[0]
    # We need n_steps - 1 returns to generate n_steps prices (including S0)
    n_returns_needed = n_steps - 1

    # Calculate number of blocks needed (using integer arithmetic for static shape compatibility)
    n_blocks = (n_returns_needed + block_size - 1) // block_size

    # Valid start indices: [0, T - block_size]
    # We sample n_paths * n_blocks starting indices
    key_starts = key
    start_indices = jax.random.randint(
        key_starts, shape=(n_paths, n_blocks), minval=0, maxval=T - block_size + 1
    )

    # Construct indices for each block
    # shape: (block_size,)
    block_offsets = jnp.arange(block_size)

    # Broadcast add: (n_paths, n_blocks, 1) + (block_size,) -> (n_paths, n_blocks, block_size)
    indices = start_indices[..., None] + block_offsets

    # Flatten to (n_paths, n_blocks * block_size)
    indices = indices.reshape(n_paths, -1)

    # Slice to exact length needed
    indices = indices[:, :n_returns_needed]

    # Gather returns
    sampled_returns = log_returns[indices]

    # Construct price paths
    log_price_path = jnp.cumsum(sampled_returns, axis=1)

    # Create zeros with matching shape for concatenation
    zeros_shape = list(log_price_path.shape)
    zeros_shape[1] = 1
    zeros = jnp.zeros(zeros_shape)

    log_price_path = jnp.concatenate([zeros, log_price_path], axis=1)

    prices = init_price * jnp.exp(log_price_path)
    return prices


def circular_block_bootstrap(
    n_paths: int,
    n_steps: int,
    init_price: Union[float, jax.Array],
    log_returns: jax.Array,
    block_size: int,
    key: jax.Array,
) -> jax.Array:
    """Generate price paths using Circular Block Bootstrap from historical data.

    Similar to Moving Block Bootstrap but wraps around the data circularly,
    eliminating edge effects that cause undersampling of returns near the
    end of the historical period.

    The algorithm treats the return series as circular:
        - Block starting at index ``T-2`` with size 5 contains indices
          ``[T-2, T-1, 0, 1, 2]`` (wrapping around)
        - All historical returns have equal probability of being sampled

    Args:
        n_paths: Number of independent bootstrap paths to generate.
        n_steps: Number of time steps per path (including initial price).
        init_price: Initial asset price $S_0$.
        log_returns: A JAX array of shape ``(T,)`` containing historical
            log returns.
        block_size: Number of consecutive returns in each resampled block.
            Can be any size; wrapping handles blocks exceeding data length.
        key: JAX PRNG key for reproducible random sampling.

    Returns:
        A JAX array of shape ``(n_paths, n_steps)`` containing bootstrapped
        price paths.

    Example:
        >>> paths = circular_block_bootstrap(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     log_returns=historical_returns,
        ...     block_size=10,
        ...     key=key,
        ... )

    Note:
        CBB introduces artificial correlation between the last and first
        observations when blocks wrap around. This is usually acceptable
        when the series is approximately stationary.

    See Also:
        - :func:`moving_block_bootstrap`: Without circular wrapping.
        - :func:`stationary_bootstrap`: Random block sizes.
    """
    T = log_returns.shape[0]
    n_returns_needed = n_steps - 1

    # Calculate number of blocks needed (using integer arithmetic for static shape compatibility)
    n_blocks = (n_returns_needed + block_size - 1) // block_size

    # Valid start indices: [0, T - 1] (Circular)
    key_starts = key
    start_indices = jax.random.randint(
        key_starts, shape=(n_paths, n_blocks), minval=0, maxval=T
    )

    block_offsets = jnp.arange(block_size)

    # (n_paths, n_blocks, block_size)
    indices = start_indices[..., None] + block_offsets

    # Wrap around indices
    indices = indices % T

    indices = indices.reshape(n_paths, -1)
    indices = indices[:, :n_returns_needed]

    sampled_returns = log_returns[indices]

    log_price_path = jnp.cumsum(sampled_returns, axis=1)

    # Create zeros with matching shape for concatenation
    zeros_shape = list(log_price_path.shape)
    zeros_shape[1] = 1
    zeros = jnp.zeros(zeros_shape)

    log_price_path = jnp.concatenate([zeros, log_price_path], axis=1)

    prices = init_price * jnp.exp(log_price_path)
    return prices


def stationary_bootstrap(
    n_paths: int,
    n_steps: int,
    init_price: Union[float, jax.Array],
    log_returns: jax.Array,
    key: jax.Array,
    p: float = 0.1,
) -> jax.Array:
    """Generate price paths using Stationary Bootstrap from historical data.

    Implements Politis & Romano's (1994) stationary bootstrap, which uses
    geometrically distributed random block lengths. This produces bootstrap
    samples that are strictly stationary (unlike fixed-block methods) while
    preserving serial dependence.

    The algorithm:
        1. At each step, with probability ``p``, start a new block at a
           random position
        2. With probability ``1-p``, continue the current block
        3. Block lengths follow Geometric(p) distribution with mean ``1/p``

    Args:
        n_paths: Number of independent bootstrap paths to generate.
        n_steps: Number of time steps per path (including initial price).
        init_price: Initial asset price $S_0$.
        log_returns: A JAX array of shape ``(T,)`` containing historical
            log returns. Treated circularly (wraps around).
        key: JAX PRNG key for reproducible random sampling.
        p: Probability of starting a new block at each step. Controls
            the expected block length: $\\mathbb{E}[\\text{block length}] = 1/p$.
            Default is 0.1 (average 10-step blocks).

    Returns:
        A JAX array of shape ``(n_paths, n_steps)`` containing bootstrapped
        price paths.

    Example:
        >>> # Generate paths with average block length of 20 days
        >>> paths = stationary_bootstrap(
        ...     n_paths=1000,
        ...     n_steps=252,
        ...     init_price=100.0,
        ...     log_returns=historical_returns,
        ...     key=key,
        ...     p=0.05,  # E[block] = 20
        ... )

    Note:
        The stationary bootstrap is theoretically preferred for inference
        because it preserves stationarity of the original series. However,
        the randomness in block lengths can increase variance compared to
        fixed-block methods.

    References:
        Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
        Journal of the American Statistical Association, 89(428), 1303-1313.

    See Also:
        - :func:`moving_block_bootstrap`: Fixed block sizes.
        - :func:`circular_block_bootstrap`: Fixed blocks with wrapping.
    """
    T = log_returns.shape[0]
    n_returns_needed = n_steps - 1

    key_p, key_start = jax.random.split(key)

    # Random variables for decision to start new block
    # (n_paths, n_returns_needed)
    u = jax.random.uniform(key_p, shape=(n_paths, n_returns_needed))

    # Random start indices if we do start a new block
    new_starts = jax.random.randint(
        key_start, shape=(n_paths, n_returns_needed), minval=0, maxval=T
    )

    # Initial indices (randomly chosen)
    # We can just use the first column of new_starts as initial indices
    init_indices = new_starts[:, 0]

    # Scan function to generate indices
    # We use a scan loop to generate indices step-by-step.
    # At each step, we decide whether to start a new block with probability p.
    # This implicitly generates block lengths following a Geometric(p) distribution,
    # which is the definition of the Stationary Bootstrap.
    # This approach is preferred over generating explicit block lengths via
    # jax.random.geometric because it ensures static array shapes for JIT compilation.
    def step_fn(current_indices, inputs):
        # inputs: (u_t, new_start_t)
        u_t, new_start_t = inputs

        # If u_t < p, start new block -> use new_start_t
        # Else, continue block -> (current_indices + 1) % T

        next_if_continue = (current_indices + 1) % T
        next_if_new = new_start_t

        decision = u_t < p
        next_indices = jnp.where(decision, next_if_new, next_if_continue)

        return next_indices, next_indices

    # We need to generate n_returns_needed indices.
    # The first index is init_indices.
    # We scan for n_returns_needed - 1 steps.
    # Actually, let's scan for all n_returns_needed steps, treating the first one as "new block" implicitly or handling it.
    # Easier: Use scan for all steps.
    # We need a dummy previous index for the first step?
    # Or just set p=1 for the first step effectively?

    # Let's refine:
    # We want a sequence of indices I_1, ..., I_N.
    # I_1 is uniform on [0, T-1].
    # I_{t+1} = (I_t + 1) % T with prob 1-p
    #           Uniform on [0, T-1] with prob p

    # So we can use scan.
    # Initial carry: dummy, doesn't matter if we force first step to be new.
    # But let's just pick I_1 explicitly.

    indices_list = []
    # First index
    indices_list.append(init_indices)

    # Scan for remaining
    # Inputs for scan should be (n_paths, n_returns_needed - 1)
    scan_u = u[:, 1:]
    scan_starts = new_starts[:, 1:]

    scan_inputs = (scan_u.T, scan_starts.T)  # Transpose for scan iteration

    _, subsequent_indices = jax.lax.scan(step_fn, init_indices, scan_inputs)

    # subsequent_indices is (n_returns_needed - 1, n_paths)
    subsequent_indices = subsequent_indices.T

    # Concatenate
    indices = jnp.concatenate([init_indices[:, None], subsequent_indices], axis=1)

    sampled_returns = log_returns[indices]

    log_price_path = jnp.cumsum(sampled_returns, axis=1)

    # Create zeros with matching shape for concatenation
    zeros_shape = list(log_price_path.shape)
    zeros_shape[1] = 1
    zeros = jnp.zeros(zeros_shape)

    log_price_path = jnp.concatenate([zeros, log_price_path], axis=1)

    prices = init_price * jnp.exp(log_price_path)
    return prices
