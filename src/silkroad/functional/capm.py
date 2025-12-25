"""The functionl module for the Capital Asset Pricing Model (CAPM).


This module provides functions to calculate key metrics related to CAPM, such as
market portfolio, market weights etc.


Note:
    - Short selling is allowid in this implementation for simplicity.
"""

import jax
import jax.numpy as jnp


@jax.jit
def market_portfolio_returns(
    risk_free_rate: float,
    mu: jax.Array,
    cov: jax.Array,
) -> jax.Array:
    """
    Calculate the market returns using the Capital Asset Pricing Model (CAPM).

    The market return is the expected return of the market portfolio, which is a
    weighted sum of the returns of all assets in the market portfolio. Note that
    covariance matrix must be symmetric and positive definite.

    Args:
        risk_free_rate (float): The risk-free rate of return.
        mu (jax.Array): Expected returns of the assets.
        cov (jax.Array): Covariance matrix of the asset returns.

    Returns:
        jax.Array: The market returns.
    """
    Vinv = jnp.linalg.inv(cov)
    ones = jnp.ones(mu.shape)
    A = ones.T @ Vinv @ ones
    B = ones.T @ Vinv @ mu
    C = mu.T @ Vinv @ mu
    return (C - B * risk_free_rate) / (B - A * risk_free_rate)


@jax.jit
def market_portfolio_risk(
    risk_free_rate: float,
    mu: jax.Array,
    cov: jax.Array,
) -> jax.Array:
    """
    Calculate the market risk using the Capital Asset Pricing Model (CAPM).

    The market risk is the standard deviation of the returns of the market portfolio,
    which is a weighted sum of the risks of all assets in the market portfolio. Note
    that covariance matrix must be symmetric and positive definite.

    Args:
        risk_free_rate (float): The risk-free rate of return.
        mu (jax.Array): Expected returns of the assets.
        cov (jax.Array): Covariance matrix of the asset returns.

    Returns:
        jax.Array: The market risk.
    """
    Vinv = jnp.linalg.inv(cov)
    ones = jnp.ones(mu.shape)
    A = ones.T @ Vinv @ ones
    B = ones.T @ Vinv @ mu
    C = mu.T @ Vinv @ mu
    sigma_squared = (A * risk_free_rate**2 - 2 * B * risk_free_rate + C) / (
        B - A * risk_free_rate
    ) ** 2
    return jnp.sqrt(sigma_squared)


@jax.jit
def market_portfolio_weights(
    risk_free_rate: float,
    mu: jax.Array,
    cov: jax.Array,
) -> jax.Array:
    """
    Calculate the market portfolio weights using the Capital Asset Pricing Model (CAPM).

    Args:
        risk_free_rate (float): The risk-free rate of return.
        mu (jax.Array): Expected returns of the assets.
        cov (jax.Array): Covariance matrix of the asset returns.

    Returns:
        jax.Array: The market portfolio weights.
    """
    Vinv = jnp.linalg.inv(cov)
    ones = jnp.ones(mu.shape)
    A = ones.T @ Vinv @ ones
    B = ones.T @ Vinv @ mu
    weights = (Vinv @ (mu - risk_free_rate * ones)) / (B - A * risk_free_rate)
    return weights


@jax.jit
def beta(
    asset_returns: jax.Array,
    market_returns: jax.Array,
) -> jax.Array:
    """
    Estimate the beta of an asset using its returns and the market returns.

    Args:
        asset_returns (jax.Array): Returns of the asset.
        market_returns (jax.Array): Returns of the market portfolio.

    Returns:
        jax.Array: The estimated beta of the asset.
    """
    covariance = jnp.cov(asset_returns, market_returns)[0, 1]
    market_variance = jnp.var(market_returns)
    beta = covariance / market_variance
    return beta
