"""Contains functions to point estimate parameters of single factor models.

In the signle factor model, asset returns are assumed to be driven by a single common factor (e.g., market return) plus idiosyncratic noise
which is uncorrelated with the market return. The following is the model formulation:

R_i = alpha_i + beta_i * F_i + epsilon_i
where R_i is the return of asset i, F_i is the common factor (e.g., market return), beta_i is the sensitivity of asset i to the factor,
and epsilon_i is the idiosyncratic noise term usually assumed to be normally distributed with mean zero and variance sigma_i^2.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "estimate_beta",
    "estimate_alpha",
    "estimate_idiosyncratic_variance",
]


@jax.jit
def estimate_beta(
    asset_returns: jax.Array,
    factor_returns: jax.Array,
) -> jax.Array:
    """
    Estimate the beta of an asset using its returns and the factor returns.

    Args:
        asset_returns (jax.Array): Returns of the asset.
        factor_returns (jax.Array): Returns of the common factor (e.g., market return).

    Returns:
        jax.Array: The estimated beta of the asset.
    """
    covariance = jnp.cov(asset_returns, factor_returns)[0, 1]
    factor_variance = jnp.var(factor_returns)
    beta_estimate = covariance / factor_variance
    return beta_estimate


@jax.jit
def estimate_alpha(
    asset_returns: jax.Array,
    factor_returns: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """
    Estimate the alpha of an asset using its returns, factor returns, and estimated beta.

    Args:
        asset_returns (jax.Array): Returns of the asset.
        factor_returns (jax.Array): Returns of the common factor (e.g., market return).
        beta (jax.Array): Estimated beta of the asset.

    Returns:
        jax.Array: The estimated alpha of the asset.
    """
    mean_asset_return = jnp.mean(asset_returns)
    mean_factor_return = jnp.mean(factor_returns)
    alpha_estimate = mean_asset_return - beta * mean_factor_return
    return alpha_estimate


@jax.jit
def estimate_idiosyncratic_variance(
    asset_returns: jax.Array,
    factor_returns: jax.Array,
    beta: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    """
    Estimate the idiosyncratic variance of an asset using its returns, factor returns, estimated beta, and estimated alpha.

    Args:
        asset_returns (jax.Array): Returns of the asset.
        factor_returns (jax.Array): Returns of the common factor (e.g., market return).
        beta (jax.Array): Estimated beta of the asset.
        alpha (jax.Array): Estimated alpha of the asset.

    Returns:
        jax.Array: The estimated idiosyncratic variance of the asset.
    """
    predicted_returns = alpha + beta * factor_returns
    residuals = asset_returns - predicted_returns
    idiosyncratic_variance = jnp.var(residuals)
    return idiosyncratic_variance


@jax.jit
def estimate_single_factor_model_params(
    asset_returns: jax.Array,
    factor_returns: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Estimate the parameters of a single factor model: alpha, beta, and idiosyncratic variance.

    Args:
        asset_returns (jax.Array): Returns of the asset.
        factor_returns (jax.Array): Returns of the common factor (e.g., market return).

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: The estimated alpha, beta, and idiosyncratic variance of the asset.
    """
    beta = estimate_beta(asset_returns, factor_returns)
    alpha = estimate_alpha(asset_returns, factor_returns, beta)
    idiosyncratic_variance = estimate_idiosyncratic_variance(
        asset_returns, factor_returns, beta, alpha
    )
    return alpha, beta, idiosyncratic_variance
