r"""Functional interface for factor-based models and principal component regression.

This module provides stateless, JAX-compatible functions for factor analysis and modeling,
including Principal Component Regression (PCR) and factor-based forecasting of asset returns
and covariances. All functions are end-to-end differentiable and compatible with JAX
transformations.

Mathematical Model:
    The core linear factor model is defined as:

    .. math::

        Y = A + X B + \epsilon

    where:

    - :math:`Y \in \mathbb{R}^{T \times N}` is the target matrix (e.g., asset returns)
    - :math:`X \in \mathbb{R}^{T \times K}` is the feature matrix (e.g., factor returns)
    - :math:`A \in \mathbb{R}^{N}` is the intercept vector
    - :math:`B \in \mathbb{R}^{K \times N}` is the coefficient matrix (factor loadings)
    - :math:`\epsilon \in \mathbb{R}^{T \times N}` is the residual matrix with :math:`\epsilon \sim \mathcal{N}(0, \text{diag}(\sigma^2))`

Principal Component Regression (PCR):
    PCR estimates the model parameters by:

    1. Center the data: :math:`\tilde{X} = X - \bar{X}`, :math:`\tilde{Y} = Y - \bar{Y}`
    2. Compute SVD: :math:`\tilde{X} = U \Sigma V^T`
    3. Select top-k components: :math:`V_k \in \mathbb{R}^{K \times k}`
    4. Project features: :math:`Z_k = \tilde{X} V_k`
    5. Fit in PCA space: :math:`B_k = (Z_k^T Z_k)^{-1} Z_k^T \tilde{Y}` (or pseudoinverse in practice)
    6. Transform back: :math:`B = V_k B_k`
    7. Compute intercept: :math:`A = \bar{Y} - \bar{X} B`

Factor-Based Forecasting:
    Given the estimated parameters :math:`(A, B, \sigma^2)` and factor forecasts, the asset
    expected returns and covariance are:

    .. math::

        \mathbb{E}[Y] = A + B^T \mathbb{E}[X]

    .. math::

        \text{Cov}[Y] = B^T \text{Cov}[X] B + \text{diag}(\sigma^2)

    This decomposition separates systematic risk (factor-driven) from idiosyncratic risk.

Properties:
    - Dimensionality reduction via k principal components (k ≤ K)
    - Mitigates multicollinearity through orthogonal PCA components
    - Separates systematic (factor) and idiosyncratic risk
    - Reduces estimation error in high-dimensional settings
    - All operations are JAX-compatible and fully differentiable

Key Functions:
    - principal_component_regression: Fit a linear model using PCA-transformed features.
    - forcast_asset_state_based_on_factor_model: Forecast asset returns and covariances based on factor models.
"""

import jax
import jax.numpy as jnp
import typing as tp
from functools import partial

__all__ = [
    "principal_component_regression",
    "forcast_asset_state_based_on_factor_model",
]


@partial(jax.jit, static_argnames=("pca_components",))
def principal_component_regression(
    X: jax.Array,
    Y: jax.Array,
    pca_components: int,
    eps: float = 1e-10,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Perform Principal Component Regression (PCR) on the given data.

    Fits a linear regression model using k principal components of the feature matrix.
    Features are standardized before PCA to ensure scale-invariance.
    See module docstring for the mathematical model and algorithm details.

    Args:
        X (jnp.ndarray): The input feature matrix of shape (T, K) where T is the number
            of time samples and K is the number of features.
        Y (jnp.ndarray): The target matrix of shape (T, N) where N is the number of
            target variables (e.g., assets).
        pca_components (int): The number of principal components k to retain (k ≤ K).
        eps (float): Small constant for numerical stability to prevent division by zero.
            Default is 1e-8.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: A tuple containing:
            - A (jax.Array): Intercept vector of shape (N,)
            - B (jax.Array): Coefficient matrix of shape (K, N) in original scale
            - residual_variance (jax.Array): Variance of residuals of shape (N,)
    """
    # Standardize X and center Y
    mu_X = jnp.mean(X, axis=0, keepdims=True)  # (1, K)
    sigma_X = jnp.std(X, axis=0, keepdims=True)  # (1, K)

    # Prevent division by zero for constant features
    sigma_X = jnp.where(sigma_X < eps, jnp.ones_like(sigma_X), sigma_X)

    X_std = (X - mu_X) / sigma_X  # (T, K) - standardized features

    mu_Y = jnp.mean(Y, axis=0, keepdims=True)  # (1, N)
    Y_c = Y - mu_Y  # (T, N) - centered targets

    # Calculate the principal components of X_std using SVD
    # Use economic SVD for numerical stability
    _, S, Vt = jnp.linalg.svd(X_std, full_matrices=False)

    # Select top-k principal components
    V_k = Vt[:pca_components, :].T  # (K, pca_components)

    # Project standardized features onto principal components
    Z_k = X_std @ V_k  # (T, pca_components)

    # Regression coefficients in the PCA space using pseudoinverse
    # Add rcond for numerical stability
    B_k = jnp.linalg.pinv(Z_k, rcond=eps) @ Y_c  # (pca_components, N)

    # Transform back to standardized space
    B_std = V_k @ B_k  # (K, N)

    # Transform coefficients to original scale
    # Use safe division
    B = B_std / sigma_X.T  # (K, N)

    # Calculate the intercept term
    A = mu_Y.squeeze() - (mu_X / sigma_X) @ B_std  # (N,)

    # Calculate the residuals and their variance
    Y_pred = A + X @ B  # (T, N)
    residuals = Y - Y_pred  # (T, N)

    # Use ddof=1 for unbiased variance estimate
    residual_variance = jnp.var(residuals, axis=0, ddof=1)  # (N,)

    # Ensure non-negative variance (numerical stability)
    residual_variance = jnp.maximum(residual_variance, 0.0)

    return A, B, residual_variance


@partial(jax.jit, static_argnames=("pca_components",))
def forcast_asset_state_based_on_factor_model(
    Y: jnp.ndarray,
    X: jnp.ndarray,
    pca_components: int,
    factor_expected_returns: jnp.ndarray,
    factor_covariance: jnp.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Forecast asset state based on a factor model.

    Uses PCR to fit factor loadings, then forecasts asset expected returns and covariance
    given factor forecasts. See module docstring for the mathematical model.

    Args:
        Y (jnp.ndarray): Historical asset returns matrix of shape (T, N) where T is the
            number of time samples and N is the number of assets.
        X (jnp.ndarray): Historical factor returns matrix of shape (T, K) where K is the
            number of factors.
        pca_components (int): The number of principal components k to use in the regression (k ≤ K).
        factor_expected_returns (jnp.ndarray): Expected returns of the factors of shape (K,).
        factor_covariance (jnp.ndarray): Covariance matrix of the factors of shape (K, K).

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing:
            - asset_expected_returns (jax.Array): Forecasted expected returns of shape (N,)
            - asset_covariance (jax.Array): Forecasted covariance matrix of shape (N, N)
    """
    A, B, diag_residual_variance = principal_component_regression(X, Y, pca_components)
    # Forecast asset expected returns
    asset_expected_returns = A + B.T @ factor_expected_returns
    # Forecast asset covariance
    asset_covariance = B.T @ factor_covariance @ B + jnp.diag(diag_residual_variance)

    return asset_expected_returns, asset_covariance
