r"""Mean-variance optimization utilities.

This module provides a thin wrapper around CVXPY for solving the
classical Markowitz mean-variance portfolio optimization problem.

The canonical optimization problem solved is:

$$
\begin{aligned}
\min_{w\in\mathbb{R}^n}\quad & w^T \Sigma w\\
	ext{subject to}\quad & \mathbf{1}^T w = 1 \\ 
& \mu^T w = r^* \
& w \ge 0 \quad \text{(optional: if short selling is not allowed)}
\end{aligned}
$$

where $\Sigma$ is the covariance matrix, $\mu$ the expected returns,
and $r^*$ the target portfolio return.

Functions
---------
mean_variance_optimization
    Build and solve the quadratic program above using CVXPY.
"""

import cvxpy as cp
from jax import Array
import typing as tp


__all__ = ["mean_variance_optimization"]


def mean_variance_optimization(
    mu: Array,
    cov: Array,
    target_return: float,
    short_selling: bool = False,
) -> tp.Dict[str, tp.Union[Array, float, str]]:
    r"""Solve a mean-variance optimization problem.

    The optimization is the Markowitz quadratic program:

    $$\min_{w} w^T \Sigma w$$
    subject to

    - $\mathbf{1}^T w = 1$ (fully invested),
    - $\mu^T w = r^*$ (expected return equals `target_return`), and
    - $w_i \ge 0$ for all i if `short_selling` is False.

    Args:
        mu: Expected returns vector (shape ``(n_assets,)``).
        cov: Covariance matrix (shape ``(n_assets, n_assets)``). Should
            be symmetric positive semidefinite.
        target_return: Target portfolio return $r^*$ (scalar).
        short_selling: If False (default), weights are constrained to be
            non-negative. If True, short positions are allowed.

    Returns:
        dict: A dictionary with the following keys:
            - ``weights``: Optimal weight vector $w^*$ (numpy or jax Array).
            - ``status``: Solver status string (e.g., ``"optimal"``).
            - ``optimal_value``: The optimal objective value
              ($w^{*T}\Sigma w^*$).
            - ``target_return``: Echo of the ``target_return`` parameter.

    Raises:
        ValueError: If the solver fails to find an optimal solution.

    Examples:
        >>> mu = jax.numpy.array([0.1, 0.12])
        >>> cov = jax.numpy.array([[0.01, 0.001], [0.001, 0.02]])
        >>> mean_variance_optimization(mu, cov, target_return=0.11)

    Notes:
        - This function uses CVXPY which returns weights as a NumPy array.
        - Type annotations use ``jax.Array`` to allow either NumPy or JAX
          arrays, although CVXPY always returns NumPy arrays.
    """
    # Determine the number of assets
    n_assets = mu.shape[0]

    # Define the optimization variable
    w = cp.Variable(n_assets)
    # Define the objective function (minimize portfolio variance)
    objective = cp.Minimize(cp.quad_form(w, cov))  # Checks PSD of cov internally
    # Define the constraints
    constraints = [cp.sum(w) == 1, cp.matmul(mu, w) == target_return]

    # Add short selling constraint if not allowed
    if not short_selling:
        constraints.append(w >= 0)

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        raise ValueError(
            f"Optimization failed with status: {problem.status}. Check problem formulation and inputs for validity."
        )

    return {
        "weights": w.value,  # type: ignore
        "status": problem.status,
        "optimal_value": problem.value,
        "target_return": target_return,
    }
