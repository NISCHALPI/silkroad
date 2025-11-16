"""Functional primitives for Silkroad.

This module provides functional programming utilities used by the Silkroad
project to simplify data processing and algorithmic code. Utilities are
designed to be stateless and side-effect-free so they can be composed into
pipelines for analysis and optimization tasks.

Functions:
    mean_variance_optimization(returns, ...):
        Perform mean-variance portfolio optimization and return optimal
        allocations for the given returns and constraints.

Notes:
    - Intended for internal use by Silkroad algorithms.
    - Do not define classes or persist module-level state here; prefer pure
      functions and lightweight helpers.

Example:
    >>> from silkroad.functional import mean_variance_optimization
    >>> allocs = mean_variance_optimization(returns, target_return=0.01)
"""

from .mean_variance import mean_variance_optimization

__all__ = ["mean_variance_optimization"]
