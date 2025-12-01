import jax
import jax.numpy as jnp
import numpy as np
import pytest
from silkroad.functional.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    expected_annualized_return,
    annulized_volatility,
    CVaR,
    VaR,
    calmar_ratio,
    omega_ratio,
    ulcer_index,
    information_ratio,
    tail_ratio,
)


@pytest.fixture
def sample_returns():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (1000,)) * 0.01


def test_sharpe_ratio_jit(sample_returns):
    jit_sharpe = jax.jit(sharpe_ratio)
    result = jit_sharpe(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_sortino_ratio_jit(sample_returns):
    jit_sortino = jax.jit(sortino_ratio)
    result = jit_sortino(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_max_drawdown_jit(sample_returns):
    jit_mdd = jax.jit(max_drawdown)
    result = jit_mdd(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert result >= 0
    assert not jnp.isnan(result)


def test_expected_annualized_return_jit(sample_returns):
    jit_ear = jax.jit(expected_annualized_return)
    result = jit_ear(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_annulized_volatility_jit(sample_returns):
    jit_vol = jax.jit(annulized_volatility)
    result = jit_vol(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert result >= 0
    assert not jnp.isnan(result)


def test_cvar_jit(sample_returns):
    jit_cvar = jax.jit(CVaR)
    result = jit_cvar(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_var_jit(sample_returns):
    jit_var = jax.jit(VaR)
    result = jit_var(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_metrics_values_simple():
    # Simple case: constant positive return
    # log returns of 0.01 approx 1% simple return
    returns = jnp.array([0.01] * 100)

    # Max drawdown should be 0 for positive returns
    assert max_drawdown(returns) == 0.0

    # VaR should be the return itself (since no variance) or close to it depending on percentile logic on constant array
    # Actually percentile of constant array is that constant.
    # returns are log returns. simple returns = exp(0.01)-1 ~= 0.01005
    simple_ret = jnp.exp(0.01) - 1
    assert jnp.isclose(VaR(returns), simple_ret, atol=1e-6)

    # CVaR should also be that constant
    assert jnp.isclose(CVaR(returns), simple_ret, atol=1e-6)


def test_calmar_ratio_jit(sample_returns):
    jit_calmar = jax.jit(calmar_ratio)
    result = jit_calmar(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_omega_ratio_jit(sample_returns):
    jit_omega = jax.jit(omega_ratio)
    result = jit_omega(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_ulcer_index_jit(sample_returns):
    jit_ulcer = jax.jit(ulcer_index)
    result = jit_ulcer(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert result >= 0
    assert not jnp.isnan(result)


def test_information_ratio_jit(sample_returns):
    benchmark = jax.random.normal(jax.random.PRNGKey(1), sample_returns.shape) * 0.01
    jit_ir = jax.jit(information_ratio)
    result = jit_ir(sample_returns, benchmark)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)


def test_tail_ratio_jit(sample_returns):
    jit_tail = jax.jit(tail_ratio)
    result = jit_tail(sample_returns)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    assert not jnp.isnan(result)
