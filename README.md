# Silkroad

**Silkroad** is a modern, high-performance quantitative trading and portfolio management framework built in Python. It leverages **JAX** for accelerated numerical computing, **Pydantic** for robust data validation, and **CVXPY** for convex optimization.

Designed for systematic trading, Silkroad provides a modular architecture separating domain models, functional kernels, and application logic.

## 🚀 Key Features

*   **Robust Data Models**: Type-safe market data handling using `UniformBarSet` and `UniformBarCollection` to ensure proper time-series alignment, timezone awareness, and validation.
*   **Functional Kernels**: Stateless, JAX-accelerated functions located in `src/silkroad/functional`:
    *   **Stochastic Processes**: Generators for Geometric Brownian Motion (GBM), Heston Model, Ornstein-Uhlenbeck (OU), and Merton Jump Diffusion (MJD).
    *   **Parameter Estimation**: Calibrate model parameters (drift, volatility, kappa, theta) directly from historical price paths.
    *   **Performance Metrics**: Efficient calculation of Sharpe Ratio, Sortino Ratio, and Max Drawdown.
*   **Portfolio Optimization**: Mean-Variance optimization wrapper around CVXPY for solving allocation problems.
*   **Feature Engineering**: Comprehensive signal generation pipeline (`feature_extrator`) integrating:
    *   Momentum & Volatility dynamics
    *   Technical Indicators (via TA-Lib)
    *   Macroeconomic factors & Calendar effects
*   **Backtesting**: Functional backtesting engine supporting drift, rebalancing logic, and transaction costs.
*   **End-to-End Differentiability**: Built on JAX, allowing for gradient-based optimization of strategy parameters directly through the backtest engine.

## 🛠 Installation

Silkroad uses `uv` for fast dependency management.

### Prerequisites
*   Python >= 3.12
*   [TA-Lib](https://github.com/mrjbq7/ta-lib) (C Library required for technical indicators)

### Setup

```bash
# Install dependencies
uv sync
```

## 📂 Project Structure

```text
src/silkroad/
├── app/              # Application logic and strategies
├── core/             # Domain models (UniformBarSet, Enums)
├── functional/       # Pure functions (JAX/NumPy)
│   ├── backtest.py       # Backtesting logic
│   ├── mean_variance.py  # Portfolio optimization
│   ├── metrics.py        # Risk & performance metrics
│   └── paths.py          # Stochastic path generators & estimators
├── logger/           # Logging configuration
├── portfolio/        # Portfolio management classes
└── preprocessors/    # Feature extraction & data pipelines
```

## 💡 Usage Examples

### Generating Synthetic Price Paths (GBM)

```python
import jax
import jax.numpy as jnp
from silkroad.functional.paths import geometric_brownian_motion

key = jax.random.PRNGKey(42)
paths = geometric_brownian_motion(
    n_paths=10,
    n_steps=252,
    init_price=100.0,
    drift=0.05,
    volatility=0.2,
    dt=1/252,
    key=key
)
```

### Calculating Metrics

```python
import numpy as np
from silkroad.functional.metrics import sharpe_ratio

# Simulated daily log returns
returns = np.random.normal(0.001, 0.02, 252)
sr = sharpe_ratio(returns, periods=252)
print(f"Sharpe Ratio: {sr:.2f}")
```

### Differentiable Backtesting

Optimize strategy parameters (e.g., rebalancing thresholds or weights) by differentiating through the backtest engine.

```python
import jax
import jax.numpy as jnp
from silkroad.functional.backtest import backtest

def loss_function(target_weights, log_returns):
    # Run backtest
    (final_val, final_w), (step_returns, costs, turnover) = backtest(
        init_value=1000.0,
        init_weights=jnp.array([0.5, 0.5]),
        log_returns=log_returns,
        target_weights=target_weights, # We want to optimize this
        transaction_cost_bp=5.0
    )
    # Minimize negative total return
    return -jnp.sum(step_returns)

# Dummy data
log_returns = jnp.array([[0.01, -0.01], [-0.01, 0.02], [0.005, 0.005]])
initial_target = jnp.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

# Compute gradients w.r.t. target_weights
grads = jax.grad(loss_function)(initial_target, log_returns)
print("Gradients:", grads)
```

## 🧪 Testing

Run the test suite using `pytest`:

```bash
uv run pytest
```

## 🗺 Roadmap

- [ ] **Reinforcement Learning**: Gym environments for RL agent training.
- [ ] **Live Trading**: Alpaca API integration for paper and live trading.
- [ ] **Advanced Optimization**: Black-Litterman and Hierarchical Risk Parity (HRP).
- [ ] **Dashboard**: Streamlit-based visualization of backtest results.

## 📄 License

This project is licensed under the MIT License.
