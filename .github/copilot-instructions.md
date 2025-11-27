# Silkroad AI Coding Instructions

You are working on **Silkroad**, a Python-based quantitative trading and portfolio management framework.

## 🏗 Architecture & Core Concepts

- **Domain Models (`src/silkroad/core`)**:
  - The system relies heavily on **Pydantic v2** for data validation.
  - `UniformBarSet`: The fundamental unit for time-series data (OHLCV). It manages both historical data (Pandas DataFrame) and streaming buffers.
  - `UniformBarCollection`: Aggregates multiple `UniformBarSet`s for multi-asset analysis.
  - **Rule**: Always use these wrapper classes instead of raw DataFrames when passing market data between components.

- **Portfolio Management (`src/silkroad/portfolio`)**:
  - `Portfolio`: Manages asset weights, cash, and rebalancing logic.
  - Uses `cvxpy` via `src/silkroad/functional/mean_variance.py` for optimization.

- **Application Logic (`src/silkroad/app`)**:
  - Contains strategies (e.g., `WeeklyRebalancer`) and feature engineering.
  - `feature_extrator.py`: Central hub for signal generation (momentum, volatility, macro). Handles `ta-lib` integration.

- **Functional Kernels (`src/silkroad/functional`)**:
  - Pure functions for math, metrics, and optimization.
  - Keep this layer stateless and testable.

## 💻 Coding Conventions

- **Type Safety**:
  - Use strict type hinting.
  - Leverage Pydantic's `Field`, `computed_field`, and validators.
  - Example: `class MyModel(BaseModel): ...`

- **Numerical Computing**:
  - Use `numpy` for vector operations.
  - Use `cvxpy` for convex optimization problems.
  - Use `pandas` for time-series alignment and manipulation within the core models.

- **Documentation**:
  - Use **Google-style docstrings**.
  - Include LaTeX math for financial formulas (e.g., `$$ \mu^T w = r^* $$`).

- **Error Handling**:
  - Use custom exceptions or standard Python exceptions with descriptive messages.
  - Handle optional dependencies (like `ta-lib`) gracefully with try/except blocks.

## 🛠 Workflows & Testing

- **Dependency Management**:
  - The project uses `uv` for build and dependency management.
  - Dependencies are defined in `pyproject.toml`.

- **Testing**:
  - Run tests with `pytest`.
  - Use fixtures to generate dummy market data (`alpaca.data.models.Bar`).
  - Example fixture pattern:
    ```python
    @pytest.fixture
    def sample_bars(symbol, start_time):
        return [create_dummy_bar(...) for i in range(5)]
    ```

## 🚨 Critical Implementation Details

- **Timezones**: Ensure all timestamps are timezone-aware (UTC) to prevent lookahead bias or alignment errors.
- **Data Alignment**: When working with multiple assets, ensure they are aligned on the same `Horizon` (time interval) using `UniformBarCollection`.
- **Feature Extraction**: When adding new features in `feature_extrator.py`, ensure they handle `NaN`s and are robust to missing data.
