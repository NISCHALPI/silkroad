# Silkroad Agent Guidelines

## commands
- **Test All**: `uv run pytest`
- **Single Test**: `uv run pytest tests/path/to/test.py::test_function_name`
- **Build**: `uv build`
- **Dependencies**: Managed via `uv`. See `pyproject.toml`.

## Code Style & Rules
- **Data Structures**: ALWAYS use `UniformBarSet`/`UniformBarCollection` for OHLCV data. NEVER pass raw DataFrames between components.
- **Typing**: Strict type hints required. Use Pydantic v2 (`BaseModel`, `Field`, `computed_field`) heavily.
- **Timezones**: ALL timestamps must be timezone-aware (UTC).
- **Libraries**: `jax` for math, `cvxpy` for optimization, `fastapi` for API.
- **Docstrings**: Google-style. Include LaTeX for math formulas.
- **Architecture**: Keep functional kernels (`src/silkroad/functional`) stateless.
- **Imports**: Group by: Standard Lib, Third Party (e.g., pandas, jax), Local (`silkroad.*`).
- **Ref**: See `.github/copilot-instructions.md` for detailed architectural concepts.
