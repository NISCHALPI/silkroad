# Silkroad Agent Guidelines

## 🛠 Commands
- **Test All**: `uv run pytest`
- **Single Test**: `uv run pytest tests/path/to/test.py::test_function`
- **Build**: `uv build` | **Deps**: Managed via `uv` (`pyproject.toml`)

## 📐 Code Style & Architecture
- **Data**: MUST use `UniformBarSet`/`UniformBarCollection` (in `silkroad.core`). NO raw DataFrames.
- **Typing**: Strict `Pydantic` v2 usage (`BaseModel`, `Field`). All timestamps MUST be UTC & timezone-aware.
- **Math/Opt**: `jax` for vector math, `cvxpy` for optimization. `numpy` for arrays.
- **Structure**: `src/silkroad/functional` must remain stateless. Imports: StdLib -> 3rd Party -> Local.
- **Docs**: Google-style docstrings. Use LaTeX for math (e.g., `$$ f(x) $$`).
- **Error Handling**: Use custom exceptions. Handle `NaN`s in feature extractors robustly.

## 🤖 Context
- **Stack**: Python 3.12+, FastAPI, ArcticDB.
