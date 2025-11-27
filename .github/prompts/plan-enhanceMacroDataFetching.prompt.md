# Task: Redesign Macroeconomic Data Module for ML Asset Allocation

## Role
You are a Senior Quantitative Developer and Data Engineer specializing in financial time-series analysis.

## Context
We currently have a basic python module (`fetch_macroeconomic_data`) that pulls VIX (Yahoo) and CPI/Unemployment (FRED). The current implementation is insufficient for a machine learning model intended to predict expected returns and covariance matrices. It lacks feature richness, proper handling of publication lags (look-ahead bias), and smooth interpolation for lower-frequency data.

## Objective
Refactor and expand the module to create a robust `MacroDataLoader` class. The output must be a Pandas DataFrame with a daily resolution, ready for ML training.

## Implementation Plan

### 1. Architecture & Configuration
- **Class-Based Design:** Create a class `MacroDataLoader` instead of standalone functions to handle configuration state.
- **Config Object:** Define a dictionary or DataClass mapping readable names (e.g., 'inflation') to their Source IDs (FRED/Yahoo), Frequency, and Publication Lag (in days).
- **Sources:** Use the "Feature List" provided below to populate the configuration.

### 2. Data Fetching Strategy
- **Concurrent Fetching:** Use `ThreadPoolExecutor` or async logic to fetch data from FRED and Yahoo simultaneously to reduce latency.
- **Error Handling:** If a specific series fails, log a warning but do not crash the entire pipeline unless critical data (like Risk-Free rate) is missing.

### 3. Advanced Data Processing (The Core Logic)
The ML model requires daily inputs, but many economic indicators are Monthly/Quarterly.
- **Step A: Lagging (Crucial):** 
  - You MUST implement a `publication_lag` shift. 
  - Example: CPI for January is released ~Feb 14th. The value for Jan 31st in the DataFrame cannot be the Jan CPI. It must be the Dec CPI. 
  - *Requirement:* Allow a `lag_days` parameter for each series (default to 30 days for monthly macro data if unknown).
- **Step B: Transformation:** 
  - Raw levels (e.g., CPI = 305.2) are non-stationary.
  - Implement a transformation flag: `calc_yoy` (Year-over-Year change) or `calc_mom` (Month-over-Month).
  - For percentages (Unemployment, Yields), keep as is or calculate `diff` (change in basis points).
- **Step C: Upsampling & Interpolation:**
  - Once lagged, the data is "safe" to use.
  - Implement two modes via a `interpolation_method` argument:
    1. `'ffill'`: Forward fill (Standard step function). Safe, creates jagged data.
    2. `'spline'`: Cubic Spline or Linear interpolation. Creates smooth gradients beneficial for Neural Networks.
    - *Note:* Ensure the interpolation does not leak future data (only interpolate between *already available* lagged data points).

### 4. Output Formatting
- **Index:** `pd.DatetimeIndex` (UTC, Normalized to midnight).
- **Columns:** Use standard snake_case naming conventions (e.g., `rate_us10y`, `infl_cpi_yoy`).
- **Cleaning:** Handle `NaN`s at the start of the series (drop or backfill specific to user request).

## Feature List (To Implement)
Implement the following dictionary structure for the fetcher:

1. **Yields:** 10Y (`DGS10`), 3M (`DGS3MO`), Spread (`T10Y2Y`).
2. **Inflation:** CPI (`CPIAUCSL`), 5Y Breakeven (`T5YIE`), Oil (`DCOILWTICO`).
3. **Growth:** Ind Production (`INDPRO`), Retail Sales (`RSAFS`), Housing Starts (`HOUST`).
4. **Liquidity/Forex:** M2 (`M2SL`), USD Index (`DX-Y.NYB` via Yahoo).
5. **Risk:** VIX (`^VIX` via Yahoo), High Yield Spread (`BAMLH0A0HYM2`).

## Constraints & Code Style
- **Type Hinting:** Strict `typing` usage.
- **Docstrings:** Google-style docstrings explaining the mathematical transformation.
- **Libraries:** `pandas`, `numpy`, `yfinance`, `fredapi`, `scipy` (for interpolation).
- **Logging:** Use the existing `silkroad.logger.logger`.

## Example Usage to Enable
```python
loader = MacroDataLoader(fred_api_key="...")
df = loader.fetch_data(
    start_date="2010-01-01", 
    end_date="2023-12-31", 
    transform_stationary=True, # Auto-converts CPI to YoY % change
    interpolation="cubic"      # Smooths monthly data to daily
)