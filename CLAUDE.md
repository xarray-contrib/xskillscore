# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**xskillscore** is a Python package for computing forecast verification metrics using xarray. It provides both deterministic and probabilistic forecast verification metrics designed to work with multi-dimensional labeled arrays, with support for Dask parallel computing.

Originally developed to parallelize forecast metrics for multi-model-multi-ensemble forecasts in the SubX project.

## Development Commands

### Testing

Run full test suite:
```bash
pytest -n 4 --cov=xskillscore --cov-report=xml --verbose
```

Run tests for a single file:
```bash
pytest xskillscore/tests/test_deterministic.py
```

Run a specific test:
```bash
pytest xskillscore/tests/test_deterministic.py::test_pearson_r -v
```

Run tests with specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m "not network"  # Skip tests requiring network
```

### Doctests

Run doctests on all modules:
```bash
python -m pytest --doctest-modules xskillscore --ignore xskillscore/tests
```

### Code Quality

Run pre-commit checks:
```bash
pre-commit run --all-files
```

Linting and formatting (via ruff):
```bash
ruff check --fix .
ruff format .
```

Type checking:
```bash
mypy xskillscore
```

### Documentation

Build documentation:
```bash
cd docs
make html
```

Test notebooks in documentation:
```bash
cd docs
nbstripout source/*.ipynb
make -j4 html
```

### Installation

Install in development mode:
```bash
pip install -e .
```

Install with test dependencies:
```bash
pip install -e ".[test]"
```

Install with all dependencies:
```bash
pip install -e ".[complete]"
```

## Architecture

### Core Module Structure

The `xskillscore/core/` directory contains the main implementation:

- **deterministic.py**: Deterministic forecast metrics (pearson_r, rmse, mae, mse, etc.)
- **probabilistic.py**: Probabilistic metrics (crps_*, brier_score, rps, rank_histogram, etc.)
- **comparative.py**: Comparative tests (sign_test, halfwidth_ci_test)
- **stattests.py**: Statistical tests (multipletests)
- **contingency.py**: Contingency table class and categorical metrics
- **resampling.py**: Resampling and bootstrapping utilities
- **accessor.py**: xarray accessor (`ds.xs.metric()`) for convenient API
- **utils.py**: Shared utilities for preprocessing dimensions, weights, and broadcasting
- **np_deterministic.py**: NumPy implementations of deterministic metrics
- **np_probabilistic.py**: NumPy implementations of probabilistic metrics
- **types.py**: Type definitions

### Key Design Patterns

1. **xarray.apply_ufunc Pattern**: All metrics use `xr.apply_ufunc` to:
   - Apply NumPy implementations to xarray objects
   - Handle broadcasting automatically
   - Enable Dask parallelization with `dask="parallelized"`
   - Preserve attributes with `keep_attrs` parameter

2. **Dimension Preprocessing**: Metrics follow this pattern:
   ```python
   dim, axis = _preprocess_dims(dim, a)  # Convert dim to list and axis tuple
   a, b = xr.broadcast(a, b, exclude=dim)  # Broadcast arrays
   a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)  # Stack multi-dims
   weights = _preprocess_weights(a, dim, new_dim, weights)  # Normalize weights
   ```

3. **Separation of xarray and NumPy logic**:
   - High-level functions in `deterministic.py`/`probabilistic.py` handle xarray objects
   - Low-level functions in `np_deterministic.py`/`np_probabilistic.py` contain pure NumPy logic
   - This enables easier testing and reuse

4. **Optional Weights**: Most metrics support optional `weights` parameter matching the dimensions being reduced.

5. **Member Dimension Convention**: Probabilistic metrics use `member_dim="member"` by default for ensemble dimensions.

### xarray Accessor

Users can access metrics via the `.xs` accessor on xarray Datasets:
```python
ds = xr.Dataset({"a": a_dataarray, "b": b_dataarray})
result = ds.xs.pearson_r("a", "b", dim="time")
```

The accessor handles converting string variable names to actual DataArrays.

### Testing Infrastructure

- **conftest.py**: Centralized pytest fixtures for test data (times, lats, lons, members, etc.)
- Test fixtures provide consistent test data across test modules
- Fixtures include regular data, NaN-masked data, dask-chunked data, and 1D timeseries
- Use `np.random.seed(42)` in doctests for deterministic examples

## Important Considerations

### Temporal Metrics

Some metrics are specifically designed for temporal dimensions:
- `effective_sample_size()`, `pearson_r_eff_p_value()`, `spearman_r_eff_p_value()`
- These raise warnings if applied to non-"time" dimensions
- They account for autocorrelation and should only be used on time series

### NumPy Version Compatibility

The codebase supports both numpy<2.0 and numpy>=2.0. When using NumPy functions:
- Use try/except for imports that changed between versions
- Example: `trapezoid` (new) vs `trapz` (old)

### Dimension Handling

- `dim=None` means reduce over all dimensions
- `dim` can be a string or list of strings
- When multiple dimensions are provided, they are stacked into a single dimension internally
- The `member` dimension in probabilistic forecasts is special and should not be included in `dim`

### NaN Handling

- Most metrics support `skipna` parameter (default: False)
- Probabilistic metrics use `_keep_nans_masked()` to preserve NaN patterns from inputs

### Dask Support

All metrics support Dask arrays via `dask="parallelized"` in `xr.apply_ufunc`. No special handling needed when adding new metrics.

## Python Support

- Minimum Python version: 3.9
- Supported versions: 3.9, 3.10, 3.11, 3.12, 3.13

## Key Dependencies

- xarray >= 2023.4.0 (core data structure)
- numpy >= 1.24
- scipy >= 1.10
- dask[array] >= 2023.4.0 (parallel computing)
- properscoring (probabilistic metrics)
- xhistogram >= 0.3.2 (histogram computations)
- statsmodels (statistical tests)

Optional acceleration:
- bottleneck (faster NaN operations)
- numba >= 0.57 (JIT compilation)

## Contributing Workflow

1. Create a new branch for your feature
2. Make changes and add tests in `xskillscore/tests/`
3. Add docstring examples (they are tested via doctest)
4. Run `pre-commit run --all-files` before committing
5. Ensure tests pass: `pytest -n 4`
6. Ensure doctests pass: `python -m pytest --doctest-modules xskillscore --ignore xskillscore/tests`
7. Update CHANGELOG.rst if appropriate
8. Submit PR to main branch

Note: CI includes tests on multiple Python versions, doctest validation, and notebook execution in docs.
