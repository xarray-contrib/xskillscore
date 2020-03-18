import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import assert_allclose
from xskillscore.core.deterministic import (
    _preprocess_dims,
    _preprocess_weights,
    effective_sample_size,
    mae,
    mape,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_eff_p_value,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_eff_p_value,
    spearman_r_p_value,
    r2,
)
from xskillscore.core.np_deterministic import (
    _effective_sample_size,
    _mae,
    _mape,
    _median_absolute_error,
    _mse,
    _pearson_r,
    _pearson_r_eff_p_value,
    _pearson_r_p_value,
    _rmse,
    _smape,
    _spearman_r,
    _spearman_r_eff_p_value,
    _spearman_r_p_value,
    _r2,
)

correlation_metrics = [
    (pearson_r, _pearson_r),
    (r2, _r2),
    (pearson_r_p_value, _pearson_r_p_value),
    (pearson_r_eff_p_value, _pearson_r_eff_p_value),
    (spearman_r, _spearman_r),
    (spearman_r_p_value, _spearman_r_p_value),
    (spearman_r_eff_p_value, _spearman_r_eff_p_value),
    (effective_sample_size, _effective_sample_size),
]
distance_metrics = [
    (mse, _mse),
    (rmse, _rmse),
    (mae, _mae),
    (median_absolute_error, _median_absolute_error),
    (mape, _mape),
    (smape, _smape),
]

AXES = ("time", "lat", "lon", ["lat", "lon"], ["time", "lat", "lon"])

temporal_only_metrics = [
    pearson_r_eff_p_value,
    spearman_r_eff_p_value,
    effective_sample_size,
]


@pytest.fixture
def a():
    times = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def b(a):
    b = a.copy()
    b.values = np.random.rand(a.shape[0], a.shape[1], a.shape[2])
    return b


@pytest.fixture
def b_nan(b):
    return b.where(b < 0.5)


@pytest.fixture
def weights(a):
    """Weighting array by cosine of the latitude."""
    a_weighted = a.copy()
    cos = np.abs(np.cos(a.lat))
    data = np.tile(cos, (a.shape[0], a.shape[2], 1)).reshape(
        a.shape[0], a.shape[1], a.shape[2]
    )
    a_weighted.values = data
    return a_weighted


@pytest.fixture
def a_dask(a):
    return a.chunk()


@pytest.fixture
def b_dask(b):
    return b.chunk()


@pytest.fixture
def weights_dask(weights):
    """
    Weighting array by cosine of the latitude.
    """
    return weights.chunk()


def adjust_weights(dim, weight_bool, weights):
    """
    Adjust the weights test data to only span the core dimension
    that the function is being applied over.
    """
    if weight_bool:
        drop_dims = [i for i in weights.dims if i not in dim]
        drop_dims = {k: 0 for k in drop_dims}
        return weights.isel(drop_dims)
    else:
        return None


@pytest.mark.parametrize("metrics", correlation_metrics)
@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight_bool", [True, False])
def test_correlation_metrics_xr(a, b, dim, weight_bool, weights, metrics):
    """Test whether correlation metric for xarray functions (from
     deterministic.py) give save numerical results as for numpy functions from
     np_deterministic.py)."""
    # unpack metrics
    metric, _metric = metrics
    # Only apply over time dimension for effective p value.
    if (dim != "time") and (metric in temporal_only_metrics):
        dim = "time"
    # Generates subsetted weights to pass in as arg to main function and for
    # the numpy testing.
    _weights = adjust_weights(dim, weight_bool, weights)
    if metric in temporal_only_metrics:
        actual = metric(a, b, dim)
    else:
        actual = metric(a, b, dim, weights=_weights)
    # check that no chunks for no chunk inputs
    assert actual.chunks is None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
        if weight_bool:
            _weights = _weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b
    _weights = _preprocess_weights(_a, dim, new_dim, _weights)

    # ensure _weights.values or None
    _weights = None if _weights is None else _weights.values

    axis = _a.dims.index(new_dim)
    if metric in temporal_only_metrics:
        res = _metric(_a.values, _b.values, axis, skipna=False)
    else:
        res = _metric(_a.values, _b.values, _weights, axis, skipna=False)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("metrics", distance_metrics)
@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight_bool", [True, False])
def test_distance_metrics_xr(a, b, dim, weight_bool, weights, metrics):
    """Test whether distance-based metric for xarray functions (from
     deterministic.py) give save numerical results as for numpy functions from
     np_deterministic.py)."""
    # unpack metrics
    metric, _metric = metrics
    # Generates subsetted weights to pass in as arg to main function and for
    # the numpy testing.
    weights = adjust_weights(dim, weight_bool, weights)
    # median absolute error has no weights argument
    if metric is median_absolute_error:
        actual = metric(a, b, dim)
    else:
        actual = metric(a, b, dim, weights=weights)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights = _preprocess_weights(_a, dim, dim, weights)
    axis = tuple(a.dims.index(d) for d in dim)
    if metric is median_absolute_error:
        res = _metric(_a.values, _b.values, axis, skipna=False)
    else:
        # ensure _weights.values or None
        _weights = None if _weights is None else _weights.values
        res = _metric(_a.values, _b.values, _weights, axis, skipna=False)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("metrics", correlation_metrics)
@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight_bool", [True, False])
def test_correlation_metrics_xr_dask(
    a_dask, b_dask, dim, weight_bool, weights_dask, metrics
):
    """Test whether correlation metric for xarray functions can be lazy when
     chunked by using dask and give same results."""
    a = a_dask
    b = b_dask
    weights = weights_dask
    # unpack metrics
    metric, _metric = metrics
    # Only apply over time dimension for effective p value.
    if (dim != "time") and (metric in temporal_only_metrics):
        dim = "time"
    # Generates subsetted weights to pass in as arg to main function and for
    # the numpy testing.
    _weights = adjust_weights(dim, weight_bool, weights)

    if metric in temporal_only_metrics:
        actual = metric(a, b, dim)
    else:
        actual = metric(a, b, dim, weights=_weights)
    # check that chunks for chunk inputs
    assert actual.chunks is not None

    if _weights is not None:
        _weights = _weights.load()

    if metric in temporal_only_metrics:
        expected = metric(a.load(), b.load(), dim)
    else:
        expected = metric(a.load(), b.load(), dim, _weights)
    assert expected.chunks is None
    assert_allclose(actual.compute(), expected)


@pytest.mark.parametrize("metrics", distance_metrics)
@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight_bool", [True, False])
def test_distance_metrics_xr_dask(
    a_dask, b_dask, dim, weight_bool, weights_dask, metrics
):
    """Test whether distance metrics for xarray functions can be lazy when
     chunked by using dask and give same results."""
    a = a_dask.copy()
    b = b_dask.copy()
    weights = weights_dask.copy()
    # unpack metrics
    metric, _metric = metrics
    # Generates subsetted weights to pass in as arg to main function and for
    # the numpy testing.
    _weights = adjust_weights(dim, weight_bool, weights)
    if _weights is not None:
        _weights = _weights.load()
    if metric is median_absolute_error:
        actual = metric(a, b, dim)
    else:
        actual = metric(a, b, dim, weights=_weights)
    # check that chunks for chunk inputs
    assert actual.chunks is not None
    if metric is median_absolute_error:
        expected = metric(a.load(), b.load(), dim)
    else:
        expected = metric(a.load(), b.load(), dim, weights=_weights)
    assert expected.chunks is None
    assert_allclose(actual.compute(), expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("metric", [smape])
def test_percentage_metric_in_interval_0_1(a, b, dim, metric):
    print(a, b)
    """Test smape to be within bounds."""
    res = metric(a, b, dim)
    assert not (res < 0).any()
    assert not (res > 1).any()
    assert not res.isnull().any()


def test_pearson_r_p_value_skipna(a, b_nan):
    """Test whether NaNs sprinkled in array will NOT yield all NaNs."""
    res = pearson_r_p_value(a, b_nan, ["lat", "lon"], skipna=True)
    assert not np.isnan(res).all()


# def test_pearson_r_eff_p_value_skipna(a, b_nan):
#     """Test whether NaNs sprinkled in array will NOT yield all NaNs."""
#     res = pearson_r_eff_p_value(a, b_nan, ["lat", "lon"], skipna=True)
#     assert not np.isnan(res).all()


def test_pearson_r_integer():
    """Test whether arrays as integers work."""
    da = xr.DataArray([0, 1, 2], dims=["time"])
    assert pearson_r(da, da, dim="time") == 1
