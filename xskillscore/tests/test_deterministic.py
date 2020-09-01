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
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_eff_p_value,
    spearman_r_p_value,
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
    _r2,
    _rmse,
    _smape,
    _spearman_r,
    _spearman_r_eff_p_value,
    _spearman_r_p_value,
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

AXES = ('time', 'lat', 'lon', ['lat', 'lon'], ['time', 'lat', 'lon'])

temporal_only_metrics = [
    pearson_r_eff_p_value,
    spearman_r_eff_p_value,
    effective_sample_size,
]


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


@pytest.mark.parametrize('metrics', correlation_metrics)
@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight_bool', [True, False])
def test_correlation_metrics_xr(a, b, dim, weight_bool, weights, metrics):
    """Test whether correlation metric for xarray functions (from
     deterministic.py) give save numerical results as for numpy functions from
     np_deterministic.py)."""
    # unpack metrics
    metric, _metric = metrics
    # Only apply over time dimension for effective p value.
    if (dim != 'time') and (metric in temporal_only_metrics):
        dim = 'time'
    # Generates subsetted weights to pass in as arg to main function and for
    # the numpy testing.
    _weights = adjust_weights(dim, weight_bool, weights)
    if metric in temporal_only_metrics:
        actual = metric(a, b, dim)
    else:
        actual = metric(a, b, dim, weights=_weights)
    # check that no chunks for no chunk inputs
    assert actual.chunks is None

    dim, _ = _preprocess_dims(dim, a)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
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


@pytest.mark.parametrize('metrics', distance_metrics)
@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight_bool', [True, False])
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

    dim, axis = _preprocess_dims(dim, a)
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


@pytest.mark.parametrize('metrics', correlation_metrics)
@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight_bool', [True, False])
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
    if (dim != 'time') and (metric in temporal_only_metrics):
        dim = 'time'
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


@pytest.mark.parametrize('metrics', distance_metrics)
@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight_bool', [True, False])
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


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('metric', [smape])
def test_percentage_metric_in_interval_0_1(a, b, dim, metric):
    """Test smape to be within bounds."""
    res = metric(a, b, dim)
    assert not (res < 0).any()
    assert not (res > 1).any()
    assert not res.isnull().any()


def test_pearson_r_p_value_skipna(a, b_nan):
    """Test whether NaNs sprinkled in array will NOT yield all NaNs."""
    res = pearson_r_p_value(a, b_nan, ['lat', 'lon'], skipna=True)
    assert not np.isnan(res).all()


# def test_pearson_r_eff_p_value_skipna(a, b_nan):
#     """Test whether NaNs sprinkled in array will NOT yield all NaNs."""
#     res = pearson_r_eff_p_value(a, b_nan, ["lat", "lon"], skipna=True)
#     assert not np.isnan(res).all()


def test_pearson_r_integer():
    """Test whether arrays as integers work."""
    da = xr.DataArray([0, 1, 2], dims=['time'])
    assert pearson_r(da, da, dim='time') == 1


@pytest.mark.parametrize('metrics', correlation_metrics + distance_metrics)
@pytest.mark.parametrize('keep_attrs', [True, False])
def test_keep_attrs(a, b, metrics, keep_attrs):
    """Test keep_attrs for all metrics."""
    metric, _metric = metrics
    # ths tests only copying attrs from a
    res = metric(a, b, 'time', keep_attrs=keep_attrs)
    if keep_attrs:
        assert res.attrs == a.attrs
    else:
        assert res.attrs == {}
    da = xr.DataArray([0, 1, 2], dims=['time'])
    assert pearson_r(da, da, dim='time') == 1


@pytest.mark.parametrize('metrics', correlation_metrics + distance_metrics)
def test_dim_None(a, b, metrics):
    """Test that `dim=None` reduces all dimensions as xr.mean(dim=None) and fails for
    effective metrics."""
    metric, _metric = metrics
    if metric in [effective_sample_size, spearman_r_eff_p_value, pearson_r_eff_p_value]:
        with pytest.raises(ValueError) as excinfo:
            metric(a, b, dim=None)
        assert (
            'Effective sample size should only be applied to a singular time dimension.'
            in str(excinfo.value)
        )
    else:
        metric, _metric = metrics
        res = metric(a, b, dim=None)
        assert len(res.dims) == 0, print(res.dims)


@pytest.mark.parametrize('metrics', correlation_metrics + distance_metrics)
def test_dim_empty_list(a, b, metrics):
    """Test that `dim=[]` reduces no dimensions as xr.mean(dim=[]) and fails for
    correlation metrics."""
    if metrics in correlation_metrics:
        metric, _metric = metrics
        with pytest.raises(ValueError) as excinfo:
            metric(a, b, dim=[])
        assert 'requires `dim` not being empty, found dim' in str(excinfo.value)
    elif metrics in distance_metrics:
        metric, _metric = metrics
        res = metric(a, b, dim=[])
        assert len(res.dims) == len(a.dims), print(res.dims)
