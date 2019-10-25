import bottleneck as bn
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import distributions
from xarray.tests import assert_allclose

from xskillscore.core.deterministic import (_preprocess_dims,
                                            _preprocess_weights, mae, mse,
                                            pearson_r, pearson_r_p_value, rmse,
                                            smape, spearman_r,
                                            spearman_r_p_value)
from xskillscore.core.np_deterministic import (_mae, _mse, _pearson_r,
                                               _pearson_r_p_value, _rmse,
                                               _spearman_r)

AXES = ("time", "lat", "lon", ("lat", "lon"), ("time", "lat", "lon"))


@pytest.fixture
def a():
    dates = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data, coords=[dates, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def b():
    dates = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data, coords=[dates, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def weights():
    """
    Weighting array by cosine of the latitude.
    """
    dates = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    cos = np.abs(np.cos(lats))
    data = np.tile(cos, (len(dates), len(lons), 1)).reshape(
        len(dates), len(lats), len(lons)
    )
    return xr.DataArray(data, coords=[dates, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def weights_dask():
    """
    Weighting array by cosine of the latitude.
    """
    dates = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    cos = np.abs(np.cos(lats))
    data = np.tile(cos, (len(dates), len(lons), 1)).reshape(
        len(dates), len(lats), len(lons)
    )
    return xr.DataArray(
        data, coords=[dates, lats, lons], dims=["time", "lat", "lon"]
    ).chunk()


@pytest.fixture
def a_dask():
    dates = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(
        data, coords=[dates, lats, lons], dims=["time", "lat", "lon"]
    ).chunk()


@pytest.fixture
def b_dask(b):
    dates = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(
        data, coords=[dates, lats, lons], dims=["time", "lat", "lon"]
    ).chunk()


def adjust_weights(dim, weight, weights):
    """
    Adjust the weights test data to only span the core dimension
    that the function is being applied over.
    """
    if weight:
        drop_dims = [i for i in weights.dims if i not in dim]
        drop_dims = {k: 0 for k in drop_dims}
        return weights.isel(drop_dims)
    else:
        return None


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_pearson_r_xr(a, b, dim, weight, weights):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights)

    actual = pearson_r(a, b, dim, weights=_weights)
    assert actual.chunks is None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
        if weight:
            _weights = _weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b
    _weights = _preprocess_weights(_a, dim, new_dim, _weights)

    axis = _a.dims.index(new_dim)
    res = _pearson_r(_a.values, _b.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_pearson_r_xr_dask(a_dask, b_dask, dim, weight, weights_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights_dask)

    actual = pearson_r(a_dask, b_dask, dim, weights=_weights)
    assert actual.chunks is not None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        _a_dask = a_dask.stack(**{new_dim: dim})
        _b_dask = b_dask.stack(**{new_dim: dim})
        if weight:
            _weights = _weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a_dask = a_dask
        _b_dask = b_dask
    _weights = _preprocess_weights(_a_dask, dim, new_dim, _weights)

    axis = _a_dask.dims.index(new_dim)
    res = _pearson_r(_a_dask.values, _b_dask.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_pearson_r_p_value_xr(a, b, dim, weight, weights):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights)

    actual = pearson_r_p_value(a, b, dim, weights=_weights)
    assert actual.chunks is None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
        if weight:
            _weights = _weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b
    _weights = _preprocess_weights(_a, dim, new_dim, _weights)

    axis = _a.dims.index(new_dim)
    res = _pearson_r_p_value(_a.values, _b.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_pearson_r_p_value_xr_dask(a_dask, b_dask, dim, weight, weights_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights_dask)

    actual = pearson_r_p_value(a_dask, b_dask, dim, weights=_weights)
    assert actual.chunks is not None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        _a_dask = a_dask.stack(**{new_dim: dim})
        _b_dask = b_dask.stack(**{new_dim: dim})
        if weight:
            _weights = _weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a_dask = a_dask
        _b_dask = b_dask
    _weights = _preprocess_weights(_a_dask, dim, new_dim, _weights)

    axis = _a_dask.dims.index(new_dim)
    res = _pearson_r_p_value(
        _a_dask.values, _b_dask.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_rmse_xr(a, b, dim, weight, weights):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights = adjust_weights(dim, weight, weights)

    actual = rmse(a, b, dim, weights=weights)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights = _preprocess_weights(_a, dim, dim, weights)
    axis = tuple(a.dims.index(d) for d in dim)
    res = _rmse(_a.values, _b.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_rmse_xr_dask(a_dask, b_dask, dim, weight, weights_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights_dask)

    actual = rmse(a_dask, b_dask, dim, weights=_weights)
    assert actual.chunks is not None

    dim, axis = _preprocess_dims(dim)
    _a_dask = a_dask
    _b_dask = b_dask
    _weights = _preprocess_weights(_a_dask, dim, dim, _weights)
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _rmse(_a_dask.values, _b_dask.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_mse_xr(a, b, dim, weight, weights):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights = adjust_weights(dim, weight, weights)

    actual = mse(a, b, dim, weights=weights)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights = _preprocess_weights(_a, dim, dim, weights)
    axis = tuple(a.dims.index(d) for d in dim)
    res = _mse(_a.values, _b.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_mse_xr_dask(a_dask, b_dask, dim, weight, weights_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights_dask)

    actual = mse(a_dask, b_dask, dim, weights=_weights)
    assert actual.chunks is not None

    dim, axis = _preprocess_dims(dim)
    _a_dask = a_dask
    _b_dask = b_dask
    _weights = _preprocess_weights(_a_dask, dim, dim, _weights)
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _mse(_a_dask.values, _b_dask.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_mae_xr(a, b, dim, weight, weights):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights = adjust_weights(dim, weight, weights)

    actual = mae(a, b, dim, weights=weights)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights = _preprocess_weights(_a, dim, dim, weights)
    axis = tuple(a.dims.index(d) for d in dim)
    res = _mae(_a.values, _b.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight", [True, False])
def test_mae_xr_dask(a_dask, b_dask, dim, weight, weights_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    _weights = adjust_weights(dim, weight, weights_dask)
    actual = mae(a_dask, b_dask, dim, weights=_weights)
    assert actual.chunks is not None

    dim, axis = _preprocess_dims(dim)
    _a_dask = a_dask
    _b_dask = b_dask
    _weights = _preprocess_weights(_a_dask, dim, dim, _weights)
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _mae(_a_dask.values, _b_dask.values, _weights.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
def test_spearman_r(a, b, dim):
    """Test spearman_r with bottleneck.rankdata and pearson_r."""
    actual = spearman_r(a, b, dim)
    # dirty fix, this only tests whether spearman_r is equal to pearson_r rankdata
    # but this tests spearman_r doesnt crash on all AXES
    if len(dim) == 1:
        axis = list(a, dim).index(dim) - 1
        a2 = bn.rankdata(a, axis)
        b2 = bn.rankdata(b, axis)
        expected = pearson_r(a2, b2, dim)
        assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
def test_spearman_r_p_value(a, b, dim):
    """Test spearman_r with bottleneck.rankdata and pearson_r."""
    actual = spearman_r_p_value(a, b, dim)
    # dirty fix, this only tests whether spearman_r is equal to pearson_r rankdata
    # but this tests spearman_r doesnt crash on all AXES
    if len(dim) == 1:
        dof = a[dim].size - 2  # degrees of freedom
        rs = _spearman_r(a, b, dim)
        t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
        expected = 2 * distributions.t.sf(np.abs(t), dof)
        assert_allclose(actual, expected)


@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("metric", [smape])
def test_percentage_metric_in_interval_0_1(a, b, dim, metric):
    """Test smape to be within bounds."""
    res = metric(a, b, dim)
    print(res.mean())
    assert not (res < 0).any()
    assert not (res > 1).any()
    assert not res.isnull().any()
