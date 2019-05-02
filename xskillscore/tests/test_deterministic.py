import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import assert_allclose

from xskillscore.core.deterministic import (
    _preprocess, mae, mse, pearson_r, pearson_r_p_value, rmse)
from xskillscore.core.np_deterministic import (
    _mae, _mse, _pearson_r, _pearson_r_p_value, _rmse)


AXES = ('time', 'lat', 'lon', ('lat', 'lon'), ('time', 'lat', 'lon'))


@pytest.fixture
def a():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def b():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def a_dask():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


@pytest.fixture
def b_dask(b):
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


@pytest.mark.parametrize('dim', AXES)
def test_pearson_r_xr(a, b, dim):
    actual = pearson_r(a, b, dim)

    dim, _ = _preprocess(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b

    axis = _a.dims.index(new_dim)
    res = _pearson_r(_a.values, _b.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_pearson_r_xr_dask(a_dask, b_dask, dim):
    actual = pearson_r(a_dask, b_dask, dim)

    dim, _ = _preprocess(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a_dask = a_dask.stack(**{new_dim: dim})
        _b_dask = b_dask.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a_dask = a_dask
        _b_dask = b_dask

    axis = _a_dask.dims.index(new_dim)
    res = _pearson_r(_a_dask.values, _b_dask.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_pearson_r_p_value_xr(a, b, dim):
    actual = pearson_r_p_value(a, b, dim)

    dim, _ = _preprocess(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b

    axis = _a.dims.index(new_dim)
    res = _pearson_r_p_value(_a.values, _b.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_pearson_r_p_value_xr_dask(a_dask, b_dask, dim):
    actual = pearson_r_p_value(a_dask, b_dask, dim)

    dim, _ = _preprocess(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a_dask = a_dask.stack(**{new_dim: dim})
        _b_dask = b_dask.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a_dask = a_dask
        _b_dask = b_dask

    axis = _a_dask.dims.index(new_dim)
    res = _pearson_r_p_value(_a_dask.values, _b_dask.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_rmse_r_xr(a, b, dim):
    actual = rmse(a, b, dim)
    dim, axis = _preprocess(dim)
    _a = a.values
    _b = b.values
    axis = tuple(a.dims.index(d) for d in dim)
    res = _rmse(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_rmse_r_xr_dask(a_dask, b_dask, dim):
    actual = rmse(a_dask, b_dask, dim)
    dim, axis = _preprocess(dim)
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _rmse(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_mse_r_xr(a, b, dim):
    actual = mse(a, b, dim)
    dim, axis = _preprocess(dim)
    _a = a.values
    _b = b.values
    axis = tuple(a.dims.index(d) for d in dim)
    res = _mse(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_mse_r_xr_dask(a_dask, b_dask, dim):
    actual = mse(a_dask, b_dask, dim)
    dim, axis = _preprocess(dim)
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _mse(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_mae_r_xr(a, b, dim):
    actual = mae(a, b, dim)
    dim, axis = _preprocess(dim)
    _a = a.values
    _b = b.values
    axis = tuple(a.dims.index(d) for d in dim)
    res = _mae(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
def test_mae_r_xr_dask(a_dask, b_dask, dim):
    actual = mae(a_dask, b_dask, dim)
    dim, axis = _preprocess(dim)
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _mae(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)
