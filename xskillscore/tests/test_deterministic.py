import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import assert_allclose

from xskillscore.core.np_deterministic import (_pearson_r, _pearson_r_p_value,
                                               _rmse, _mse, _mae)


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


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_pearson_r_xr(a, b, dim):
    actual = xr.apply_ufunc(_pearson_r, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a = a.values
    _b = b.values
    axis = a.dims.index(dim)
    res = _pearson_r(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_pearson_r_xr_dask(a_dask, b_dask, dim):
    actual = xr.apply_ufunc(_pearson_r, a_dask, b_dask,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = a_dask.dims.index(dim)
    res = _pearson_r(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_pearson_r_p_value_xr(a, b, dim):
    actual = xr.apply_ufunc(_pearson_r_p_value, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a = a.values
    _b = b.values
    axis = a.dims.index(dim)
    res = _pearson_r_p_value(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_pearson_r_p_value_xr_dask(a_dask, b_dask, dim):
    actual = xr.apply_ufunc(_pearson_r_p_value, a_dask, b_dask,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = a_dask.dims.index(dim)
    res = _pearson_r_p_value(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_rmse_r_xr(a, b, dim):
    actual = xr.apply_ufunc(_rmse, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a = a.values
    _b = b.values
    axis = a.dims.index(dim)
    res = _rmse(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_rmse_r_xr_dask(a_dask, b_dask, dim):
    actual = xr.apply_ufunc(_rmse, a_dask, b_dask,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = a_dask.dims.index(dim)
    res = _rmse(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_mse_r_xr(a, b, dim):
    actual = xr.apply_ufunc(_mse, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a = a.values
    _b = b.values
    axis = a.dims.index(dim)
    res = _mse(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_mse_r_xr_dask(a_dask, b_dask, dim):
    actual = xr.apply_ufunc(_mse, a_dask, b_dask,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = a_dask.dims.index(dim)
    res = _mse(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_mae_r_xr(a, b, dim):
    actual = xr.apply_ufunc(_mae, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a = a.values
    _b = b.values
    axis = a.dims.index(dim)
    res = _mae(_a, _b, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_mae_r_xr_dask(a_dask, b_dask, dim):
    actual = xr.apply_ufunc(_mae, a_dask, b_dask,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1},
                            dask='allowed')
    _a_dask = a_dask.values
    _b_dask = b_dask.values
    axis = a_dask.dims.index(dim)
    res = _mae(_a_dask, _b_dask, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)
