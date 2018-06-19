from xarray.tests import assert_allclose
import xarray as xr
import pandas as pd
import numpy as np
import pytest


from.np_deterministic import _pearson_r, _rmse


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
    
@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_pearson_r_xr(a, b, dim):
    actual = xr.apply_ufunc(_pearson_r, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1})
    _a = a.values
    _b = b.values
    axis = a.dims.index('time')
    expected = _pearson_r(_a, _b, axis)
    assert_allclose(actual, expected)
    
@pytest.mark.parametrize('dim', ('time', 'lat', 'lon'))
def test_rmse_r_xr(a, b, dim):
    actual = xr.apply_ufunc(_rmse, a, b,
                            input_core_dims=[[dim], [dim]],
                            kwargs={'axis': -1})
    _a = a.values
    _b = b.values
    axis = a.dims.index('time')
    expected = _rmse(_a, _b, axis)
    assert_allclose(actual, expected)    