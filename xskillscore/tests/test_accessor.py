import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose

from xskillscore.core.deterministic import mae, pearson_r, rmse
from xskillscore.core.probabilistic import xr_crps_gaussian


@pytest.fixture
def ds_dask():
    dates = xr.cftime_range('2000-01-01', '2000-01-03', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    a = xr.DataArray(
        np.random.rand(len(dates), len(lats), len(lons)),
        coords=[dates, lats, lons],
        dims=['time', 'lat', 'lon'],
    ).chunk()
    b = xr.DataArray(
        np.random.rand(len(dates), len(lats), len(lons)),
        coords=[dates, lats, lons],
        dims=['time', 'lat', 'lon'],
    ).chunk()

    ds = xr.Dataset()
    ds['a'] = a
    ds['b'] = b
    return ds


@pytest.fixture
def mu():
    return xr.DataArray(5)


@pytest.fixture
def sigma():
    return xr.DataArray(6)


def test_pearson_r_accessor(ds_dask):
    ds = ds_dask.load()

    dim = 'time'
    actual = pearson_r(ds['a'], ds['b'], dim)
    expected = ds.xs.pearson_r('a', 'b', dim)
    assert_allclose(actual, expected)


def test_rmse_accessor_dask(ds_dask):
    dim = 'lon'
    actual = rmse(ds_dask['a'], ds_dask['b'], dim).compute()
    expected = ds_dask.xs.rmse('a', 'b', dim).compute()
    assert_allclose(actual, expected)


def test_mae_accessor_outer_array(ds_dask):
    ds = ds_dask.load()
    b = ds['b']
    ds = ds.drop_vars('b')
    dim = 'lat'

    actual = mae(ds['a'], b, dim)
    expected = ds.xs.mae('a', b, dim)
    assert_allclose(actual, expected)


def test_crps_gaussian_accessor(ds_dask, mu, sigma):
    ds = ds_dask.load()
    ds['mu'] = mu
    ds['sigma'] = sigma

    actual = xr_crps_gaussian(ds['a'], mu, sigma)
    expected = ds.xs.crps_gaussian('a', 'mu', 'sigma')
    assert_allclose(actual, expected)


def test_crps_gaussian_accessor_outer_array(ds_dask, mu, sigma):
    ds = ds_dask.load()
    actual = xr_crps_gaussian(ds['a'], mu, sigma)
    expected = ds.xs.crps_gaussian('a', mu, sigma)
    assert_allclose(actual, expected)
