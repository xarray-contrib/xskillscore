import numpy as np
import pytest
import xarray as xr

from xskillscore import Contingency


@pytest.fixture
def a():
    times = xr.cftime_range(start='2000', freq='D', periods=10)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(data, coords=[times, lats, lons], dims=['time', 'lat', 'lon'])


@pytest.fixture
def b(a):
    b = a.copy()
    b.values = np.random.rand(a.shape[0], a.shape[1], a.shape[2])
    return b


@pytest.fixture
def category_edges_a():
    return np.linspace(-2, 2, 5)


@pytest.fixture
def category_edges_b():
    return np.linspace(-3, 3, 5)


@pytest.fixture
def a_dask(a):
    return a.chunk()


@pytest.fixture
def b_dask(b):
    return b.chunk()


DIMS = [['time'], ['lon'], ['lat'], 'time', ['time', 'lon', 'lat']]


@pytest.mark.parametrize('dim', DIMS)
def test_Contingency(a, b, category_edges_a, category_edges_b, dim):
    Contingency(a, b, category_edges_a, category_edges_b, dim=dim)
