import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def o():
    """Observation."""
    times = xr.cftime_range(start='2000', periods=3, freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(lats), len(lons), len(times))
    return xr.DataArray(data, coords=[lats, lons, times], dims=['lat', 'lon', 'times'])


@pytest.fixture
def f_prob():
    """Probabilistic forecast containing also a member dimension."""
    times = xr.cftime_range(start='2000', periods=3, freq='D')
    members = np.arange(3)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(members), len(lats), len(lons), len(times))
    return xr.DataArray(
        data,
        coords=[members, lats, lons, times],
        dims=['member', 'lat', 'lon', 'times'],
    )


@pytest.fixture
def f(f_prob):
    """Deterministic forecast matching observation a."""
    return f_prob.isel(member=0, drop=True)


@pytest.fixture
def a(o):
    return o


@pytest.fixture
def b(f):
    return f


@pytest.fixture
def weights(a):
    """Weighting array by cosine of the latitude."""
    return xr.ones_like(a) * np.abs(np.cos(a.lat))
