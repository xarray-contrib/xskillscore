import numpy as np
import pytest
import xarray as xr

PERIODS = 10  # effective_p_value produces nans for shorter periods


@pytest.fixture
def o():
    """Observation."""
    times = xr.cftime_range(start='2000', periods=PERIODS, freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(
        data,
        coords=[times, lats, lons],
        dims=['time', 'lat', 'lon'],
        attrs={'source': 'test'},
    )


@pytest.fixture
def o_dask(o):
    return o.chunk()


@pytest.fixture
def f_prob():
    """Probabilistic forecast containing also a member dimension."""
    times = xr.cftime_range(start='2000', periods=PERIODS, freq='D')
    members = np.arange(3)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(members), len(times), len(lats), len(lons))
    return xr.DataArray(
        data,
        coords=[members, times, lats, lons],
        dims=['member', 'time', 'lat', 'lon'],
        attrs={'source': 'test'},
    )


@pytest.fixture
def f(f_prob):
    """Deterministic forecast matching observation o."""
    return f_prob.isel(member=0, drop=True)


@pytest.fixture
def f_prob_dask(f_prob):
    return f_prob.chunk()


@pytest.fixture
def a(o):
    return o


@pytest.fixture
def b(f):
    return f


@pytest.fixture
def a_1d(a):
    """Timeseries of a"""
    return a.isel(lon=0, lat=0, drop=True)


@pytest.fixture
def b_1d(b):
    """Timeseries of b"""
    return b.isel(lon=0, lat=0, drop=True)


@pytest.fixture
def b_nan(b):
    """Masked"""
    b = b.copy()
    return b.where(b < 0.5)


@pytest.fixture
def a_dask(a):
    """Chunked"""
    return a.chunk()


@pytest.fixture
def b_dask(b):
    return b.chunk()


@pytest.fixture
def weights(a):
    """Weighting array by cosine of the latitude."""
    return xr.ones_like(a) * np.abs(np.cos(a.lat))


@pytest.fixture
def weights_dask(weights):
    """
    Weighting array by cosine of the latitude.
    """
    return weights.chunk()
