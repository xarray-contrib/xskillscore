import numpy as np
import pytest
import xarray as xr

from xskillscore import Contingency

PERIODS = 12  # effective_p_value produces nans for shorter periods

np.random.seed(42)


@pytest.fixture
def o():
    """Observation."""
    times = xr.cftime_range(start="2000", periods=PERIODS, freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(
        data,
        coords=[times, lats, lons],
        dims=["time", "lat", "lon"],
        attrs={"source": "test"},
    )


@pytest.fixture
def f_prob():
    """Probabilistic forecast containing also a member dimension."""
    times = xr.cftime_range(start="2000", periods=PERIODS, freq="D")
    members = np.arange(3)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(members), len(times), len(lats), len(lons))
    return xr.DataArray(
        data,
        coords=[members, times, lats, lons],
        dims=["member", "time", "lat", "lon"],
        attrs={"source": "test"},
    )


@pytest.fixture
def f(f_prob):
    """Deterministic forecast matching observation o."""
    return f_prob.isel(member=0, drop=True)


# a vs. b in deterministic, o vs. f in probabilistic
@pytest.fixture
def a(o):
    return o


@pytest.fixture
def b(f):
    return f


# nan
@pytest.fixture
def a_nan(a):
    """Masked"""
    return a.copy().where(a < 0.5)


@pytest.fixture
def b_nan(b):
    """Masked"""
    return b.copy().where(b < 0.5)


# with zeros
@pytest.fixture
def a_with_zeros(a):
    """Zeros"""
    return a.copy().where(a < 0.5, 0)


# dask
@pytest.fixture
def a_dask(a):
    """Chunked"""
    return a.chunk()


@pytest.fixture
def b_dask(b):
    return b.chunk()


@pytest.fixture
def o_dask(o):
    return o.chunk()


@pytest.fixture
def f_prob_dask(f_prob):
    return f_prob.chunk()


# 1D time
@pytest.fixture
def a_1d(a):
    """Timeseries of a"""
    return a.isel(lon=0, lat=0, drop=True)


@pytest.fixture
def b_1d(b):
    """Timeseries of b"""
    return b.isel(lon=0, lat=0, drop=True)


@pytest.fixture
def a_1d_nan():
    time = xr.cftime_range("2000-01-01", "2000-01-03", freq="D")
    return xr.DataArray([3, np.nan, 5], dims=["time"], coords=[time])


@pytest.fixture
def b_1d_nan(a_1d_nan):
    b = a_1d_nan.copy()
    b.values = [7, 2, np.nan]
    return b


@pytest.fixture
def a_1d_with_zeros(a_with_zeros):
    """Timeseries of a with zeros"""
    return a_with_zeros.isel(lon=0, lat=0, drop=True)


# weights
@pytest.fixture
def weights(a):
    """Weighting array by cosine of the latitude."""
    return xr.ones_like(a) * np.abs(np.cos(a.lat))


@pytest.fixture
def weights_lonlat(a):
    weights = np.cos(np.deg2rad(a.lat))
    _, weights = xr.broadcast(a, weights)
    return weights.isel(time=0, drop=True)


@pytest.fixture
def weights_time():
    time = xr.cftime_range("2000-01-01", "2000-01-03", freq="D")
    return xr.DataArray([1, 2, 3], dims=["time"], coords=[time])


@pytest.fixture
def weights_dask(weights):
    """
    Weighting array by cosine of the latitude.
    """
    return weights.chunk()


@pytest.fixture
def category_edges():
    """Category bin edges between 0 and 1."""
    return np.linspace(0, 1 + 1e-8, 6)


@pytest.fixture
def forecast_3d():
    """Random 3D forecast used for testing Contingency."""
    times = xr.cftime_range(start="2000", freq="D", periods=10)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.randint(0, 10, size=(len(times), len(lats), len(lons)))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def observation_3d(forecast_3d):
    """Random 3D observation used for testing Contingency."""
    b = forecast_3d.copy()
    b.values = np.random.randint(0, 10, size=(b.shape[0], b.shape[1], b.shape[2]))
    return b


@pytest.fixture
def dichotomous_Contingency_1d():
    """Contingency of fixed, dichotomous 1d forecast and observations."""
    observations = xr.DataArray(
        np.array(2 * [0] + 2 * [1] + 1 * [0] + 2 * [1]), coords=[("x", np.arange(7))]
    )
    forecasts = xr.DataArray(
        np.array(2 * [0] + 2 * [0] + 1 * [1] + 2 * [1]), coords=[("x", np.arange(7))]
    )
    category_edges = np.array([-np.inf, 0.5, np.inf])
    return Contingency(
        observations, forecasts, category_edges, category_edges, dim=["x"]
    )


@pytest.fixture
def symmetric_edges():
    """Category bin edges between 0 and 1."""
    return np.linspace(-2, 2, 11)


np.random.seed(42)


@pytest.fixture
def forecast_1d_long():
    """Forecasts normally distributed around 0."""
    s = 100
    return xr.DataArray(np.random.normal(size=(s)), coords=[("time", np.arange(s))])


@pytest.fixture
def observation_1d_long():
    """Observations normally distributed around 0."""
    s = 100
    return xr.DataArray(np.random.normal(size=(s)), coords=[("time", np.arange(s))])
