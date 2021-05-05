import numpy as np
import pytest
import xarray as xr

import xskillscore as xs
from xskillscore import Contingency

PERIODS = 12  # effective_p_value produces nans for shorter periods


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """imports for doctest"""
    doctest_namespace["np"] = np
    doctest_namespace["xr"] = xr
    doctest_namespace["xs"] = xs

    # always seed numpy.random to make the examples deterministic
    np.random.seed(42)


@pytest.fixture
def times():
    return xr.cftime_range(start="2000", periods=PERIODS, freq="D")


@pytest.fixture
def lats():
    return np.arange(4)


@pytest.fixture
def lons():
    return np.arange(5)


@pytest.fixture
def members():
    return np.arange(3)


# o vs. f in probabilistic
@pytest.fixture
def o(times, lats, lons):
    """Observation."""
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(
        data,
        coords=[times, lats, lons],
        dims=["time", "lat", "lon"],
        attrs={"source": "test"},
    )


@pytest.fixture
def f_prob(times, lats, lons, members):
    """Probabilistic forecast containing also a member dimension."""
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


# a vs. b in deterministic
@pytest.fixture
def a(o):
    return o


@pytest.fixture
def b(f):
    return f


# nan
@pytest.fixture
def a_rand_nan(a):
    """Masked"""
    return a.where(a < 0.5)


@pytest.fixture
def b_rand_nan(b):
    """Masked"""
    return b.where(b < 0.5)


@pytest.fixture
def a_fixed_nan(a):
    """Masked block"""
    a.data[:, 1:3, 1:3] = np.nan
    return a


@pytest.fixture
def b_fixed_nan(b):
    """Masked block"""
    b.data[:, 1:3, 1:3] = np.nan
    return b


# with zeros
@pytest.fixture
def a_with_zeros(a):
    """Zeros"""
    return a.where(a < 0.5, 0)


# dask
@pytest.fixture
def a_dask(a):
    """Chunked"""
    return a.chunk()


@pytest.fixture
def b_dask(b):
    """Chunked"""
    return b.chunk()


@pytest.fixture
def a_rand_nan_dask(a_rand_nan):
    """Chunked"""
    return a_rand_nan.chunk()


@pytest.fixture
def b_rand_nan_dask(b_rand_nan):
    """Chunked"""
    return b_rand_nan.chunk()


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
def a_1d_fixed_nan():
    time = xr.cftime_range("2000-01-01", "2000-01-03", freq="D")
    return xr.DataArray([3, np.nan, 5], dims=["time"], coords=[time])


@pytest.fixture
def b_1d_fixed_nan(a_1d_fixed_nan):
    b = a_1d_fixed_nan.copy()
    b.values = [7, 2, np.nan]
    return b


@pytest.fixture
def a_1d_with_zeros(a_with_zeros):
    """Timeseries of a with zeros"""
    return a_with_zeros.isel(lon=0, lat=0, drop=True)


# weights
@pytest.fixture
def weights_cos_lat(a):
    """Weighting array by cosine of the latitude."""
    return xr.ones_like(a) * np.abs(np.cos(a.lat))


@pytest.fixture
def weights_linear_time(a):
    """Weighting array by linear (1 -> 0) of the time."""
    weights = np.linspace(1, 0, num=len(a.time))
    return xr.ones_like(a) * xr.DataArray(weights, dims="time")


@pytest.fixture
def weights_linear_time_1d(weights_linear_time):
    """Timeseries of weights_linear_time"""
    return weights_linear_time.isel(lon=0, lat=0, drop=True)


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
def weights_cos_lat_dask(weights_cos_lat):
    """
    Weighting array by cosine of the latitude.
    """
    return weights_cos_lat.chunk()


@pytest.fixture
def category_edges():
    """Category bin edges between 0 and 1."""
    return np.linspace(0, 1, 6)


@pytest.fixture
def forecast_3d_int():
    """Random integer 3D forecast used for testing Contingency."""
    times = xr.cftime_range(start="2000", freq="D", periods=10)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.randint(0, 10, size=(len(times), len(lats), len(lons)))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def observation_3d_int(forecast_3d_int):
    """Random integer 3D observation used for testing Contingency."""
    b = forecast_3d_int.copy()
    b.values = np.random.randint(0, 10, size=(b.shape[0], b.shape[1], b.shape[2]))
    return b


@pytest.fixture
def forecast_3d():
    """Random 3D forecast used for testing Contingency."""
    times = xr.cftime_range(start="2000", freq="D", periods=10)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.normal(0, 1, size=(len(times), len(lats), len(lons)))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def observation_3d(forecast_3d):
    """Random 3D observation used for testing Contingency."""
    b = forecast_3d.copy()
    b.values = np.random.normal(0, 1, size=(b.shape[0], b.shape[1], b.shape[2]))
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
    return np.linspace(-3, 3, 11)


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
