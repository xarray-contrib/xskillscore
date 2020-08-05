import numpy as np
import pytest
import xarray as xr
from sklearn.metrics import confusion_matrix

from xskillscore import Contingency

DIMS = [['time'], ['lon'], ['lat'], 'time', ['lon', 'lat', 'time']]


@pytest.fixture
def forecast():
    times = xr.cftime_range(start='2000', freq='D', periods=10)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(data, coords=[times, lats, lons], dims=['time', 'lat', 'lon'])


@pytest.fixture
def observation(forecast):
    b = forecast.copy()
    b.values = np.random.rand(b.shape[0], b.shape[1], b.shape[2])
    return b


@pytest.fixture
def category_edges_a():
    return np.linspace(-2, 2, 5)


@pytest.fixture
def category_edges_b():
    return np.linspace(-3, 3, 5)


@pytest.mark.parametrize('type', ['da', 'ds', 'chunked_da', 'chunked_ds'])
@pytest.mark.parametrize('dim', DIMS)
def test_Contingency_values(
    forecast, observation, category_edges_a, category_edges_b, dim, type
):
    if 'ds' in type:
        name = 'var'
        forecast = forecast.to_dataset(name=name)
        observation = observation.to_dataset(name=name)
    if 'chunked' in type:
        forecast = forecast.chunk()
        observation = observation.chunk()
    cont_table = Contingency(
        forecast, observation, category_edges_a, category_edges_b, dim=dim
    )
    assert cont_table
    # test against sklearn.metrics.confusion_matrix
    if type == 'da' and dim == 'time':  # only tests for one dim

        def logical(ds):
            return ds < 0.5

        for lon in forecast.lon:
            for lat in forecast.lat:
                forecast_1d = logical(forecast.sel(lon=lon, lat=lat))
                observation_1d = logical(observation.sel(lon=lon, lat=lat))
                sklearn_cont_table_1d = confusion_matrix(forecast_1d, observation_1d)
                expected == cont_table.sel(lon=lon, lat=lat).values
                assert sklearn_cont_table_1d == expected, print(
                    int(lon), int(lat), sklearn_cont_table_1d, expected
                )
