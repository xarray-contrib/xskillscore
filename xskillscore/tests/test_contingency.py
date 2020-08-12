import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from sklearn.metrics import confusion_matrix

from xskillscore import Contingency

DIMS = (['time'], ['lon'], ['lat'], 'time', ['lon', 'lat', 'time'])
CATEGORY_EDGES = [
    np.array([-np.inf, 0.5, np.inf]),
    np.array([-np.inf, 0.5, 1.5, np.inf]),
]


@pytest.fixture
def forecast():
    times = xr.cftime_range(start='2000', freq='D', periods=10)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.randint(0, 10, size=(len(times), len(lats), len(lons)))
    return xr.DataArray(data, coords=[times, lats, lons], dims=['time', 'lat', 'lon'])


@pytest.fixture
def observation(forecast):
    b = forecast.copy()
    b.values = np.random.randint(0, 10, size=(b.shape[0], b.shape[1], b.shape[2]))
    return b


@pytest.fixture
def dichotomous_Contingency():
    observations = xr.DataArray(
        np.array(2 * [0] + 2 * [1] + 1 * [0] + 2 * [1]), coords=[('x', np.arange(7))]
    )
    forecasts = xr.DataArray(
        np.array(2 * [0] + 2 * [0] + 1 * [1] + 2 * [1]), coords=[('x', np.arange(7))]
    )
    category_edges = np.array([-np.inf, 0.5, np.inf])
    return Contingency(
        observations, forecasts, category_edges, category_edges, dim=['x']
    )


@pytest.mark.parametrize('type', ['da', 'ds', 'chunked_da', 'chunked_ds'])
@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('category_edges', CATEGORY_EDGES)
def test_Contingency_table(observation, forecast, category_edges, dim, type):
    """Test that contingency table builds successfully
    """
    if 'ds' in type:
        name = 'var'
        observation = observation.to_dataset(name=name)
        forecast = forecast.to_dataset(name=name)
    if 'chunked' in type:
        observation = observation.chunk()
        forecast = forecast.chunk()
    cont_table = Contingency(
        observation, forecast, category_edges, category_edges, dim=dim
    )
    assert cont_table


@pytest.mark.parametrize('category_edges', CATEGORY_EDGES)
def test_Contingency_table_values(observation, forecast, category_edges):
    """Test contingency table values against sklearn.metrics.confusion_matrix for 1D data
    """

    def logical(ds, edges):
        """Convert input xarray DataArray or Dataset to integers corresponding to which bin the data falls in.
            In first bin -> 0; in second bin -> 1; in third bin -> 2; etc
        """
        ds_out = 0 * ds.copy()
        for i in range(1, len(edges) - 1):
            ds_out += i * ((ds > edges[i]) & (ds <= edges[i + 1]))
        return ds_out

    cont_table = Contingency(
        observation, forecast, category_edges, category_edges, dim='time'
    )
    for lon in forecast.lon:
        for lat in forecast.lat:
            observation_1d = logical(observation.sel(lon=lon, lat=lat), category_edges)
            forecast_1d = logical(forecast.sel(lon=lon, lat=lat), category_edges)
            sklearn_cont_table_1d = confusion_matrix(
                observation_1d, forecast_1d, labels=range(len(category_edges) - 1)
            )
            xs_cont_table_1d = (
                cont_table.table.sel(lon=lon, lat=lat)
                .transpose('observations_category', 'forecasts_category')
                .values
            )
            npt.assert_allclose(sklearn_cont_table_1d, xs_cont_table_1d)


@pytest.mark.parametrize(
    'method, expected',
    [
        ('hits', 2),
        ('misses', 2),
        ('false_alarms', 1),
        ('correct_negatives', 2),
        ('bias_score', 3 / 4),
        ('hit_rate', 1 / 2),
        ('false_alarm_ratio', 1 / 3),
        ('false_alarm_rate', 1 / 3),
        ('success_ratio', 2 / 3),
        ('threat_score', 2 / 5),
        ('equit_threat_score', (2 - 12 / 7) / (5 - 12 / 7)),
        ('odds_ratio', 2),
        ('odds_ratio_skill_score', 1 / 3),
        ('accuracy', 4 / 7),
        ('heidke_score', (4 - 24 / 7) / (7 - 24 / 7)),
        ('peirce_score', 1 / 2 - 1 / 3),
        ('gerrity_score', (2 * (3 / 4) + 1 * -1 + 2 * -1 + 2 * (4 / 3)) / 7),
    ],
)
def test_dichotomous_scores(dichotomous_Contingency, method, expected):
    """Test score for simple 2x2 contingency table against hand-computed values
        Scores are for H/TP: 2, M/FN: 2, FA/FP: 1, CN/TN: 2
    """
    xs_score = getattr(dichotomous_Contingency, method)().item()
    npt.assert_almost_equal(xs_score, expected)
