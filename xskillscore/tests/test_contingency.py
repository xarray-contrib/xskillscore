import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from xskillscore import Contingency, roc

DIMS = (["time"], ["lon"], ["lat"], "time", ["lon", "lat", "time"])
CATEGORY_EDGES = [
    np.array([-np.inf, 0.5, np.inf]),
    np.array([-np.inf, 0.5, 1.5, np.inf]),
]


@pytest.mark.parametrize("type", ["da", "ds", "chunked_da", "chunked_ds"])
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("category_edges", CATEGORY_EDGES)
def test_Contingency_table(observation_3d, forecast_3d, category_edges, dim, type):
    """Test that contingency table builds successfully"""
    if "ds" in type:
        name = "var"
        observation_3d = observation_3d.to_dataset(name=name)
        forecast_3d = forecast_3d.to_dataset(name=name)
    if "chunked" in type:
        observation_3d = observation_3d.chunk()
        forecast_3d = forecast_3d.chunk()
    cont_table = Contingency(
        observation_3d, forecast_3d, category_edges, category_edges, dim=dim
    )
    assert cont_table


@pytest.mark.parametrize("category_edges", CATEGORY_EDGES)
def test_Contingency_table_values(observation_3d, forecast_3d, category_edges):
    """Test contingency table values against sklearn.metrics.confusion_matrix
    for 1D data"""

    def logical(ds, edges):
        """Convert input xarray DataArray or Dataset to integers corresponding
        to which bin the data falls in.
        In first bin -> 0; in second bin -> 1; in third bin -> 2; etc
        """
        ds_out = 0 * ds.copy()
        for i in range(1, len(edges) - 1):
            ds_out += i * ((ds > edges[i]) & (ds <= edges[i + 1]))
        return ds_out

    cont_table = Contingency(
        observation_3d, forecast_3d, category_edges, category_edges, dim="time"
    )
    for lon in forecast_3d.lon:
        for lat in forecast_3d.lat:
            observation_1d = logical(
                observation_3d.sel(lon=lon, lat=lat), category_edges
            )
            forecast_1d = logical(forecast_3d.sel(lon=lon, lat=lat), category_edges)
            sklearn_cont_table_1d = confusion_matrix(
                observation_1d, forecast_1d, labels=range(len(category_edges) - 1)
            )
            xs_cont_table_1d = (
                cont_table.table.sel(lon=lon, lat=lat)
                .transpose("observations_category", "forecasts_category")
                .values
            )
            npt.assert_allclose(sklearn_cont_table_1d, xs_cont_table_1d)


@pytest.mark.parametrize(
    "method, expected",
    [
        ("hits", 2),
        ("misses", 2),
        ("false_alarms", 1),
        ("correct_negatives", 2),
        ("bias_score", 3 / 4),
        ("hit_rate", 1 / 2),
        ("false_alarm_ratio", 1 / 3),
        ("false_alarm_rate", 1 / 3),
        ("success_ratio", 2 / 3),
        ("threat_score", 2 / 5),
        ("equit_threat_score", (2 - 12 / 7) / (5 - 12 / 7)),
        ("odds_ratio", 2),
        ("odds_ratio_skill_score", 1 / 3),
        ("accuracy", 4 / 7),
        ("heidke_score", (4 - 24 / 7) / (7 - 24 / 7)),
        ("peirce_score", 1 / 2 - 1 / 3),
        ("gerrity_score", (2 * (3 / 4) + 1 * -1 + 2 * -1 + 2 * (4 / 3)) / 7),
    ],
)
def test_dichotomous_scores(dichotomous_Contingency_1d, method, expected):
    """Test score for simple 2x2 contingency table against hand-computed values
    Scores are for H/TP: 2, M/FN: 2, FA/FP: 1, CN/TN: 2
    """
    xs_score = getattr(dichotomous_Contingency_1d, method)().item()
    npt.assert_almost_equal(xs_score, expected)


@pytest.mark.parametrize(
    "forecast, observation",
    [
        (
            pytest.lazy_fixture("forecast_1d_long"),
            pytest.lazy_fixture("observation_1d_long"),
        ),
        (
            pytest.lazy_fixture("forecast_3d"),
            pytest.lazy_fixture("observation_3d"),
        ),
    ],
)
@pytest.mark.parametrize("dim", ["time", None])
@pytest.mark.parametrize("drop_intermediate", [True, False])
@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "DataArray"])
@pytest.mark.parametrize(
    "return_results", ["all_as_tuple", "area", "all_as_metric_dim"]
)
def test_roc_returns(
    forecast,
    observation,
    symmetric_edges,
    dim,
    return_results,
    input,
    chunk,
    drop_intermediate,
):
    """testing keywords and inputs"""
    if "Dataset" in input:
        name = "var"
        forecast = forecast.to_dataset(name=name)
        observation = observation.to_dataset(name=name)
    if chunk:
        forecast = forecast.chunk()
        observation = observation.chunk()

    roc(
        forecast,
        observation,
        symmetric_edges,
        dim=dim,
        drop_intermediate=drop_intermediate,
        return_results=return_results,
    )


def test_roc_auc_score_random_forecast(
    forecast_1d_long, observation_1d_long, symmetric_edges
):
    """Test that ROC AUC around 0.5 for random forecast."""
    area = roc(
        forecast_1d_long,
        observation_1d_long,
        symmetric_edges,
        dim="time",
        return_results="area",
    )
    assert area < 0.6
    assert area > 0.4


def test_roc_auc_score_perfect_forecast(forecast_1d_long, symmetric_edges):
    """Test that ROC AUC equals 1 for perfect forecast."""
    area = roc(
        forecast_1d_long,
        forecast_1d_long,
        symmetric_edges,
        dim="time",
        return_results="area",
    )
    assert area == 1.0


def test_roc_auc_score_constant_forecast(forecast_1d_long, symmetric_edges):
    """Test that ROC AUC equals 0 or 0.5. for constant forecast."""
    area = roc(
        forecast_1d_long,
        xr.ones_like(forecast_1d_long) * 10,
        symmetric_edges,
        dim="time",
        return_results="area",
    )
    assert float(area) in [0.0, 0.5]


@pytest.mark.parametrize("drop_intermediate", [False, True])
def test_roc_bin_edges_continuous_against_sklearn(
    forecast_1d_long, observation_1d_long, drop_intermediate
):
    """Test xs.roc against sklearn.metrics.roc_curve/auc_score."""
    fb = forecast_1d_long > 0  # binary
    op = np.clip(observation_1d_long, 0, 1)  # prob
    # sklearn
    sk_fpr, sk_tpr, _ = roc_curve(fb, op, drop_intermediate=drop_intermediate)
    sk_area = roc_auc_score(fb, op)
    # xs
    xs_fpr, xs_tpr, xs_area = roc(
        fb,
        op,
        "continuous",
        drop_intermediate=drop_intermediate,
        return_results="all_as_tuple",
    )
    np.testing.assert_allclose(xs_area, sk_area)
    if not drop_intermediate:  # drops sometimes one too much or too little
        assert (xs_fpr == sk_fpr).all()
        assert (xs_tpr == sk_tpr).all()


def test_roc_bin_edges_drop_intermediate(forecast_1d_long, observation_1d_long):
    """Test that drop_intermediate reduces probability_bins in xs.roc ."""
    fb = forecast_1d_long > 0  # binary
    op = np.clip(observation_1d_long, 0, 1)  # prob
    # xs
    txs_fpr, txs_tpr, txs_area = roc(
        fb, op, "continuous", drop_intermediate=True, return_results="all_as_tuple"
    )
    fxs_fpr, fxs_tpr, fxs_area = roc(
        fb, op, "continuous", drop_intermediate=False, return_results="all_as_tuple"
    )
    # same area
    np.testing.assert_allclose(fxs_area, txs_area)
    # same or less probability_bins
    assert len(fxs_fpr) >= len(txs_fpr)
    assert len(fxs_tpr) >= len(txs_tpr)


def test_roc_multi_dim(forecast_3d, observation_3d):
    roc(forecast_3d, observation_3d, bin_edges=np.linspace(0, 10 + 1e-8, 6))
