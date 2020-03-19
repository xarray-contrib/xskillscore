import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import assert_allclose


from xskillscore.core.deterministic import (
    median_absolute_error,
    mae,
    mape,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
    r2,
)

WEIGHTED_METRICS = [
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    mae,
    mse,
    mape,
    smape,
    rmse,
    r2,
]

NON_WEIGHTED_METRICS = [median_absolute_error]


@pytest.fixture
def a():
    time = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    da = xr.DataArray([3, np.nan, 5], dims=["time"], coords=[time])
    return da


@pytest.fixture
def b(a):
    b = a.copy()
    b.values = [7, 2, np.nan]
    return b


@pytest.fixture
def weights():
    time = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    da = xr.DataArray([1, 2, 3], dims=["time"], coords=[time])
    return da


@pytest.fixture
def a_3d():
    da = xr.DataArray(
        np.repeat(np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9]]), 3).reshape(3, 3, 3),
        coords=[
            pd.date_range("1/1/2000", "1/3/2000", freq="D"),
            np.linspace(-89.5, 89.5, 3),
            np.linspace(-179.5, 179.5, 3),
        ],
        dims=["time", "lat", "lon"],
    )
    return da


@pytest.fixture
def b_3d(a_3d):
    return a_3d.copy()


@pytest.fixture
def weights_3d(a_3d):
    weights = np.cos(np.deg2rad(a_3d.lat))
    _, weights = xr.broadcast(a_3d, weights)
    weights = weights.isel(time=0)
    return weights


def drop_nans(a, b, weights=None, dim="time"):
    """
    Masks a and b where they have pairwise nans.
    """
    a = a.where(b.notnull())
    b = b.where(a.notnull())
    if weights is not None:
        weights = weights.where(a.notnull())
        weights = weights.dropna(dim)
    return a.dropna(dim), b.dropna(dim), weights


@pytest.mark.parametrize("metric", WEIGHTED_METRICS + NON_WEIGHTED_METRICS)
def test_skipna_returns_same_value_as_dropped_pairwise_nans(a, b, metric):
    """Tests that DataArrays with pairwise nans return the same result
    as the same two with those nans dropped."""
    a_dropped, b_dropped, _ = drop_nans(a, b)
    res_with_nans = metric(a, b, "time", skipna=True)
    res_dropped_nans = metric(a_dropped, b_dropped, "time")
    assert_allclose(res_with_nans, res_dropped_nans)


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
def test_skipna_returns_same_value_as_dropped_pairwise_nans_with_weights(
    a, b, weights, metric
):
    """Tests that DataArrays with pairwise nans return the same result
    as the same two with those nans dropped."""
    a_dropped, b_dropped, weights_dropped = drop_nans(a, b, weights)
    res_with_nans = metric(a, b, "time", skipna=True, weights=weights)
    res_dropped_nans = metric(a_dropped, b_dropped, "time", weights=weights_dropped)
    assert_allclose(res_with_nans, res_dropped_nans)


@pytest.mark.parametrize("metric", WEIGHTED_METRICS + NON_WEIGHTED_METRICS)
def test_skipna_returns_nan_when_false(a, b, metric):
    """Tests that nan is returned if there's any nans in the time series
    and skipna is False."""
    res = metric(a, b, "time", skipna=False)
    assert np.isnan(res).all()


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
def test_skipna_broadcast_weights_assignment_destination(
    a_3d, b_3d, weights_3d, metric
):
    """Tests that 'assignment destination is read-only' is not raised
    https://github.com/raybellwaves/xskillscore/issues/79"""
    a_3d_nan = a_3d.where(a_3d > 0)
    b_3d_nan = b_3d.where(b_3d > 0)
    res = metric(a_3d_nan, b_3d_nan, ["lat", "lon"], weights=weights_3d, skipna=True)
    assert all([not np.isnan(r) for r in res])
