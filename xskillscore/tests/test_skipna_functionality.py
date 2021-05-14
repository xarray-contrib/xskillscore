from typing import Callable, List

import dask
import numpy as np
import pytest
from xarray.tests import CountingScheduler, assert_allclose, raise_if_dask_computes

from xskillscore.core.deterministic import (
    linslope,
    mae,
    mape,
    me,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_p_value,
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
)

WEIGHTED_METRICS: List[Callable] = [
    linslope,
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    mae,
    mse,
    mape,
    smape,
    me,
    rmse,
    r2,
]

NON_WEIGHTED_METRICS: List[Callable] = [median_absolute_error]


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
def test_skipna_returns_same_value_as_dropped_pairwise_nans(
    a_1d_fixed_nan, b_1d_fixed_nan, metric
):
    """Tests that DataArrays with pairwise nans return the same result
    as the same two with those nans dropped."""
    a_dropped, b_dropped, _ = drop_nans(a_1d_fixed_nan, b_1d_fixed_nan)
    with raise_if_dask_computes():
        res_with_nans = metric(a_1d_fixed_nan, b_1d_fixed_nan, "time", skipna=True)
        res_dropped_nans = metric(a_dropped, b_dropped, "time")
    assert_allclose(res_with_nans, res_dropped_nans)


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
def test_skipna_returns_same_value_as_dropped_pairwise_nans_with_weights(
    a_1d_fixed_nan, b_1d_fixed_nan, weights_time, metric
):
    """Tests that DataArrays with pairwise nans return the same result
    as the same two with those nans dropped."""
    a_dropped, b_dropped, weights_time_dropped = drop_nans(
        a_1d_fixed_nan, b_1d_fixed_nan, weights_time
    )
    with raise_if_dask_computes():
        res_with_nans = metric(
            a_1d_fixed_nan, b_1d_fixed_nan, "time", skipna=True, weights=weights_time
        )
        res_dropped_nans = metric(
            a_dropped, b_dropped, "time", weights=weights_time_dropped
        )
    assert_allclose(res_with_nans, res_dropped_nans)


@pytest.mark.parametrize("metric", WEIGHTED_METRICS + NON_WEIGHTED_METRICS)
def test_skipna_returns_nan_when_false(a_1d_fixed_nan, b_1d_fixed_nan, metric):
    """Tests that nan is returned if there's any nans in the time series
    and skipna is False."""
    with raise_if_dask_computes():
        res = metric(a_1d_fixed_nan, b_1d_fixed_nan, "time", skipna=False)
    assert np.isnan(res).all()


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
def test_skipna_broadcast_weights_assignment_destination(
    a_rand_nan, b_rand_nan, weights_lonlat, metric
):
    """Tests that 'assignment destination is read-only' is not raised
    https://github.com/xarray-contrib/xskillscore/issues/79"""
    with raise_if_dask_computes():
        metric(
            a_rand_nan, b_rand_nan, ["lat", "lon"], weights=weights_lonlat, skipna=True
        )


def test_nan_skipna(a, b):
    # Randomly add some nans to a
    a = a.where(np.random.random(a.shape) < 0.5)
    with raise_if_dask_computes():
        pearson_r(a, b, dim="lat", skipna=True)
