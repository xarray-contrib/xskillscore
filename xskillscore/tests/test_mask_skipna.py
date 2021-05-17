from typing import Callable, List

import numpy as np
import pytest

from xskillscore.core.deterministic import (
    linslope,
    mae,
    mape,
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

# Should only have masking issues when pulling in masked
# grid cells over space.
AXES = ("time", "lat", "lon", ["lat", "lon"], ["time", "lat", "lon"])

distance_metrics: List[Callable] = [mae, mape, median_absolute_error, mse, rmse, smape]
correlation_metrics: List[Callable] = [
    linslope,
    pearson_r,
    pearson_r_p_value,
    r2,
    spearman_r,
    spearman_r_p_value,
]


@pytest.mark.parametrize("metric", correlation_metrics + distance_metrics)
@pytest.mark.parametrize("dim", AXES)
def test_metrics_masked(a_fixed_nan, b_fixed_nan, dim, metric):
    """Test for all distance-based metrics whether result of skipna does not
    contain any nans when applied along dim with nans."""
    a = a_fixed_nan
    b = b_fixed_nan
    res_skipna = metric(a, b, dim, skipna=True)
    res_no_skipna = metric(a, b, dim, skipna=False)

    if "lon" in dim or "lat" in dim:  # metric is applied along axis with nans
        # res_skipna shouldnt have nans
        if metric not in [spearman_r_p_value, pearson_r_p_value]:
            assert not np.isnan(res_skipna).any()
        # res_no_skipna should have different result then skipna
        assert (res_no_skipna != res_skipna).any()
    else:  # metric is applied along axis without nans
        res_skipna_where_masked = res_skipna.isel(lon=[1, 2], lat=[1, 2])
        res_no_skipna_where_masked = res_no_skipna.isel(lon=[1, 2], lat=[1, 2])

        assert np.isnan(res_skipna_where_masked).all()
        assert np.isnan(res_no_skipna_where_masked).all()
        # res_skipna should have a few nans
        assert np.isnan(res_skipna).any()
        # res_no_skipna should have a few nans
        assert np.isnan(res_no_skipna).any()
        # # res_no_skipna should have different result then skipna
        assert (res_no_skipna != res_skipna).any()
