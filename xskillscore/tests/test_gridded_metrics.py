import numpy as np
import pytest

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

METRICS = [
    linslope,
    mae,
    mse,
    median_absolute_error,
    mape,
    smape,
    me,
    rmse,
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    r2,
]


def mask_data(da):
    """
    Masks sample data arbitrarily like a block of land.
    """
    da.data[:, 1:3, 1:3] = np.nan
    da.data[0:2, 0, 0] = np.nan
    return da


@pytest.mark.parametrize("metric", METRICS)
def test_single_grid_cell_matches_individual_time_series(a, b, metric):
    """Test that a single grid cell result from a gridded dataset matches the
    result for just passing through the individual time series"""
    for x in range(a["lon"].size):
        for y in range(a["lat"].size):
            ts_a = a.isel(lat=y, lon=x)
            ts_b = b.isel(lat=y, lon=x)
            gridded_res = metric(a, b, "time").isel(lat=y, lon=x)
            ts_res = metric(ts_a, ts_b, "time")
            assert np.allclose(gridded_res, ts_res)


@pytest.mark.parametrize("metric", METRICS)
def test_single_grid_cell_matches_individual_time_series_nans(a, b, metric):
    """Test that a single grid cell result from a gridded dataset with nans matches the
    result for just passing through the individual time series"""
    a_masked = mask_data(a)
    b_masked = mask_data(b)
    for x in range(a_masked["lon"].size):
        for y in range(a_masked["lat"].size):
            ts_a_masked = a_masked.isel(lat=y, lon=x)
            ts_b_masked = b_masked.isel(lat=y, lon=x)
            gridded_res = metric(a_masked, b_masked, "time", skipna=True).isel(
                lat=y, lon=x
            )
            ts_res = metric(ts_a_masked, ts_b_masked, "time", skipna=True)
            assert np.allclose(gridded_res, ts_res, equal_nan=True)
