import numpy as np
import pandas as pd
import pytest
import xarray as xr

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

METRICS = [
    mae,
    mse,
    median_absolute_error,
    mape,
    smape,
    rmse,
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    r2,
]


@pytest.fixture
def gridded_a():
    times = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def gridded_b(gridded_a):
    b = gridded_a.copy()
    b.values = np.random.rand(b.shape[0], b.shape[1], b.shape[2])
    return b


def mask_data(da):
    """
    Masks sample data arbitrarily like a block of land.
    """
    da.data[:, 1:3, 1:3] = np.nan
    da.data[0:2, 0, 0] = np.nan
    return da


@pytest.mark.parametrize("metric", METRICS)
def test_single_grid_cell_matches_individual_time_series(gridded_a, gridded_b, metric):
    """Test that a single grid cell result from a gridded dataset matches the
    result for just passing through the individual time series"""
    for x in range(gridded_a["lon"].size):
        for y in range(gridded_a["lat"].size):
            ts_a = gridded_a.isel(lat=y, lon=x)
            ts_b = gridded_b.isel(lat=y, lon=x)
            gridded_res = metric(gridded_a, gridded_b, "time").isel(lat=y, lon=x)
            ts_res = metric(ts_a, ts_b, "time")
            assert np.allclose(gridded_res, ts_res)


@pytest.mark.parametrize("metric", METRICS)
def test_single_grid_cell_matches_individual_time_series_nans(
    gridded_a, gridded_b, metric
):
    """Test that a single grid cell result from a gridded dataset with nans matches the
    result for just passing through the individual time series"""
    gridded_a_masked = mask_data(gridded_a)
    gridded_b_masked = mask_data(gridded_b)
    for x in range(gridded_a_masked["lon"].size):
        for y in range(gridded_a_masked["lat"].size):
            ts_a_masked = gridded_a_masked.isel(lat=y, lon=x)
            ts_b_masked = gridded_b_masked.isel(lat=y, lon=x)
            gridded_res = metric(
                gridded_a_masked, gridded_b_masked, "time", skipna=True
            ).isel(lat=y, lon=x)
            ts_res = metric(ts_a_masked, ts_b_masked, "time", skipna=True)
            assert np.allclose(gridded_res, ts_res, equal_nan=True)
