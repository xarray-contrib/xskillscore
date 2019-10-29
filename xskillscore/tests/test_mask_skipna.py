import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xskillscore.core.deterministic import (
    mad,
    mae,
    mape,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
)

# Should only have masking issues when pulling in masked
# grid cells over space.
AXES = [['time'], ['lat'], ['lon'], ('lat', 'lon'), ('time', 'lat', 'lon')]

distance_metrics = [mae, mse, mad, mape, smape, rmse]
correlation_metrics = [
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
]


@pytest.fixture
def a():
    time = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(time), len(lats), len(lons))
    da = xr.DataArray(
        data, coords=[time, lats, lons], dims=['time', 'lat', 'lon']
    )
    return da


@pytest.fixture
def b():
    time = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(time), len(lats), len(lons))
    da = xr.DataArray(
        data, coords=[time, lats, lons], dims=['time', 'lat', 'lon']
    )
    return da


def mask_land_data(da):
    """
    Masks sample data arbitrarily like a block of land.
    """
    da.data[:, 1:3, 1:3] = np.nan
    return da


@pytest.mark.parametrize('metric', correlation_metrics + distance_metrics)
@pytest.mark.parametrize('dim', AXES)
def test_metrics_masked(a, b, dim, metric):
    """Test for all distance-based metrics whether result if skipna do not
     contain any nans when applied along dim with nans."""
    a_masked = mask_land_data(a)
    b_masked = mask_land_data(b)
    res_skipna = metric(a_masked, b_masked, dim, skipna=True)
    res_no_skipna = metric(a_masked, b_masked, dim, skipna=False)

    return_for_nan_pearson_r_p_value = 1.0
    return_for_nan_spearman_r = 1.0

    if 'lon' in dim or 'lat' in dim:  # metric is applied along axis with nans
        # res_skipna shouldnt have nans
        assert not np.isnan(res_skipna).any()
        # res_no_skipna should have nans except for pearson_r_p_value 1.0
        if metric is pearson_r_p_value:
            assert (res_no_skipna == return_for_nan_pearson_r_p_value).any()
        else:
            assert np.isnan(res_no_skipna).any()
    else:  # metric is applied along axis without nans
        res_skipna_where_masked = res_skipna.isel(lon=[1, 2], lat=[1, 2])
        res_no_skipna_where_masked = res_no_skipna.isel(lon=[1, 2], lat=[1, 2])

        # where masked should be all nan
        if metric in [spearman_r]:
            assert (res_skipna_where_masked == return_for_nan_spearman_r).all()
        else:
            assert np.isnan(res_skipna_where_masked).all()
        assert np.isnan(res_no_skipna_where_masked).all()
        # res_skipna should have a few nans
        if metric in [spearman_r]:
            assert (res_skipna == return_for_nan_spearman_r).any()
        else:
            assert np.isnan(res_skipna).any()
        # res_no_skipna should have a few nans
        assert np.isnan(res_no_skipna).any()
