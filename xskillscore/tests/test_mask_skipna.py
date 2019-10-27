import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xskillscore.core.deterministic import (
    mae,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
)


# Should only have masking issues when pulling in masked
# grid cells over space.
AXES = [["time"], ["lat"], ["lon"], ("lat", "lon"), ("time", "lat", "lon")]


@pytest.fixture
def a():
    time = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(time), len(lats), len(lons))
    da = xr.DataArray(
        data, coords=[time, lats, lons], dims=["time", "lat", "lon"]
    )
    return da


@pytest.fixture
def b():
    time = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(time), len(lats), len(lons))
    da = xr.DataArray(
        data, coords=[time, lats, lons], dims=["time", "lat", "lon"]
    )
    return da


def mask_data(da, dim):
    """
    Masks sample data arbitrarily over specified dim.
    """
    # Mask an arbitrary region with NaNs (like a block of land).
    return da.where(da[dim] > da[dim].isel({dim: 0}))


@pytest.mark.parametrize("dim", AXES)
def test_pearson_r_masked(a, b, dim):
    a_masked = mask_data(a, dim[0])
    b_masked = mask_data(b, dim[0])

    res_skipna = pearson_r(a_masked, b_masked, dim, skipna=True)
    res_no_skipna = pearson_r(a_masked, b_masked, dim, skipna=False)
    assert np.isnan(res_no_skipna).all()
    assert not np.isnan(res_skipna).any()


@pytest.mark.parametrize("dim", AXES)
def test_pearson_r_p_value_masked(a, b, dim):
    a_masked = mask_data(a, dim[0])
    b_masked = mask_data(b, dim[0])

    res_skipna = pearson_r_p_value(a_masked, b_masked, dim, skipna=True)
    res_no_skipna = pearson_r_p_value(a_masked, b_masked, dim, skipna=False)
    # p-value defaults to exactly 1.0 instead of NaNs.
    assert (res_no_skipna == 1.0).all()
    assert not np.isnan(res_skipna).any()


@pytest.mark.parametrize("dim", AXES)
def test_rmse_masked(a, b, dim):
    a_masked = mask_data(a, dim[0])
    b_masked = mask_data(b, dim[0])

    res_skipna = rmse(a_masked, b_masked, dim, skipna=True)
    res_no_skipna = rmse(a_masked, b_masked, dim, skipna=False)
    assert np.isnan(res_no_skipna).all()
    assert not np.isnan(res_skipna).any()


@pytest.mark.parametrize("dim", AXES)
def test_mse_masked(a, b, dim):
    a_masked = mask_data(a, dim[0])
    b_masked = mask_data(b, dim[0])

    res_skipna = mse(a_masked, b_masked, dim, skipna=True)
    res_no_skipna = mse(a_masked, b_masked, dim, skipna=False)
    assert np.isnan(res_no_skipna).all()
    assert not np.isnan(res_skipna).any()


@pytest.mark.parametrize("dim", AXES)
def test_mae_masked(a, b, dim):
    a_masked = mask_data(a, dim[0])
    b_masked = mask_data(b, dim[0])

    res_skipna = mae(a_masked, b_masked, dim, skipna=True)
    res_no_skipna = mae(a_masked, b_masked, dim, skipna=False)
    assert np.isnan(res_no_skipna).all()
    assert not np.isnan(res_skipna).any()
