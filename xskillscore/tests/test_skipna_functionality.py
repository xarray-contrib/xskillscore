import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import assert_allclose


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

METRICS = [
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    mae,
    mse,
    mad,
    mape,
    smape,
    rmse,
]


@pytest.fixture
def a():
    time = pd.date_range("1/1/2000", "1/5/2000", freq="D")
    da = xr.DataArray([3, np.nan, 5, 7, 9], dims=["time"], coords=[time])
    return da


@pytest.fixture
def b():
    time = pd.date_range("1/1/2000", "1/5/2000", freq="D")
    da = xr.DataArray([7, 2, np.nan, 2, 4], dims=["time"], coords=[time])
    return da


def drop_nans(a, b, dim="time"):
    """
    Masks a and b where they have pairwise nans.
    """
    a = a.where(b.notnull())
    b = b.where(a.notnull())
    return a.dropna(dim), b.dropna(dim)

# ADD WEIGHTS
@pytest.mark.parametrize("metric", METRICS)
def test_skipna_returns_same_value_as_dropped_pairwise_nans(a, b, metric):
    """Tests that DataArrays with pairwise nans return the same result
    as the same two with those nans dropped."""
    a_dropped, b_dropped = drop_nans(a, b)
    res_with_nans = metric(a, b, "time", skipna=True)
    res_dropped_nans = metric(a_dropped, b_dropped, "time")
    assert_allclose(res_with_nans, res_dropped_nans)


@pytest.mark.parametrize("metric", METRICS)
def test_skipna_returns_nan_when_false(a, b, metric):
    """Tests that nan is returned if there's any nans in the time series
    and skipna is False."""
    res = metric(a, b, "time", skipna=False)
    assert np.isnan(res).all()