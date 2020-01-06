import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xskillscore.core.deterministic import (
    pearson_r_p_value,
    pearson_r_eff_p_value,
    spearman_r_p_value,
    spearman_r_eff_p_value,
    effective_sample_size,
)

DIM = "time"


@pytest.fixture
def a():
    times = pd.date_range("1/1/2000", "2/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def b(a):
    b = a.copy()
    b.values = np.random.rand(a.shape[0], a.shape[1], a.shape[2])
    return b


def test_eff_sample_size_smaller_than_n(a, b):
    """Tests that the effective sample size is less than or equal to the normal
    sample size."""
    # Can just use `.size` here since we don't
    # have any NaNs in the test.
    N = a[DIM].size
    eff_N = effective_sample_size(a, b, DIM)
    assert (eff_N <= N).all()


def test_eff_pearson_p_greater_or_equal_to_normal_p(a, b):
    """Tests that the effective Pearson p value is greater than or equal to the
    normal Pearson p value."""
    normal_p = pearson_r_p_value(a, b, "time")
    eff_p = pearson_r_eff_p_value(a, b, "time")
    assert (eff_p >= normal_p).all()


def test_eff_spearman_p_greater_or_equal_to_normal_p(a, b):
    """Tests that the effective Spearman p value is greater than or equal to the
    normal Spearman p value."""
    normal_p = spearman_r_p_value(a, b, "time")
    eff_p = spearman_r_eff_p_value(a, b, "time")
    assert (eff_p >= normal_p).all()
