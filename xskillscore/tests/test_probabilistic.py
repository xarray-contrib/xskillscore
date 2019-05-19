import numpy as np
import pandas as pd
import pytest
import xarray as xr
from properscoring import (crps_ensemble, crps_gaussian,
                           threshold_brier_score)
from xarray.tests import assert_identical

from xskillscore.core.probabilistic import (xr_crps_ensemble, xr_crps_gaussian,
                                            xr_threshold_brier_score)


@pytest.fixture
def a_dask():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


@pytest.fixture
def b_dask():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


def test_xr_crps_ensemble_dask(a_dask, b_dask):
    actual = xr_crps_ensemble(a_dask, b_dask)
    expected = crps_ensemble(a_dask, b_dask)
    expected = xr.DataArray(expected, coords=a_dask.coords)
    # test for numerical identity of xr_crps and crps
    assert_identical(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None


def test_xr_crps_gaussian_dask(a_dask, b_dask):
    mu = b_dask.mean('time')
    sig = b_dask.std('time')
    actual = xr_crps_gaussian(a_dask, mu, sig)
    expected = crps_gaussian(a_dask, mu, sig)
    expected = xr.DataArray(expected, coords=a_dask.coords)
    # test for numerical identity of xr_crps and crps
    assert_identical(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None


def test_xr_threshold_brier_score_dask(a_dask, b_dask):
    threshold = .5
    actual = xr_threshold_brier_score(a_dask, b_dask, threshold)
    expected = threshold_brier_score(a_dask, b_dask, threshold)
    expected = xr.DataArray(expected, coords=a_dask.coords)
    # test for numerical identity of xr_threshold and threshold
    assert_identical(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None
