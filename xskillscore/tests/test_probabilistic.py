import numpy as np
import pytest
import xarray as xr
from properscoring import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    threshold_brier_score,
)
from scipy.stats import norm
from xarray.tests import assert_allclose, assert_identical
from xskillscore.core.probabilistic import (
    xr_brier_score,
    xr_crps_ensemble,
    xr_crps_gaussian,
    xr_crps_quadrature,
    xr_threshold_brier_score,
)


@pytest.fixture
def o_dask():
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(lats), len(lons))
    return xr.DataArray(
        data, coords=[lats, lons], dims=["lat", "lon"],
        attrs={"source": "test"}
    ).chunk()


@pytest.fixture
def f_dask():
    members = np.arange(3)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(members), len(lats), len(lons))
    return xr.DataArray(
        data, coords=[members, lats, lons], dims=["member", "lat", "lon"]
    ).chunk()

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_crps_ensemble_dask(o_dask, f_dask, keep_attrs):
    actual = xr_crps_ensemble(o_dask, f_dask, keep_attrs=keep_attrs)
    expected = crps_ensemble(o_dask, f_dask, axis=0)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xr_crps and crps
    assert_allclose(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_crps_gaussian_dask(o_dask, f_dask, keep_attrs):
    mu = f_dask.mean("member")
    sig = f_dask.std("member")
    actual = xr_crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    expected = crps_gaussian(o_dask, mu, sig)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xr_crps and crps
    assert_allclose(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_crps_quadrature_dask(o_dask, f_dask, keep_attrs):
    cdf_or_dist = norm
    actual = xr_crps_quadrature(o_dask, cdf_or_dist, keep_attrs=keep_attrs)
    expected = crps_quadrature(o_dask, cdf_or_dist)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xr_crps and crps
    assert_allclose(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_crps_quadrature_args(o_dask, f_dask, keep_attrs):
    xmin, xmax, tol = -10, 10, 1e-6
    cdf_or_dist = norm
    actual = xr_crps_quadrature(o_dask, cdf_or_dist, xmin, xmax, tol, keep_attrs=keep_attrs)
    expected = crps_quadrature(o_dask, cdf_or_dist, xmin, xmax, tol)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xr_crps and crps
    assert_allclose(actual, expected)
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_threshold_brier_score_dask(o_dask, f_dask, keep_attrs):
    threshold = 0.5
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    expected = threshold_brier_score(o_dask, f_dask, threshold, axis=0)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xr_threshold and threshold
    assert_identical(actual, expected.assign_attrs(**actual.attrs))
    # test that xr_crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_crps_gaussian_dask_b_int(o_dask, keep_attrs):
    mu = 0
    sig = 1
    actual = xr_crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_threshold_brier_score_dask_b_float(o_dask, f_dask, keep_attrs):
    threshold = 0.5
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_threshold_brier_score_dask_b_int(o_dask, f_dask, keep_attrs):
    threshold = 0
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_threshold_brier_score_multiple_thresholds_list(
    o_dask, f_dask, keep_attrs
):
    threshold = [0.1, 0.3, 0.5]
    actual = xr_threshold_brier_score(o_dask.compute(), f_dask.compute(), threshold, keep_attrs=keep_attrs)
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_threshold_brier_score_multiple_thresholds_xr(
    o_dask, f_dask, keep_attrs
):
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims="threshold")
    actual = xr_threshold_brier_score(o_dask.compute(), f_dask.compute(), threshold, keep_attrs=keep_attrs)
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_threshold_brier_score_multiple_thresholds_dask(
    o_dask, f_dask, keep_attrs
):
    threshold = xr.DataArray([0.1, 0.3, 0.5, 0.7], dims="threshold").chunk()
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_brier_score(o_dask, f_dask, keep_attrs):
    actual = xr_brier_score(
        (o_dask > 0.5).compute().assign_attrs(**o_dask.attrs),
        (f_dask > 0.5).mean("member").compute(), keep_attrs=keep_attrs
    )
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}

@pytest.mark.parametrize("keep_attrs", [True, False])
def test_xr_brier_score_dask(o_dask, f_dask, keep_attrs):
    actual = xr_brier_score(
        (o_dask > 0.5).assign_attrs(**o_dask.attrs),
        (f_dask > 0.5).mean("member"),
        keep_attrs=keep_attrs)
    assert actual.chunks is not None
    expected = brier_score((o_dask > 0.5), (f_dask > 0.5).mean("member"))
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xr_brier_score and brier_score
    assert_allclose(actual, expected)
    # test that xr_brier_score returns chunks
    assert actual.chunks is not None
    # show that brier_score returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}
