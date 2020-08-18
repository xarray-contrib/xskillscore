import numpy as np
import properscoring
import pytest
import xarray as xr
from scipy.stats import norm
from xarray.tests import assert_allclose, assert_identical

from xskillscore.core.probabilistic import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    threshold_brier_score,
)

DIMS = ['lon', 'lat', ['lon', 'lat'], None, []]


@pytest.fixture
def o():
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(lats), len(lons))
    return xr.DataArray(
        data, coords=[lats, lons], dims=['lat', 'lon'], attrs={'source': 'test'}
    ).chunk()


@pytest.fixture
def o_dask(o):
    return o.chunk()


@pytest.fixture
def weights(o):
    """Latitudinal weighting"""
    return o.lat


@pytest.fixture
def f():
    """Forecast has same dimensions as observation and member dimension."""
    members = np.arange(3)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(members), len(lats), len(lons))
    return xr.DataArray(
        data, coords=[members, lats, lons], dims=['member', 'lat', 'lon']
    )


@pytest.fixture
def f_dask(f):
    return f.chunk()


def assert_only_dim_reduced(dim, actual, obs):
    """Check that actual is only reduced by dims in dim."""
    if not isinstance(dim, list):
        dim = [dim]
    for d in dim:
        assert d not in actual.dims
    for d in obs.dims:
        if d not in dim:
            assert d in actual.dims


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_ensemble_dask(o_dask, f_dask, keep_attrs):
    actual = crps_ensemble(o_dask, f_dask, keep_attrs=keep_attrs)
    expected = properscoring.crps_ensemble(o_dask, f_dask, axis=0)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xskillscore crps and properscoring crps
    assert_allclose(actual, expected)
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('dim', DIMS)
def test_crps_ensemble_dim(o, f, dim):
    actual = crps_ensemble(o, f, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_ensemble_weighted(o, f, weights, keep_attrs):
    dim = ['lon', 'lat']
    actual = crps_ensemble(o, f, dim=dim, weights=weights, keep_attrs=keep_attrs)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_gaussian_dask(o_dask, f_dask, keep_attrs):
    mu = f_dask.mean('member')
    sig = f_dask.std('member')
    actual = crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    expected = properscoring.crps_gaussian(o_dask, mu, sig)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xskillscore crps and properscoring crps
    assert_allclose(actual, expected)
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None


@pytest.mark.parametrize('dim', DIMS)
def test_crps_gaussian_dim(o, f, dim):
    mu = f.mean('member')
    sig = f.std('member')
    actual = crps_gaussian(o, mu, sig, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_quadrature_dask(o_dask, keep_attrs):
    cdf_or_dist = norm
    actual = crps_quadrature(o_dask, cdf_or_dist, keep_attrs=keep_attrs)
    expected = properscoring.crps_quadrature(o_dask, cdf_or_dist)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xskillscore crps and properscoring crps
    assert_allclose(actual, expected)
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_quadrature_args(o_dask, f_dask, keep_attrs):
    xmin, xmax, tol = -10, 10, 1e-6
    cdf_or_dist = norm
    actual = crps_quadrature(
        o_dask, cdf_or_dist, xmin, xmax, tol, keep_attrs=keep_attrs
    )
    expected = properscoring.crps_quadrature(o_dask, cdf_or_dist, xmin, xmax, tol)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xskillscore crps and crps
    assert_allclose(actual, expected)
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_quadrature_dim(o, f, dim, keep_attrs):
    cdf_or_dist = norm
    actual = crps_quadrature(o, cdf_or_dist, dim=dim, keep_attrs=keep_attrs)
    assert_only_dim_reduced(dim, actual, o)
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_dask(o_dask, f_dask, keep_attrs):
    threshold = 0.5
    actual = threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    expected = properscoring.threshold_brier_score(o_dask, f_dask, threshold, axis=0)
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of xskillscore threshold and properscorin threshold
    assert_identical(actual, expected.assign_attrs(**actual.attrs))
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_gaussian_dask_b_int(o_dask, keep_attrs):
    mu = 0
    sig = 1
    actual = crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_dask_b_float(o_dask, f_dask, keep_attrs):
    threshold = 0.5
    actual = threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_dask_b_int(o_dask, f_dask, keep_attrs):
    threshold = 0
    actual = threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_multiple_thresholds_list(o_dask, f_dask, keep_attrs):
    threshold = [0.1, 0.3, 0.5]
    actual = threshold_brier_score(
        o_dask.compute(), f_dask.compute(), threshold, keep_attrs=keep_attrs
    )
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_multiple_thresholds_xr(o_dask, f_dask, keep_attrs):
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims='threshold')
    actual = threshold_brier_score(
        o_dask.compute(), f_dask.compute(), threshold, keep_attrs=keep_attrs
    )
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_multiple_thresholds_dask(o_dask, f_dask, keep_attrs):
    threshold = xr.DataArray([0.1, 0.3, 0.5, 0.7], dims='threshold').chunk()
    actual = threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_brier_score(o_dask, f_dask, keep_attrs):
    actual = brier_score(
        (o_dask > 0.5).compute().assign_attrs(**o_dask.attrs),
        (f_dask > 0.5).mean('member').compute(),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('dim', DIMS)
def test_threshold_brier_score_dim(o, f, dim):
    actual = threshold_brier_score(o, f, threshold=0.5, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize(
    'threshold', [0, 0.5, [0.1, 0.3, 0.5]], ids=['int', 'flat', 'list']
)
def test_threshold_brier_score_dask_threshold(o_dask, f_dask, threshold):
    actual = threshold_brier_score(o_dask, f_dask, threshold)
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
def test_brier_score_dim(o, f, dim):
    actual = brier_score((o > 0.5), (f > 0.5).mean('member'), dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_brier_score_dask(o_dask, f_dask, keep_attrs):
    actual = brier_score(
        (o_dask > 0.5).assign_attrs(**o_dask.attrs),
        (f_dask > 0.5).mean('member'),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is not None
    expected = properscoring.brier_score((o_dask > 0.5), (f_dask > 0.5).mean('member'))
    expected = xr.DataArray(expected, coords=o_dask.coords)
    # test for numerical identity of brier_score and properscoring brier_score
    assert_allclose(actual, expected)
    # test that xskillscore brier_score returns chunks
    assert actual.chunks is not None
    # show that properscoring brier_score returns no chunks
    assert expected.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}
