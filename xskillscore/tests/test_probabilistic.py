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
    discrimination,
    rank_histogram,
    xr_brier_score,
    xr_crps_ensemble,
    xr_crps_gaussian,
    xr_crps_quadrature,
    xr_threshold_brier_score,
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


@pytest.mark.parametrize('dim', DIMS)
def test_xr_crps_ensemble_dim(o, f, dim):
    actual = xr_crps_ensemble(o, f, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_crps_ensemble_weighted(o, f, weights, keep_attrs):
    dim = ['lon', 'lat']
    actual = xr_crps_ensemble(o, f, dim=dim, weights=weights, keep_attrs=keep_attrs)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_crps_gaussian_dask(o_dask, f_dask, keep_attrs):
    mu = f_dask.mean('member')
    sig = f_dask.std('member')
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


@pytest.mark.parametrize('dim', DIMS)
def test_xr_crps_gaussian_dim(o, f, dim):
    mu = f.mean('member')
    sig = f.std('member')
    actual = xr_crps_gaussian(o, mu, sig, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_crps_quadrature_dask(o_dask, keep_attrs):
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


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_crps_quadrature_args(o_dask, f_dask, keep_attrs):
    xmin, xmax, tol = -10, 10, 1e-6
    cdf_or_dist = norm
    actual = xr_crps_quadrature(
        o_dask, cdf_or_dist, xmin, xmax, tol, keep_attrs=keep_attrs
    )
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


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_crps_quadrature_dim(o, f, dim, keep_attrs):
    cdf_or_dist = norm
    actual = xr_crps_quadrature(o, cdf_or_dist, dim=dim, keep_attrs=keep_attrs)
    assert_only_dim_reduced(dim, actual, o)
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
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


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_crps_gaussian_dask_b_int(o_dask, keep_attrs):
    mu = 0
    sig = 1
    actual = xr_crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_threshold_brier_score_dask_b_float(o_dask, f_dask, keep_attrs):
    threshold = 0.5
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_threshold_brier_score_dask_b_int(o_dask, f_dask, keep_attrs):
    threshold = 0
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_threshold_brier_score_multiple_thresholds_list(o_dask, f_dask, keep_attrs):
    threshold = [0.1, 0.3, 0.5]
    actual = xr_threshold_brier_score(
        o_dask.compute(), f_dask.compute(), threshold, keep_attrs=keep_attrs
    )
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_threshold_brier_score_multiple_thresholds_xr(o_dask, f_dask, keep_attrs):
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims='threshold')
    actual = xr_threshold_brier_score(
        o_dask.compute(), f_dask.compute(), threshold, keep_attrs=keep_attrs
    )
    assert actual.chunks is None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_threshold_brier_score_multiple_thresholds_dask(o_dask, f_dask, keep_attrs):
    threshold = xr.DataArray([0.1, 0.3, 0.5, 0.7], dims='threshold').chunk()
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_brier_score(o_dask, f_dask, keep_attrs):
    actual = xr_brier_score(
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
def test_xr_threshold_brier_score_dim(o, f, dim):
    actual = xr_threshold_brier_score(o, f, threshold=0.5, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize(
    'threshold', [0, 0.5, [0.1, 0.3, 0.5]], ids=['int', 'flat', 'list']
)
def test_xr_threshold_brier_score_dask_threshold(o_dask, f_dask, threshold):
    actual = xr_threshold_brier_score(o_dask, f_dask, threshold)
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
def test_xr_brier_score_dim(o, f, dim):
    actual = xr_brier_score((o > 0.5), (f > 0.5).mean('member'), dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_xr_brier_score_dask(o_dask, f_dask, keep_attrs):
    actual = xr_brier_score(
        (o_dask > 0.5).assign_attrs(**o_dask.attrs),
        (f_dask > 0.5).mean('member'),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is not None
    expected = brier_score((o_dask > 0.5), (f_dask > 0.5).mean('member'))
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
        

@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_rank_histogram_sum(o, f, dim, obj):
    """Test that the number of samples in the rank histogram is correct
    """
    if 'ds' in obj:
        name = 'var'
        o = o.to_dataset(name=name)
        f = f.to_dataset(name=name)
    if 'chunked' in obj:
        o = o.chunk()
        f = f.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            rank_histogram(o, f, dim=dim)
    else:
        assert rank_histogram(o, f, dim=dim).sum() == o.count()


def test_rank_histogram_values(o, f):
    """Test values in extreme cases (observations all smaller/larger \
        than forecasts)
    """
    assert rank_histogram((f.min() - 1) + 0 * o, f)[0] == o.size
    assert rank_histogram((f.max() + 1) + 0 * o, f)[-1] == o.size


def test_rank_histogram_dask(o_dask, f_dask):
    """Test that rank_histogram returns dask array if provided dask array
    """
    actual = rank_histogram(o_dask, f_dask)
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_discrimination_sum(o, f, dim, obj):
    """Test that the number of samples in the rank histogram is correct
    """
    if 'ds' in obj:
        name = 'var'
        o = o.to_dataset(name=name)
        f = f.to_dataset(name=name)
    if 'chunked' in obj:
        o = o.chunk()
        f = f.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            discrimination(o > 0.5, (f > 0.5).mean('member'), dim=dim)
    else:
        hist_event, hist_no_event = discrimination(
            o > 0.5, (f > 0.5).mean('member'), dim=dim
        )
        if 'ds' in obj:
            hist_event = hist_event[name]
            hist_no_event = hist_no_event[name]
        hist_event_sum = hist_event.sum('forecast_probability', skipna=False).values
        hist_no_event_sum = hist_no_event.sum(
            'forecast_probability', skipna=False
        ).values
        # Note, xarray's assert_allclose is already imported but won't compare to scalar
        assert np.allclose(hist_event_sum[~np.isnan(hist_event_sum)], 1)
        assert np.allclose(hist_no_event_sum[~np.isnan(hist_no_event_sum)], 1)


def test_discrimination_values(o):
    """Test values for perfect forecast
    """
    f = xr.concat(10 * [o], dim='member')
    hist_event, hist_no_event = discrimination(o > 0.5, (f > 0.5).mean('member'))
    assert np.allclose(hist_event[-1], 1)
    assert np.allclose(hist_event[:-1], 0)
    assert np.allclose(hist_no_event[0], 1)
    assert np.allclose(hist_no_event[1:], 0)


def test_discrimination_dask(o_dask, f_dask):
    """Test that rank_histogram returns dask array if provided dask array
    """
    hist_event, hist_no_event = discrimination(
        o_dask > 0.5, (f_dask > 0.5).mean('member')
    )
    assert hist_event.chunks is not None
    assert hist_no_event.chunks is not None