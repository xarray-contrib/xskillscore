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
    discrimination,
    rank_histogram,
    threshold_brier_score,
)

DIMS = ['lon', 'lat', ['lon', 'lat'], None, []]


def assert_only_dim_reduced(dim, actual, obs):
    """Check that actual is only reduced by dims in dim."""
    if dim is None:
        dim = list(obs.dims)
    elif not isinstance(dim, list):
        dim = [dim]
    for d in dim:
        assert d not in actual.dims
    for d in obs.dims:
        if d not in dim:
            assert d in actual.dims


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_ensemble_dask(o_dask, f_prob_dask, keep_attrs):
    actual = crps_ensemble(o_dask, f_prob_dask, keep_attrs=keep_attrs)
    expected = properscoring.crps_ensemble(o_dask, f_prob_dask, axis=0)
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
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
def test_crps_ensemble_dim(o, f_prob, dim):
    actual = crps_ensemble(o, f_prob, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_ensemble_weighted(o, f_prob, weights, keep_attrs):
    dim = ['lon', 'lat']
    actual = crps_ensemble(o, f_prob, dim=dim, weights=weights, keep_attrs=keep_attrs)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_gaussian_dask(o_dask, f_prob_dask, keep_attrs):
    mu = f_prob_dask.mean('member')
    sig = f_prob_dask.std('member')
    actual = crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    expected = properscoring.crps_gaussian(o_dask, mu, sig)
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
    # test for numerical identity of xskillscore crps and properscoring crps
    assert_allclose(actual, expected)
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None


@pytest.mark.parametrize('dim', DIMS)
def test_crps_gaussian_dim(o, f_prob, dim):
    mu = f_prob.mean('member')
    sig = f_prob.std('member')
    actual = crps_gaussian(o, mu, sig, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.slow
@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_quadrature_dask(o_dask, keep_attrs):
    cdf_or_dist = norm
    actual = crps_quadrature(o_dask, cdf_or_dist, keep_attrs=keep_attrs)
    expected = properscoring.crps_quadrature(o_dask, cdf_or_dist)
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
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


@pytest.mark.slow
@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_quadrature_args(o_dask, f_prob_dask, keep_attrs):
    xmin, xmax, tol = -10, 10, 1e-6
    cdf_or_dist = norm
    actual = crps_quadrature(
        o_dask, cdf_or_dist, xmin, xmax, tol, keep_attrs=keep_attrs
    )
    expected = properscoring.crps_quadrature(o_dask, cdf_or_dist, xmin, xmax, tol)
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
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
def test_crps_quadrature_dim(o, dim, keep_attrs):
    cdf_or_dist = norm
    actual = crps_quadrature(o, cdf_or_dist, dim=dim, keep_attrs=keep_attrs)
    assert_only_dim_reduced(dim, actual, o)
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_dask(o_dask, f_prob_dask, keep_attrs):
    threshold = 0.5
    actual = threshold_brier_score(
        o_dask, f_prob_dask, threshold, keep_attrs=keep_attrs
    )
    expected = properscoring.threshold_brier_score(
        o_dask, f_prob_dask, threshold, axis=0
    )
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
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
def test_threshold_brier_score_dask_b_float(o_dask, f_prob_dask, keep_attrs):
    threshold = 0.5
    actual = threshold_brier_score(
        o_dask, f_prob_dask, threshold, keep_attrs=keep_attrs
    )
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_dask_b_int(o_dask, f_prob_dask, keep_attrs):
    threshold = 0
    actual = threshold_brier_score(
        o_dask, f_prob_dask, threshold, keep_attrs=keep_attrs
    )
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_multiple_thresholds_list(o, f_prob, keep_attrs):
    threshold = [0.1, 0.3, 0.5]
    actual = threshold_brier_score(o, f_prob, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is None or actual.chunks == ()
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_multiple_thresholds_xr(o, f_prob, keep_attrs):
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims='threshold')
    actual = threshold_brier_score(o, f_prob, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is None or actual.chunks == ()
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_threshold_brier_score_multiple_thresholds_dask(
    o_dask, f_prob_dask, keep_attrs
):
    threshold = xr.DataArray([0.1, 0.3, 0.5, 0.7], dims='threshold').chunk()
    actual = threshold_brier_score(
        o_dask, f_prob_dask, threshold, keep_attrs=keep_attrs
    )
    assert actual.chunks is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_brier_score(o, f_prob, keep_attrs):
    actual = brier_score(
        (o > 0.5).assign_attrs(**o.attrs),
        (f_prob > 0.5).mean('member'),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is None or actual.chunks == ()
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize('dim', DIMS)
def test_threshold_brier_score_dim(o, f_prob, dim):
    actual = threshold_brier_score(o, f_prob, threshold=0.5, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize(
    'threshold', [0, 0.5, [0.1, 0.3, 0.5]], ids=['int', 'flat', 'list']
)
def test_threshold_brier_score_dask_threshold(o_dask, f_prob_dask, threshold):
    actual = threshold_brier_score(o_dask, f_prob_dask, threshold)
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
def test_brier_score_dim(o, f_prob, dim):
    actual = brier_score((o > 0.5), (f_prob > 0.5).mean('member'), dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_brier_score_dask(o_dask, f_prob_dask, keep_attrs):
    actual = brier_score(
        (o_dask > 0.5).assign_attrs(**o_dask.attrs),
        (f_prob_dask > 0.5).mean('member'),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is not None
    expected = properscoring.brier_score(
        (o_dask > 0.5), (f_prob_dask > 0.5).mean('member')
    )
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
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


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_rank_histogram_sum(o, f_prob, dim, obj):
    """Test that the number of samples in the rank histogram is correct"""
    if 'ds' in obj:
        name = 'var'
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
    if 'chunked' in obj:
        o = o.chunk()
        f_prob = f_prob.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            rank_histogram(o, f_prob, dim=dim)
    else:
        rank_hist = rank_histogram(o, f_prob, dim=dim)
        if 'ds' in obj:
            rank_hist = rank_hist[name]
            o = o[name]
        assert_allclose(rank_hist.sum(), o.count())


def test_rank_histogram_values(o, f):
    """Test values in extreme cases that observations all smaller/larger than forecasts
    """
    assert rank_histogram((f.min() - 1) + 0 * o, f)[0] == o.size
    assert rank_histogram((f.max() + 1) + 0 * o, f)[-1] == o.size


def test_rank_histogram_dask(o_dask, f_prob_dask):
    """Test that rank_histogram returns dask array if provided dask array"""
    actual = rank_histogram(o_dask, f_prob_dask)
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_discrimination_sum(o, f_prob, dim, obj):
    """Test that the number of samples in the rank histogram is correct"""
    if 'ds' in obj:
        name = 'var'
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
    if 'chunked' in obj:
        o = o.chunk()
        f_prob = f_prob.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            discrimination(o > 0.5, (f_prob > 0.5).mean('member'), dim=dim)
    else:
        hist_event, hist_no_event = discrimination(
            o > 0.5, (f_prob > 0.5).mean('member'), dim=dim
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
    """Test values for perfect forecast"""
    f = xr.concat(10 * [o], dim='member')
    hist_event, hist_no_event = discrimination(o > 0.5, (f > 0.5).mean('member'))
    assert np.allclose(hist_event[-1], 1)
    assert np.allclose(hist_event[:-1], 0)
    assert np.allclose(hist_no_event[0], 1)
    assert np.allclose(hist_no_event[1:], 0)


def test_discrimination_dask(o_dask, f_prob_dask):
    """Test that rank_histogram returns dask array if provided dask array"""
    hist_event, hist_no_event = discrimination(
        o_dask > 0.5, (f_prob_dask > 0.5).mean('member')
    )
    assert hist_event.chunks is not None
    assert hist_no_event.chunks is not None
