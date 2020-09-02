import numpy as np
import numpy.testing as npt
import properscoring
import pytest
import xarray as xr
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from xarray.tests import assert_allclose, assert_identical

from xskillscore.core.probabilistic import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    discrimination,
    rank_histogram,
    reliability,
    rps,
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
    # to speed things up
    o_dask = o_dask.isel(time=0, drop=True)
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
    # to speed things up
    o_dask = o_dask.isel(time=0, drop=True)
    f_prob_dask = f_prob_dask.isel(time=0, drop=True)
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


@pytest.mark.slow
@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('keep_attrs', [True, False])
def test_crps_quadrature_dim(o, dim, keep_attrs):
    # to speed things up
    o = o.isel(time=0, drop=True)
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


def test_rank_histogram_values(o, f_prob):
    """Test values in extreme cases that observations all smaller/larger than forecasts
    """
    assert rank_histogram((f_prob.min() - 1) + 0 * o, f_prob)[0] == o.size
    assert rank_histogram((f_prob.max() + 1) + 0 * o, f_prob)[-1] == o.size


def test_rank_histogram_dask(o_dask, f_prob_dask):
    """Test that rank_histogram returns dask array if provided dask array"""
    actual = rank_histogram(o_dask, f_prob_dask)
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_discrimination_sum(o, f_prob, dim, obj):
    """Test that the probabilities sum to 1"""
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
        disc = discrimination(o > 0.5, (f_prob > 0.5).mean('member'), dim=dim)
        if 'ds' in obj:
            disc = disc[name]
        hist_event_sum = (
            disc.sel(event=True).sum('forecast_probability', skipna=False).values
        )
        hist_no_event_sum = (
            disc.sel(event=False).sum('forecast_probability', skipna=False).values
        )
        # Note, xarray's assert_allclose is already imported but won't compare to scalar
        assert np.allclose(hist_event_sum[~np.isnan(hist_event_sum)], 1)
        assert np.allclose(hist_no_event_sum[~np.isnan(hist_no_event_sum)], 1)


def test_discrimination_perfect_values(o):
    """Test values for perfect forecast
    """
    f = xr.concat(10 * [o], dim='member')
    disc = discrimination(o > 0.5, (f > 0.5).mean('member'))
    assert np.allclose(disc.sel(event=True)[-1], 1)
    assert np.allclose(disc.sel(event=True)[:-1], 0)
    assert np.allclose(disc.sel(event=False)[0], 1)
    assert np.allclose(disc.sel(event=False)[1:], 0)


def test_discrimination_dask(o_dask, f_prob_dask):
    """Test that discrimination returns dask array if provided dask array"""
    disc = discrimination(o_dask > 0.5, (f_prob_dask > 0.5).mean('member'))
    assert disc.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_reliability(o, f_prob, dim, obj):
    """Test that reliability object can be generated"""
    if 'ds' in obj:
        name = 'var'
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
    if 'chunked' in obj:
        o = o.chunk()
        f_prob = f_prob.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            reliability(o > 0.5, (f_prob > 0.5).mean('member'), dim)
    else:
        reliability(o > 0.5, (f_prob > 0.5).mean('member'), dim=dim)


def test_reliability_values(o, f_prob):
    """Test 1D reliability values against sklearn calibration_curve"""
    for lon in f_prob.lon:
        for lat in f_prob.lat:
            o_1d = o.sel(lon=lon, lat=lat) > 0.5
            f_1d = (f_prob.sel(lon=lon, lat=lat) > 0.5).mean('member')
            actual = reliability(o_1d, f_1d)
            expected, _ = calibration_curve(
                o_1d, f_1d, normalize=False, n_bins=5, strategy='uniform'
            )
            npt.assert_allclose(actual.where(actual.notnull(), drop=True), expected)
            npt.assert_allclose(actual['samples'].sum(), o_1d.size)


def test_reliability_perfect_values(o):
    """Test values for perfect forecast"""
    f_prob = xr.concat(10 * [o], dim='member')
    actual = reliability(o > 0.5, (f_prob > 0.5).mean('member'))
    expected_true_samples = (o > 0.5).sum()
    expected_false_samples = (o <= 0.5).sum()
    assert np.allclose(actual[0], 0)
    assert np.allclose(actual[-1], 1)
    assert np.allclose(actual['samples'][0], expected_false_samples)
    assert np.allclose(actual['samples'][-1], expected_true_samples)
    assert np.allclose(actual['samples'].sum(), o.size)


def test_reliability_dask(o_dask, f_prob_dask):
    """Test that reliability returns dask array if provided dask array"""
    actual = reliability(o_dask > 0.5, (f_prob_dask > 0.5).mean('member'))
    assert actual.chunks is not None


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('obj', ['da', 'ds', 'chunked_da', 'chunked_ds'])
def test_rps(o, f_prob, category_edges, dim, obj):
    """Test that rps reduced dim and works for (chunked) ds and da"""
    actual = rps(o, f_prob, category_edges=category_edges, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


def test_rps_wilks_example():
    """Test with values from Wilks, D. S. (2006). Statistical methods in the
    atmospheric sciences (2nd ed, Vol. 91). Amsterdam ; Boston: Academic Press. p.301
    """
    category_edges = np.array([-0.01, 0.01, 0.24, 10])
    Obs = xr.DataArray([0.0001])
    F1 = xr.DataArray([0] * 2 + [0.1] * 5 + [0.3] * 3, dims='member')
    F2 = xr.DataArray([0] * 2 + [0.1] * 3 + [0.3] * 5, dims='member')
    np.testing.assert_allclose(rps(Obs, F2, category_edges), 0.89)
    np.testing.assert_allclose(rps(Obs, F1, category_edges), 0.73)


def test_2_category_rps_equals_brier_score(o, f_prob):
    """Test that RPS for two categories equals the Brier Score."""
    category_edges = np.array([0.0, 0.5, 1.0])
    assert_allclose(
        rps(o, f_prob, category_edges=category_edges, dim=None),
        brier_score(o > 0.5, (f_prob > 0.5).mean('member'), dim=None),
    )


def test_rps_perfect_values(o, category_edges):
    """Test values for perfect forecast
    """
    f = xr.concat(10 * [o], dim='member')
    res = rps(o, f, category_edges=category_edges)
    assert (res == 0).all()


def test_rps_dask(o_dask, f_prob_dask, category_edges):
    """Test that rps returns dask array if provided dask array
    """
    assert rps(o_dask, f_prob_dask, category_edges=category_edges).chunks is not None
