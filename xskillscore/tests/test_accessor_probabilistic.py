import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm
from xarray.tests import assert_allclose

from xskillscore.core.probabilistic import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    discrimination,
    rank_histogram,
    reliability,
    threshold_brier_score,
)


@pytest.fixture
def threshold():
    return 0.5


@pytest.mark.parametrize('outer_bool', [False, True])
def test_crps_gaussian_accessor(o, f_prob, outer_bool):
    mu = f_prob.mean('member')
    sig = f_prob.std('member')
    actual = crps_gaussian(o, mu, sig)
    ds = xr.Dataset()
    ds['o'] = o
    ds['mu'] = mu
    ds['sig'] = sig
    if outer_bool:
        ds = ds.drop_vars('mu')
        expected = ds.xs.crps_gaussian('o', mu, sig)
    else:
        expected = ds.xs.crps_gaussian('o', 'mu', 'sig')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
def test_crps_ensemble_accessor(o, f_prob, outer_bool):
    actual = crps_ensemble(o, f_prob)
    ds = xr.Dataset()
    ds['o'] = o
    ds['f_prob'] = f_prob
    if outer_bool:
        ds = ds.drop_vars('f_prob')
        expected = ds.xs.crps_ensemble('o', f_prob)
    else:
        expected = ds.xs.crps_ensemble('o', 'f_prob')
    assert_allclose(actual, expected)


@pytest.mark.slow
def test_crps_quadrature_accessor(o):
    # to speed things up
    o = o.isel(time=0, drop=True)
    cdf_or_dist = norm
    actual = crps_quadrature(o, cdf_or_dist)
    ds = xr.Dataset()
    ds['o'] = o
    expected = ds.xs.crps_quadrature('o', cdf_or_dist)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
def test_threshold_brier_score_accessor(o, f_prob, threshold, outer_bool):
    actual = threshold_brier_score(o, f_prob, threshold)
    ds = xr.Dataset()
    ds['o'] = o
    ds['f_prob'] = f_prob
    if outer_bool:
        ds = ds.drop_vars('f_prob')
        expected = ds.xs.threshold_brier_score('o', f_prob, threshold)
    else:
        expected = ds.xs.threshold_brier_score('o', 'f_prob', threshold)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
def test_brier_score_accessor(o, f_prob, threshold, outer_bool):
    actual = brier_score(o > threshold, (f_prob > threshold).mean('member'))
    ds = xr.Dataset()
    ds['o'] = o > threshold
    ds['f_prob'] = (f_prob > threshold).mean('member')
    if outer_bool:
        ds = ds.drop_vars('f_prob')
        expected = ds.xs.brier_score('o', (f_prob > threshold).mean('member'))
    else:
        expected = ds.xs.brier_score('o', 'f_prob')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
def test_rank_histogram_accessor(o, f_prob, outer_bool):
    actual = rank_histogram(o, f_prob)
    ds = xr.Dataset()
    ds['o'] = o
    ds['f_prob'] = f_prob
    if outer_bool:
        ds = ds.drop_vars('f_prob')
        expected = ds.xs.rank_histogram('o', f_prob)
    else:
        expected = ds.xs.rank_histogram('o', 'f_prob')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
def test_discrimination_accessor(o, f_prob, threshold, outer_bool):
    hist_event_actual, hist_no_event_actual = discrimination(
        o > threshold, (f_prob > threshold).mean('member')
    )
    ds = xr.Dataset()
    ds['o'] = o > threshold
    ds['f_prob'] = (f_prob > threshold).mean('member')
    if outer_bool:
        ds = ds.drop_vars('f_prob')
        hist_event_expected, hist_no_event_expected = ds.xs.discrimination(
            'o', (f_prob > threshold).mean('member')
        )
    else:
        hist_event_expected, hist_no_event_expected = ds.xs.discrimination(
            'o', 'f_prob'
        )
    assert_allclose(hist_event_actual, hist_event_expected)
    assert_allclose(hist_no_event_actual, hist_no_event_expected)


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_reliability_accessor(o, f, threshold, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    rel_actual, samp_actual = reliability(o > threshold, (f > threshold).mean('member'))

    ds = xr.Dataset()
    ds['o'] = o > threshold
    ds['f'] = (f > threshold).mean('member')
    if outer_bool:
        ds = ds.drop_vars('f')
        rel_expected, samp_expected = ds.xs.reliability(
            'o', (f > threshold).mean('member')
        )
    else:
        rel_expected, samp_expected = ds.xs.reliability('o', 'f')
    assert_allclose(rel_actual, rel_expected)
    assert_allclose(samp_actual, samp_expected)
