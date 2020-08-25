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
    threshold_brier_score,
)


@pytest.fixture
def o():
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(lats), len(lons))
    return xr.DataArray(data, coords=[lats, lons], dims=['lat', 'lon'])


@pytest.fixture
def f():
    members = np.arange(3)
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(members), len(lats), len(lons))
    return xr.DataArray(
        data, coords=[members, lats, lons], dims=['member', 'lat', 'lon']
    )


@pytest.fixture
def threshold():
    return 0.5


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_crps_gaussian_accessor(o, f, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    mu = f.mean('member')
    sig = f.std('member')
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
@pytest.mark.parametrize('dask_bool', [False, True])
def test_crps_ensemble_accessor(o, f, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    actual = crps_ensemble(o, f)

    ds = xr.Dataset()
    ds['o'] = o
    ds['f'] = f
    if outer_bool:
        ds = ds.drop_vars('f')
        expected = ds.xs.crps_ensemble('o', f)
    else:
        expected = ds.xs.crps_ensemble('o', 'f')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_crps_quadrature_accessor(o, dask_bool, outer_bool):
    cdf_or_dist = norm
    if dask_bool:
        o = o.chunk()
    actual = crps_quadrature(o, cdf_or_dist)

    ds = xr.Dataset()
    ds['o'] = o
    ds['cdf_or_dist'] = cdf_or_dist
    if outer_bool:
        ds = ds.drop_vars('cdf_or_dist')
        expected = ds.xs.crps_quadrature('o', cdf_or_dist)
    else:
        expected = ds.xs.crps_quadrature('o', 'cdf_or_dist')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_threshold_brier_score_accessor(o, f, threshold, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    actual = threshold_brier_score(o, f, threshold)

    ds = xr.Dataset()
    ds['o'] = o
    ds['f'] = f
    if outer_bool:
        ds = ds.drop_vars('f')
        expected = ds.xs.threshold_brier_score('o', f, threshold)
    else:
        expected = ds.xs.threshold_brier_score('o', 'f', threshold)
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_brier_score_accessor(o, f, threshold, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    actual = brier_score(o > threshold, (f > threshold).mean('member'))

    ds = xr.Dataset()
    ds['o'] = o > threshold
    ds['f'] = (f > threshold).mean('member')
    if outer_bool:
        ds = ds.drop_vars('f')
        expected = ds.xs.brier_score('o', (f > threshold).mean('member'))
    else:
        expected = ds.xs.brier_score('o', 'f')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_rank_histogram_accessor(o, f, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    actual = rank_histogram(o, f)

    ds = xr.Dataset()
    ds['o'] = o
    ds['f'] = f
    if outer_bool:
        ds = ds.drop_vars('f')
        expected = ds.xs.rank_histogram('o', f)
    else:
        expected = ds.xs.rank_histogram('o', 'f')
    assert_allclose(actual, expected)


@pytest.mark.parametrize('outer_bool', [False, True])
@pytest.mark.parametrize('dask_bool', [False, True])
def test_discrimination_accessor(o, f, threshold, dask_bool, outer_bool):
    if dask_bool:
        o = o.chunk()
        f = f.chunk()
    hist_event_actual, hist_no_event_actual = discrimination(
        o > threshold, (f > threshold).mean('member')
    )

    ds = xr.Dataset()
    ds['o'] = o > threshold
    ds['f'] = (f > threshold).mean('member')
    if outer_bool:
        ds = ds.drop_vars('f')
        hist_event_expected, hist_no_event_expected = ds.xs.discrimination(
            'o', (f > threshold).mean('member')
        )
    else:
        hist_event_expected, hist_no_event_expected = ds.xs.discrimination('o', 'f')
    assert_allclose(hist_event_actual, hist_event_expected)
    assert_allclose(hist_no_event_actual, hist_no_event_expected)
