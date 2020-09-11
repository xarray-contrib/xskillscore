import numpy as np
import pytest

import xskillscore as xs
from xskillscore import sign_test

OFFSET = -1


@pytest.fixture
def a_1d_worse(a_1d):
    return a_1d + OFFSET


def logical(ds):
    return ds > 0.5


def test_sign_test_raw(a_1d, a_1d_worse, b_1d):
    """Test sign_test where significance crossed (for np.random.seed(42) values)."""
    actual = sign_test(a_1d, a_1d_worse, b_1d, dim='time', alpha=0.05)
    walk_larger_significance = actual.sel(results='sign_test') > actual.sel(
        results='confidence'
    )
    crossing_after_timesteps = walk_larger_significance.argmax(dim='time')
    # check timesteps after which sign_test larger confidence
    assert crossing_after_timesteps == 3


def test_sign_test_categorical(a_1d, a_1d_worse, b_1d):
    """Test sign_test categorical."""
    a_1d = logical(a_1d)
    a_1d_worse = logical(a_1d_worse)
    b_1d = logical(b_1d)
    sign_test(a_1d, a_1d_worse, b_1d, dim='time', categorical=True)


@pytest.mark.parametrize('categorical', [True, False])
def test_sign_test_identical(a_1d, a_1d_worse, b_1d, categorical):
    """Test that identical forecasts show no walk step."""
    identicals = [1, 7]
    for i in identicals:
        # set a_1d_worse = a_1d identical for time=i
        a_1d_worse = a_1d_worse.where(a_1d_worse != a_1d_worse.isel(time=i)).fillna(
            a_1d.isel(time=i)
        )
    if categorical:
        a_1d = logical(a_1d)
        a_1d_worse = logical(a_1d_worse)
        b_1d = logical(b_1d)
    actual = sign_test(a_1d, a_1d_worse, b_1d, dim='time', categorical=categorical)
    # check flat
    assert (
        actual.sel(results='sign_test')
        .diff(dim='time')
        .isel(time=[i - 1 for i in identicals])
        == 0
    ).all()


def test_sign_test_alpha(a_1d, a_1d_worse, b_1d):
    """Test that larger alpha leads to small confidence bounds in sign_test."""
    actual_large_alpha = sign_test(a_1d, a_1d_worse, b_1d, dim='time', alpha=0.1)
    actual_small_alpha = sign_test(a_1d, a_1d_worse, b_1d, dim='time', alpha=0.01)
    # check difference in confidence
    assert (
        actual_large_alpha.sel(results='confidence')
        < actual_small_alpha.sel(results='confidence')
    ).all()
    # check identical sign_test
    assert (
        actual_large_alpha.sel(results='sign_test')
        .drop('alpha')
        .equals(actual_small_alpha.sel(results='sign_test').drop('alpha'))
    )


def test_sign_test_user_function(a_1d, a_1d_worse, b_1d):
    """Test sign_test with user function as metric."""

    def mse(a, b, dim):
        return ((a - b) ** 2).mean(dim)

    actual = sign_test(
        a_1d, a_1d_worse, b_1d, dim='time', orientation='negative', metric=mse
    )
    assert actual.sel(results='sign_test').isel(time=-1) > 0


@pytest.mark.parametrize('metric', ['mae', 'rmse', 'mse'])
def test_sign_test_already_compared_orientation_negative(
    a_1d, a_1d_worse, b_1d, metric
):
    """Test sign_test with forecasts already previously evaluated with observation for
    negative orientation (smaller distances mean better forecast)."""
    a_b_diff = getattr(xs, metric)(a_1d, b_1d, dim=[])
    a_worse_b_diff = getattr(xs, metric)(a_1d_worse, b_1d, dim=[])
    actual = sign_test(
        a_b_diff, a_worse_b_diff, observation=None, dim='time', orientation='negative'
    )
    print(actual)
    assert actual.sel(results='sign_test').isel(time=-1) > 0


def crpss(o, f_prob, dim=None):
    return 1 - xs.crps_ensemble(o, f_prob, dim=dim) / xs.crps_gaussian(
        o, o.mean('time'), o.std('time'), dim=[]
    )


@pytest.mark.parametrize('metric', [crpss], ids=['crpss'])
def test_sign_test_already_compared_orientation_positive_probabilistic(
    f_prob, o, metric
):
    """Test sign_test for probabilistic crpss metric with positive orientation."""
    o = o.isel(lon=0, lat=0, drop=True)
    f_prob = f_prob.isel(lon=0, lat=0, drop=True)
    f_prob_worse = f_prob + OFFSET
    f_o_diff = metric(o, f_prob, dim=[])
    f_worse_o_diff = metric(o, f_prob_worse, dim=[])
    actual = sign_test(
        f_o_diff, f_worse_o_diff, observation=None, dim='time', orientation='positive'
    )
    assert actual.sel(results='sign_test').isel(time=-1) > 0


@pytest.mark.parametrize('metric', ['no_valid_metric_string', (), 1])
def test_sign_test_invalid_metric_fails(metric, a_1d, a_1d_worse, b_1d):
    """Sign_test fails because of invalid metric keyword."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, metric=metric)
    assert 'metric' in str(e.value)


@pytest.mark.parametrize('orientation', ['categorical'])
def test_sign_test_invalid_orientation_fails(orientation, a_1d, a_1d_worse, b_1d):
    """Sign_test fails because of invalid orientation keyword."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, orientation=orientation)
    assert '`orientation` requires to be either "positive" or' in str(e.value)
