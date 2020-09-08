import numpy as np
import pytest

from xskillscore import sign_test

OFFSET = -1


@pytest.fixture
def a_1d_worse(a_1d):
    return a_1d + OFFSET


def logical(ds):
    return ds > 0.5


def test_sign_test_raw(a_1d, a_1d_worse, b_1d):
    actual = sign_test(a_1d, a_1d_worse, b_1d, dim='time', alpha=0.05)
    walk_larger_significance = actual.sel(results='sign_test') > actual.sel(
        results='confidence'
    )
    crossing_after_timesteps = walk_larger_significance.argmax(dim='time')
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
    assert (
        actual_large_alpha.sel(results='confidence')
        < actual_small_alpha.sel(results='confidence')
    ).all()
    assert actual_large_alpha.sel(results='sign_test').equals(
        actual_small_alpha.sel(results='sign_test')
    )
