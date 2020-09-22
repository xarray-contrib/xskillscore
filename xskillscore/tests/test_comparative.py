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
    actual = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim='time', alpha=0.05, metric='mae'
    )
    walk_larger_significance = actual > actual.confidence
    crossing_after_timesteps = walk_larger_significance.argmax(dim='time')
    # check timesteps after which sign_test larger confidence
    assert crossing_after_timesteps == 3


@pytest.mark.parametrize('observation', [True, False])
def test_sign_test_categorical(a_1d, a_1d_worse, b_1d, observation):
    """Test sign_test categorical."""
    a_1d = logical(a_1d)
    a_1d_worse = logical(a_1d_worse)
    b_1d = logical(b_1d)
    sign_test(a_1d, a_1d_worse, b_1d, time_dim='time', metric='categorical')


@pytest.mark.parametrize('metric', ['categorical', 'mae'])
def test_sign_test_identical(a_1d, a_1d_worse, b_1d, metric):
    """Test that identical forecasts show no walk step."""
    identicals = [1, 7]
    for i in identicals:
        # set a_1d_worse = a_1d identical for time=i
        a_1d_worse = a_1d_worse.where(a_1d_worse != a_1d_worse.isel(time=i)).fillna(
            a_1d.isel(time=i)
        )
    if metric == 'categorical':
        a_1d = logical(a_1d)
        a_1d_worse = logical(a_1d_worse)
        b_1d = logical(b_1d)
    actual = sign_test(a_1d, a_1d_worse, b_1d, time_dim='time', metric=metric)
    # check flat
    assert (actual.diff(dim='time').isel(time=[i - 1 for i in identicals]) == 0).all()


def test_sign_test_alpha(a_1d, a_1d_worse, b_1d):
    """Test that larger alpha leads to small confidence bounds in sign_test."""
    actual_large_alpha = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim='time', alpha=0.1, metric='mae'
    )
    actual_small_alpha = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim='time', alpha=0.01, metric='mae'
    )
    # check difference in confidence
    assert (actual_large_alpha.confidence < actual_small_alpha.confidence).all()
    # check identical sign_test
    assert actual_large_alpha.drop(['alpha', 'confidence']).equals(
        actual_small_alpha.drop(['alpha', 'confidence'])
    )


def test_sign_test_user_function(a_1d, a_1d_worse, b_1d):
    """Test sign_test with user function as metric."""

    def mse(a, b, dim):
        return ((a - b) ** 2).mean(dim)

    actual = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim='time', orientation='negative', metric=mse
    )
    assert actual.isel(time=-1) > 0


@pytest.mark.parametrize('orientation', ['negative', 'positive'])
def test_sign_test_orientation(a_1d, a_1d_worse, b_1d, orientation):
    """Test sign_test orientation."""
    actual = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim='time', orientation=orientation, metric='mae'
    )
    if orientation == 'negative':
        # a_1d wins
        assert actual.isel(time=-1) > 0
    elif orientation == 'positive':
        # a_1d_worse wins because of corrupted metric orientation
        print(actual.isel(time=-1))
        assert actual.isel(time=-1) < 0


@pytest.mark.parametrize('metric', ['mae', 'rmse', 'mse'])
def test_sign_test_already_compared_orientation_negative(
    a_1d, a_1d_worse, b_1d, metric
):
    """Test sign_test with forecasts already previously evaluated with observation for
    negative orientation (smaller distances mean better forecast)."""
    a_b_diff = getattr(xs, metric)(a_1d, b_1d, dim=[])
    a_worse_b_diff = getattr(xs, metric)(a_1d_worse, b_1d, dim=[])
    actual = sign_test(
        a_b_diff, a_worse_b_diff, None, time_dim='time', orientation='negative'
    )
    assert actual.isel(time=-1) > 0


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
        f_o_diff, f_worse_o_diff, None, time_dim='time', orientation='positive'
    )
    assert actual.isel(time=-1) > 0


@pytest.mark.parametrize('metric', ['no_valid_metric_string', (), 1])
def test_sign_test_invalid_metric_fails(metric, a_1d, a_1d_worse, b_1d):
    """Sign_test fails because of invalid metric keyword."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, metric=metric)
    assert 'metric' in str(e.value)


def test_sign_test_observations_None_metric_fails(a_1d, a_1d_worse):
    """Sign_test fails because observations None but metric provided."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, None, metric='mae')
    assert 'observations must be provided when metric' in str(e.value)


@pytest.mark.parametrize('orientation', ['categorical', None])
def test_sign_test_invalid_orientation_fails(orientation, a_1d, a_1d_worse, b_1d):
    """Sign_test fails because of invalid orientation keyword."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, orientation=orientation, metric=None)
    assert '`orientation` requires to be either "positive" or' in str(e.value)


@pytest.mark.filterwarnings('ignore:Ignoring provided observation')
def test_sign_test_no_metric_but_observation_warns(a_1d, a_1d_worse, b_1d):
    """Sign_test warns if no metric but observation, ignores observation."""
    with_obs = sign_test(a_1d, a_1d_worse, b_1d, orientation='positive', metric=None)
    without_obs = sign_test(a_1d, a_1d_worse, None, orientation='positive', metric=None)
    assert (without_obs == with_obs).all()


def test_sign_test_dim(a, b):
    """Sign_test with dim specified."""
    a_worse = a + OFFSET
    actual = sign_test(
        a,
        a_worse,
        b,
        orientation='positive',
        metric='mse',
        dim=['lon', 'lat'],
        time_dim='time',
    )
    # check result reduced by dim
    assert len(actual.dims) == 1


@pytest.mark.xfail()
def test_sign_test_dim_fails(a_1d, a_1d_worse, b_1d):
    """Sign_test fails if no time_dim in dim."""
    sign_test(a_1d, a_1d_worse, b_1d, time_dim='time', dim='time')
