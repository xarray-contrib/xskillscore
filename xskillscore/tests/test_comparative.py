import numpy as np
import pytest
from dask import is_dask_collection

import xskillscore as xs
from xskillscore import sign_test

OFFSET = -1


@pytest.fixture
def a_1d_worse(a_1d):
    return a_1d + OFFSET


@pytest.fixture
def a_worse(a):
    return a + OFFSET


def logical(ds):
    return ds > 0.5


@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "DataArray"])
def test_sign_test_inputs(a_1d, a_1d_worse, b_1d, input, chunk):
    """Test sign_test with xr inputs and chunked."""
    if input == "Dataset":
        name = "var"
        a_1d = a_1d.to_dataset(name=name)
        a_1d_worse = a_1d_worse.to_dataset(name=name)
        b_1d = b_1d.to_dataset(name=name)
    if chunk:
        a_1d = a_1d.chunk()
        a_1d_worse = a_1d_worse.chunk()
        b_1d = b_1d.chunk()
    actual = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", alpha=0.05, metric="mae"
    )
    # check dask collection preserved
    assert is_dask_collection(actual) if chunk else not is_dask_collection(actual)


@pytest.mark.parametrize("observation", [True, False])
def test_sign_test_categorical(a_1d, a_1d_worse, b_1d, observation):
    """Test sign_test categorical."""
    a_1d = logical(a_1d)
    a_1d_worse = logical(a_1d_worse)
    b_1d = logical(b_1d)
    sign_test(a_1d, a_1d_worse, b_1d, time_dim="time", metric="categorical")


@pytest.mark.parametrize("metric", ["categorical", "mae"])
def test_sign_test_identical(a_1d, a_1d_worse, b_1d, metric):
    """Test that identical forecasts show no walk step."""
    identicals = [1, 7]
    for i in identicals:
        # set a_1d_worse = a_1d identical for time=i
        a_1d_worse = a_1d_worse.where(a_1d_worse != a_1d_worse.isel(time=i)).fillna(
            a_1d.isel(time=i)
        )
    if metric == "categorical":
        a_1d = logical(a_1d)
        a_1d_worse = logical(a_1d_worse)
        b_1d = logical(b_1d)
    actual = sign_test(a_1d, a_1d_worse, b_1d, time_dim="time", metric=metric)
    # check flat
    assert (actual.diff(dim="time").isel(time=[i - 1 for i in identicals]) == 0).all()


def test_sign_test_alpha(a_1d, a_1d_worse, b_1d):
    """Test that larger alpha leads to small confidence bounds in sign_test."""
    actual_large_alpha = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", alpha=0.1, metric="mae"
    )
    actual_small_alpha = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", alpha=0.01, metric="mae"
    )
    # check difference in confidence
    assert (actual_large_alpha.confidence < actual_small_alpha.confidence).all()
    # check identical sign_test
    assert actual_large_alpha.drop(["alpha", "confidence"]).equals(
        actual_small_alpha.drop(["alpha", "confidence"])
    )


def test_sign_test_user_function(a_1d, a_1d_worse, b_1d):
    """Test sign_test with user function as metric."""

    def mse(a, b, dim):
        return ((a - b) ** 2).mean(dim)

    actual = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", orientation="negative", metric=mse
    )
    assert actual.isel(time=-1) > 0


@pytest.mark.parametrize("orientation", ["negative", "positive"])
def test_sign_test_orientation(a_1d, a_1d_worse, b_1d, orientation):
    """Test sign_test orientation."""
    actual = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", orientation=orientation, metric="mae"
    )
    if orientation == "negative":
        # a_1d wins
        assert actual.isel(time=-1) > 0
    elif orientation == "positive":
        # a_1d_worse wins because of corrupted metric orientation
        print(actual.isel(time=-1))
        assert actual.isel(time=-1) < 0


@pytest.mark.parametrize("metric", ["mae", "rmse", "mse"])
def test_sign_test_already_compared_orientation_negative(
    a_1d, a_1d_worse, b_1d, metric
):
    """Test sign_test with forecasts already previously evaluated with observation for
    negative orientation (smaller distances mean better forecast)."""
    a_b_diff = getattr(xs, metric)(a_1d, b_1d, dim=[])
    a_worse_b_diff = getattr(xs, metric)(a_1d_worse, b_1d, dim=[])
    actual = sign_test(
        a_b_diff, a_worse_b_diff, None, time_dim="time", orientation="negative"
    )
    assert actual.isel(time=-1) > 0


def crpss(o, f_prob, dim=None):
    return 1 - xs.crps_ensemble(o, f_prob, dim=dim) / xs.crps_gaussian(
        o, o.mean("time"), o.std("time"), dim=[]
    )


@pytest.mark.parametrize("metric", [crpss], ids=["crpss"])
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
        f_o_diff, f_worse_o_diff, None, time_dim="time", orientation="positive"
    )
    assert actual.isel(time=-1) > 0


@pytest.mark.parametrize("metric", ["no_valid_metric_string", (), 1])
def test_sign_test_invalid_metric_fails(metric, a_1d, a_1d_worse, b_1d):
    """Sign_test fails because of invalid metric keyword."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, metric=metric)
    assert "metric" in str(e.value)


def test_sign_test_observations_None_metric_fails(a_1d, a_1d_worse):
    """Sign_test fails because observations None but metric provided."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, None, metric="mae")
    assert "observations must be provided when metric" in str(e.value)


@pytest.mark.parametrize("orientation", ["categorical", None])
def test_sign_test_invalid_orientation_fails(orientation, a_1d, a_1d_worse, b_1d):
    """Sign_test fails because of invalid orientation keyword."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, orientation=orientation, metric=None)
    assert '`orientation` requires to be either "positive" or' in str(e.value)


@pytest.mark.filterwarnings("ignore:Ignoring provided observation")
def test_sign_test_no_metric_but_observation_warns(a_1d, a_1d_worse, b_1d):
    """Sign_test warns if no metric but observation, ignores observation."""
    with_obs = sign_test(a_1d, a_1d_worse, b_1d, orientation="positive", metric=None)
    without_obs = sign_test(a_1d, a_1d_worse, None, orientation="positive", metric=None)
    assert (without_obs == with_obs).all()


def test_sign_test_dim(a, a_worse, b):
    """Sign_test with dim specified."""
    actual = sign_test(
        a,
        a_worse,
        b,
        orientation="positive",
        metric="mse",
        dim=["lon", "lat"],
        time_dim="time",
    )
    # check result reduced by dim
    assert len(actual.dims) == 1


@pytest.mark.xfail()
def test_sign_test_dim_fails(a_1d, a_1d_worse, b_1d):
    """Sign_test fails if no time_dim in dim."""
    with pytest.raises(ValueError) as e:
        sign_test(a_1d, a_1d_worse, b_1d, time_dim="time", dim="time")
    assert "`dim` cannot contain `time_dim`" in str(e.value)


def test_sign_test_metric_correlation(a, a_worse, b):
    """Sign_test work for correlation metrics over other dimensions that time_dim."""
    sign_test(a, a_worse, b, time_dim="time", dim=["lon", "lat"], metric="pearson_r")


def test_sign_test_NaNs_confidence(a, a_worse, b):
    """Sign_test confidence with NaNs."""
    actual = sign_test(a, a_worse, b, time_dim="time", metric="mse")
    a_nan = a.copy()
    a_nan[1:3, 1:3, 1:3] = np.nan
    actual_nan = sign_test(a_nan, a_worse, b, time_dim="time", metric="mse")
    assert not (actual_nan.confidence == actual.confidence).all()
