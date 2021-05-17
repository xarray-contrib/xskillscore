import numpy as np
import pytest
from dask import is_dask_collection
from xarray.testing import assert_equal

import xskillscore as xs
from xskillscore import halfwidth_ci_test, mae, sign_test

OFFSET = -1
METRIC = "mae"


@pytest.fixture
def a_1d_worse(a_1d):
    """a_1d worsened by constant offset."""
    return a_1d + OFFSET


@pytest.fixture
def a_1d_worse_less_corr(a_1d):
    """similar but not perfectly correlated a_1d."""
    a_1d_worse = a_1d.copy()
    step = 3
    a_1d_worse[::step] = a_1d[::step].values + 0.1
    return a_1d_worse


@pytest.fixture
def a_worse(a):
    """a worsened by constant offset."""
    return a + OFFSET


def logical(ds):
    """returns results from [0, 1]."""
    return ds > 0.5


@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "multidim Dataset", "DataArray", "mixed"])
def test_sign_test_inputs(a_1d, a_1d_worse, b_1d, input, chunk):
    """Test sign_test with xr inputs and chunked."""
    if "Dataset" in input:
        name = "var"
        a_1d = a_1d.to_dataset(name=name)
        a_1d_worse = a_1d_worse.to_dataset(name=name)
        b_1d = b_1d.to_dataset(name=name)
        if input == "multidim Dataset":
            a_1d["var2"] = a_1d["var"] * 2
            a_1d_worse["var2"] = a_1d_worse["var"] * 2
            b_1d["var2"] = b_1d["var"] * 2
    elif input == "mixed":
        name = "var"
        a_1d = a_1d.to_dataset(name=name)
    if chunk:
        a_1d = a_1d.chunk()
        a_1d_worse = a_1d_worse.chunk()
        b_1d = b_1d.chunk()
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", alpha=0.05, metric="mae"
    )
    # check dask collection preserved
    for actual in [actual_significantly_different, actual_walk, actual_confidence]:
        assert is_dask_collection(actual) if chunk else not is_dask_collection(actual)


@pytest.mark.parametrize("observation", [True, False])
def test_sign_test_categorical(a_1d, a_1d_worse, b_1d, observation):
    """Test sign_test categorical."""
    a_1d = logical(a_1d)
    a_1d_worse = logical(a_1d_worse)
    b_1d = logical(b_1d)
    sign_test(
        a_1d,
        a_1d_worse,
        b_1d,
        time_dim="time",
        metric="categorical",
        orientation="positive",
    )


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
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", metric=metric, orientation="positive"
    )
    # check flat
    assert (
        actual_walk.diff(dim="time").isel(time=[i - 1 for i in identicals]) == 0
    ).all()


def test_sign_test_alpha(a_1d, a_1d_worse, b_1d):
    """Test that larger alpha leads to small confidence bounds in sign_test."""
    (
        actual_significantly_different_large_alpha,
        actual_walk_large_alpha,
        actual_confidence_large_alpha,
    ) = sign_test(a_1d, a_1d_worse, b_1d, time_dim="time", alpha=0.1, metric="mae")
    (
        actual_significantly_different_small_alpha,
        actual_walk_small_alpha,
        actual_confidence_small_alpha,
    ) = sign_test(a_1d, a_1d_worse, b_1d, time_dim="time", alpha=0.01, metric="mae")
    # check difference in confidence
    assert (actual_confidence_large_alpha < actual_confidence_small_alpha).all()
    # check identical sign_test
    assert actual_walk_large_alpha.equals(actual_walk_small_alpha)


def test_sign_test_user_function(a_1d, a_1d_worse, b_1d):
    """Test sign_test with user function as metric."""

    def mse(a, b, dim):
        return ((a - b) ** 2).mean(dim)

    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", orientation="negative", metric=mse
    )
    assert actual_walk.isel(time=-1) > 0


@pytest.mark.parametrize("orientation", ["negative", "positive"])
def test_sign_test_orientation(a_1d, a_1d_worse, b_1d, orientation):
    """Test sign_test orientation."""
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a_1d, a_1d_worse, b_1d, time_dim="time", orientation=orientation, metric="mae"
    )
    if orientation == "negative":
        # a_1d wins
        assert actual_walk.isel(time=-1) > 0
    elif orientation == "positive":
        # a_1d_worse wins because of corrupted metric orientation
        print(actual_walk.isel(time=-1))
        assert actual_walk.isel(time=-1) < 0


@pytest.mark.parametrize("metric", ["mae", "rmse", "mse"])
def test_sign_test_already_compared_orientation_negative(
    a_1d, a_1d_worse, b_1d, metric
):
    """Test sign_test with forecasts already previously evaluated with observation for
    negative orientation (smaller distances mean better forecast)."""
    a_b_diff = getattr(xs, metric)(a_1d, b_1d, dim=[])
    a_worse_b_diff = getattr(xs, metric)(a_1d_worse, b_1d, dim=[])
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a_b_diff, a_worse_b_diff, None, time_dim="time", orientation="negative"
    )
    assert actual_walk.isel(time=-1) > 0


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
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        f_o_diff, f_worse_o_diff, None, time_dim="time", orientation="positive"
    )
    assert actual_walk.isel(time=-1) > 0


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
    (
        actual_significantly_different_with_obs,
        actual_walk_with_obs,
        actual_confidence_with_obs,
    ) = sign_test(a_1d, a_1d_worse, b_1d, orientation="positive", metric=None)
    (
        actual_significantly_different_without_obs,
        actual_walk_without_obs,
        actual_confidence_without_obs,
    ) = sign_test(a_1d, a_1d_worse, None, orientation="positive", metric=None)
    assert (actual_walk_without_obs == actual_walk_with_obs).all()


def test_sign_test_dim(a, a_worse, b):
    """Sign_test with dim specified."""
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a,
        a_worse,
        b,
        orientation="positive",
        metric="mse",
        dim=["lon", "lat"],
        time_dim="time",
    )
    # check result reduced by dim
    assert len(actual_walk.dims) == 1


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
    actual_significantly_different, actual_walk, actual_confidence = sign_test(
        a, a_worse, b, time_dim="time", metric="mse"
    )
    a_nan = a.copy()
    a_nan[1:3, 1:3, 1:3] = np.nan
    (
        actual_significantly_different_nan,
        actual_walk_nan,
        actual_confidence_nan,
    ) = sign_test(a_nan, a_worse, b, time_dim="time", metric="mse")
    assert not (actual_confidence_nan == actual_confidence).all()


@pytest.mark.parametrize("alpha", [0.05])
@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "multidim Dataset", "DataArray", "mixed"])
def test_halfwidth_ci_test_inputs(
    a_1d, a_1d_worse_less_corr, b_1d, input, chunk, alpha
):
    """Test halfwidth_ci_test with xr inputs and chunked."""
    if "Dataset" in input:
        name = "var"
        a_1d = a_1d.to_dataset(name=name)
        a_1d_worse_less_corr = a_1d_worse_less_corr.to_dataset(name=name)
        b_1d = b_1d.to_dataset(name=name)
        if input == "multidim Dataset":
            a_1d["var2"] = a_1d["var"] * 2
            a_1d_worse_less_corr["var2"] = a_1d_worse_less_corr["var"] * 2
            b_1d["var2"] = b_1d["var"] * 2
    elif input == "mixed":
        name = "var"
        a_1d = a_1d.to_dataset(name=name)
    if chunk:
        a_1d = a_1d.chunk()
        a_1d_worse_less_corr = a_1d_worse_less_corr.chunk()
        b_1d = b_1d.chunk()
    actual_significantly_different, actual_diff, actual_hwci = halfwidth_ci_test(
        a_1d, a_1d_worse_less_corr, b_1d, METRIC, alpha=alpha
    )
    # check dask collection preserved
    for actual in [actual_significantly_different, actual_diff, actual_hwci]:
        assert is_dask_collection(actual) if chunk else not is_dask_collection(actual)


@pytest.mark.parametrize("alpha", [0.0, 0, 1.0, 1.0, 5.0, 5])
def test_halfwidth_ci_test_alpha(a_1d, a_1d_worse_less_corr, b_1d, alpha):
    """Test halfwidth_ci_test alpha error messages."""
    with pytest.raises(ValueError) as e:
        halfwidth_ci_test(a_1d, a_1d_worse_less_corr, b_1d, METRIC, alpha=alpha)
    assert "`alpha` must be between 0 and 1 or `return_p`" in str(e.value)


@pytest.mark.parametrize("metric", ["mape", "pearson_r"])
def test_halfwidth_ci_test_metric_error(a_1d, a_1d_worse_less_corr, b_1d, metric):
    """Test halfwidth_ci_test alpha error messages."""
    with pytest.raises(ValueError) as e:
        halfwidth_ci_test(a_1d, a_1d_worse_less_corr, b_1d, metric)
    assert "`metric` should be a valid distance metric function" in str(e.value)


@pytest.mark.parametrize("alpha", [0.05])
@pytest.mark.parametrize(
    "metric", ["me", "rmse", "mse", "mae", "median_absolute_error", "smape"]
)
def test_halfwidth_ci_test(a_1d, a_1d_worse_less_corr, b_1d, metric, alpha):
    """Test halfwidth_ci_test favors better forecast."""
    a_1d_worse_less_corr = a_1d.copy()
    # make a_worse worse every second timestep
    step = 3
    a_1d_worse_less_corr[::step] = a_1d_worse_less_corr[::step] + OFFSET * 3
    actual_significantly_different, actual_diff, actual_hwci = halfwidth_ci_test(
        a_1d, a_1d_worse_less_corr, b_1d, metric, alpha=alpha
    )
    assert actual_significantly_different


def test_halfwidth_ci_test_climpred(a_1d, b_1d):
    """Test halfwidth_ci_test as climpred would use it with observations=None."""
    a_1d_worse_less_corr = a_1d.copy()
    # make a_worse worse every second timestep
    a_1d_worse_less_corr[::2] = a_1d_worse_less_corr[::2] + OFFSET
    # calc skill before as in climpred
    dim = []
    time_dim = "time"
    mae_f1o = mae(a_1d, b_1d, dim=dim)
    mae_f2o = mae(a_1d_worse_less_corr, b_1d, dim=dim)

    actual_significantly_different, actual_diff, actual_hwci = halfwidth_ci_test(
        mae_f1o,
        mae_f2o,
        observations=None,
        metric=None,
        dim=dim,
        time_dim=time_dim,
    )
    expected_significantly_different, expected_diff, expected_hwci = halfwidth_ci_test(
        a_1d, a_1d_worse_less_corr, b_1d, METRIC, dim=dim, time_dim=time_dim
    )
    assert_equal(actual_significantly_different, expected_significantly_different)
    assert_equal(actual_diff, expected_diff)
    assert_equal(actual_hwci, expected_hwci)


@pytest.mark.parametrize("dim", [[], ["lon", "lat"]])
def test_halfwidth_ci_test_dim(a, b, dim):
    """Test halfwidth_ci_test for different dim on ."""
    a_worse = a.copy()
    # make a_worse worse every second timestep
    a_worse[::2, :, :] = a_worse[::2, :, :] + OFFSET
    actual_significantly_different, actual_diff, actual_hwci = halfwidth_ci_test(
        a, a_worse, b, METRIC, dim=dim
    )
    # difference larger than half width ci
    assert (actual_diff > actual_hwci).mean() > 0.5
    for d in dim:
        assert d not in actual_diff.dims, print(d, "found, but shouldnt")


def test_halfwidth_ci_test_alpha_hwci(a_1d, a_1d_worse_less_corr, b_1d):
    """Test halfwidth_ci_test with larger alpha leads to smaller hwci."""
    (
        actual_large_alpha_significantly_different,
        actual_large_alpha_diff,
        actual_large_alpha_alpha,
    ) = halfwidth_ci_test(a_1d, a_1d_worse_less_corr, b_1d, METRIC, alpha=0.1)
    (
        actual_small_alpha_significantly_different,
        actual_small_alpha_diff,
        actual_small_alpha_alpha,
    ) = halfwidth_ci_test(a_1d, a_1d_worse_less_corr, b_1d, METRIC, alpha=0.01)
    assert actual_large_alpha_alpha < actual_small_alpha_alpha
