import numpy as np
import numpy.testing as npt
import properscoring
import pytest
import xarray as xr
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from xarray.tests import assert_allclose, assert_identical

from xskillscore.core.probabilistic import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    discrimination,
    rank_histogram,
    reliability,
    roc,
    rps,
    threshold_brier_score,
)

DIMS = ["lon", "lat", ["lon", "lat"], None, []]


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


@pytest.mark.parametrize("keep_attrs", [True, False])
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


@pytest.mark.parametrize("dim", DIMS)
def test_crps_ensemble_dim(o, f_prob, dim):
    actual = crps_ensemble(o, f_prob, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_ensemble_weighted(o, f_prob, weights_cos_lat, keep_attrs):
    dim = ["lon", "lat"]
    actual = crps_ensemble(
        o, f_prob, dim=dim, weights=weights_cos_lat, keep_attrs=keep_attrs
    )
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_gaussian_dask(o_dask, f_prob_dask, keep_attrs):
    mu = f_prob_dask.mean("member")
    sig = f_prob_dask.std("member")
    actual = crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    expected = properscoring.crps_gaussian(o_dask, mu, sig)
    expected = xr.DataArray(expected, coords=o_dask.coords).mean()
    # test for numerical identity of xskillscore crps and properscoring crps
    assert_allclose(actual, expected)
    # test that xskillscore crps_ensemble returns chunks
    assert actual.chunks is not None
    # show that properscoring crps_ensemble returns no chunks
    assert expected.chunks is None


@pytest.mark.parametrize("dim", DIMS)
def test_crps_gaussian_dim(o, f_prob, dim):
    mu = f_prob.mean("member")
    sig = f_prob.std("member")
    actual = crps_gaussian(o, mu, sig, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.slow
@pytest.mark.parametrize("keep_attrs", [True, False])
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
@pytest.mark.parametrize("keep_attrs", [True, False])
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
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keep_attrs", [True, False])
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


@pytest.mark.parametrize("keep_attrs", [True, False])
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


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_gaussian_dask_b_int(o_dask, keep_attrs):
    mu = 0
    sig = 1
    actual = crps_gaussian(o_dask, mu, sig, keep_attrs=keep_attrs)
    assert actual is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize("keep_attrs", [True, False])
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


@pytest.mark.parametrize("keep_attrs", [True, False])
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


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_threshold_brier_score_multiple_thresholds_list(o, f_prob, keep_attrs):
    threshold = [0.1, 0.3, 0.5]
    actual = threshold_brier_score(o, f_prob, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is None or actual.chunks == ()
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_threshold_brier_score_multiple_thresholds_xr(o, f_prob, keep_attrs):
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims="threshold")
    actual = threshold_brier_score(o, f_prob, threshold, keep_attrs=keep_attrs)
    assert actual.chunks is None or actual.chunks == ()
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_threshold_brier_score_multiple_thresholds_dask(
    o_dask, f_prob_dask, keep_attrs
):
    threshold = xr.DataArray([0.1, 0.3, 0.5, 0.7], dims="threshold").chunk()
    actual = threshold_brier_score(
        o_dask, f_prob_dask, threshold, keep_attrs=keep_attrs
    )
    assert actual.chunks is not None
    if keep_attrs:
        assert actual.attrs == o_dask.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_brier_score(o, f_prob, keep_attrs):
    actual = brier_score(
        (o > 0.5).assign_attrs(**o.attrs),
        (f_prob > 0.5).mean("member"),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is None or actual.chunks == ()
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


@pytest.mark.parametrize("dim", DIMS)
def test_threshold_brier_score_dim(o, f_prob, dim):
    actual = threshold_brier_score(o, f_prob, threshold=0.5, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize(
    "threshold", [0, 0.5, [0.1, 0.3, 0.5]], ids=["int", "flat", "list"]
)
def test_threshold_brier_score_dask_threshold(o_dask, f_prob_dask, threshold):
    actual = threshold_brier_score(o_dask, f_prob_dask, threshold)
    assert actual.chunks is not None


@pytest.mark.parametrize("dim", DIMS)
def test_brier_score_dim(o, f_prob, dim):
    actual = brier_score((o > 0.5), (f_prob > 0.5).mean("member"), dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_brier_score_dask(o_dask, f_prob_dask, keep_attrs):
    actual = brier_score(
        (o_dask > 0.5).assign_attrs(**o_dask.attrs),
        (f_prob_dask > 0.5).mean("member"),
        keep_attrs=keep_attrs,
    )
    assert actual.chunks is not None
    expected = properscoring.brier_score(
        (o_dask > 0.5), (f_prob_dask > 0.5).mean("member")
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


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("obj", ["da", "ds", "chunked_da", "chunked_ds"])
def test_rank_histogram_sum(o, f_prob, dim, obj):
    """Test that the number of samples in the rank histogram is correct"""
    if "ds" in obj:
        name = "var"
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
    if "chunked" in obj:
        o = o.chunk()
        f_prob = f_prob.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            rank_histogram(o, f_prob, dim=dim)
    else:
        rank_hist = rank_histogram(o, f_prob, dim=dim)
        if "ds" in obj:
            rank_hist = rank_hist[name]
            o = o[name]
        assert_allclose(rank_hist.sum(), o.count())


def test_rank_histogram_values(o, f_prob):
    """Test values in extreme cases that observations \
        all smaller/larger than forecasts"""
    assert rank_histogram((f_prob.min() - 1) + 0 * o, f_prob)[0] == o.size
    assert rank_histogram((f_prob.max() + 1) + 0 * o, f_prob)[-1] == o.size


def test_rank_histogram_dask(o_dask, f_prob_dask):
    """Test that rank_histogram returns dask array if provided dask array"""
    actual = rank_histogram(o_dask, f_prob_dask)
    assert actual.chunks is not None


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("obj", ["da", "ds", "chunked_da", "chunked_ds"])
def test_discrimination_sum(o, f_prob, dim, obj):
    """Test that the probabilities sum to 1"""
    if "ds" in obj:
        name = "var"
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
    if "chunked" in obj:
        o = o.chunk()
        f_prob = f_prob.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            discrimination(o > 0.5, (f_prob > 0.5).mean("member"), dim=dim)
    else:
        disc = discrimination(o > 0.5, (f_prob > 0.5).mean("member"), dim=dim)
        if "ds" in obj:
            disc = disc[name]
        hist_event_sum = (
            disc.sel(event=True).sum("forecast_probability", skipna=False).values
        )
        hist_no_event_sum = (
            disc.sel(event=False).sum("forecast_probability", skipna=False).values
        )
        # Note, xarray's assert_allclose is already imported but won't compare to scalar
        assert np.allclose(hist_event_sum[~np.isnan(hist_event_sum)], 1)
        assert np.allclose(hist_no_event_sum[~np.isnan(hist_no_event_sum)], 1)


def test_discrimination_perfect_values(o):
    """Test values for perfect forecast"""
    f = xr.concat(10 * [o], dim="member")
    disc = discrimination(o > 0.5, (f > 0.5).mean("member"))
    assert np.allclose(disc.sel(event=True)[-1], 1)
    assert np.allclose(disc.sel(event=True)[:-1], 0)
    assert np.allclose(disc.sel(event=False)[0], 1)
    assert np.allclose(disc.sel(event=False)[1:], 0)


def test_discrimination_dask(o_dask, f_prob_dask):
    """Test that discrimination returns dask array if provided dask array"""
    disc = discrimination(o_dask > 0.5, (f_prob_dask > 0.5).mean("member"))
    assert disc.chunks is not None


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("obj", ["da", "ds", "chunked_da", "chunked_ds"])
def test_reliability(o, f_prob, dim, obj):
    """Test that reliability object can be generated"""
    if "ds" in obj:
        name = "var"
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
    if "chunked" in obj:
        o = o.chunk()
        f_prob = f_prob.chunk()
    if dim == []:
        with pytest.raises(ValueError):
            reliability(o > 0.5, (f_prob > 0.5).mean("member"), dim)
    else:
        reliability(o > 0.5, (f_prob > 0.5).mean("member"), dim=dim)


def test_reliability_values(o, f_prob):
    """Test 1D reliability values against sklearn calibration_curve"""
    for lon in f_prob.lon:
        for lat in f_prob.lat:
            o_1d = o.sel(lon=lon, lat=lat) > 0.5
            f_1d = (f_prob.sel(lon=lon, lat=lat) > 0.5).mean("member")
            # scipy bins are only left-edge inclusive and 1e-8 is added to the last bin, whereas
            # xhistogram the rightmost edge of xhistogram bins is included - mimic scipy behaviour
            actual = reliability(
                o_1d, f_1d, probability_bin_edges=np.linspace(0, 1 + 1e-8, 6)
            )
            expected, _ = calibration_curve(
                o_1d, f_1d, normalize=False, n_bins=5, strategy="uniform"
            )
            npt.assert_allclose(actual.where(actual.notnull(), drop=True), expected)
            npt.assert_allclose(actual["samples"].sum(), o_1d.size)


def test_reliability_perfect_values(o):
    """Test values for perfect forecast"""
    f_prob = xr.concat(10 * [o], dim="member")
    # scipy bins are only left-edge inclusive and 1e-8 is added to the last bin, whereas
    # xhistogram the rightmost edge of xhistogram bins is included - mimic scipy behaviour
    actual = reliability(
        o > 0.5,
        (f_prob > 0.5).mean("member"),
        probability_bin_edges=np.linspace(0, 1 + 1e-8, 6),
    )
    expected_true_samples = (o > 0.5).sum()
    expected_false_samples = (o <= 0.5).sum()
    assert np.allclose(actual[0], 0)
    assert np.allclose(actual[-1], 1)
    assert np.allclose(actual["samples"][0], expected_false_samples)
    assert np.allclose(actual["samples"][-1], expected_true_samples)
    assert np.allclose(actual["samples"].sum(), o.size)


def test_reliability_dask(o_dask, f_prob_dask):
    """Test that reliability returns dask array if provided dask array"""
    actual = reliability(o_dask > 0.5, (f_prob_dask > 0.5).mean("member"))
    assert actual.chunks is not None


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("obj", ["da", "ds", "chunked_da", "chunked_ds"])
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
    F1 = xr.DataArray([0] * 2 + [0.1] * 5 + [0.3] * 3, dims="member")
    F2 = xr.DataArray([0] * 2 + [0.1] * 3 + [0.3] * 5, dims="member")
    np.testing.assert_allclose(rps(Obs, F2, category_edges), 0.89)
    np.testing.assert_allclose(rps(Obs, F1, category_edges), 0.73)


def test_2_category_rps_equals_brier_score(o, f_prob):
    """Test that RPS for two categories equals the Brier Score."""
    category_edges = np.array([0.0, 0.5, 1.0])
    assert_allclose(
        rps(o, f_prob, category_edges=category_edges, dim=None),
        brier_score(o > 0.5, (f_prob > 0.5).mean("member"), dim=None),
    )


def test_rps_perfect_values(o, category_edges):
    """Test values for perfect forecast"""
    f = xr.concat(10 * [o], dim="member")
    res = rps(o, f, category_edges=category_edges)
    assert (res == 0).all()


def test_rps_dask(o_dask, f_prob_dask, category_edges):
    """Test that rps returns dask array if provided dask array"""
    assert rps(o_dask, f_prob_dask, category_edges=category_edges).chunks is not None


@pytest.mark.parametrize(
    "observation,forecast",
    [
        (
            pytest.lazy_fixture("observation_1d_long"),
            pytest.lazy_fixture("forecast_1d_long"),
        ),
        (
            pytest.lazy_fixture("observation_3d"),
            pytest.lazy_fixture("forecast_3d"),
        ),
    ],
)
@pytest.mark.parametrize("dim", ["time", None])
@pytest.mark.parametrize("drop_intermediate_bool", [True, False])
@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "DataArray"])
@pytest.mark.parametrize(
    "return_results", ["all_as_tuple", "area", "all_as_metric_dim"]
)
def test_roc_returns(
    observation,
    forecast,
    symmetric_edges,
    dim,
    return_results,
    input,
    chunk,
    drop_intermediate_bool,
):
    """testing keywords and inputs pass"""
    if "Dataset" in input:
        name = "var"
        forecast = forecast.to_dataset(name=name)
        observation = observation.to_dataset(name=name)
    if chunk:
        forecast = forecast.chunk()
        observation = observation.chunk()

    roc(
        observation,
        forecast,
        symmetric_edges,
        dim=dim,
        drop_intermediate=drop_intermediate_bool,
        return_results=return_results,
    )


def test_roc_auc_score_random_forecast(
    forecast_1d_long, observation_1d_long, symmetric_edges
):
    """Test that ROC AUC around 0.5 for random forecast."""
    area = roc(
        observation_1d_long,
        forecast_1d_long,
        symmetric_edges,
        dim="time",
        return_results="area",
    )
    assert area < 0.6
    assert area > 0.4


def test_roc_auc_score_perfect_forecast(forecast_1d_long, symmetric_edges):
    """Test that ROC AUC equals 1 for perfect forecast."""
    area = roc(
        forecast_1d_long,
        forecast_1d_long,
        symmetric_edges,
        drop_intermediate=False,
        dim="time",
        return_results="area",
    )
    assert area == 1.0


@pytest.mark.parametrize("drop_intermediate_bool", [False, True])
def test_roc_auc_score_out_of_range_forecast(
    forecast_1d_long, observation_1d_long, symmetric_edges, drop_intermediate_bool
):
    """Test that ROC AUC equals 0.0 for out of range forecast."""
    area = roc(
        observation_1d_long,
        xr.ones_like(forecast_1d_long) + 100,
        symmetric_edges,
        drop_intermediate=drop_intermediate_bool,
        dim="time",
        return_results="area",
    )
    assert float(area) in [0.0, 0.5]  # expect 0.0 but sometimes evaluates at 0.5 in CI


@pytest.mark.parametrize("drop_intermediate_bool", [False, True])
def test_roc_auc_score_out_of_range_observation(
    forecast_1d_long, observation_1d_long, symmetric_edges, drop_intermediate_bool
):
    """Test that ROC AUC equals 0.0 for out of range observation."""
    area = roc(
        xr.ones_like(observation_1d_long) + 100,
        forecast_1d_long,
        symmetric_edges,
        drop_intermediate=drop_intermediate_bool,
        dim="time",
        return_results="area",
    )
    np.testing.assert_almost_equal(area, 0.0, decimal=2)


@pytest.mark.parametrize("drop_intermediate_bool", [False, True])
def test_roc_auc_score_out_of_range_edges(
    forecast_1d_long, observation_1d_long, symmetric_edges, drop_intermediate_bool
):
    """Test that ROC AUC equals 0.5 for out of range edges."""
    area = roc(
        observation_1d_long,
        forecast_1d_long,
        symmetric_edges + 100,
        drop_intermediate=drop_intermediate_bool,
        dim="time",
        return_results="area",
    )
    assert float(area) == 0.5


@pytest.mark.parametrize("drop_intermediate_bool", [False, True])
def test_roc_auc_score_constant_forecast(
    forecast_1d_long, observation_1d_long, symmetric_edges, drop_intermediate_bool
):
    """Test that ROC AUC equals 0.5 for constant forecast."""
    xs_area = roc(
        observation_1d_long > 0,
        forecast_1d_long * 0,
        symmetric_edges,
        drop_intermediate=drop_intermediate_bool,
        dim="time",
        return_results="area",
    )
    sk_area = roc_auc_score(observation_1d_long > 0, forecast_1d_long * 0)
    np.testing.assert_allclose(xs_area, sk_area)


@pytest.mark.parametrize("drop_intermediate_bool", [False, True])
def test_roc_bin_edges_continuous_against_sklearn(
    forecast_1d_long, observation_1d_long, drop_intermediate_bool
):
    """Test xs.roc against sklearn.metrics.roc_curve/auc_score."""
    fp = np.clip(forecast_1d_long, 0, 1)  # prob
    ob = observation_1d_long > 0  # binary
    # sklearn
    sk_fpr, sk_tpr, _ = roc_curve(ob, fp, drop_intermediate=drop_intermediate_bool)
    sk_area = roc_auc_score(ob, fp)
    # xs
    xs_fpr, xs_tpr, xs_area = roc(
        ob,
        fp,
        "continuous",
        drop_intermediate=drop_intermediate_bool,
        return_results="all_as_tuple",
    )
    np.testing.assert_allclose(xs_area, sk_area)
    if not drop_intermediate_bool:  # drops sometimes one too much or too little
        assert (xs_fpr == sk_fpr).all()
        assert (xs_tpr == sk_tpr).all()


def test_roc_bin_edges_drop_intermediate(forecast_1d_long, observation_1d_long):
    """Test that drop_intermediate reduces probability_bins in xs.roc ."""
    fp = np.clip(forecast_1d_long, 0, 1)  # prob
    ob = observation_1d_long > 0  # binary
    # xs
    txs_fpr, txs_tpr, txs_area = roc(
        ob, fp, "continuous", drop_intermediate=True, return_results="all_as_tuple"
    )
    fxs_fpr, fxs_tpr, fxs_area = roc(
        ob, fp, "continuous", drop_intermediate=False, return_results="all_as_tuple"
    )
    # same area
    np.testing.assert_allclose(fxs_area, txs_area)
    # same or less probability_bins
    assert len(fxs_fpr) >= len(txs_fpr)
    assert len(fxs_tpr) >= len(txs_tpr)


def test_roc_keeps_probability_bin_as_coord(
    observation_1d_long, forecast_1d_long, symmetric_edges
):
    """Test that roc keeps probability_bin as coords."""
    fpr, tpr, area = roc(
        observation_1d_long,
        forecast_1d_long,
        symmetric_edges,
        drop_intermediate=False,
        return_results="all_as_tuple",
    )
    assert (tpr.probability_bin == symmetric_edges).all()
    assert (fpr.probability_bin == symmetric_edges).all()


def test_roc_bin_edges_symmetric_asc_or_desc(
    observation_1d_long, forecast_1d_long, symmetric_edges
):
    """Test that roc bin_edges works increasing or decreasing order."""
    fpr, tpr, area = roc(
        observation_1d_long,
        forecast_1d_long,
        symmetric_edges,
        drop_intermediate=False,
        return_results="all_as_tuple",
    )
    fpr2, tpr2, area2 = roc(
        observation_1d_long,
        forecast_1d_long,
        symmetric_edges[::-1],
        drop_intermediate=False,
        return_results="all_as_tuple",
    )
    assert_identical(fpr, fpr2.sortby(fpr.probability_bin))
    assert_identical(tpr, tpr2.sortby(tpr.probability_bin))
    assert_identical(area, area2)
