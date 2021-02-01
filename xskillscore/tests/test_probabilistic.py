import numpy as np
import numpy.testing as npt
import properscoring
import pytest
import xarray as xr
from dask import is_dask_collection
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

DIMS = ["lon", "lat", ["lon", "lat"], None, []]


def modify_inputs(o, f_prob, input_type, chunk_bool):
    """Modify inputs depending on input_type and chunk_bool."""
    if "Dataset" in input_type:
        name = "var"
        o = o.to_dataset(name=name)
        f_prob = f_prob.to_dataset(name=name)
        if input_type == "multidim Dataset":
            o["var2"] = o["var"] ** 2
            f_prob["var2"] = f_prob["var"] ** 2
    if chunk_bool:
        o = o.chunk()
        f_prob = f_prob.chunk()
    return o, f_prob


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


def assert_chunk(actual, chunk_bool):
    """check that actual is chunked when chunk_bool==True."""
    if chunk_bool:
        assert is_dask_collection(actual)
    else:
        assert not is_dask_collection(actual)


def assert_keep_attrs(actual, o, keep_attrs):
    """check that actual kept attributes only if keep_attrs==True."""
    if keep_attrs:
        assert actual.attrs == o.attrs
    else:
        assert actual.attrs == {}


def assign_type_input_output(actual, o):
    assert type(o) == type(actual)


@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_ensemble_api_and_inputs(o, f_prob, keep_attrs, input_type, chunk_bool):
    """Test that crps_ensemble keeps attributes, chunking, input types and equals
    properscoring.crps_ensemble."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    actual = crps_ensemble(o, f_prob, keep_attrs=keep_attrs)
    if input_type == "DataArray":  # properscoring allows only DataArrays
        expected = properscoring.crps_ensemble(o, f_prob, axis=0)
        expected = xr.DataArray(expected, coords=o.coords).mean()
        # test for numerical identity of xskillscore crps and properscoring crps
        assert_allclose(actual, expected)
    # test that returns chunks
    assert_chunk(actual, chunk_bool)
    # test that attributes are kept
    assert_keep_attrs(actual, o, keep_attrs)
    # test that input types equal output types
    assign_type_input_output(actual, o)


@pytest.mark.parametrize("dim", DIMS)
def test_crps_ensemble_dim(o, f_prob, dim):
    """Check that crps_ensemble reduces only dim."""
    actual = crps_ensemble(o, f_prob, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_gaussian_api_and_inputs(o, f_prob, keep_attrs, input_type, chunk_bool):
    """Test that crps_gaussian keeps attributes, chunking, input types and equals
    properscoring.crps_gaussian."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    mu = f_prob.mean("member")
    sig = f_prob.std("member")
    actual = crps_gaussian(o, mu, sig, keep_attrs=keep_attrs)
    if input_type == "DataArray":  # properscoring allows only DataArrays
        expected = properscoring.crps_gaussian(o, mu, sig)
        expected = xr.DataArray(expected, coords=o.coords).mean()
        # test for numerical identity of xskillscore crps and properscoring crps
        assert_allclose(actual, expected)
    # test that returns chunks
    assert_chunk(actual, chunk_bool)
    # test that attributes are kept
    assert_keep_attrs(actual, o, keep_attrs)
    # test that input types equal output types
    assign_type_input_output(actual, o)


@pytest.mark.parametrize("dim", DIMS)
def test_crps_gaussian_dim(o, f_prob, dim):
    """Check that crps_gaussian reduces only dim."""
    mu = f_prob.mean("member")
    sig = f_prob.std("member")
    actual = crps_gaussian(o, mu, sig, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.slow
@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_quadrature_api_and_inputs(o, f_prob, keep_attrs, input_type, chunk_bool):
    """Test that crps_quadrature keeps attributes, chunking, input types and equals
    properscoring.crps_quadrature."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    # to speed things up
    o = o.isel(time=0, drop=True)
    cdf_or_dist = norm
    actual = crps_quadrature(o, cdf_or_dist, keep_attrs=keep_attrs)
    if input_type == "DataArray":  # properscoring allows only DataArrays
        expected = properscoring.crps_quadrature(o, cdf_or_dist)
        expected = xr.DataArray(expected, coords=o.coords).mean()
        # test for numerical identity of xskillscore crps and properscoring crps
        assert_allclose(actual, expected)
    # test that returns chunks
    assert_chunk(actual, chunk_bool)
    # test that attributes are kept
    assert_keep_attrs(actual, o, keep_attrs)
    # test that input types equal output types
    assign_type_input_output(actual, o)


@pytest.mark.slow
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_quadrature_args(o_dask, f_prob_dask, keep_attrs):
    """Test crps_quadrature args."""
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
    assert_chunk(actual, True)


@pytest.mark.slow
@pytest.mark.parametrize("dim", DIMS)
def test_crps_quadrature_dim(o, dim):
    """Check that crps_ensemble reduces only dim."""
    # to speed things up
    o = o.isel(time=0, drop=True)
    cdf_or_dist = norm
    actual = crps_quadrature(o, cdf_or_dist, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize("input_type", ["float", "int", "numpy.array", "xr.DataArray"])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_crps_gaussian_args(o, keep_attrs, input_type):
    """Test that crps_gaussian accepts various data types as args."""
    mu = 0
    sig = 1
    if "input_type" == "float":
        mu = float(mu)
        sig = float(sig)
    elif "input_type" == "xr.DataArray":
        mu = xr.DataArray(mu)
        sig = xr.DataArray(sig)
    elif "input_type" == "np.array":
        mu = np.array(mu)
        sig = np.array(sig)
    actual = crps_gaussian(o, mu, sig, keep_attrs=keep_attrs)
    assert_keep_attrs(actual, o, keep_attrs)


@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_threshold_brier_score_api_and_inputs(
    o, f_prob, keep_attrs, input_type, chunk_bool
):
    """Test that threshold_brier_score keeps attributes, chunking, input types and
    equals properscoring.threshold_brier_score."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    threshold = 0.5
    actual = threshold_brier_score(o, f_prob, threshold, keep_attrs=keep_attrs)
    if input_type == "DataArray":  # properscoring allows only DataArrays
        expected = properscoring.threshold_brier_score(o, f_prob, threshold, axis=0)
        expected = xr.DataArray(expected, coords=o.coords).mean()
        expected["threshold"] = threshold
        # test for numerical identity of xs threshold and properscoring threshold
        if keep_attrs:
            expected = expected.assign_attrs(**actual.attrs)
        assert_identical(actual, expected)
    # test that returns chunks
    assert_chunk(actual, chunk_bool)
    # test that attributes are kept
    assert_keep_attrs(actual, o, keep_attrs)
    # test that input types equal output types
    assign_type_input_output(actual, o)


@pytest.mark.parametrize("dim", DIMS)
def test_threshold_brier_score_dim(o, f_prob, dim):
    """Check that threshold_brier_score reduces only dim."""
    actual = threshold_brier_score(o, f_prob, threshold=0.5, dim=dim)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize(
    "threshold",
    [0, 0.5, [0.1, 0.3, 0.5], xr.DataArray([0.1, 0.3, 0.5], dims="threshold")],
    ids=["int", "flat", "list", "xr.DataArray"],
)
def test_threshold_brier_score_threshold(o_dask, f_prob_dask, threshold):
    """Test that threshold_brier_score accepts different kinds of thresholds."""
    actual = threshold_brier_score(o_dask, f_prob_dask, threshold)
    assert (actual.threshold == threshold).all()


@pytest.mark.skip(reason="not implemented")
def test_threshold_brier_score_threshold_dataset(o_dask, f_prob_dask):
    """Test that threshold_brier_score accepts xr.Dataset thresholds."""
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims="threshold").to_dataset(name="var")
    threshold["var2"] = xr.DataArray([0.2, 0.4, 0.6], dims="threshold")
    o_dask = o_dask.to_dataset(name="var")
    o_dask["var2"] = o_dask["var"] ** 2
    f_prob_dask = f_prob_dask.to_dataset(name="var")
    f_prob_dask["var2"] = f_prob_dask["var"] ** 2
    actual = threshold_brier_score(o_dask, f_prob_dask, threshold)
    assert actual
    # test thresholds in each DataArray
    # assert (actual.threshold == threshold).all()


def test_threshold_brier_score_dataset(o_dask, f_prob_dask):
    """Test that threshold_brier_score accepts xr.Datasets."""
    threshold = xr.DataArray([0.1, 0.3, 0.5], dims="threshold")
    o_dask = o_dask.to_dataset(name="var")
    o_dask["var2"] = o_dask["var"] ** 2
    f_prob_dask = f_prob_dask.to_dataset(name="var")
    f_prob_dask["var2"] = f_prob_dask["var"] ** 2
    actual = threshold_brier_score(o_dask, f_prob_dask, threshold)
    assert (actual.threshold == threshold).all()


@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("fair_bool", [True, False])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_brier_score_api_and_inputs(
    o, f_prob, keep_attrs, fair_bool, chunk_bool, input_type
):
    """Test that brier_score keeps attributes, chunking, input types."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    f_prob > 0.5
    if not fair_bool:
        f_prob = f_prob.mean("member")
    o = (o > 0.5).assign_attrs(**o.attrs)
    actual = brier_score(
        o,
        f_prob,
        keep_attrs=keep_attrs,
        fair=fair_bool,
    )
    # test that returns chunks
    assert_chunk(actual, chunk_bool)
    # test that attributes are kept
    assert_keep_attrs(actual, o, keep_attrs)
    # test that input types equal output types
    assign_type_input_output(actual, o)


@pytest.mark.parametrize("fair_bool", [True, False])
@pytest.mark.parametrize("dim", DIMS)
def test_brier_score_dim(o, f_prob, dim, fair_bool):
    """Check that brier_score reduces only dim."""
    f_prob > 0.5
    if not fair_bool:
        f_prob = f_prob.mean("member")
    o = o > 0.5
    actual = brier_score(o, f_prob, dim=dim, fair=fair_bool)
    assert_only_dim_reduced(dim, actual, o)


@pytest.mark.parametrize("dim", DIMS)
def test_brier_score_vs_fair_brier_score(o, f_prob, dim):
    """Test that brier_score scores lower for limited ensemble than bias brier_score."""
    fbs = brier_score((o > 0.5), (f_prob > 0.5), dim=dim, fair=True)
    bs = brier_score((o > 0.5), (f_prob > 0.5).mean("member"), dim=dim, fair=False)
    assert (fbs <= bs).all(), print("fairBS", fbs, "\nBS", bs)


@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["DataArray", "Dataset", "multidim Dataset"])
@pytest.mark.parametrize("dim", DIMS)
def test_rank_histogram_sum(o, f_prob, dim, chunk_bool, input_type):
    """Test that the number of samples in the rank histogram is correct"""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    if dim == []:
        with pytest.raises(ValueError):
            rank_histogram(o, f_prob, dim=dim)
    else:
        rank_hist = rank_histogram(o, f_prob, dim=dim)
        if "Dataset" in input_type:
            rank_hist = rank_hist[list(o.data_vars)[0]]
            o = o[list(o.data_vars)[0]]
            assert_allclose(rank_hist.sum(), o.count())
        assert_allclose(rank_hist.sum(), o.count())
        # test that returns chunks
        assert_chunk(rank_hist, chunk_bool)
        # test that attributes are kept # TODO: add
        # assert_keep_attrs(rank_hist, o, keep_attrs)
        # test that input types equal output types
        assign_type_input_output(rank_hist, o)


def test_rank_histogram_values(o, f_prob):
    """Test values in extreme cases that observations \
        all smaller/larger than forecasts"""
    assert rank_histogram((f_prob.min() - 1) + 0 * o, f_prob)[0] == o.size
    assert rank_histogram((f_prob.max() + 1) + 0 * o, f_prob)[-1] == o.size


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["DataArray", "Dataset", "multidim Dataset"])
def test_discrimination_sum(o, f_prob, dim, chunk_bool, input_type):
    """Test that the probabilities sum to 1"""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    if dim == []:
        with pytest.raises(ValueError):
            discrimination(o > 0.5, (f_prob > 0.5).mean("member"), dim=dim)
    else:
        disc = discrimination(o > 0.5, (f_prob > 0.5).mean("member"), dim=dim)
        # test that input types equal output types
        assign_type_input_output(disc, o)
        if "Dataset" in input_type:
            disc = disc[list(o.data_vars)[0]]
        hist_event_sum = (
            disc.sel(event=True).sum("forecast_probability", skipna=False).values
        )
        hist_no_event_sum = (
            disc.sel(event=False).sum("forecast_probability", skipna=False).values
        )
        # Note, xarray's assert_allclose is already imported but won't compare to scalar
        assert np.allclose(hist_event_sum[~np.isnan(hist_event_sum)], 1)
        assert np.allclose(hist_no_event_sum[~np.isnan(hist_no_event_sum)], 1)

        # test that returns chunks
        assert_chunk(disc, chunk_bool)
        # test that attributes are kept # TODO: add
        # assert_keep_attrs(disc, o, keep_attrs)


def test_discrimination_perfect_values(o):
    """Test values for perfect forecast"""
    f = xr.concat(10 * [o], dim="member")
    disc = discrimination(o > 0.5, (f > 0.5).mean("member"))
    assert np.allclose(disc.sel(event=True)[-1], 1)
    assert np.allclose(disc.sel(event=True)[:-1], 0)
    assert np.allclose(disc.sel(event=False)[0], 1)
    assert np.allclose(disc.sel(event=False)[1:], 0)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["DataArray", "Dataset", "multidim Dataset"])
def test_reliability_api_and_inputs(o, f_prob, dim, chunk_bool, input_type):
    """Test that reliability keeps chunking and input types."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    if dim == []:
        with pytest.raises(ValueError):
            reliability(o > 0.5, (f_prob > 0.5).mean("member"), dim)
    else:
        actual = reliability(o > 0.5, (f_prob > 0.5).mean("member"), dim=dim)
        # test that returns chunks
        assert_chunk(actual, chunk_bool)
        # test that attributes are kept
        # assert_keep_attrs(actual, o, keep_attrs) # TODO: implement
        # test that input types equal output types
        assign_type_input_output(actual, o)


def test_reliability_values(o, f_prob):
    """Test 1D reliability values against sklearn calibration_curve"""
    for lon in f_prob.lon:
        for lat in f_prob.lat:
            o_1d = o.sel(lon=lon, lat=lat) > 0.5
            f_1d = (f_prob.sel(lon=lon, lat=lat) > 0.5).mean("member")
            actual = reliability(o_1d, f_1d)
            expected, _ = calibration_curve(
                o_1d, f_1d, normalize=False, n_bins=5, strategy="uniform"
            )
            npt.assert_allclose(actual.where(actual.notnull(), drop=True), expected)
            npt.assert_allclose(actual["samples"].sum(), o_1d.size)


def test_reliability_perfect_values(o):
    """Test values for perfect forecast"""
    f_prob = xr.concat(10 * [o], dim="member")
    actual = reliability(o > 0.5, (f_prob > 0.5).mean("member"))
    expected_true_samples = (o > 0.5).sum()
    expected_false_samples = (o <= 0.5).sum()
    assert np.allclose(actual[0], 0)
    assert np.allclose(actual[-1], 1)
    assert np.allclose(actual["samples"][0], expected_false_samples)
    assert np.allclose(actual["samples"][-1], expected_true_samples)
    assert np.allclose(actual["samples"].sum(), o.size)


@pytest.mark.parametrize("fair_bool", [True, False])
@pytest.mark.parametrize("dim", DIMS)
def test_rps_reduce_dim(o, f_prob, category_edges, dim, fair_bool):
    """Test that rps reduced dim and works for (chunked) ds and da"""
    actual = rps(o, f_prob, category_edges=category_edges, dim=dim, fair=fair_bool)
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


@pytest.mark.parametrize("fair_bool", [True, False])
def test_rps_perfect_values(o, category_edges, fair_bool):
    """Test values for perfect forecast"""
    f = xr.concat(10 * [o], dim="member")
    res = rps(o, f, category_edges=category_edges, fair=fair_bool)
    assert (res == 0).all()


@pytest.mark.parametrize("fair_bool", [True, False])
def test_rps_dask(o_dask, f_prob_dask, category_edges, fair_bool):
    """Test that rps returns dask array if provided dask array"""
    assert (
        rps(o_dask, f_prob_dask, category_edges=category_edges, fair=fair_bool).chunks
        is not None
    )


@pytest.mark.parametrize("dim", DIMS)
def test_rps_vs_fair_rps(o, f_prob, category_edges, dim):
    """Test that fair rps is smaller or equal than rps due to ensemble-size
    adjustment."""
    frps = rps(o, f_prob, dim=dim, fair=True, category_edges=category_edges)
    ufrps = rps(o, f_prob, dim=dim, fair=False, category_edges=category_edges)
    assert (frps <= ufrps).all(), print("fairrps", frps, "\nufrps", ufrps)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("fair_bool", [True, False])
def test_rps_limits(o, f_prob, category_edges, fair_bool, dim):
    """Test rps between 0 and 1. Note: this only works because np.clip(rps,0,1)"""
    res = rps(o, f_prob, dim=dim, fair=fair_bool, category_edges=category_edges)
    assert (res <= 1.0).all(), print(res.max())
    assert (res >= 0).all(), print(res.min())
