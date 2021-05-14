import numpy as np
import numpy.testing as npt
import properscoring
import pytest
import xarray as xr
from dask import is_dask_collection
from pytest_lazyfixture import lazy_fixture
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
from xskillscore.core.utils import suppress_warnings

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


def test_crps_ensemble_weighted(o, f_prob, weights_cos_lat):
    """Test that weighted changes results of crps_ensemble."""
    dim = ["lon", "lat"]
    actual = crps_ensemble(o, f_prob, dim=dim, weights=weights_cos_lat)
    actual_no_weights = crps_ensemble(o, f_prob, dim=dim)
    assert not (actual_no_weights == actual).all()


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


def test_brier_score_forecast_member_dim(o, f_prob):
    """Check that brier_score allows forecasts with member dim and binary/boolean
    data."""
    f_prob > 0.5
    o = o > 0.5
    assert_identical(brier_score(o, f_prob), brier_score(o, f_prob.mean("member")))


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
        # dont understand the error message here, but it appeared
        with suppress_warnings("invalid value encountered in true_divide"):
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
    actual = reliability(
        o > 0.5,
        (f_prob > 0.5).mean("member"),
    )
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


def test_rps_category_edges_None_fails(o, f_prob):
    """Test that rps expects inputs to have category_edges dim if category_edges is None."""
    with pytest.raises(ValueError, match="Expected dimension"):
        rps(o, f_prob, category_edges=None, dim=[], input_distributions="c")


@pytest.mark.parametrize("input_distributions", ["c", "p"])
def test_rps_category_edges_None_works(o, f_prob, input_distributions):
    """Test that rps expects inputs to have category_edges dim if category_edges is None."""
    o = o.rename({"time": "category"})
    f_prob = f_prob.rename({"time": "category"}).mean("member")
    rps(o, f_prob, category_edges=None, dim=[], input_distributions=input_distributions)


@pytest.mark.parametrize("chunk_bool", [True, False])
@pytest.mark.parametrize("input_type", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("keep_attrs", [True, False])
def test_rps_api_and_inputs(
    o, f_prob, category_edges, keep_attrs, input_type, chunk_bool
):
    """Test that rps keeps attributes, chunking, input types."""
    o, f_prob = modify_inputs(o, f_prob, input_type, chunk_bool)
    category_edges = xr.DataArray(category_edges, dims="category_edge")
    if "Dataset" in input_type:
        category_edges = category_edges.to_dataset(name="var")
        if "multidim" in input_type:
            category_edges["var2"] = category_edges["var"]
    actual = rps(o, f_prob, category_edges, keep_attrs=keep_attrs)
    # test that returns chunks
    assert_chunk(actual, chunk_bool)
    # test that attributes are kept
    assert_keep_attrs(actual, o, keep_attrs)
    # test that input types equal output types
    assign_type_input_output(actual, o)


def rps_xhist(
    observations,
    forecasts,
    category_edges,
    dim=None,
    fair=False,
    weights=None,
    keep_attrs=False,
    member_dim="member",
):
    """Old way to calculate RPS with xhistogram.

    category_edges : array_like, xr.Dataset, xr.DataArray, None

        - array_like: Category bin edges used to compute the CDFs. Similar to
          np.histogram, all but the last (righthand-most) bin include the
          left edge and exclude the right edge. The last bin includes both edges.
          CDFs based on boolean or logical (True or 1 for event occurance, False or 0
          for non-occurance) observations.
          If ``fair==False``, forecasts should be between 0 and 1 without a dimension
          ``member_dim`` or boolean / binary containing a member dimension
          (probabilities will be internally calculated by ``.mean(member_dim))``.
          If ``fair==True``, forecasts must be boolean / binary containing dimension
          ``member_dim``."""
    from xskillscore.core.contingency import _get_category_bounds
    from xskillscore.core.utils import _keep_nans_masked, histogram

    bin_names = ["category"]
    bin_dim = f"{bin_names[0]}_edge"
    M = forecasts[member_dim].size

    assert isinstance(category_edges, np.ndarray)

    # histogram(dim=[]) not allowed therefore add fake member dim
    # to apply over when multi-dim observations
    if len(observations.dims) == 1:
        observations_bins = histogram(
            observations,
            bins=[category_edges],
            bin_names=["category_edge"],
            dim=None,
        )
    else:
        observations_bins = histogram(
            observations.expand_dims(member_dim),
            bins=[category_edges],
            bin_names=["category_edge"],
            dim=[member_dim],
        )
    if "category_edge_bin" in observations_bins.dims:
        observations_bins = observations_bins.rename(
            {"category_edge_bin": "category_edge"}
        )

    forecasts = histogram(
        forecasts,
        bins=[category_edges],
        bin_names=["category_edge"],
        dim=[member_dim],
    )
    if "category_edge_bin" in forecasts.dims:
        forecasts = forecasts.rename({"category_edge_bin": "category_edge"})

    # normalize f.sum()=1 to make cdf
    forecasts = forecasts / forecasts.sum(bin_dim)

    Fc = forecasts.cumsum(bin_dim)
    Oc = observations_bins.cumsum(bin_dim)

    # RPS formulas
    if fair:
        Ec = Fc * M
        res = ((Ec / M - Oc) ** 2 - Ec * (M - Ec) / (M ** 2 * (M - 1))).sum(bin_dim)
    else:
        res = ((Fc - Oc) ** 2).sum(bin_dim)

    if weights is not None:
        res = res.weighted(weights)

    res = res.mean(dim, keep_attrs=keep_attrs)
    # add bin edges as coords
    res = res.assign_coords(
        {"forecasts_category_edge": ", ".join(_get_category_bounds(category_edges))}
    )
    res = res.assign_coords(
        {"observations_category_edge": ", ".join(_get_category_bounds(category_edges))}
    )

    # keep nans and prevent 0 for all nan grids
    res = _keep_nans_masked(observations, res, dim, ignore=["category_edge"])
    return res


def test_rps_wilks_example():
    """Test with values from Wilks, D. S. (2006). Statistical methods in the
    atmospheric sciences (2nd ed, Vol. 91). Amsterdam ; Boston: Academic Press. p.301.
    """
    category_edges = np.array([0.0, 0.01, 0.24, 1.0])
    # first example
    # xhistogram way with np.array category_edges
    Obs = xr.DataArray([0.0001])  # .expand_dims('time')  # no precip
    F1 = xr.DataArray(
        [0] * 2 + [0.1] * 5 + [0.3] * 3, dims="member"
    )  # .expand_dims('time')
    F2 = xr.DataArray(
        [0] * 2 + [0.1] * 3 + [0.3] * 5, dims="member"
    )  # .expand_dims('time')
    np.testing.assert_allclose(rps_xhist(Obs, F1, category_edges), 0.73)
    np.testing.assert_allclose(rps_xhist(Obs, F2, category_edges), 0.89)
    # xr way with xr.DataArray category_edges
    xr_category_edges = xr.DataArray(
        category_edges, dims="category_edge", coords={"category_edge": category_edges}
    )
    assert_allclose(rps(Obs, F1, category_edges), rps(Obs, F1, xr_category_edges))
    assert_allclose(rps(Obs, F2, category_edges), rps(Obs, F2, xr_category_edges))

    # second example
    Obs = xr.DataArray([0.3])  # larger than 0.25
    np.testing.assert_allclose(rps_xhist(Obs, F1, category_edges), 0.53)
    np.testing.assert_allclose(rps_xhist(Obs, F2, category_edges), 0.29)
    # xr way with xr.DataArray category_edges
    assert_allclose(rps(Obs, F1, category_edges), rps(Obs, F1, xr_category_edges))
    assert_allclose(rps(Obs, F2, category_edges), rps(Obs, F2, xr_category_edges))


def test_rps_wilks_example_pdf():
    """Test xs.rps(category_edges=None, input_distributions='p') with values from
    Wilks, D. S. (2006). Statistical methods in the atmospheric sciences (2nd ed,
    Vol. 91). Amsterdam ; Boston: Academic Press. p.301.
    """
    Obs = xr.DataArray([1.0, 0.0, 0.0], dims="category")  # no precip
    F1 = xr.DataArray([0.2, 0.5, 0.3], dims="category")
    F2 = xr.DataArray([0.2, 0.3, 0.5], dims="category")
    np.testing.assert_allclose(
        rps(Obs, F1, category_edges=None, input_distributions="p"), 0.73
    )
    np.testing.assert_allclose(
        rps(Obs, F2, category_edges=None, input_distributions="p"), 0.89
    )

    # second example
    Obs = xr.DataArray([0.0, 0.0, 1.0], dims="category")  # larger than 0.25
    np.testing.assert_allclose(
        rps(Obs, F1, category_edges=None, input_distributions="p"), 0.53
    )
    np.testing.assert_allclose(
        rps(Obs, F2, category_edges=None, input_distributions="p"), 0.29
    )


@pytest.mark.parametrize("fair_bool", [True, False])
def test_2_category_rps_equals_brier_score(o, f_prob, fair_bool):
    """Test that RPS for two categories equals the Brier Score."""
    category_edges = np.array([0.0, 0.5, 1.0])
    assert_allclose(
        rps(
            o.rename({"time": "category"}),
            f_prob.rename({"time": "category"}),
            category_edges=category_edges,
            dim=None,
            fair=fair_bool,
        ).drop(["forecasts_category_edge", "observations_category_edge"]),
        brier_score(o > 0.5, (f_prob > 0.5), dim=None, fair=fair_bool),
    )


def test_rps_fair_category_edges_None(o, f_prob):
    """Test that RPS without category_edges works for fair==True if forecast[member]
    set."""
    rps(
        o.rename({"time": "category"}),
        f_prob.mean("member")
        .rename({"time": "category"})
        .assign_coords(member=f_prob.member.size),
        category_edges=None,
        dim=None,
        fair=True,
        input_distributions="p",
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
    """Test that fair rps is smaller (e.g. better) or equal than rps due to ensemble-
    size adjustment."""
    frps = rps(o, f_prob, dim=dim, fair=True, category_edges=category_edges)
    ufrps = rps(o, f_prob, dim=dim, fair=False, category_edges=category_edges)
    assert (frps <= ufrps).all()


@pytest.mark.parametrize("fair_bool", [True, False])
def test_rps_category_edges_xrDataArray(o, f_prob, fair_bool):
    """Test rps with category_edges as xrDataArray for forecast and observations edges."""
    category_edges = f_prob.quantile(
        q=[0.2, 0.4, 0.6, 0.8], dim=["time", "member"]
    ).rename({"quantile": "category_edge"})
    actual = rps(
        o,
        f_prob,
        dim="time",
        fair=fair_bool,
        category_edges=category_edges,
    )
    assert set(["lon", "lat"]) == set(actual.dims)
    assert "category_edge" not in actual.dims


@pytest.mark.parametrize("fair_bool", [True, False])
def test_rps_category_edges_xrDataset(o, f_prob, fair_bool):
    """Test rps with category_edges as xrDataArray for forecast and observations edges."""
    o = o.to_dataset(name="var")
    o["var2"] = o["var"] ** 2
    f_prob = f_prob.to_dataset(name="var")
    f_prob["var2"] = f_prob["var"] ** 2
    category_edges = f_prob.quantile(
        q=[0.2, 0.4, 0.6, 0.8], dim=["time", "member"]
    ).rename({"quantile": "category_edge"})
    actual = rps(
        o,
        f_prob,
        dim="time",
        fair=fair_bool,
        category_edges=category_edges,
    )
    assert set(["lon", "lat"]) == set(actual.dims)
    assert "category_edge" not in actual.dims


@pytest.mark.parametrize("fair_bool", [True, False])
def test_rps_category_edges_tuple(o, f_prob, fair_bool):
    """Test rps with category_edges as tuple of xrDataArray for forecast and observations edges separately."""
    o_edges = o.quantile(q=[0.2, 0.4, 0.6, 0.8], dim="time").rename(
        {"quantile": "category_edge"}
    )
    f_edges = f_prob.quantile(q=[0.2, 0.4, 0.6, 0.8], dim=["time", "member"]).rename(
        {"quantile": "category_edge"}
    )
    actual = rps(
        o,
        f_prob,
        dim="time",
        fair=fair_bool,
        category_edges=(o_edges, f_edges),
    )
    assert set(["lon", "lat"]) == set(actual.dims)
    assert "category_edge" not in actual.dims


@pytest.mark.parametrize("input_distributions", ["c", "p"])
def test_rps_category_edges_None(o, f_prob, input_distributions):
    """Test rps with category_edges as None expecting o and f_prob are already CDFs."""
    e = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_dim = "category"
    edges = xr.DataArray(e, dims=bin_dim, coords={bin_dim: e})
    o_c = o < edges  # CDF
    f_c = (f_prob < edges).mean("member")  # CDF
    actual = rps(
        o_c,
        f_c,
        dim="time",
        fair=False,
        category_edges=None,
        input_distributions=input_distributions,
    )
    assert set(["lon", "lat"]) == set(actual.dims)
    assert "quantile" not in actual.dims
    assert "member" not in actual.dims


@pytest.mark.parametrize(
    "category_edges",
    [
        xr.DataArray(
            [0.2, 0.4, 0.6, 0.8],
            dims="category_edge",
            coords={"category_edge": [0.2, 0.4, 0.6, 0.8]},
        ),
        np.array([0.2, 0.4, 0.6, 0.8]),
    ],
    ids=["edge xr", "edge np"],
)
@pytest.mark.parametrize("fair_bool", [True, False], ids=["fair=True", "fair=False"])
def test_rps_keeps_masked(o, f_prob, fair_bool, category_edges):
    """Test rps keeps NaNs."""
    o = o.where(o.lat > 1)
    f_prob = f_prob.where(f_prob.lat > 1)
    actual = rps(o, f_prob, dim="time", category_edges=category_edges)
    assert set(["lon", "lat"]) == set(actual.dims)
    assert actual.isel(lat=[0, 1]).isnull().all()
    assert actual.isel(lat=slice(2, None)).notnull().all()
    # test forecasts_category_edge no repeats
    assert (
        "[-np.inf, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, np.inf]"
        in actual.coords["forecasts_category_edge"].values
    )
    # one more category internally used than category_edges provided
    assert len(category_edges) + 1 == str(
        actual.coords["forecasts_category_edge"].values
    ).count("[")


@pytest.mark.parametrize("fair_bool", [True, False], ids=["bool=fair", "fair=False"])
def test_rps_new_identical_old_xhistogram(o, f_prob, fair_bool):
    """Test that new rps algorithm is identical to old algorithm with xhistogram.
    Makes a difference whether full range of f_prob is covered or not."""
    category_edges_np = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    category_edges_xr = xr.DataArray(
        category_edges_np,
        dims="category_edge",
        coords={"category_edge": category_edges_np},
    )
    dim = "time"
    actual = rps(o, f_prob, dim=dim, category_edges=category_edges_xr)
    expected = rps_xhist(o, f_prob, dim=dim, category_edges=category_edges_np)
    drop = ["observations_category_edge", "forecasts_category_edge"]
    assert_allclose(
        actual.rename("histogram_category_edge").drop(drop), expected.drop(drop)
    )


def test_rps_last_edge_included(o, f_prob):
    """Test that last edges is included."""
    category_edges_np = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    o = xr.ones_like(o)
    f_prob = xr.ones_like(f_prob)
    res_actual = rps(o, f_prob, dim="time", category_edges=category_edges_np)
    assert (res_actual == 0).all()


@pytest.mark.parametrize(
    "observation,forecast",
    [
        (
            lazy_fixture("observation_1d_long"),
            lazy_fixture("forecast_1d_long"),
        ),
        (
            lazy_fixture("observation_3d"),
            lazy_fixture("forecast_3d"),
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
    assert area < 0.65
    assert area > 0.35


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
    fp = forecast_1d_long.clip(0, 1)  # prob
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
    fp = forecast_1d_long.clip(0, 1)  # prob
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
