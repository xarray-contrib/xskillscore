import numpy as np
import pytest
import xarray as xr
from dask import is_dask_collection

from xskillscore.core.resampling import (
    _gen_idx,
    resample_iterations,
    resample_iterations_idx,
)

resample_iterations_funcs = [resample_iterations, resample_iterations_idx]
ITERATIONS = 2


def assert_dim_coords(a, b):
    """check dim and coord entries and order."""
    # check dim and coord size and entries
    for d in a.dims:
        assert a[d].size == b[d].size, print(a[d], "!=", b[d])
    for c in a.coords:
        assert (a[c] == b[c]).all(), print(a[c], "!=", b[c])
    # check dim and coords entries and order
    assert list(a.dims) == list(b.dims), print(a.dims, "!=", b.dims)
    assert list(a.coords) == list(b.coords), print(a.coords, "!=", b.coords)


def test_resampling_identical_dim(f_prob):
    """check that resampling functions have the same ordering of dims and coords."""
    da = f_prob
    r1 = resample_iterations(da, iterations=ITERATIONS)
    r2 = resample_iterations_idx(da, iterations=ITERATIONS)
    assert_dim_coords(r1, r2)


def test_resampling_roughly_identical_mean(f_prob):
    """check that resampling functions result in allclose iteration mean."""
    da = f_prob.isel(lon=0, lat=0, drop=True)
    iterations = 1000
    r1 = resample_iterations(da, iterations=iterations, replace=True)
    r2 = resample_iterations_idx(da, iterations=iterations, replace=True)
    xr.testing.assert_allclose(
        r1.mean("iteration"), r2.mean("iteration"), atol=0.05, rtol=0.05
    )


@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("func", resample_iterations_funcs)
def test_resampling_same_dim_coord_order_as_input(func, f_prob, iterations):
    """check whether resampling function maintain dim and coord order and add iteration
    dimension to the end."""
    da = f_prob
    r1 = resample_iterations(da, iterations=iterations)
    # same as without having iteration
    assert_dim_coords(r1.isel(iteration=0, drop=True), da)
    # same as with having iteration added in the end as in expand_dims
    assert_dim_coords(
        r1,
        da.expand_dims("iteration")
        .transpose(..., "iteration")
        .isel(iteration=[0] * iterations)
        .assign_coords(iteration=range(iterations)),
    )


@pytest.mark.skip(reason="fails, dont know why and slow")
@pytest.mark.parametrize("func", resample_iterations_funcs)
def test_resampling_replace_True_larger_std_than_replace_False(f_prob, func):
    """check that resampling functions result in allclose iteration mean."""
    da = f_prob.isel(lon=0, lat=0, member=0, drop=True)
    iterations = 10000
    # I would hope to get the same random seed for both but probably thats unrealistic
    np.random.seed(42)
    rreplace = func(da, dim="time", iterations=iterations, replace=True)
    rnoreplace = func(da, dim="time", iterations=iterations, replace=False)
    assert (rreplace.std("iteration") > rnoreplace.std("iteration")).all(), print(
        rreplace.std("iteration"),
        "\nnot greater\n",
        rnoreplace.std("iteration"),
        "\nbut should\n",
        rreplace.std("iteration") > rnoreplace.std("iteration"),
    )


@pytest.mark.parametrize("replace", [True, False])
def test_gen_idx_replace(f_prob, replace):
    """check whether replace=True creates duplicate idx."""

    def find_duplicate(da, dim="iteration"):
        for i in da[dim]:
            if da.sel({dim: i}).to_index().duplicated().any():
                return True
        return False

    actual = _gen_idx(
        f_prob, "member", 10, f_prob["member"].size, replace, f_prob["member"]
    )
    if replace:  # find identical values
        assert find_duplicate(actual)
    else:  # find no identicals
        assert not find_duplicate(actual)


@pytest.mark.skip(reason="this is a bug, test fails and should be resolved.")
def test_resample_iterations_dix_no_squeeze(f_prob):
    """Test _resample_iteration_idx with singular dimension.

    Currently this fails for dimensions with just a single index as we use `squeeze` in
    the code and not using squeeze doesnt maintain functionality. This means that
    ``_resample_iteration_idx`` should not be called on singleton dimension inputs
    (which is not critical and can be circumvented when using squeeze before.).
    """
    da = f_prob.expand_dims("test_dim")
    actual = resample_iterations_idx(da, iterations=ITERATIONS)
    assert "test_dim" in actual.dims


@pytest.mark.parametrize("func", resample_iterations_funcs)
def test_resample_replace_False_once_same_mean(f_prob, func):
    """resample replace=False on one dimension once gets same mean as origin."""
    da = f_prob.isel(lon=0, lat=0, member=0, drop=True)
    once = func(da, dim="time", replace=False, iterations=1).squeeze(drop=True)
    xr.testing.assert_allclose(once.mean("time"), da.mean("time"))


@pytest.mark.parametrize("func", resample_iterations_funcs)
@pytest.mark.parametrize("dim_max", [None, 5])
def test_resample_dim_max(f_prob, func, dim_max):
    """resample dim_max=x returns only x items in dim."""
    da = f_prob.isel(lon=0, lat=0, member=0, drop=True)
    dim = "time"
    actual = func(da, dim=dim, dim_max=dim_max, iterations=ITERATIONS)
    if dim_max:
        assert actual[dim].size == dim_max
    else:
        assert actual[dim].size == da[dim].size


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "multidim Dataset", "DataArray"])
@pytest.mark.parametrize("func", resample_iterations_funcs)
def test_resample_inputs(a_1d, func, input, chunk, replace):
    """Test sign_test with xr inputs and chunked."""
    if "Dataset" in input:
        name = "var"
        a_1d = a_1d.to_dataset(name=name)
        if input == "multidim Dataset":
            a_1d["var2"] = a_1d["var"] * 2
    if chunk:
        a_1d = a_1d.chunk()
    actual = func(a_1d, ITERATIONS, dim="time")
    # check dask collection preserved
    assert is_dask_collection(actual) if chunk else not is_dask_collection(actual)
    # input type preserved
    assert type(actual) == type(a_1d)
