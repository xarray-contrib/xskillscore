import numpy as np
import pytest
from dask import is_dask_collection
from xarray.testing import assert_equal

import xskillscore as xs
from xskillscore import multipletests


@pytest.fixture
def r_p(a, b):
    return xs.pearson_r_p_value(a, b, "time")


@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("input", ["Dataset", "multidim Dataset", "DataArray"])
def test_multipletests_inputs(r_p, input, chunk):
    """Test multipletests with xr inputs and chunked."""
    method = "fdr_bh"
    alpha = 0.05
    if "Dataset" in input:
        name = "var"
        r_p = r_p.to_dataset(name=name)
        if input == "multidim Dataset":
            r_p["var2"] = r_p["var"] * 2
    if chunk:
        r_p = r_p.chunk()
    ret = multipletests(
        r_p, method=method, alpha=alpha, return_results="all_as_result_dim"
    )
    # check dask collection preserved
    assert is_dask_collection(ret) if chunk else not is_dask_collection(ret)
    # check coords added
    assert ret.coords["multipletests_method"] == method
    assert ret.coords["multipletests_alpha"] == alpha


def test_multipletests_alpha(r_p):
    """Test that larger alpha leads to more rejected in multipletests."""
    method = "fdr_bh"
    reject = multipletests(r_p, alpha=0.05, method=method).sel(result="reject")
    reject_larger_alpha = multipletests(r_p, alpha=0.5, method=method).sel(
        result="reject"
    )
    # check more reject
    assert reject_larger_alpha.sum() > reject.sum()


def test_multipletests_method(r_p):
    """Test that multipletests is sensitive to method."""
    ret_fdr_bh = multipletests(r_p, method="fdr_bh")
    ret_hs = multipletests(r_p, method="hs")
    assert not ret_hs.equals(ret_fdr_bh)


def test_multipletests_return_results(r_p):
    """Test that multipletests is sensitive to return_results."""
    for return_results in ["pvals_corrected", "all_as_tuple", "all_as_result_dim"]:
        ret = multipletests(r_p, method="fdr_bh", return_results=return_results)
        if return_results == "pvals_corrected":
            assert isinstance(ret, type(r_p))
            assert "result" not in ret.dims
            assert ret.coords["result"] == "pvals_corrected"
        elif return_results == "all_as_tuple":
            assert isinstance(ret, tuple)
            assert all(isinstance(r, type(r_p)) for r in ret)
            print(ret)
            assert ret[1].coords["result"] == "pvals_corrected"
        elif return_results == "all_as_result_dim":
            assert isinstance(ret, type(r_p))
            assert "result" in ret.dims
            assert "pvals_corrected" in ret.result
