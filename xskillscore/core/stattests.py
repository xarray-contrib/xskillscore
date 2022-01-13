from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests
from typing_extensions import Literal

from .types import XArray

MULTIPLE_TESTS = [
    "bonferroni",
    "sidak",
    "holm-sidak",
    "hs",
    "holm",
    "simes-hochberg",
    "hommel",
    "fdr_bh",
    "fdr_by",
    "fdr_tsbh",
    "fdr_tsbky",
]


def multipletests(
    p: XArray,
    alpha: float = 0.05,
    method: Literal[MULTIPLE_TESTS] = None,
    keep_attrs=True,
    return_results: Literal[
        "pvals_corrected", "all_as_results_dim", "all_as_tuple"
    ] = "all_as_results_dim",
    **multipletests_kwargs,
) -> Tuple[XArray, XArray]:
    """Apply statsmodels.stats.multitest.multipletests for multi-dimensional
    xr.DataArray, xr.Datasets.

    Args:
        p (xr.DataArray, xr.Dataset): uncorrected p-values.
        alpha (optional float): FWER, family-wise error rate. Defaults to 0.05.
        method (str): Method used for testing and adjustment of pvalues. Can be
            either the full name or initial letters.  Available methods are:
            - "bonferroni" : one-step correction
            - "sidak" : one-step correction
            - "holm-sidak" : step down method using Sidak adjustments (alias "hs")
            - "holm" : step-down method using Bonferroni adjustments
            - "simes-hochberg" : step-up method (independent)
            - "hommel" : closed method based on Simes tests (non-negative)
            - "fdr_bh" : Benjamini/Hochberg (non-negative)
            - "fdr_by" : Benjamini/Yekutieli (negative)
            - "fdr_tsbh" : two stage fdr correction (non-negative)
            - "fdr_tsbky" : two stage fdr correction (non-negative)
        **multipletests_kwargs (optional dict): is_sorted, returnsorted
           see statsmodels.stats.multitest.multitest
        keep_attrs (bool):
        return_results (str, default='area')
            Specify how return is structed:

                - 'pvals_corrected': return only the corrected p values

                - 'all_as_tuple': return (reject, pvals_corrected, alphacSidak, alphacBonf) as tuple

                - 'all_as_results_dim': return (reject, pvals_corrected, alphacSidak, alphacBonf)
                  concatenated into new ``results`` dimension


    Returns:
        reject (xr.DataArray, xr.Dataset): true for hypothesis that can be rejected for given
            alpha
        pvals_corrected (xr.DataArray, xr.Dataset): p-values corrected for multiple tests
        alphacSidak (xr.DataArray, xr.Dataset): corrected alpha for Sidak method
        alphacBonf (xr.DataArray, xr.Dataset): corrected alpha for Bonferroni method

    Example:
        >>> reject, xpvals_corrected = xs.multipletests(p, method='fdr_bh')
    """

    if method is None:
        raise ValueError(
            f"Please indicate a method using the 'method=...' keyword. "
            f"Select from {MULTIPLE_TESTS}"
        )
    elif method not in MULTIPLE_TESTS:
        raise ValueError(
            f"Your method '{method}' is not in the accepted methods: {MULTIPLE_TESTS}"
        )

    allowed_return_results = ["all_as_tuple", " pvals_corrected", " all_as_results_dim"]
    if return_results not in allowed_return_results:
        raise NotImplementedError(
            f"expect `return_results` from {allowed_return_results}, "
            f"found {return_results}"
        )

    ret = xr.apply_ufunc(
        statsmodels_multipletests,
        p.stack(s=p.dims),
        input_core_dims=[[]],
        vectorize=True,
        output_core_dims=[[]] * 4,
        output_dtypes=[bool, float, float, float],
        kwargs=dict(method=method, alpha=alpha, **multipletests_kwargs),
        dask="parallelized",
        keep_attrs=keep_attrs,
    )

    ret = tuple(r.unstack("s").transpose(*p.dims, ...) for r in ret)

    def _add_kwargs_as_coords(ret):
        ret.coords["multipletests_method"] = method
        ret.coords["multipletests_alpha"] = alpha
        return ret

    ret = tuple(_add_kwargs_as_coords(r) for r in ret)

    returns = ["reject", "pvals_corrected", "alphacSidak", "alphacBonf"]

    if return_results == "all_as_results_dim":
        return xr.concat(ret, "result").assign_coords(result=returns)
    elif return_results == "all_as_tuple":
        for i, r in enumerate(ret):
            r.coords["result"] = returns[i]
        return ret
    elif return_results == "pvals_corrected":
        return ret[1].assign_coords(result="pvals_corrected")
