from __future__ import annotations

from typing import Literal, Mapping, Optional, Tuple, Union

import xarray as xr
from statsmodels.stats.multitest import multipletests as statsmodels_multipletests

from .types import XArray


def multipletests(
    p: XArray,
    alpha: float = 0.05,
    method: Optional[
        Literal[
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
    ] = None,
    keep_attrs=True,
    return_results: Literal[
        "pvals_corrected", "all_as_result_dim", "all_as_tuple"
    ] = "all_as_result_dim",
    **multipletests_kwargs: Mapping,
) -> Union[XArray, Tuple[XArray, ...]]:
    """Apply statsmodels.stats.multitest.multipletests for controlling the false
    discovery rate for multiple hypothesis tests for multidimensional
    xr.DataArray and xr.Datasets.

    Parameters
    ----------
    p : xarray.Dataset or xarray.DataArray
        uncorrected p-values.
    alpha : float, optional
        family-wise error rate (FWER). Defaults to 0.05.
    method : str
        Method used for testing and adjustment of pvalues. Can be
        either the full name or initial letters. Available methods are:
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
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.
    return_results : str
        Specify how return is structed:

            - 'pvals_corrected': return only the corrected p values

            - 'all_as_tuple': return (reject, pvals_corrected, alphacSidak,
              alphacBonf) as tuple

            - 'all_as_result_dim': return (reject, pvals_corrected, alphacSidak,
              alphacBonf) concatenated into new ``result`` dimension. (Default)

    **multipletests_kwargs : dict, optional
        is_sorted, returnsorted, see statsmodels.stats.multitest.multitest

    Returns
    -------
    reject : xarray.Dataset or xarray.DataArray
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : xarray.Dataset or xarray.DataArray
        p-values corrected for multiple tests
    alphacSidak : xarray.Dataset or xarray.DataArray
        corrected alpha for Sidak method
    alphacBonf : xarray.Dataset or xarray.DataArray
        corrected alpha for Bonferroni method

    References
    ----------
    * Wilks, D. S. (2016). “The Stippling Shows Statistically Significant Grid Points”:
      How Research Results are Routinely Overstated and Overinterpreted, and What to Do
      about It. Bulletin of the American Meteorological Society, 97(12), 2263–2273.
      https://www.doi.org/10/f9mvth
    * Benjamini, Y., & Hochberg, Y. (1994). Controlling the False Discovery Rate:
      A Practical and Powerful Approach to Multiple Testing.
      Journal of the Royal Statistical Society: Series B (Methodological), 57(1), 13.
      https://www.doi.org/10.1111/j.2517-6161.1995.tb02031.x

    Examples
    --------
    >>> p = xr.DataArray(
    ...     np.random.normal(size=(3, 3)),
    ...     coords=[("x", np.arange(3)), ("y", np.arange(3))],
    ... )
    >>> result = xs.multipletests(p, alpha=0.1, method="fdr_bh", return_results="all_as_result_dim")
    >>> result
    <xarray.DataArray (result: 4, x: 3, y: 3)> Size: 288B
    array([[[ 0.        ,  1.        ,  0.        ],
            [ 0.        ,  1.        ,  1.        ],
            [ 0.        ,  0.        ,  1.        ]],
    <BLANKLINE>
           [[ 0.49671415, -0.1382643 ,  0.64768854],
            [ 1.        , -0.23415337, -0.23413696],
            [ 1.        ,  0.76743473, -0.46947439]],
    <BLANKLINE>
           [[ 0.1       ,  0.1       ,  0.1       ],
            [ 0.1       ,  0.1       ,  0.1       ],
            [ 0.1       ,  0.1       ,  0.1       ]],
    <BLANKLINE>
           [[ 0.1       ,  0.1       ,  0.1       ],
            [ 0.1       ,  0.1       ,  0.1       ],
            [ 0.1       ,  0.1       ,  0.1       ]]])
    Coordinates:
      * x                     (x) int64 24B 0 1 2
      * y                     (y) int64 24B 0 1 2
        multipletests_method  <U6 24B 'fdr_bh'
        multipletests_alpha   float64 8B 0.1
      * result                (result) <U15 240B 'reject' ... 'alphacBonf'
    """
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
    msg = "Alpha must be float between 0.0 and 1.0."
    if not isinstance(alpha, float):
        raise ValueError(msg)
    elif alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(msg)

    if method is None:
        raise ValueError(
            f"Please indicate a method using the 'method=...' keyword. Select from {MULTIPLE_TESTS}"
        )
    elif method not in MULTIPLE_TESTS:
        raise ValueError(f"Your method '{method}' is not in the accepted methods: {MULTIPLE_TESTS}")

    allowed_return_results = ["all_as_tuple", "pvals_corrected", "all_as_result_dim"]
    if return_results not in allowed_return_results:
        raise ValueError(
            f"Expected `return_results` from {allowed_return_results}, found {return_results}"
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

    def _add_kwargs_as_coords(r: XArray):
        r.coords["multipletests_method"] = method
        r.coords["multipletests_alpha"] = alpha
        return r

    ret = tuple(_add_kwargs_as_coords(r) for r in ret)

    returns = ["reject", "pvals_corrected", "alphacSidak", "alphacBonf"]

    if return_results == "all_as_result_dim":
        return xr.concat(ret, "result").assign_coords(result=returns)
    elif return_results == "all_as_tuple":
        for i, r in enumerate(ret):
            r.coords["result"] = returns[i]
        return ret
    else:
        return ret[1].assign_coords(result="pvals_corrected")
