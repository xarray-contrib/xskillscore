"""These metrics in xskillscore.xr.deterministic, entirely written in xarray functions,
are identical to the well documented metrics in xskillscore.core.deterministic, which
are based on numpy functions applied to xarray objects by xarray.apply_ufunc. As the xr
metrics are only faster for small data, their use is not encouraged, as the numpy-based
metrics are 20-40% faster on large data."""
import bottleneck as bn
import numpy as np
import xarray as xr


def xr_mse(a, b, dim=None, skipna=True, weights=None):
    res = (a - b) ** 2
    if weights is not None:
        res = res.weighted(weights)
    res = res.mean(dim=dim, skipna=skipna)
    return res


def xr_mae(a, b, dim=None, skipna=True, weights=None):
    res = np.abs(a - b)
    if weights is not None:
        res = res.weighted(weights)
    res = res.mean(dim=dim, skipna=skipna)
    return res


def xr_me(a, b, dim=None, skipna=True, weights=None):
    res = a - b
    if weights is not None:
        res = res.weighted(weights)
    res = res.mean(dim=dim, skipna=skipna)
    return res


def xr_rmse(a, b, dim=None, skipna=True, weights=None):
    res = (a - b) ** 2
    if weights is not None:
        res = res.weighted(weights)
    res = res.mean(dim=dim, skipna=skipna)
    res = np.sqrt(res)
    return res


def xr_pearson_r(a, b, dim=None, **kwargs):
    return xr.corr(a, b, dim)


def _rankdata(o, dim):
    if isinstance(dim, str):
        dim = [dim]
    elif dim == None:
        dim = list(o.dims)
    if len(dim) == 1:
        return xr.apply_ufunc(
            bn.nanrankdata,
            o,
            input_core_dims=[[]],
            kwargs={"axis": o.get_axis_num(dim[0])},
            dask="allowed",
        )
    elif len(dim) > 1:
        # stack rank unstack
        return xr.apply_ufunc(
            bn.nanrankdata,
            o.stack(ndim=dim),
            input_core_dims=[[]],
            kwargs={"axis": -1},
            dask="allowed",
        ).unstack("ndim")


def xr_spearman_r(a, b, dim=None, **kwargs):
    return xr.corr(_rankdata(a, dim), _rankdata(b, dim), dim)
