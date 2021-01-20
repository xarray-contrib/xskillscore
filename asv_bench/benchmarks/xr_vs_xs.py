# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import os
import shutil

import bottleneck as bn
import numpy as np
import pandas as pd
import xarray as xr

from xskillscore import (
    mae as xs_mae,
    mse as xs_mse,
    pearson_r as xs_pearson_r,
    rmse as xs_rmse,
    spearman_r as xs_spearman_r,
)

from . import Generate, parameterized, requires_dask

# These metrics in xskillscore.xr.deterministic, entirely written in xarray functions,
# are identical to the well documented metrics in xskillscore.core.deterministic, which
# are based on numpy functions applied to xarray objects by xarray.apply_ufunc. As the
# xr metrics are only faster for small data, their use is not encouraged, as the
# numpy-based metrics are 20-40% faster on large data."""


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
    elif dim is None:
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


METRICS = [
    xs_mse,
    xr_mse,
    xs_rmse,
    xr_rmse,
    xs_mae,
    xr_mae,
    xs_pearson_r,
    xr_pearson_r,
    xs_spearman_r,
    xr_spearman_r,
]

DIMS = ["member", ["lon", "lat"]]

large_lon_lat = 2000
large_lon_lat_chunksize = large_lon_lat // 4
ntime = 24


class Compute_small(Generate):
    """
    A benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(ntime, 1, 1)  # no grid

    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def time_xskillscore_metric(self, metric, dim):
        """Take time for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim).compute()

    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def peakmem_xskillscore_metric(self, metric, dim):
        """Take memory peak for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim).compute()


class Compute_large(Compute_small):
    """
    A benchmark xskillscore.metric for large xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(ntime, large_lon_lat, large_lon_lat)


class Compute_large_dask(Compute_small):
    """
    A benchmark xskillscore.metric for large xr.DataArrays with dask."""

    def setup(self, *args, **kwargs):
        requires_dask()
        self.make_ds(
            ntime, large_lon_lat, large_lon_lat, chunks={"lon": large_lon_lat_chunksize}
        )
