# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import os
import shutil

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
from xskillscore.xr import xr_mae, xr_mse, xr_pearson_r, xr_rmse, xr_spearman_r

from . import Generate, parameterized, requires_dask

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
