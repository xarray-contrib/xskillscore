# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import os

import numpy as np
import xarray as xr

from xskillscore import mae, mse, pearson_r, pearson_r_p_value, rmse

from . import Generate, parameterized, requires_dask

DETERMINISTIC_METRICS = [rmse, pearson_r, mae, mse, pearson_r_p_value]

large_lon_lat = 2000
large_lon_lat_chunksize = large_lon_lat // 4
nmember = 4


class Compute_small(Generate):
    """
    A benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(nmember, 1, 1)  # no grid

    @parameterized("metric", DETERMINISTIC_METRICS)
    def time_xskillscore_metric(self, metric):
        """Take time for xskillscore.metric."""
        dim = "member"
        metric(self.ds["tos"], self.ds["sos"], dim=dim).compute()

    @parameterized("metric", DETERMINISTIC_METRICS)
    def peakmem_xskillscore_metric(self, metric):
        dim = "member"
        """Take memory peak for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim).compute()


class Compute_large(Compute_small):
    """
    A benchmark xskillscore.metric for large xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(nmember, large_lon_lat, large_lon_lat)


class Compute_large_dask(Compute_large):
    """
    A benchmark xskillscore.metric for large xr.DataArrays with dask."""

    def setup(self, *args, **kwargs):
        requires_dask()
        self.make_ds(
            nmember,
            large_lon_lat,
            large_lon_lat,
            chunks={"lon": large_lon_lat_chunksize},
        )
