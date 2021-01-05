# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


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

from . import parameterized, randn, requires_dask

METRICS = [
    xs_mse,
    xr_mse,
    xs_rmse,
    xr_rmse,
    xs_mae,
    xr_mae,
    xs_pearson_r,
    xr_spearman_r,
]

DIMS = ["time", ["lon", "lat"]]

large_lon_lat = 2000
large_lon_lat_chunksize = large_lon_lat // 4
ntime = 24


class Generate:
    """
    Generate random xr.Dataset ds to be benchmarked.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_ds(self, ntime, nx, ny):

        # ds
        self.ds = xr.Dataset()
        self.ntime = ntime
        self.nx = nx  # 4 deg
        self.ny = ny  # 4 deg

        frac_nan = 0.0

        times = pd.date_range(
            start="1/1/2000",
            periods=ntime,
            freq="D",
        )

        lons = xr.DataArray(
            np.linspace(0, 360, self.nx),
            dims=("lon",),
            attrs={"units": "degrees east", "long_name": "longitude"},
        )
        lats = xr.DataArray(
            np.linspace(-90, 90, self.ny),
            dims=("lat",),
            attrs={"units": "degrees north", "long_name": "latitude"},
        )
        self.ds["tos"] = xr.DataArray(
            randn((self.ntime, self.nx, self.ny), frac_nan=frac_nan),
            coords={"time": times, "lon": lons, "lat": lats},
            dims=("time", "lon", "lat"),
            name="tos",
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds["sos"] = xr.DataArray(
            randn((self.ntime, self.nx, self.ny), frac_nan=frac_nan),
            coords={"time": times, "lon": lons, "lat": lats},
            dims=("time", "lon", "lat"),
            name="sos",
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds.attrs = {"history": "created for xskillscore benchmarking"}

        # set nans for land sea mask
        self.ds = self.ds.where(
            (abs(self.ds.lat) > 20) | (self.ds.lat < 100) | (self.ds.lat > 160)
        )


class Compute_small(Generate):
    """
    A benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(ntime, 1, 1)  # no grid

    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def time_xskillscore_metric_small(self, metric, dim):
        """Take time for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim)

    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def peakmem_xskillscore_metric_small(self, metric, dim):
        """Take memory peak for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim)


class Compute_large(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays"""

    def setup_cache(self, *args, **kwargs):
        self.make_ds(ntime, large_lon_lat, large_lon_lat)
        self.ds.to_netcdf("large.nc")

    def setup(self, *args, **kwargs):
        self.ds = xr.open_dataset("large.nc")

    # can get inherited from Compute_small, remove
    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def time_xskillscore_metric_large(self, metric, dim):
        """Take time for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim)

    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def peakmem_xskillscore_metric_large(self, metric, dim):
        """Take memory peak for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim)


class Compute_large_dask(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays with dask."""

    # rewrite with zarr
    def setup_cache(self, *args, **kwargs):
        requires_dask()
        self.make_ds(ntime, large_lon_lat, large_lon_lat)
        self.ds.to_netcdf("large.nc")

    def setup(self, *args, **kwargs):
        self.ds = xr.open_dataset("large.nc", chunks={"lon": large_lon_lat_chunksize})

    # inherit
    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def time_xskillscore_metric_large_dask(self, metric, dim):
        """Take time for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim).compute()

    @parameterized(["metric", "dim"], (METRICS, DIMS))
    def peakmem_xskillscore_metric_large_dask(self, metric, dim):
        """Take memory peak for xskillscore.metric."""
        metric(self.ds["tos"], self.ds["sos"], dim=dim).compute()
