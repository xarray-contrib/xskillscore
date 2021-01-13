# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import numpy as np
import xarray as xr
from scipy.stats import norm

from xskillscore import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    threshold_brier_score,
)

from . import Generate, parameterized, requires_dask

PROBABILISTIC_METRICS = [
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    brier_score,
    threshold_brier_score,
]

including_crps_quadrature = False

large_lon_lat = 2000
large_lon_lat_chunksize = large_lon_lat // 2
nmember = 4


class Compute_small(Generate):
    """
    A  benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(nmember, 1, 1)  # no grid
        self.ds["tos"] = self.ds["tos"].isel(member=0, drop=True)

    @parameterized("metric", PROBABILISTIC_METRICS)
    def time_xskillscore_metric(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.ds["tos"], mu, sig).compute()
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.ds["tos"], cdf_or_dist, xmin, xmax, tol).compute()
        elif metric is crps_ensemble:
            metric(self.ds["tos"], self.ds["sos"]).compute()
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.ds["tos"], self.ds["sos"], threshold).compute()
        elif metric is brier_score:
            metric(
                self.ds["tos"] > 0.5, (self.ds["sos"] > 0.5).mean("member")
            ).compute()

    @parameterized("metric", PROBABILISTIC_METRICS)
    def peakmem_xskillscore_metric(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.ds["tos"], mu, sig).compute()
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.ds["tos"], cdf_or_dist, xmin, xmax, tol).compute()
        elif metric is crps_ensemble:
            metric(self.ds["tos"], self.ds["sos"]).compute()
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.ds["tos"], self.ds["sos"], threshold).compute()
        elif metric is brier_score:
            metric(
                self.ds["tos"] > 0.5, (self.ds["sos"] > 0.5).mean("member")
            ).compute()


class Compute_large(Compute_small):
    """
    A benchmark xskillscore.metric for large xr.DataArrays."""

    def setup(self, *args, **kwargs):
        self.make_ds(nmember, large_lon_lat, large_lon_lat)
        self.ds["tos"] = self.ds["tos"].isel(member=0, drop=True)


class Compute_large_dask(Compute_small):
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
        self.ds["tos"] = self.ds["tos"].isel(member=0, drop=True)
