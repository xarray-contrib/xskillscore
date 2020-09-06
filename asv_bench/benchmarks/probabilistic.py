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

from . import parameterized, randn, requires_dask

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


class Generate:
    """
    Generate random fct and obs to be benckmarked.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_ds(self, nmember, nx, ny):

        # ds
        self.obs = xr.Dataset()
        self.fct = xr.Dataset()
        self.nmember = nmember
        self.nx = nx  # 4 deg
        self.ny = ny  # 4 deg

        frac_nan = 0.0

        members = np.arange(1, 1 + self.nmember)

        lons = xr.DataArray(
            np.linspace(0, 360, self.nx),
            dims=('lon',),
            attrs={'units': 'degrees east', 'long_name': 'longitude'},
        )
        lats = xr.DataArray(
            np.linspace(-90, 90, self.ny),
            dims=('lat',),
            attrs={'units': 'degrees north', 'long_name': 'latitude'},
        )
        self.fct['tos'] = xr.DataArray(
            randn((self.nmember, self.nx, self.ny), frac_nan=frac_nan),
            coords={'member': members, 'lon': lons, 'lat': lats},
            dims=('member', 'lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )
        self.obs['tos'] = xr.DataArray(
            randn((self.nx, self.ny), frac_nan=frac_nan),
            coords={'lon': lons, 'lat': lats},
            dims=('lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )

        self.fct.attrs = {'history': 'created for xarray benchmarking'}
        self.obs.attrs = {'history': 'created for xarray benchmarking'}

        # set nans for land sea mask
        self.fct = self.fct.where(
            (abs(self.fct.lat) > 20) | (self.fct.lat < 100) | (self.fct.lat > 160)
        )
        self.obs = self.obs.where(
            (abs(self.obs.lat) > 20) | (self.obs.lat < 100) | (self.obs.lat > 160)
        )


class Compute_small(Generate):
    """
    A  benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(nmember, 90, 45)  # 4 degree grid

    @parameterized('metric', PROBABILISTIC_METRICS)
    def time_xskillscore_probabilistic_small(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.obs['tos'], mu, sig)
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.obs['tos'], cdf_or_dist, xmin, xmax, tol)
        elif metric is crps_ensemble:
            metric(self.obs['tos'], self.fct['tos'])
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.obs['tos'], self.fct['tos'], threshold)
        elif metric is brier_score:
            metric(self.obs['tos'] > 0.5, (self.fct['tos'] > 0.5).mean('member'))

    @parameterized('metric', PROBABILISTIC_METRICS)
    def peakmem_xskillscore_probabilistic_small(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.obs['tos'], mu, sig)
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.obs['tos'], cdf_or_dist, xmin, xmax, tol)
        elif metric is crps_ensemble:
            metric(self.obs['tos'], self.fct['tos'])
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.obs['tos'], self.fct['tos'], threshold)
        elif metric is brier_score:
            metric(self.obs['tos'] > 0.5, (self.fct['tos'] > 0.5).mean('member'))


class Compute_large(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays."""

    def setup(self, *args, **kwargs):
        self.make_ds(nmember, large_lon_lat, large_lon_lat)

    @parameterized('metric', PROBABILISTIC_METRICS)
    def time_xskillscore_probabilistic_large(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.obs['tos'], mu, sig)
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.obs['tos'], cdf_or_dist, xmin, xmax, tol)
        elif metric is crps_ensemble:
            metric(self.obs['tos'], self.fct['tos'])
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.obs['tos'], self.fct['tos'], threshold)
        elif metric is brier_score:
            metric(self.obs['tos'] > 0.5, (self.fct['tos'] > 0.5).mean('member'))

    @parameterized('metric', PROBABILISTIC_METRICS)
    def peakmem_xskillscore_probabilistic_large(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.obs['tos'], mu, sig)
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.obs['tos'], cdf_or_dist, xmin, xmax, tol)
        elif metric is crps_ensemble:
            metric(self.obs['tos'], self.fct['tos'])
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.obs['tos'], self.fct['tos'], threshold)
        elif metric is brier_score:
            metric(self.obs['tos'] > 0.5, (self.fct['tos'] > 0.5).mean('member'))


class Compute_large_dask(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays with dask."""

    def setup(self, *args, **kwargs):
        requires_dask()
        self.make_ds(nmember, large_lon_lat, large_lon_lat)
        self.obs = self.obs.chunk(
            {'lon': large_lon_lat_chunksize, 'lat': large_lon_lat_chunksize}
        )
        self.fct = self.fct.chunk(
            {'lon': large_lon_lat_chunksize, 'lat': large_lon_lat_chunksize}
        )

    @parameterized('metric', PROBABILISTIC_METRICS)
    def time_xskillscore_probabilistic_large_dask(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.obs['tos'], mu, sig).compute()
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.obs['tos'], cdf_or_dist, xmin, xmax, tol).compute()
        elif metric is crps_ensemble:
            metric(self.obs['tos'], self.fct['tos']).compute()
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.obs['tos'], self.fct['tos'], threshold).compute()
        elif metric is brier_score:
            metric(
                self.obs['tos'] > 0.5, (self.fct['tos'] > 0.5).mean('member')
            ).compute()

    @parameterized('metric', PROBABILISTIC_METRICS)
    def peakmem_xskillscore_probabilistic_large_dask(self, metric):
        """Take time for xskillscore.metric."""
        if metric is crps_gaussian:
            mu = 0.5
            sig = 0.2
            metric(self.obs['tos'], mu, sig).compute()
        elif metric is crps_quadrature:
            if not including_crps_quadrature:
                pass
            else:
                xmin, xmax, tol = -10, 10, 1e-6
                cdf_or_dist = norm
                metric(self.obs['tos'], cdf_or_dist, xmin, xmax, tol).compute()
        elif metric is crps_ensemble:
            metric(self.obs['tos'], self.fct['tos']).compute()
        elif metric is threshold_brier_score:
            threshold = 0.5
            metric(self.obs['tos'], self.fct['tos'], threshold).compute()
        elif metric is brier_score:
            metric(
                self.obs['tos'] > 0.5, (self.fct['tos'] > 0.5).mean('member')
            ).compute()
