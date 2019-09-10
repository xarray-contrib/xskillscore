# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import numpy as np
import xarray as xr

from xskillscore import mae, mse, pearson_r, pearson_r_p_value, rmse

from . import parameterized, randn, requires_dask

DETERMINISTIC_METRICS = [rmse, pearson_r, mae, mse, pearson_r_p_value]


class Generate:
    """
    Generate random ds, control to be benckmarked.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_ds(self, nmember, nx, ny):

        # ds
        self.ds = xr.Dataset()
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
        self.ds['tos'] = xr.DataArray(
            randn((self.nmember, self.nx, self.ny), frac_nan=frac_nan),
            coords={'member': members, 'lon': lons, 'lat': lats},
            dims=('member', 'lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )
        self.ds['sos'] = xr.DataArray(
            randn((self.nmember, self.nx, self.ny), frac_nan=frac_nan),
            coords={'member': members, 'lon': lons, 'lat': lats},
            dims=('member', 'lon', 'lat'),
            name='sos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )

        self.ds.attrs = {'history': 'created for xarray benchmarking'}


class Compute_small(Generate):
    """
    A benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(3, 90, 45)  # 4 degree grid

    @parameterized('metric', DETERMINISTIC_METRICS)
    def time_xskillscore_metric_small(self, metric):
        """Take time for xskillscore.metric."""
        dim = 'member'
        metric(self.ds['tos'], self.ds['sos'], dim=dim)

    @parameterized('metric', DETERMINISTIC_METRICS)
    def peakmem_xskillscore_metric_small(self, metric):
        dim = 'member'
        """Take memory peak for xskillscore.metric."""
        metric(self.ds['tos'], self.ds['sos'], dim=dim)


class Compute_large(Generate):
    """
    A benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(5, 1000, 1000)

    @parameterized('metric', DETERMINISTIC_METRICS)
    def time_xskillscore_metric_large(self, metric):
        """Take time for xskillscore.metric."""
        dim = 'member'
        metric(self.ds['tos'], self.ds['sos'], dim=dim).compute()

    @parameterized('metric', DETERMINISTIC_METRICS)
    def peakmem_xskillscore_metric_large(self, metric):
        dim = 'member'
        """Take memory peak for xskillscore.metric."""
        metric(self.ds['tos'], self.ds['sos'], dim=dim).compute()


class Compute_large_dask(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays with dask."""

    def setup(self, *args, **kwargs):
        requires_dask()
        self.make_ds(5, 1000, 1000)
        self.ds = self.ds.chunk({'lon': 500, 'lat': 500})

    @parameterized('metric', DETERMINISTIC_METRICS)
    def time_xskillscore_metric_large_dask(self, metric):
        """Take time for xskillscore.metric."""
        dim = 'member'
        metric(self.ds['tos'], self.ds['sos'], dim=dim).compute()

    @parameterized('metric', DETERMINISTIC_METRICS)
    def peakmem_xskillscore_metric_large_dask(self, metric):
        dim = 'member'
        """Take memory peak for xskillscore.metric."""
        metric(self.ds['tos'], self.ds['sos'], dim=dim).compute()
