# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import numpy as np
import pandas as pd
import xarray as xr

from xskillscore import mse as xs_mse, pearson_r as xs_pearson_r

from . import parameterized, randn, requires_dask


def xr_mse(a, b, dim):
    """mse implementation using xarray only."""
    return ((a - b) ** 2).mean(dim)


def covariance_gufunc(x, y):
    return (
        (x - x.mean(axis=-1, keepdims=True)) * (y - y.mean(axis=-1, keepdims=True))
    ).mean(axis=-1)


def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


def xr_pearson_r(x, y, dim):
    """pearson_r implementation using xarray and minimal numpy only."""
    return xr.apply_ufunc(
        pearson_correlation_gufunc,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
    )


METRICS = [xs_mse, xr_mse, xs_pearson_r, xr_pearson_r]

large_lon_lat = 2000
large_lon_lat_chunksize = large_lon_lat // 4
ntime = 4


class Generate:
    """
    Generate random ds to be benckmarked.
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

        times = pd.date_range(start='1/1/2000', periods=ntime, freq='D',)

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
            randn((self.ntime, self.nx, self.ny), frac_nan=frac_nan),
            coords={'time': times, 'lon': lons, 'lat': lats},
            dims=('time', 'lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )
        self.ds['sos'] = xr.DataArray(
            randn((self.ntime, self.nx, self.ny), frac_nan=frac_nan),
            coords={'time': times, 'lon': lons, 'lat': lats},
            dims=('time', 'lon', 'lat'),
            name='sos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )
        self.ds.attrs = {'history': 'created for xarray benchmarking'}

        # set nans for land sea mask
        self.ds = self.ds.where(
            (abs(self.ds.lat) > 20) | (self.ds.lat < 100) | (self.ds.lat > 160)
        )


class Compute_small(Generate):
    """
    A benchmark xskillscore.metric for small xr.DataArrays"""

    def setup(self, *args, **kwargs):
        self.make_ds(ntime, 90, 45)  # 4 degree grid

    @parameterized('metric', METRICS)
    def time_xskillscore_metric_small(self, metric):
        """Take time for xskillscore.metric."""
        dim = 'time'
        metric(self.ds['tos'], self.ds['sos'], dim=dim)

    @parameterized('metric', METRICS)
    def peakmem_xskillscore_metric_small(self, metric):
        dim = 'time'
        """Take memory peak for xskillscore.metric."""
        metric(self.ds['tos'], self.ds['sos'], dim=dim)


class Compute_large(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays"""

    def setup_cache(self, *args, **kwargs):
        self.make_ds(ntime, large_lon_lat, large_lon_lat)
        self.ds.to_netcdf('large.nc')

    def setup(self, *args, **kwargs):
        self.ds = xr.open_dataset('large.nc')

    @parameterized('metric', METRICS)
    def time_xskillscore_metric_large(self, metric):
        """Take time for xskillscore.metric."""
        dim = 'time'
        metric(self.ds['tos'], self.ds['sos'], dim=dim)

    @parameterized('metric', METRICS)
    def peakmem_xskillscore_metric_large(self, metric):
        dim = 'time'
        """Take memory peak for xskillscore.metric."""
        metric(self.ds['tos'], self.ds['sos'], dim=dim)


class Compute_large_dask(Generate):
    """
    A benchmark xskillscore.metric for large xr.DataArrays with dask."""

    def setup_cache(self, *args, **kwargs):
        requires_dask()
        self.make_ds(ntime, large_lon_lat, large_lon_lat)
        self.ds.to_netcdf('large.nc')

    def setup(self, *args, **kwargs):
        self.ds = xr.open_dataset('large.nc', chunks={'lon': large_lon_lat_chunksize})

    @parameterized('metric', METRICS)
    def time_xskillscore_metric_large_dask(self, metric):
        """Take time for xskillscore.metric."""
        dim = 'time'
        metric(self.ds['tos'], self.ds['sos'], dim=dim).compute()

    @parameterized('metric', METRICS)
    def peakmem_xskillscore_metric_large_dask(self, metric):
        dim = 'time'
        """Take memory peak for xskillscore.metric."""
        metric(self.ds['tos'], self.ds['sos'], dim=dim).compute()
