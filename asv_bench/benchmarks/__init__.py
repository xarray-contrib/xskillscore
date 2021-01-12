# https://github.com/pydata/xarray/blob/master/asv_bench/benchmarks/__init__.py

import numpy as np
import xarray as xr


def parameterized(names, params):
    def decorator(func):
        func.param_names = names
        func.params = params
        return func

    return decorator


def requires_dask():
    try:
        import dask  # noqa
    except ImportError:
        raise NotImplementedError


def randn(shape, frac_nan=None, chunks=None, seed=0):
    rng = np.random.RandomState(seed)
    if chunks is None:
        x = rng.standard_normal(shape)
    else:
        import dask.array as da

        rng = da.random.RandomState(seed)
        x = rng.standard_normal(shape, chunks=chunks)

    if frac_nan is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_nan))
        x.flat[inds] = np.nan

    return x


def randint(low, high=None, size=None, frac_minus=None, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randint(low, high, size)
    if frac_minus is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_minus))
        x.flat[inds] = -1

    return x


class Generate:
    """
    Generate random xr.Dataset ds to be benckmarked.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_ds(self, nmember, nx, ny, chunks=None):

        # ds
        self.ds = xr.Dataset()
        self.nmember = nmember
        self.nx = nx
        self.ny = ny

        frac_nan = 0.0

        members = np.arange(1, 1 + self.nmember)

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
            randn((self.nmember, self.nx, self.ny), frac_nan=frac_nan, chunks=chunks),
            coords={"member": members, "lon": lons, "lat": lats},
            dims=("member", "lon", "lat"),
            name="tos",
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds["sos"] = xr.DataArray(
            randn((self.nmember, self.nx, self.ny), frac_nan=frac_nan, chunks=chunks),
            coords={"member": members, "lon": lons, "lat": lats},
            dims=("member", "lon", "lat"),
            name="sos",
            attrs={"units": "foo units", "description": "a description"},
        )
        self.ds.attrs = {"history": "created for xskillscore benchmarking"}

        # set nans for land sea mask
        self.ds = self.ds.where(
            (abs(self.ds.lat) > 20) | (self.ds.lat < 100) | (self.ds.lat > 160)
        )
