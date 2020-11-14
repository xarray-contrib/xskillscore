import dask
import numpy as np
# import dask.array as da
import xarray as xr

CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}


def _gen_idx(forecast, dim, iterations, select_dim_items, replace, new_dim):
    """Generate indices to select from. replace decides whether resampling is with or
    without replacement."""
    if replace:
        idx = np.random.randint(0, forecast[dim].size, (iterations, select_dim_items))
    elif not replace:
        # create 2d np.arange
        idx = np.linspace(
            (np.arange(select_dim_items)),
            (np.arange(select_dim_items)),
            iterations,
            dtype="int",
        )
        # shuffle each line
        for ndx in np.arange(iterations):
            np.random.shuffle(idx[ndx])
    idx_da = xr.DataArray(
        idx,
        dims=("iteration", dim),
        coords=({"iteration": range(iterations), dim: new_dim}),
    )
    return idx_da


def resample_iterations(forecast, iterations, dim="member", dim_max=None, replace=True):
    """Resample over ``dim`` by index ``iterations`` times.

    .. note::
        This gives the same result as `resample_iterations_idx`. When using ``dask``,
        the number of tasks in ``resample_iterations` will scale with ``iterations`` but
        constant chunksize, whereas the tasks in `resample_iterations_idx` will stay
        constant with increasing chunksize.

    Parameters
    ----------
    forecast : xr.DataArray, xr.Dataset)
        forecast.
    iterations :int
        Number of resampling iterations.
    dim : str
        Dimension name to resamplw over. Defaults to ``'member'``.
    dim_max : int
        Number of items to select in `dim`.
    replace : bool
        Resampling with or without replacement. Defaults to ``True``.

    Returns
    -------
    forecast_smp : xr.DataArray, xr.Dataset
        resampled data resample along dimension ``dim`` with additional
        ``dim='iteration'``.

    References
    ----------
    * Mason, S. J., & Mimmack, G. M. (1992). The use of bootstrap confidence intervals
      for the correlation coefficient in climatology. Theoretical and Applied
      Climatology, 45(4), 229–233. doi: 10/b6fnsv

    * Mason, S. J. (2008). Understanding forecast verification statistics.
      Meteorological Applications, 15(1), 31–40. doi: 10/bgvgnz

    * Goddard, L., Kumar, A., Solomon, A., Smith, D., Boer, G., Gonzalez, P.,
      Kharin, V., Merryfield, W., Deser, C., Mason, S. J., Kirtman, B. P., Msadek, R.,
      Sutton, R., Hawkins, E., Fricker, T., Hegerl, G., Ferro, C. a. T.,
      Stephenson, D. B., Meehl, G. A., … Delworth, T. (2013). A verification framework
      for interannual-to-decadal predictions experiments. Climate Dynamics, 40(1–2),
      245–272. doi: 10/f4jjvf
    """
    if dim_max is not None and dim_max <= forecast[dim].size:
        # select only dim_max items
        select_dim_items = dim_max
        new_dim = forecast[dim].isel({dim: slice(None, dim_max)})
    else:
        select_dim_items = forecast[dim].size
        new_dim = forecast[dim]

    # generate random indices to select from
    idx_da = _gen_idx(forecast, dim, iterations, select_dim_items, replace, new_dim)
    # select those indices in a loop
    forecast_smp = []
    for i in np.arange(iterations):
        idx = idx_da.sel(iteration=i).data
        forecast_smp2 = forecast.isel({dim: idx}).assign_coords({dim: new_dim})
        forecast_smp.append(forecast_smp2)
    forecast_smp = xr.concat(forecast_smp, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)
    return forecast_smp.transpose(..., "iteration")


def resample_iterations_idx(
    forecast, iterations, dim="member", replace=True, dim_max=None
):
    """Resample over ``dim`` by index ``iterations`` times.

    .. note::
        This is a much faster way to bootstrap than resampling each iteration
        individually and applying the function to it. However, this will create a
        DataArray with dimension ``iteration`` of size ``iterations``. It is probably
        best to do this out-of-memory with ``dask`` if you are doing a large number
        of iterations or using spatial output (i.e., not time series data).

    Parameters
    ----------
        forecast : xr.DataArray, xr.Dataset
            forecast.
        iterations : int
            Number of resamling iterations.
        dim : str
            Dimension name to resample over. Defaults to ``'member'``.
        replace : bool
            Resampling with or without replacement. Defaults to ``True``.
        dim_max : int
            Number of indices from ``dim`` to return.

    Returns
    -------
    forecast_smp : xr.DataArray, xr.Dataset
        resampled data resample along dimension ``dim`` with additional
        ``dim='iteration'``.

    References
    ----------
    * Mason, S. J., & Mimmack, G. M. (1992). The use of bootstrap confidence intervals
      for the correlation coefficient in climatology. Theoretical and Applied
      Climatology, 45(4), 229–233. doi: 10/b6fnsv

    * Mason, S. J. (2008). Understanding forecast verification statistics.
      Meteorological Applications, 15(1), 31–40. doi: 10/bgvgnz

    * Goddard, L., Kumar, A., Solomon, A., Smith, D., Boer, G., Gonzalez, P.,
      Kharin, V., Merryfield, W., Deser, C., Mason, S. J., Kirtman, B. P., Msadek, R.,
      Sutton, R., Hawkins, E., Fricker, T., Hegerl, G., Ferro, C. a. T.,
      Stephenson, D. B., Meehl, G. A., … Delworth, T. (2013). A verification framework
      for interannual-to-decadal predictions experiments. Climate Dynamics, 40(1–2),
      245–272. doi: 10/f4jjvf

    """
    select_dim_items = forecast[dim].size
    new_dim = forecast[dim]

    if dask.is_dask_collection(forecast):
        forecast = forecast.chunk({"lead": -1, "member": -1})  # needed?

    def select_bootstrap_indices_ufunc(x, idx):
        """Selects multi-level indices ``idx`` from xarray object ``x`` for all
        iterations."""
        # `apply_ufunc` sometimes adds a singleton dimension on the end, so we squeeze
        # it out here. This leverages multi-level indexing from numpy, so we can
        # select a different set of, e.g., ensemble members for each iteration and
        # construct one large DataArray with ``iterations`` as a dimension.
        return np.moveaxis(x.squeeze()[idx.squeeze().transpose()], 0, -1)

    # generate random indices to select from
    idx_da = _gen_idx(forecast, dim, iterations, select_dim_items, replace, new_dim)
    # select those indices in one go
    transpose_kwargs = (
        {"transpose_coords": False} if isinstance(forecast, xr.DataArray) else {}
    )
    # bug fix when only one iteration
    if iterations == 1:
        return forecast.isel({dim: idx_da.isel(iteration=0, drop=True).values})
    else:
        forecast_smp = xr.apply_ufunc(
            select_bootstrap_indices_ufunc,
            forecast.transpose(dim, ..., **transpose_kwargs),
            idx_da,
            dask="parallelized",
            output_dtypes=[float],
        )
        if dim_max is not None and dim_max <= forecast[dim].size:
            forecast_smp = forecast_smp.isel({dim: slice(None, dim_max)})
        return forecast_smp
