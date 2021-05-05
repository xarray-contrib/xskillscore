import dask
import numpy as np
import xarray as xr

CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}


def _gen_idx(forecast, dim, iterations, select_dim_items, replace, new_dim):
    """Generate indices to select from. Replace decides whether resampling is with or
    without replacement.

    Args: (selected)
        select_dim_items(int) : number of new items in dim to select
        new_dim (xr.DataArray) : new dimension of above
    """
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
        This gives the same result as
        :py:func:`~xskillscore.resampling.resample_iterations_idx` but slower.
        When using ``dask``, the number of tasks in
        :py:func:`~xskillscore.resampling.resample_iterations` will scale with
        ``iterations`` but constant chunksize, whereas the tasks in
        :py:func:`~xskillscore.resampling.resample_iterations_idx` will stay constant
        with increasing chunksize.

    Parameters
    ----------
    forecast : xr.DataArray, xr.Dataset
        Forecast.
    iterations : int
        Number of resampling iterations.
    dim : str
        Dimension name to resample over. Defaults to ``'member'``.
    dim_max : int
        Number of items to select in `dim`.
    replace : bool
        Resampling with or without replacement. Defaults to ``True``.

    Returns
    -------
    forecast_smp : xr.DataArray, xr.Dataset
        data resampled along dimension ``dim`` with additional ``dim='iteration'``.

    Examples
    --------
    >>> a = xr.DataArray(np.random.rand(1000, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.resample_iterations(a, 500, 'time')  # doctest: +SKIP
    <xarray.DataArray (time: 1000, x: 3, y: 3, iteration: 500)>

    See also
    --------
    :py:func:`~xskillscore.resampling.resample_iterations_idx`

    References
    ----------
    * Mason, S. J., & Mimmack, G. M. (1992). The use of bootstrap confidence intervals
      for the correlation coefficient in climatology. Theoretical and Applied
      Climatology, 45(4), 229–233. https://doi.org/10/b6fnsv

    * Mason, S. J. (2008). Understanding forecast verification statistics.
      Meteorological Applications, 15(1), 31–40. https://doi.org/10/bgvgnz

    * Goddard, L., Kumar, A., Solomon, A., Smith, D., Boer, G., Gonzalez, P.,
      Kharin, V., Merryfield, W., Deser, C., Mason, S. J., Kirtman, B. P., Msadek, R.,
      Sutton, R., Hawkins, E., Fricker, T., Hegerl, G., Ferro, C. a. T.,
      Stephenson, D. B., Meehl, G. A., … Delworth, T. (2013). A verification framework
      for interannual-to-decadal predictions experiments. Climate Dynamics, 40(1–2),
      245–272. https://doi.org/10/f4jjvf
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
        forecast_smp.append(forecast.isel({dim: idx}).assign_coords({dim: new_dim}))
    forecast_smp = xr.concat(forecast_smp, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)
    return forecast_smp.transpose(..., "iteration")


def resample_iterations_idx(
    forecast, iterations, dim="member", replace=True, dim_max=None
):
    """Resample over ``dim`` by index ``iterations`` times.

    .. note::
        This is a much faster way to bootstrap/resample each iteration
        individually and applying the function to it. This will create a
        DataArray with dimension ``iteration`` of size ``iterations``.
        When using ``dask``, the number of tasks in
        :py:func:`~xskillscore.resampling.resample_iterations` will scale with
        ``iterations`` but constant chunksize, whereas the tasks in
        :py:func:`~xskillscore.resampling.resample_iterations_idx` will stay constant
        with increasing chunksize.

    Parameters
    ----------
        forecast : xr.DataArray, xr.Dataset
            Forecast.
        iterations : int
            Number of resampling iterations.
        dim : str
            Dimension name to resample over. Defaults to ``'member'``.
        replace : bool
            Resampling with or without replacement. Defaults to ``True``.
        dim_max : int
            Number of item from ``dim`` to return.

    Returns
    -------
    forecast_smp : xr.DataArray, xr.Dataset
        data resampled along dimension ``dim`` with additional ``dim='iteration'``.

    Examples
    --------
    >>> a = xr.DataArray(np.random.rand(1000, 3, 3),
    ...                  coords=[("time", np.arange(1000)),
    ...                          ("x", np.arange(3)),
    ...                          ("y", np.arange(3))])
    >>> xs.resample_iterations_idx(a, 500, 'time') # doctest: +SKIP
    <xarray.DataArray (time: 1000, x: 3, y: 3, iteration: 500)>

    See also
    --------
    :py:func:`~xskillscore.resampling.resample_iterations`

    References
    ----------
    * Mason, S. J., & Mimmack, G. M. (1992). The use of bootstrap confidence intervals
      for the correlation coefficient in climatology. Theoretical and Applied
      Climatology, 45(4), 229–233. https://doi.org/10/b6fnsv

    * Mason, S. J. (2008). Understanding forecast verification statistics.
      Meteorological Applications, 15(1), 31–40. https://doi.org/10/bgvgnz

    * Goddard, L., Kumar, A., Solomon, A., Smith, D., Boer, G., Gonzalez, P.,
      Kharin, V., Merryfield, W., Deser, C., Mason, S. J., Kirtman, B. P., Msadek, R.,
      Sutton, R., Hawkins, E., Fricker, T., Hegerl, G., Ferro, C. a. T.,
      Stephenson, D. B., Meehl, G. A., … Delworth, T. (2013). A verification framework
      for interannual-to-decadal predictions experiments. Climate Dynamics, 40(1–2),
      245–272. https://doi.org/10/f4jjvf
    """
    # equivalent to above
    select_dim_items = forecast[dim].size
    new_dim = forecast[dim]

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
        forecast_smp = forecast.isel({dim: idx_da.isel(iteration=0, drop=True).values})
    else:
        forecast_smp = xr.apply_ufunc(
            select_bootstrap_indices_ufunc,
            forecast.transpose(dim, ..., **transpose_kwargs),
            idx_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    # return only dim_max members
    if dim_max is not None and dim_max <= forecast[dim].size:
        forecast_smp = forecast_smp.isel({dim: slice(None, dim_max)})
    return forecast_smp
