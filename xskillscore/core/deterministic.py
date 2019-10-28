import xarray as xr

from .np_deterministic import (_mad, _mae, _mape, _mse, _pearson_r,
                               _pearson_r_p_value, _rmse, _smape, _spearman_r,
                               _spearman_r_p_value)

__all__ = ["pearson_r", "pearson_r_p_value", "rmse", "mse", "mae",
           "mad", "smape", "mape", "spearman_r", "spearman_r_p_value"]


def _preprocess_dims(dim):
    """Preprocesses dimensions to prep for stacking.

    Parameters
    ----------
    dim : str, list
        The dimension(s) to apply the function along.
    """
    if isinstance(dim, str):
        dim = [dim]
    axis = tuple(range(-1, -len(dim) - 1, -1))
    return dim, axis


def _preprocess_weights(a, dim, new_dim, weights):
    """Preprocesses weights array to prepare for numpy computation.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        One of the arrays over which the function will be applied.
    dim : str, list
        The original dimension(s) to apply the function along.
    new_dim : str
        The newly named dimension after running ``_preprocess_dims``
    weights : xarray.Dataset or xarray.DataArray
        Weights to apply to function, matching the dimension size of
        ``new_dim``.
    """
    if weights is None:
        return xr.full_like(a, None)  # Return nan weighting array.
    else:
        # Throw error if there are negative weights.
        if weights.min() < 0:
            raise ValueError(
                "Weights has a minimum below 0. Please submit a weights array "
                "of positive numbers."
            )
        # Scale weights to vary from 0 to 1.
        weights = weights / weights.max()
        # Check that the weights array has the same size
        # dimension(s) as those being applied over.
        drop_dims = {k: 0 for k in a.dims if k not in new_dim}
        if dict(weights.sizes) != dict(a.isel(drop_dims).sizes):
            raise ValueError(
                f"weights dimension(s) {dim} of size {dict(weights.sizes)} "
                f"does not match DataArray's size "
                f"{dict(a.isel(drop_dims).sizes)}"
            )
        if dict(weights.sizes) != dict(a.sizes):
            # Broadcast weights to full size of main object.
            _, weights = xr.broadcast(a, weights)
        return weights


def pearson_r(a, b, dim, weights=None, skipna=False):
    """
    Pearson's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr
    xarray.apply_ufunc

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        if weights is not None:
            weights = weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
    weights = _preprocess_weights(a, dim, new_dim, weights)

    return xr.apply_ufunc(
        _pearson_r,
        a,
        b,
        weights,
        input_core_dims=[[new_dim], [new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def pearson_r_p_value(a, b, dim, weights=None, skipna=False):
    """
    2-tailed p-value associated with pearson's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr
    xarray.apply_ufunc

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        if weights is not None:
            weights = weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
    weights = _preprocess_weights(a, dim, new_dim, weights)

    return xr.apply_ufunc(
        _pearson_r_p_value,
        a,
        b,
        weights,
        input_core_dims=[[new_dim], [new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def spearman_r(a, b, dim, weights=None, skipna=False):
    """
    Spearman's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Spearman's correlation coefficient.

    See Also
    --------
    scipy.stats.spearman_r
    xarray.apply_ufunc
    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        if weights is not None:
            weights = weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
    weights = _preprocess_weights(a, dim, new_dim, weights)

    return xr.apply_ufunc(
        _spearman_r,
        a,
        b,
        weights,
        input_core_dims=[[new_dim], [new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def spearman_r_p_value(a, b, dim, weights=None, skipna=False):
    """
    2-tailed p-value associated with Spearman's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Spearman's correlation coefficient.

    See Also
    --------
    scipy.stats.spearman_r
    xarray.apply_ufunc
    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        if weights is not None:
            weights = weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
    weights = _preprocess_weights(a, dim, new_dim, weights)

    return xr.apply_ufunc(
        _spearman_r_p_value,
        a,
        b,
        weights,
        input_core_dims=[[new_dim], [new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def rmse(a, b, dim, weights=None, skipna=False):
    """
    Root Mean Squared Error.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the rmse along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Root Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error
    xarray.apply_ufunc

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)

    return xr.apply_ufunc(
        _rmse,
        a,
        b,
        weights,
        input_core_dims=[dim, dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def mse(a, b, dim, weights=None, skipna=False):
    """
    Mean Squared Error.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mse along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error
    xarray.apply_ufunc

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)

    return xr.apply_ufunc(
        _mse,
        a,
        b,
        weights,
        input_core_dims=[dim, dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def mae(a, b, dim, weights=None, skipna=False):
    """
    Mean Absolute Error.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mae along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Absolute Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error
    xarray.apply_ufunc

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)

    return xr.apply_ufunc(
        _mae,
        a,
        b,
        weights,
        input_core_dims=[dim, dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def mad(a, b, dim, skipna=False):
    """
    Median Absolute Deviation.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mae along.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Median Absolute Deviation.

    See Also
    --------
    sklearn.metrics.median_absolute_error
    xarray.apply_ufunc

    Reference
    ---------
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    dim, axis = _preprocess_dims(dim)

    return xr.apply_ufunc(
        _mad,
        a,
        b,
        input_core_dims=[dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def mape(a, b, dim, weights=None, skipna=False):
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        (Truth which will be divided by)
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mae along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Absolute Percentage Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error / a * 100
    xarray.apply_ufunc

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)

    return xr.apply_ufunc(
        _mape,
        a,
        b,
        weights,
        input_core_dims=[dim, dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def smape(a, b, dim, weights=None, skipna=False):
    """
    Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        (Truth which will be divided by)
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mae along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Symmetric Mean Absolute Percentage Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error / (np.abs(a) + np.abs(b)) * 100
    xarray.apply_ufunc

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)

    return xr.apply_ufunc(
        _smape,
        a,
        b,
        weights,
        input_core_dims=[dim, dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )
