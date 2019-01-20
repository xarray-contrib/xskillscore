import xarray as xr

from.np_deterministic import _pearson_r, _pearson_r_p_value, _rmse, _mse, _mae


__all__ = ['pearson_r', 'pearson_r_p_value', 'rmse', 'mse', 'mae']


def pearson_r(a, b, dim):
    """
    Pearson's correlation coefficient.

    Parameters
    ----------
    a : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    b : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.
        Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr
    xarray.apply_ufunc

    """
    return xr.apply_ufunc(_pearson_r, a, b,
                          input_core_dims=[[dim], [dim]],
                          kwargs={'axis': -1},
                          dask='allowed')


def pearson_r_p_value(a, b, dim):
    """
    2-tailed p-value associated with pearson's correlation coefficient.

    Parameters
    ----------
    a : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    b : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.
        2-tailed p-value.

    See Also
    --------
    scipy.stats.pearsonr
    xarray.apply_ufunc

    """
    return xr.apply_ufunc(_pearson_r_p_value, a, b,
                          input_core_dims=[[dim], [dim]],
                          kwargs={'axis': -1},
                          dask='allowed')


def rmse(a, b, dim):
    """
    Root Mean Squared Error.

    Parameters
    ----------
    a : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    b : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.
        Root Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error
    xarray.apply_ufunc

    """
    return xr.apply_ufunc(_rmse, a, b,
                          input_core_dims=[[dim], [dim]],
                          kwargs={'axis': -1},
                          dask='allowed')


def mse(a, b, dim):
    """
    Mean Squared Error.

    Parameters
    ----------
    a : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    b : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.
        Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error
    xarray.apply_ufunc

    """
    return xr.apply_ufunc(_mse, a, b,
                          input_core_dims=[[dim], [dim]],
                          kwargs={'axis': -1},
                          dask='allowed')


def mae(a, b, dim):
    """
    Mean Absolute Error.

    Parameters
    ----------
    a : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    b : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    dim : str
        The dimension to apply the correlation along.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.
        Mean Absolute Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error
    xarray.apply_ufunc

    """
    return xr.apply_ufunc(_mae, a, b,
                          input_core_dims=[[dim], [dim]],
                          kwargs={'axis': -1},
                          dask='allowed')
