import xarray as xr
import warnings

from .np_deterministic import (
    _mae,
    _mape,
    _median_absolute_error,
    _mse,
    _pearson_r,
    _pearson_r_p_value,
    _pearson_r_eff_p_value,
    _rmse,
    _smape,
    _spearman_r,
    _spearman_r_p_value,
    _spearman_r_eff_p_value,
    _effective_sample_size,
    _r2,
)

__all__ = [
    "pearson_r",
    "pearson_r_p_value",
    "pearson_r_eff_p_value",
    "rmse",
    "mse",
    "mae",
    "median_absolute_error",
    "smape",
    "mape",
    "spearman_r",
    "spearman_r_p_value",
    "spearman_r_eff_p_value",
    "effective_sample_size",
    "r2",
]


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


def _stack_input_if_needed(a, b, dim, weights):
    """
    Stack input arrays a, b if needed in correlation metrics.
    Adapt dim and weights accordingly.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.

    Returns
    -------
    a : xarray.Dataset or xarray.DataArray stacked with new_dim
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray stacked with new_dim
        Labeled array(s) over which to apply the function.
    new_dim : str
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray stacked with new_dim or None
        Weights matching dimensions of ``dim`` to apply during the function.

    """
    if len(dim) > 1:
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        if weights is not None:
            weights = weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
    return a, b, new_dim, weights


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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights to apply to function, matching the dimension size of
        ``new_dim``.
    """
    if weights is None:
        return None
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


def _determine_input_core_dims(dim, weights):
    """
    Determine input_core_dims based on type of dim and weights.

    Parameters
    ----------
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.

    Returns
    -------
    list of lists
        input_core_dims used for xr.apply_ufunc.
    """
    if not isinstance(dim, list):
        dim = [dim]
    # build input_core_dims depending on weights
    if weights is None:
        input_core_dims = [dim, dim, [None]]
    else:
        input_core_dims = [dim, dim, dim]
    return input_core_dims


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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Pearson's correlation coefficient.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.pearsonr
    xskillscore.core.np_deterministic._pearson_r

    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    """
    dim, _ = _preprocess_dims(dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)

    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _pearson_r,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def r2(a, b, dim, weights=None, skipna=False):
    """
    R^2 (coefficient of determination) score.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        R^2 (coefficient of determination) score.

    See Also
    --------
    xarray.apply_ufunc
    sklearn.metrics.r2_score

    Reference
    ---------
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    dim, _ = _preprocess_dims(dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)

    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _r2,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Pearson's correlation coefficient.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.pearsonr
    xskillscore.core.np_deterministic._pearson_r_p_value

    """
    dim, _ = _preprocess_dims(dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)
    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _pearson_r_p_value,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def effective_sample_size(a, b, dim, skipna=False):
    """Effective sample size for temporally correlated data.

    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.

    The effective sample size extracts the number of independent samples
    between two time series being correlated. This is derived by assessing
    the magnitude of the lag-1 autocorrelation coefficient in each of the time series
    being correlated. A higher autocorrelation induces a lower effective sample
    size which raises the correlation coefficient for a given p value.

     .. math::
        N_{eff} = N\\left( \\frac{1 -
                   \\rho_{f}\\rho_{o}}{1 + \\rho_{f}\\rho_{o}} \\right),

    where :math:`\\rho_{f}` and :math:`\\rho_{o}` are the lag-1 autocorrelation
    coefficients for the forecast and observations.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the function along.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Effective sample size.

    Reference
    ---------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        raise ValueError(
            "Effective sample size should only be applied to a singular time dimension."
        )
    else:
        new_dim = dim[0]
    if new_dim != "time":
        warnings.warn(
            f"{dim} is not 'time'. Make sure that you are applying this over a "
            f"temporal dimension."
        )

    return xr.apply_ufunc(
        _effective_sample_size,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def pearson_r_eff_p_value(a, b, dim, skipna=False):
    """
    2-tailed p-value associated with Pearson's correlation coefficient,
    accounting for autocorrelation.

    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.

    The effective p value is computed by replacing the sample size :math:`N` in the
    t-statistic with the effective sample size, :math:`N_{eff}`. The same Pearson
    product-moment correlation coefficient :math:`r` is used as when computing the
    standard p value.

    .. math::
        t = r\\sqrt{ \\frac{N_{eff} - 2}{1 - r^{2}} },

    where :math:`N_{eff}` is computed via the autocorrelation in the forecast and
    observations.

    .. math::
        N_{eff} = N\\left( \\frac{1 -
                   \\rho_{f}\\rho_{o}}{1 + \\rho_{f}\\rho_{o}} \\right),

    where :math:`\\rho_{f}` and :math:`\\rho_{o}` are the lag-1 autocorrelation
    coefficients for the forecast and observations.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to compute the p value over.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Pearson's correlation coefficient, accounting
        for autocorrelation.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.pearsonr
    xskillscore.core.np_deterministic._pearson_r_eff_p_value

    Reference
    ---------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        raise ValueError(
            "Effective sample size should only be applied to a singular time dimension."
        )
    else:
        new_dim = dim[0]
    if new_dim != "time":
        warnings.warn(
            f"{dim} is not 'time'. Make sure that you are applying this over a "
            f"temporal dimension."
        )

    return xr.apply_ufunc(
        _pearson_r_eff_p_value,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Spearman's correlation coefficient.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.spearman_r
    xskillscore.core.np_deterministic._spearman_r

    Reference
    ---------
    https://github.com/scipy/scipy/blob/v1.3.1/scipy/stats/stats.py#L3613-L3764
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    """
    dim, _ = _preprocess_dims(dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)
    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _spearman_r,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Spearman's correlation coefficient.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.spearman_r
    xskillscore.core.np_deterministic._spearman_r_p_value

    """
    dim, _ = _preprocess_dims(dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)
    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _spearman_r_p_value,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def spearman_r_eff_p_value(a, b, dim, skipna=False):
    """
    2-tailed p-value associated with Spearman rank correlation coefficient,
    accounting for autocorrelation.

    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.

    The effective p value is computed by replacing the sample size :math:`N` in the
    t-statistic with the effective sample size, :math:`N_{eff}`. The same Spearman's
    rank correlation coefficient :math:`r` is used as when computing the standard p
    value.

    .. math::
        t = r\\sqrt{ \\frac{N_{eff} - 2}{1 - r^{2}} },

    where :math:`N_{eff}` is computed via the autocorrelation in the forecast and
    observations.

    .. math::
        N_{eff} = N\\left( \\frac{1 -
                   \\rho_{f}\\rho_{o}}{1 + \\rho_{f}\\rho_{o}} \\right),

    where :math:`\\rho_{f}` and :math:`\\rho_{o}` are the lag-1 autocorrelation
    coefficients for the forecast and observations.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to compute the p value over.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Spearman's correlation coefficient, accounting for
        autocorrelation.

    Reference
    ---------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.spearman_r
    xskillscore.core.np_deterministic._spearman_r_eff_p_value

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        raise ValueError(
            "Effective sample size should only be applied to a singular time dimension."
        )
    else:
        new_dim = dim[0]
    if new_dim != "time":
        warnings.warn(
            f"{dim} is not 'time'. Make sure that you are applying this over a "
            f"temporal dimension."
        )

    return xr.apply_ufunc(
        _spearman_r_eff_p_value,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Root Mean Squared Error.

    See Also
    --------
    xarray.apply_ufunc
    sklearn.metrics.mean_squared_error
    xskillscore.core.np_deterministic._rmse

    Reference
    ---------
    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _rmse,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Squared Error.

    See Also
    --------
    xarray.apply_ufunc
    sklearn.metrics.mean_squared_error
    xskillscore.core.np_deterministic._mse

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_squared_error

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _mse,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Absolute Error.

    See Also
    --------
    xarray.apply_ufunc
    sklearn.metrics.mean_absolute_error
    xskillscore.core.np_deterministic._mae

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_error

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _mae,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )


def median_absolute_error(a, b, dim, skipna=False):
    """
    Median Absolute Error.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the median absolute error along.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Median Absolute Error.

    See Also
    --------
    sklearn.metrics.median_absolute_error
    xarray.apply_ufunc
    xskillscore.core.np_deterministic._median_absolute_error

    """
    dim, axis = _preprocess_dims(dim)

    return xr.apply_ufunc(
        _median_absolute_error,
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Absolute Percentage Error.

    See Also
    --------
    xarray.apply_ufunc
    xskillscore.core.np_deterministic._mape

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _mape,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
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
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Symmetric Mean Absolute Percentage Error.

    See Also
    --------
    xarray.apply_ufunc
    xskillscore.core.np_deterministic._smape

    Reference
    ---------
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    """
    dim, axis = _preprocess_dims(dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _smape,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
    )
