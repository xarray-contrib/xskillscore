import warnings

import xarray as xr

from .np_deterministic import (
    _effective_sample_size,
    _mae,
    _mape,
    _median_absolute_error,
    _mse,
    _pearson_r,
    _pearson_r_eff_p_value,
    _pearson_r_p_value,
    _r2,
    _rmse,
    _smape,
    _spearman_r,
    _spearman_r_eff_p_value,
    _spearman_r_p_value,
)
from .utils import (
    _fail_if_dim_empty,
    _preprocess_dims,
    _preprocess_weights,
    _stack_input_if_needed,
)

__all__ = [
    'pearson_r',
    'pearson_r_p_value',
    'pearson_r_eff_p_value',
    'rmse',
    'mse',
    'mae',
    'median_absolute_error',
    'smape',
    'mape',
    'spearman_r',
    'spearman_r_p_value',
    'spearman_r_eff_p_value',
    'effective_sample_size',
    'r2',
]


def _determine_input_core_dims(dim, weights):
    """
    Determine input_core_dims based on type of dim and weights.

    Parameters
    ----------
    dim : str, list
        The dimension(s) to apply the metric along.
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


def pearson_r(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Pearson's correlation coefficient.

    .. math::
        r_{ab} = \\frac{ \\sum_{i=i}^{n} (a_{i} - \\bar{a}) (b_{i} - \\bar{b}) }
                 {\\sqrt{ \\sum_{i=1}^{n} (a_{i} - \\bar{a})^{2} }
                  \\sqrt{ \\sum_{i=1}^{n} (b_{i} - \\bar{b})^{2} }}

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr

    References
    ----------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import pearson_r
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> pearson_r(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)

    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _pearson_r,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def r2(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """R^2 (coefficient of determination) score.

    We first take the total sum of squares of our known vector, a.

    .. math::
        SS_{\\mathrm{tot}} = \\sum_{i=1}^{n} (a_{i} - \\bar{a})^{2}

    Next, we take the sum of squares of the error between our known vector
    a and the predicted vector, b.

    .. math::
        SS_{\\mathrm{res}} = \\sum_{i=1}^{n} (a_{i} - b_{i})^{2}

    Lastly we compute the coefficient of determiniation using these two
    terms.

    .. math::
        R^{2} = 1 - \\frac{SS_{\\mathrm{res}}}{SS_{\\mathrm{tot}}}

    .. note::
        The coefficient of determination is *not* symmetric. In other words,
        ``r2(a, b) != r2(b, a)``. Be careful and note that by our
        convention, ``b`` is the modeled/predicted vector and ``a`` is the
        observed vector.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        R^2 (coefficient of determination) score.

    See Also
    --------
    sklearn.metrics.r2_score

    References
    ----------
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import r2
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> r2(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)

    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _r2,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def pearson_r_p_value(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """2-tailed p-value associated with pearson's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import pearson_r_p_value
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> pearson_r_p_value(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)
    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _pearson_r_p_value,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def effective_sample_size(a, b, dim='time', skipna=False, keep_attrs=False):
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
        The dimension(s) to apply the function along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Effective sample size.

    References
    ----------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import effective_sample_size
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> effective_sample_size(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

    if len(dim) > 1:
        raise ValueError(
            'Effective sample size should only be applied to a singular time dimension.'
        )
    else:
        new_dim = dim[0]
    if new_dim != 'time':
        warnings.warn(
            f"{dim} is not 'time'. Make sure that you are applying this over a "
            f'temporal dimension.'
        )

    return xr.apply_ufunc(
        _effective_sample_size,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def pearson_r_eff_p_value(a, b, dim=None, skipna=False, keep_attrs=False):
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
        The dimension(s) to compute the p value over. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Pearson's correlation coefficient, accounting
        for autocorrelation.

    See Also
    --------
    scipy.stats.pearsonr

    References
    ----------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import pearson_r_eff_p_value
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> pearson_r_eff_p_value(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

    if len(dim) > 1:
        raise ValueError(
            'Effective sample size should only be applied to a singular time dimension.'
        )
    else:
        new_dim = dim[0]
    if new_dim != 'time':
        warnings.warn(
            f"{dim} is not 'time'. Make sure that you are applying this over a "
            f'temporal dimension.'
        )

    return xr.apply_ufunc(
        _pearson_r_eff_p_value,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def spearman_r(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Spearman's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Spearman's correlation coefficient.

    See Also
    --------
    scipy.stats.spearman_r

    References
    ----------
    https://github.com/scipy/scipy/blob/v1.3.1/scipy/stats/stats.py#L3613-L3764
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import spearman_r
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> spearman_r(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)
    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _spearman_r,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def spearman_r_p_value(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """2-tailed p-value associated with Spearman's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Spearman's correlation coefficient.

    See Also
    --------
    scipy.stats.spearman_r

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import spearman_r_p_value
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> spearman_r_p_value(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)
    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _spearman_r_p_value,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def spearman_r_eff_p_value(a, b, dim=None, skipna=False, keep_attrs=False):
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
        The dimension(s) to compute the p value over. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Spearman's correlation coefficient, accounting for
        autocorrelation.

    See Also
    --------
    scipy.stats.spearman_r

    References
    ----------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.



    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import spearman_r_eff_p_value
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> spearman_r_eff_p_value(a, b, dim='time')
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

    if len(dim) > 1:
        raise ValueError(
            'Effective sample size should only be applied to a singular time dimension.'
        )
    else:
        new_dim = dim[0]
    if new_dim != 'time':
        warnings.warn(
            f"{dim} is not 'time'. Make sure that you are applying this over a "
            f'temporal dimension.'
        )

    return xr.apply_ufunc(
        _spearman_r_eff_p_value,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def rmse(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Root Mean Squared Error.

    .. math::
        \\mathrm{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(a_{i} - b_{i})^{2}}

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the rmse along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Root Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error

    References
    ----------
    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import rmse
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> rmse(a, b, dim='time')
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _rmse,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': axis, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def mse(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Mean Squared Error.

    .. math::
        \\mathrm{MSE} = \\frac{1}{n}\\sum_{i=1}^{n}(a_{i} - b_{i})^{2}

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mse along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_squared_error

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import mse
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> mse(a, b, dim='time')
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _mse,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': axis, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def mae(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Mean Absolute Error.

    .. math::
        \\mathrm{MAE} = \\frac{1}{n}\\sum_{i=1}^{n}\\vert a - b\\vert

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the mae along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Absolute Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_absolute_error

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import mae
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> mae(a, b, dim='time')
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _mae,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': axis, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def median_absolute_error(a, b, dim=None, skipna=False, keep_attrs=False):
    """
    Median Absolute Error.

    .. math::
        \\mathrm{median}(\\vert a - b\\vert)

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the median absolute error along.
        Note that this dimension will be reduced as a result. Defaults to None reducing all dimensions.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Median Absolute Error.

    See Also
    --------
    sklearn.metrics.median_absolute_error

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import median_absolute_error
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> median_absolute_error(a, b, dim='time')
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

    return xr.apply_ufunc(
        _median_absolute_error,
        a,
        b,
        input_core_dims=[dim, dim],
        kwargs={'axis': axis, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def mape(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Mean Absolute Percentage Error.

    .. math::
        \\mathrm{MAPE} = \\frac{1}{n} \\sum_{i=1}^{n}
                         \\frac{\\vert a_{i} - b_{i} \\vert}
                               {\\vert a_{i} \\vert}

    .. note::
        The percent error is calculated in reference to ``a``. Percent
        error is reported as decimal percent. I.e., a value of 1 is
        100%.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        (Truth which will be divided by)
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply mape along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mean Absolute Percentage Error.

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import mape
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> mape(a, b, dim='time')
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _mape,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': axis, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def smape(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Symmetric Mean Absolute Percentage Error.

    .. math::
        \\mathrm{SMAPE} = \\frac{1}{n} \\sum_{i=1}^{n}
                          \\frac{ \\vert a_{i} - b_{i} \\vert }
                          { \\vert a_{i} \\vert + \\vert b_{i} \\vert  }

    .. note::
        Percent error is reported as decimal percent. I.e., a value of 1 is
        100%.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        (Truth which will be divided by)
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the smape along. Note that this dimension will
        be reduced as a result. Defaults to None reducing all dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Symmetric Mean Absolute Percentage Error.

    References
    ----------
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xskillscore import smape
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
                        dims=['time', 'x', 'y'])
    >>> smape(a, b, dim='time')
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _smape,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={'axis': axis, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
