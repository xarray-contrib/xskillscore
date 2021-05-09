import warnings

import xarray as xr

from .np_deterministic import (
    _effective_sample_size,
    _linslope,
    _mae,
    _mape,
    _me,
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
    "effective_sample_size",
    "linslope",
    "mae",
    "mape",
    "me",
    "median_absolute_error",
    "mse",
    "pearson_r",
    "pearson_r_eff_p_value",
    "pearson_r_p_value",
    "r2",
    "rmse",
    "smape",
    "spearman_r",
    "spearman_r_eff_p_value",
    "spearman_r_p_value",
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
        input_core_dims = [dim, dim, []]
    else:
        input_core_dims = [dim, dim, dim]
    return input_core_dims


def linslope(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Slope of linear fit.

    .. math::
        s_{ab} = \\frac{ \\sum_{i=i}^{n} (a_{i} - \\bar{a}) (b_{i} - \\bar{b}) }
                 { \\sum_{i=1}^{n} (a_{i} - \\bar{a})^{2} }

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the slope of linear fit along. Note that this
        dimension will be reduced as a result. Defaults to None reducing all
        dimensions.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    skipna : bool
        If True, skip NaNs when computing function.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied from the first input to the
        new one. If False (default), the new object will be returned without
        attributes.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Slope of linear fit.

    See Also
    --------
    scipy.stats.linregress

    Examples
    --------
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> xs.linslope(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[-0.30948771, -0.21562529, -0.63141304],
           [ 0.31446077,  2.23858011,  0.44743617],
           [-0.22243944,  0.47034784,  1.08512859]])
    Dimensions without coordinates: x, y
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    a, b, new_dim, weights = _stack_input_if_needed(a, b, dim, weights)
    weights = _preprocess_weights(a, dim, new_dim, weights)

    input_core_dims = _determine_input_core_dims(new_dim, weights)

    return xr.apply_ufunc(
        _linslope,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> xs.pearson_r(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[-0.17455755, -0.26648379, -0.74265833],
           [ 0.32535918,  0.42496646,  0.1940647 ],
           [-0.3203094 ,  0.33207755,  0.89250429]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> xs.pearson_r_p_value(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.77888033, 0.66476199, 0.15051516],
           [0.59316935, 0.47567465, 0.75446898],
           [0.59925464, 0.58509064, 0.04161894]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def effective_sample_size(a, b, dim="time", skipna=False, keep_attrs=False):
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> xs.effective_sample_size(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[4., 0., 4.],
           [3., 4., 4.],
           [3., 4., 2.]])
    Dimensions without coordinates: x, y
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3),
    ...                  dims=['time', 'x', 'y'])
    >>> xs.pearson_r_eff_p_value(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.82544245,        nan, 0.25734167],
           [0.78902959, 0.57503354, 0.8059353 ],
           [0.79242625, 0.66792245,        nan]])
    Dimensions without coordinates: x, y
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.spearman_r(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[-0.6, -0.5, -0.7],
           [ 0.4,  0.3,  0.3],
           [-0.3, -0.1,  0.9]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.spearman_r_p_value(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.28475698, 0.39100222, 0.1881204 ],
           [0.50463158, 0.62383766, 0.62383766],
           [0.62383766, 0.87288857, 0.03738607]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.spearman_r_eff_p_value(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.4       ,        nan, 0.3       ],
           [0.73802024, 0.7       , 0.7       ],
           [0.80602663, 0.9       ,        nan]])
    Dimensions without coordinates: x, y
    """
    _fail_if_dim_empty(dim)
    dim, _ = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> r2(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[ -3.77828319,  -1.25687543,  -2.52495914],
           [ -0.67280201, -39.45271514,  -5.78241791],
           [ -1.66615797,  -1.56749317,   0.09843265]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def me(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Mean Error.

    .. math::
        \\mathrm{ME} = \\frac{1}{n}\\sum_{i=1}^{n}(a_{i} - b_{i})

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the me along. Note that this dimension will
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
        Mean Error.

    Examples
    --------
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> me(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[ 0.01748202, -0.14165293,  0.22455357],
           [ 0.13893709, -0.23513353, -0.18174132],
           [-0.29317762,  0.16887445, -0.17297527]])
    Dimensions without coordinates: x, y
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)
    weights = _preprocess_weights(a, dim, dim, weights)
    input_core_dims = _determine_input_core_dims(dim, weights)

    return xr.apply_ufunc(
        _me,
        a,
        b,
        weights,
        input_core_dims=input_core_dims,
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.rmse(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.30366872, 0.5147618 , 0.57410211],
           [0.2963848 , 0.37177283, 0.40563885],
           [0.55686111, 0.38189299, 0.21317579]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
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
    --------
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.mse(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.09221469, 0.26497971, 0.32959323],
           [0.08784395, 0.13821504, 0.16454288],
           [0.31009429, 0.14584225, 0.04544392]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> mae(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.26014863, 0.40137207, 0.48871634],
           [0.18809417, 0.30197826, 0.2984658 ],
           [0.52934554, 0.19820357, 0.17335851]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
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
        Note that this dimension will be reduced as a result.
        Defaults to None reducing all dimensions.
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.median_absolute_error(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.28798217, 0.23322591, 0.62067468],
           [0.12146232, 0.20314509, 0.23442927],
           [0.59041981, 0.03289321, 0.21343494]])
    Dimensions without coordinates: x, y
    """
    dim, axis = _preprocess_dims(dim, a)
    a, b = xr.broadcast(a, b, exclude=dim)

    return xr.apply_ufunc(
        _median_absolute_error,
        a,
        b,
        input_core_dims=[dim, dim],
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )


def mape(a, b, dim=None, weights=None, skipna=False, keep_attrs=False):
    """Mean Absolute Percentage Error.

    .. math::
        \\mathrm{MAPE} = \\frac{1}{n} \\sum_{i=1}^{n}
                         \\frac{\\vert a_{i} - b_{i} \\vert}
                               {max(\epsilon, \\vert a_{i} \\vert)}

    .. note::
        The percent error is calculated in reference to ``a``. Percent
        error is reported as decimal percent. I.e., a value of 1 is
        100%. :math:`\epsilon` is an arbitrary small yet strictly positive
        number to avoid undefined results when ``a`` is zero.

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

    See Also
    --------
    sklearn.metrics.mean_absolute_percentage_error

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Examples
    --------
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.mape(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.6268041 , 9.45134297, 3.28717608],
           [0.27099746, 1.58105176, 1.48258713],
           [6.55806162, 0.22271096, 0.39302745]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
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
    >>> a = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> b = xr.DataArray(np.random.rand(5, 3, 3), dims=['time', 'x', 'y'])
    >>> xs.smape(a, b, dim='time')
    <xarray.DataArray (x: 3, y: 3)>
    array([[0.35591619, 0.43662087, 0.55372571],
           [0.1864336 , 0.45831965, 0.38473469],
           [0.58730494, 0.18081757, 0.14960832]])
    Dimensions without coordinates: x, y
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
        kwargs={"axis": axis, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
