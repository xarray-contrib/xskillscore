import bottleneck as bn
import numpy as np
from scipy import special
from scipy.stats import distributions

from .utils import suppress_warnings

__all__ = [
    "_effective_sample_size",
    "_linslope",
    "_mae",
    "_mape",
    "_me",
    "_median_absolute_error",
    "_mse",
    "_pearson_r",
    "_pearson_r_eff_p_value",
    "_pearson_r_p_value",
    "_r2",
    "_rmse",
    "_smape",
    "_spearman_r",
    "_spearman_r_eff_p_value",
    "_spearman_r_p_value",
]


def _match_nans(a, b, weights):
    """
    Considers missing values pairwise. If a value is missing
    in a, the corresponding value in b is turned to nan, and
    vice versa.

    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with nans placed at
        pairwise locations.

    """
    if np.isnan(a).any() or np.isnan(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isnan(a), np.isnan(b))
        a[idx], b[idx] = np.nan, np.nan
        # https://github.com/xarray-contrib/xskillscore/issues/168
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights[idx] = np.nan
    return a, b, weights


def _get_numpy_funcs(skipna):
    """
    Returns nansum and nanmean if skipna is True;
    Returns sum and mean if skipna is False.
    """
    if skipna:
        return np.nansum, np.nanmean
    else:
        return np.sum, np.mean


def _check_weights(weights):
    """
    Quick check if weights are all NaN. If so,
    return None to guide weighting scheme.
    """
    if weights is None:
        return None
    # catch if np.ndarray values are None
    elif (weights == None).all():
        return None
    elif np.all(np.isnan(weights)):
        return None
    else:
        if weights.min() < 0:
            raise ValueError(
                "Weights has a minimum below 0. Please submit a weights array "
                "of positive numbers."
            )
        return weights


def __compute_anomalies(a, b, weights, axis, skipna):
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    # Only do weighted sums if there are weights. Cannot have a
    # single generic function with weights of all ones, because
    # the denominator gets inflated when there are masked regions.
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            ma = sumfunc(a * weights, axis=axis) / sumfunc(weights, axis=axis)
            mb = sumfunc(b * weights, axis=axis) / sumfunc(weights, axis=axis)
    else:
        with suppress_warnings("Mean of empty slice"):
            ma = meanfunc(a, axis=axis)
            mb = meanfunc(b, axis=axis)
    am, bm = a - ma, b - mb
    return am, bm


def _effective_sample_size(a, b, axis, skipna):
    """Effective sample size for temporally correlated data.

    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to compute the effective sample size over.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    n_eff : ndarray
        Effective sample size.

    References
    ----------
    Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    if skipna:
        a, b, _ = _match_nans(a, b, None)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)

    # count total number of samples that are non-nan.
    n = np.count_nonzero(~np.isnan(a), axis=0)

    # compute lag-1 autocorrelation.
    am, bm = __compute_anomalies(a, b, weights=None, axis=0, skipna=skipna)
    a_auto = _pearson_r(am[0:-1], am[1::], weights=None, axis=0, skipna=skipna)
    b_auto = _pearson_r(bm[0:-1], bm[1::], weights=None, axis=0, skipna=skipna)

    # compute effective sample size per Bretherton et al. 1999
    with suppress_warnings("divide by zero encountered in true_divide"):
        n_eff = n * (1 - a_auto * b_auto) / (1 + a_auto * b_auto)
    n_eff = np.floor(n_eff)
    n_eff = np.clip(n_eff, 0, n)
    return n_eff


def _linslope(a, b, weights, axis, skipna):
    """ndarray implementation of scipy.stats.linregress[slope].

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the linear slope along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        slope of linear fit.

    See Also
    --------
    scipy.stats.linregress
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    if weights is not None:
        weights = np.rollaxis(weights, axis)

    am, bm = __compute_anomalies(a, b, weights=weights, axis=0, skipna=skipna)

    if weights is not None:
        s_num = sumfunc(weights * am * bm, axis=0)
        s_den = sumfunc(weights * am * am, axis=0)
    else:
        s_num = sumfunc(am * bm, axis=0)
        s_den = sumfunc(am * am, axis=0)

    with suppress_warnings("invalid value encountered in true_divide"):
        with suppress_warnings("invalid value encountered in double_scalars"):
            res = s_num / s_den
    return res


def _r2(a, b, weights, axis, skipna):
    """ndarray implementation of sklearn.metrics.r2_score.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the r2_score along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        R^2 (coefficient of determination) score.

    See Also
    --------
    sklearn.metrics.r2_score

    References
    ----------
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    if weights is not None:
        weights = np.rollaxis(weights, axis)

    am, bm = __compute_anomalies(a, b, weights=weights, axis=0, skipna=skipna)

    if weights is not None:
        squared_error = weights * ((a - b) ** 2)
        am_squared = weights * (am ** 2)
    else:
        squared_error = (a - b) ** 2
        am_squared = am ** 2
    num = sumfunc(squared_error, axis=0)
    den = sumfunc(am_squared, axis=0)
    with suppress_warnings("invalid value encountered in true_divide"):
        with suppress_warnings("divide by zero encountered in true_divide"):
            with suppress_warnings("divide by zero encountered in double_scalars"):
                with suppress_warnings("invalid value encountered in double_scalars"):
                    r2 = 1 - (num / den)
    return r2


def _pearson_r(a, b, weights, axis, skipna):
    """ndarray implementation of scipy.stats.pearsonr.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the correlation along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    if weights is not None:
        weights = np.rollaxis(weights, axis)

    am, bm = __compute_anomalies(a, b, weights=weights, axis=0, skipna=skipna)

    if weights is not None:
        r_num = sumfunc(weights * am * bm, axis=0)
        r_den = np.sqrt(
            sumfunc(weights * am * am, axis=0) * sumfunc(weights * bm * bm, axis=0)
        )
    else:
        r_num = sumfunc(am * bm, axis=0)
        r_den = np.sqrt(sumfunc(am * am, axis=0) * sumfunc(bm * bm, axis=0))

    with suppress_warnings("invalid value encountered in true_divide"):
        with suppress_warnings("invalid value encountered in double_scalars"):
            r = r_num / r_den
    res = np.clip(r, -1.0, 1.0)
    return res


def _pearson_r_p_value(a, b, weights, axis, skipna):
    """ndarray implementation of scipy.stats.pearsonr.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the compute the p value over.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    See Also
    --------
    scipy.stats.pearsonr
    """
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    r = _pearson_r(a, b, weights, axis, skipna)
    if np.isnan(r).all():
        return r
    else:
        # no nans or some nans
        a = np.rollaxis(a, axis)
        b = np.rollaxis(b, axis)
        # count non-nans
        dof = np.count_nonzero(~np.isnan(a), axis=0) - 2
        with suppress_warnings("invalid value encountered in true_divide"):
            t_squared = r ** 2 * (dof / ((1.0 - r) * (1.0 + r)))
            _x = dof / (dof + t_squared)
        _x = np.asarray(_x)
        _x = np.where(_x < 1.0, _x, 1.0)
        _a = 0.5 * dof
        _b = 0.5
        res = special.betainc(_a, _b, _x)
        # reset masked values to nan
        # raises  <__array_function__ internals>:5: DeprecationWarning: Calling nonzero
        # on 0d arrays is deprecated, as it behaves surprisingly. Use
        # `atleast_1d(cond).nonzero()` if the old behavior was intended. If the context
        # of this warning is of the form `arr[nonzero(cond)]`, just use `arr[cond]`.
        nan_locs = np.where(np.isnan(r))
        if len(nan_locs[0]) > 0:
            res[nan_locs] = np.nan
        return res


def _pearson_r_eff_p_value(a, b, axis, skipna):
    """Pearson r p value accounting for autocorrelation.

    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to compute the p value over.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    References
    ----------
    Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    if skipna:
        a, b, _ = _match_nans(a, b, None)
    r = _pearson_r(a, b, None, axis, skipna)
    if np.isnan(r).all():
        return r
    else:
        dof = _effective_sample_size(a, b, axis, skipna) - 2
        t_squared = r ** 2 * (dof / ((1.0 - r) * (1.0 + r)))
        with suppress_warnings("invalid value encountered in true_divide"):
            _x = dof / (dof + t_squared)
        _x = np.asarray(_x)
        _x = np.where(_x < 1.0, _x, 1.0)
        _a = 0.5 * dof
        _b = 0.5
        res = special.betainc(_a, _b, _x)
        # reset masked values to nan
        nan_locs = np.where(np.isnan(r))
        if len(nan_locs[0]) > 0:
            res[nan_locs] = np.nan
        return res


def _spearman_r(a, b, weights, axis, skipna):
    """ndarray implementation of scipy.stats.spearmanr.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the correlation along.
    weights : ndarray
        Input array.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Spearmanr's correlation coefficient.

    See Also
    --------
    scipy.stats.spearmanr
    """
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    _a = bn.nanrankdata(a, axis=axis)
    _b = bn.nanrankdata(b, axis=axis)
    return _pearson_r(_a, _b, weights, axis, skipna)


def _spearman_r_p_value(a, b, weights, axis, skipna):
    """ndarray implementation of scipy.stats.spearmanr.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the correlation along.
    weights : ndarray
        Input array.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    See Also
    --------
    scipy.stats.spearmanr

    References
    ----------
    https://github.com/scipy/scipy/blob/v1.3.1/scipy/stats/stats.py#L3613-L3764
    """
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    rs = _spearman_r(a, b, weights, axis, skipna)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    # count non-nans
    dof = np.count_nonzero(~np.isnan(a), axis=0) - 2
    with suppress_warnings("invalid value encountered in true_divide"):
        with suppress_warnings("divide by zero encountered in true_divide"):
            t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
    p = 2 * distributions.t.sf(np.abs(t), dof)
    return p


def _spearman_r_eff_p_value(a, b, axis, skipna):
    """Spearman rank correlation p value, accounting for autocorrelation.

    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    weights : ndarray
        Input array.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    See Also
    --------
    scipy.stats.spearmanr

    References
    ----------
    Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    if skipna:
        a, b, _ = _match_nans(a, b, None)
    rs = _spearman_r(a, b, None, axis, skipna)
    dof = _effective_sample_size(a, b, axis, skipna) - 2
    with suppress_warnings("invalid value encountered in true_divide"):
        with suppress_warnings("divide by zero encountered in true_divide"):
            t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
    p = 2 * distributions.t.sf(np.abs(t), dof)
    return p


def _me(a, b, weights, axis, skipna):
    """Mean Error.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the me along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Mean Error.
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)

    error = a - b
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            mean_error = sumfunc(error * weights, axis=axis) / sumfunc(
                weights, axis=axis
            )
    else:
        with suppress_warnings("Mean of empty slice"):
            mean_error = meanfunc(error, axis=axis)
    return mean_error


def _rmse(a, b, weights, axis, skipna):
    """Root Mean Squared Error.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the rmse along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Root Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)

    squared_error = (a - b) ** 2
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            mean_squared_error = sumfunc(squared_error * weights, axis=axis) / sumfunc(
                weights, axis=axis
            )
    else:
        with suppress_warnings("Mean of empty slice"):
            mean_squared_error = meanfunc(squared_error, axis=axis)
    res = np.sqrt(mean_squared_error)
    return res


def _mse(a, b, weights, axis, skipna):
    """Mean Squared Error.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the mse along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)

    squared_error = (a - b) ** 2
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            return sumfunc(squared_error * weights, axis=axis) / sumfunc(
                weights, axis=axis
            )
    else:
        with suppress_warnings("Mean of empty slice"):
            return meanfunc(squared_error, axis=axis)


def _mae(a, b, weights, axis, skipna):
    """Mean Absolute Error.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the mae along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Mean Absolute Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)

    absolute_error = np.absolute(a - b)
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            return sumfunc(absolute_error * weights, axis=axis) / sumfunc(
                weights, axis=axis
            )
    else:
        with suppress_warnings("Mean of empty slice"):
            return meanfunc(absolute_error, axis=axis)


def _median_absolute_error(a, b, axis, skipna):
    """Median Absolute Error.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the median absolute error along.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Median Absolute Error.

    See Also
    --------
    sklearn.metrics.median_absolute_error
    """
    medianfunc = np.nanmedian if skipna else np.median
    if skipna:
        a, b, _ = _match_nans(a, b, None)
    absolute_error = np.absolute(a - b)
    with suppress_warnings("All-NaN slice encountered"):
        return medianfunc(absolute_error, axis=axis)


def _mape(a, b, weights, axis, skipna):
    """Mean Absolute Percentage Error.

    .. math::
        \\mathrm{MAPE} = \\frac{1}{n} \\sum_{i=1}^{n}
                         \\frac{\\vert a_{i} - b_{i} \\vert}
                               {max(\epsilon, \\vert a_{i} \\vert)}

    Parameters
    ----------
    a : ndarray
        Input array (truth to be divided by).
    b : ndarray
        Input array.
    axis : int
        The axis to apply the mape along.
    weights : ndarray
        Input array.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Mean Absolute Percentage Error.

    Notes
    -----
    The percent error is calculated in reference to ``a``.

    Percent error is reported as decimal percent. I.e., a value of
    1 is 100%.

    \epsilon is an arbitrary small yet strictly positive number to avoid
    undefined results when ``a`` is zero.

    See Also
    --------
    sklearn.metrics.mean_absolute_percentage_error

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)
    epsilon = np.finfo(np.float64).eps
    mape = np.absolute(a - b) / np.maximum(np.absolute(a), epsilon)
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            return sumfunc(mape * weights, axis=axis) / sumfunc(weights, axis=axis)
    else:
        with suppress_warnings("Mean of empty slice"):
            return meanfunc(mape, axis=axis)


def _smape(a, b, weights, axis, skipna):
    """Symmetric Mean Absolute Percentage Error.

    .. math::
        \\mathrm{SMAPE} = \\frac{1}{n} \\sum_{i=1}^{n}
                          \\frac{ \\vert a_{i} - b_{i} \\vert }
                          { \\vert a_{i} \\vert + \\vert b_{i} \\vert  }

    Parameters
    ----------
    a : ndarray
        Input array (truth to be divided by).
    b : ndarray
        Input array.
    axis : int
        The axis to apply the mae along.
    weights : ndarray
        Input array.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Symmetric Mean Absolute Percentage Error.

    Notes
    -----
    Symmetric percent error is reported as decimal percent. I.e., a value of 1
    is 100%.

    References
    ----------
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    if skipna:
        a, b, weights = _match_nans(a, b, weights)
    weights = _check_weights(weights)
    smape = np.absolute(a - b) / (np.absolute(a) + np.absolute(b))
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            return sumfunc(smape * weights, axis=axis) / sumfunc(weights, axis=axis)
    else:
        with suppress_warnings("Mean of empty slice"):
            return meanfunc(smape, axis=axis)
