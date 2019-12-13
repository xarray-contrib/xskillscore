import bottleneck as bn
import numpy as np
from scipy import special
from scipy.stats import distributions

__all__ = [
    "_pearson_r",
    "_pearson_r_p_value",
    "_rmse",
    "_mse",
    "_mae",
    "_mad",
    "_smape",
    "_mape",
    "_spearman_r",
    "_spearman_r_p_value",
]


def _trim_nans(a, b, weights):
    """
    Considers missing values pairwise. If a value is missing
    in a, the corresponding value in b is also dropped, and
    vice versa.

    Returns
    -------
    a_trimmed, b_trimmed, weights_trimmed : ndarray
        a, b, and weights (if not None) with values removed
        if there is a nan at the given index in a or b.
    all_nan : bool
        True if either a or b are all nans.
    """
    if np.isnan(a).all() or np.isnan(b).all():
        all_nan = True
        a_trimmed, b_trimmed, weights_trimmed = np.nan, np.nan, np.nan
    else:
        all_nan = False
        # Find pairwise indices in a and b that do not have nans.
        idx = np.logical_and(~np.isnan(a), ~np.isnan(b))
        a_trimmed, b_trimmed = a[idx], b[idx]
        if weights is None:
            weights_trimmed = None
        else:
            weights_trimmed = weights[idx]
    return a_trimmed, b_trimmed, weights_trimmed, all_nan


def _check_weights(weights):
    """
    Quick check if weights are all NaN. If so,
    return None to guide weighting scheme.
    """
    if weights is None:
        return weights
    elif np.all(np.isnan(weights)):
        return None
    else:
        return weights


def _pearson_r(a, b, weights, axis, skipna):
    """
    ndarray implementation of scipy.stats.pearsonr.

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
    if skipna:
        a, b, weights, all_nan = _trim_nans(a, b, weights)
        if all_nan:
            return np.nan
    weights = _check_weights(weights)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)

    # Only do weighted sums if there are weights. Cannot have a
    # single generic function with weights of all ones, because
    # the denominator gets inflated when there are masked regions.
    if weights is not None:
        weights = np.rollaxis(weights, axis)
        ma = np.sum(a * weights, axis=0) / np.sum(weights, axis=0)
        mb = np.sum(b * weights, axis=0) / np.sum(weights, axis=0)
    else:
        ma = np.mean(a, axis=0)
        mb = np.mean(b, axis=0)

    am, bm = a - ma, b - mb

    if weights is not None:
        r_num = np.sum(weights * am * bm, axis=0)
        r_den = np.sqrt(
            np.sum(weights * am * am, axis=0) * np.sum(weights * bm * bm, axis=0)
        )
    else:
        r_num = np.sum(am * bm, axis=0)
        r_den = np.sqrt(np.sum(am * am, axis=0) * np.sum(bm * bm, axis=0))

    r = r_num / r_den
    res = np.clip(r, -1.0, 1.0)
    return res


def _pearson_r_p_value(a, b, weights, axis, skipna):
    """
    ndarray implementation of scipy.stats.pearsonr.

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
        2-tailed p-value.

    See Also
    --------
    scipy.stats.pearsonr

    """
    r = _pearson_r(a, b, weights, axis, skipna)
    if np.isnan(r).all():
        return r
    else:
        # no nans or some nans
        a = np.rollaxis(a, axis)
        b = np.rollaxis(b, axis)
        dof = np.apply_over_axes(np.sum, np.isnan(a * b), 0).squeeze() - 2
        dof = np.where(dof > 1.0, dof, a.shape[0] - 2)
        t_squared = r ** 2 * (dof / ((1.0 - r) * (1.0 + r)))
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
    """
    ndarray implementation of scipy.stats.spearmanr.

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
    rankfunc = bn.nanrankdata
    _a = rankfunc(a, axis=axis)
    _b = rankfunc(b, axis=axis)
    return _pearson_r(_a, _b, weights, axis, skipna)


def _spearman_r_p_value(a, b, weights, axis, skipna):
    """
    ndarray implementation of scipy.stats.spearmanr.

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

    Reference
    ---------
    https://github.com/scipy/scipy/blob/v1.3.1/scipy/stats/stats.py#L3613-L3764

    """
    rs = _spearman_r(a, b, weights, axis, skipna)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    dof = np.apply_over_axes(np.sum, np.isnan(a * b), 0).squeeze() - 2
    dof = np.where(dof > 1.0, dof, a.shape[0] - 2)
    t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
    p = 2 * distributions.t.sf(np.abs(t), dof)
    return p


def _rmse(a, b, weights, axis, skipna):
    """
    Root Mean Squared Error.

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
    if skipna:
        a, b, weights, all_nan = _trim_nans(a, b, weights)
        if all_nan:
            return np.nan
    weights = _check_weights(weights)

    squared_error = (a - b) ** 2
    if weights is not None:
        mean_squared_error = np.sum(squared_error * weights, axis=axis) / np.sum(
            weights, axis=axis
        )
    else:
        mean_squared_error = np.mean(((a - b) ** 2), axis=axis)
    res = np.sqrt(mean_squared_error)
    return res


def _mse(a, b, weights, axis, skipna):
    """
    Mean Squared Error.

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
    if skipna:
        a, b, weights, all_nan = _trim_nans(a, b, weights)
        if all_nan:
            return np.nan
    weights = _check_weights(weights)

    squared_error = (a - b) ** 2
    if weights is not None:
        return np.sum(squared_error * weights, axis=axis) / np.sum(weights, axis=axis)
    else:
        return np.mean(squared_error, axis=axis)


def _mae(a, b, weights, axis, skipna):
    """
    Mean Absolute Error.

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
    if skipna:
        a, b, weights, all_nan = _trim_nans(a, b, weights)
        if all_nan:
            return np.nan
    weights = _check_weights(weights)

    absolute_error = np.absolute(a - b)
    if weights is not None:
        return np.sum(absolute_error * weights, axis=axis) / np.sum(
            weights, axis=axis
        )
    else:
        return np.mean(absolute_error, axis=axis)


def _mad(a, b, axis, skipna):
    """
    Median Absolute Error.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the mae along.
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
    if skipna:
        a, b, _, all_nan = _trim_nans(a, b, None)
        if all_nan:
            return np.nan
    absolute_error = np.absolute(a - b)
    return np.median(absolute_error, axis=axis)


def _mape(a, b, weights, axis, skipna):
    """
    Mean Absolute Percentage Error.

    :: math MAPE = 1/n \sum \frac{|F_t-A_t|}{|A_t|}

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
        Mean Absolute Percentage Error.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    if skipna:
        a, b, weights, all_nan = _trim_nans(a, b, weights)
        if all_nan:
            return np.nan
    weights = _check_weights(weights)
    # replace divided by 0 with nan
    mape = np.absolute(a - b) / np.absolute(np.where(a != 0, a, np.nan))
    if weights is not None:
        return np.sum(mape * weights, axis=axis) / np.sum(weights, axis=axis)
    else:
        return np.mean(mape, axis=axis)


def _smape(a, b, weights, axis, skipna):
    """
    Symmetric Mean Absolute Percentage Error.

    :: math SMAPE = 1/n \sum \frac{|F_t-A_t|}{(|A_t|+|F_t|)}

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

    Reference
    ---------
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    """
    if skipna:
        a, b, weights, all_nan = _trim_nans(a, b, weights)
        if all_nan:
            return np.nan
    weights = _check_weights(weights)
    smape = np.absolute(a - b) / (np.absolute(a) + np.absolute(b))
    if weights is not None:
        return np.sum(smape * weights, axis=axis) / np.sum(weights, axis=axis)
    else:
        return np.mean(smape, axis=axis)
