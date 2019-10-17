import numpy as np
from scipy import special


__all__ = ['_pearson_r', '_pearson_r_p_value', '_rmse', '_mse', '_mae']


def _pearson_r(a, b, weights, axis):
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
        Input array.

    Returns
    -------
    res : ndarray
        Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr

    """
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    weights = np.rollaxis(weights, axis)
    ma = np.sum(a * weights, axis=0) / np.sum(weights, axis=0)
    mb = np.sum(b * weights, axis=0) / np.sum(weights, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(weights * am * bm, axis=0)
    r_den = np.sqrt(np.sum(weights * am * am, axis=0) * np.sum(weights * bm * bm, axis=0))
    r = r_num / r_den
    res = np.clip(r, -1.0, 1.0)
    return res


def _pearson_r_p_value(a, b, weights, axis):
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
        Input array.

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    See Also
    --------
    scipy.stats.pearsonr

    """
    r = _pearson_r(a, b, weights, axis)
    a = np.rollaxis(a, axis)
    df = a.shape[0] - 2
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    _x = df/(df+t_squared)
    _x = np.asarray(_x)
    _x = np.where(_x < 1.0, _x, 1.0)
    _a = 0.5*df
    _b = 0.5
    res = special.betainc(_a, _b, _x)
    return res


def _rmse(a, b, weights, axis):
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
        Input array.

    Returns
    -------
    res : ndarray
        Root Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error

    """
    squared_error = (a - b) ** 2
    mean_squared_error = np.sum(squared_error * weights, axis=axis) / np.sum(weights, axis=axis)
    res = np.sqrt(mean_squared_error)
    return res


def _mse(a, b, weights, axis):
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
        Input array.

    Returns
    -------
    res : ndarray
        Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error

    """
    squared_error = (a - b) **2
    res = np.sum(squared_error * weights, axis=axis) / np.sum(weights, axis=axis)
    return res


def _mae(a, b, weights, axis):
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
        Input array.

    Returns
    -------
    res : ndarray
        Mean Absolute Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error

    """
    absolute_error = np.absolute(a - b)
    res = np.sum(absolute_error * weights, axis=axis) / np.sum(weights, axis=axis)
    return res
