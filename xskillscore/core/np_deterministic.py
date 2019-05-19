import numpy as np
from scipy import special


__all__ = ['_pearson_r', '_pearson_r_p_value', '_rmse', '_mse', '_mae']


def _pearson_r(a, b, axis):
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
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    res = np.clip(r, -1.0, 1.0)
    return res


def _pearson_r_p_value(a, b, axis):
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

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    See Also
    --------
    scipy.stats.pearsonr

    """
    r = _pearson_r(a, b, axis)
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


def _rmse(a, b, axis):
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

    Returns
    -------
    res : ndarray
        Root Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error

    """
    res = np.sqrt(((a - b) ** 2).mean(axis=axis))
    return res


def _mse(a, b, axis):
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

    Returns
    -------
    res : ndarray
        Mean Squared Error.

    See Also
    --------
    sklearn.metrics.mean_squared_error

    """
    res = ((a - b) ** 2).mean(axis=axis)
    return res


def _mae(a, b, axis):
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

    Returns
    -------
    res : ndarray
        Mean Absolute Error.

    See Also
    --------
    sklearn.metrics.mean_absolute_error

    """
    res = (np.absolute(a - b)).mean(axis=axis)
    return res
