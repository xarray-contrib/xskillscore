import numpy as np


__all__ = ['_pearson_r', '_rmse']


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
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    res = np.sqrt(((a - b) ** 2).mean(axis=0))
    return res 
