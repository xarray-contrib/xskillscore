import numpy as np

__all__ = ["_gerrity_s"]

# TODO: Currently computes the Gerrity scoring matrix using nested for-loops.
# Is it possible to remove these?


def _gerrity_s(table):
    """Returns Gerrity scoring matrix, s"""
    p_o = (table.sum(axis=-1).T / table.sum(axis=(-2, -1)).T).T
    p_sum = np.cumsum(p_o, axis=-1)
    a = (1.0 - p_sum) / p_sum
    k = a.shape[-1]
    s = np.zeros(table.shape, dtype=float)
    for (i, j) in np.ndindex(*s.shape[-2:]):
        if i == j:
            s[..., i, j] = (
                1.0
                / (k - 1.0)
                * (
                    np.sum(1.0 / a[..., 0:j], axis=-1)
                    + np.sum(a[..., j : k - 1], axis=-1)
                )
            )
        elif i < j:
            s[..., i, j] = (
                1.0
                / (k - 1.0)
                * (
                    np.sum(1.0 / a[..., 0:i], axis=-1)
                    - (j - i)
                    + np.sum(a[..., j : k - 1], axis=-1)
                )
            )
        else:
            s[..., i, j] = s[..., j, i]
    return s
