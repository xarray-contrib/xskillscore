import warnings

import numpy as np

__all__ = ["_reliability"]


def _reliability(o, f, bin_edges):
    """Return the reliability and number of samples per bin"""
    # I couldn't get dask='parallelized' working in this case
    # so dealing with dask arrays explicitly
    # if true, imports dask.array later
    is_dask_array = not isinstance(o, np.ndarray) or not isinstance(f, np.ndarray)

    if is_dask_array:
        r = []
        N = []
    else:
        r = np.zeros((*o.shape[:-1], len(bin_edges) - 1), dtype=float)
        N = np.zeros_like(r)

    for i in range(len(bin_edges) - 1):
        # Follow xhistogram behaviour: all bins are half-open,
        # except for the right-most bin
        # which adds an epsilon to the right edge
        # see https://github.com/xgcm/xhistogram/issues/18
        if i == (len(bin_edges) - 2):
            f_in_bin = (f >= bin_edges[i]) & (f < (bin_edges[i + 1] + 1e-8))
        else:
            f_in_bin = (f >= bin_edges[i]) & (f < bin_edges[i + 1])
        o_f_in_bin = o & f_in_bin
        N_f_in_bin = f_in_bin.sum(axis=-1)
        N_o_f_in_bin = o_f_in_bin.sum(axis=-1)
        if is_dask_array:
            r.append(N_o_f_in_bin / N_f_in_bin)
            N.append(N_f_in_bin)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r[..., i] = N_o_f_in_bin / N_f_in_bin
                N[..., i] = N_f_in_bin

    if is_dask_array:
        import dask.array as da

        return (
            da.stack(r, axis=-1).rechunk({-1: -1}),
            da.stack(N, axis=-1).rechunk({-1: -1}),
        )
    else:
        return r, N
