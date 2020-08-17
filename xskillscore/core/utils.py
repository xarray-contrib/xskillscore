import numpy as np
import xarray as xr
from xhistogram.xarray import histogram as xhist

__all__ = ['get_bin_edges', 'histogram']


def get_bin_edges(bins):
    """
        Returns bin edges of provided bins

        Parameters
        ----------
        bins : array_like
            One-dimensional array of bin values to compute bin edges

        Returns
        -------
        edges : np.array
            Array of bin edges where the first and last edge are computed using the spacing between \
                    the first-and-second and second-last-and-last bins, respectively. This array is one\
                    element larger than the input array

        Examples
        --------
        >>> bins = np.linspace(-2,2,10)
        >>> bin_edges = get_bin_edges(bins)
        array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])
    """

    dbin = np.diff(bins) / 2
    bin_edges = np.concatenate(
        ([bins[0] - dbin[0]], bins[:-1] + dbin, [bins[-1] + dbin[-1]])
    )

    return bin_edges


def histogram(*args, bins=None, bin_names=None, **kwargs):
    """Wrapper on xhistogram to deal with Datasets appropriately
    """
    # xhistogram expects a list for the dim input
    if 'dim' in kwargs:
        if isinstance(kwargs['dim'], str):
            kwargs['dim'] = [kwargs['dim']]
    for bin in bins:
        assert isinstance(bin, np.ndarray), 'all bins must be numpy arrays'

    if isinstance(args[0], xr.Dataset):
        # Get list of variables that are shared across all Datasets
        overlapping_vars = set.intersection(*map(set, [arg.data_vars for arg in args]))
        if overlapping_vars:
            # If bin_names not provided, use default ----
            if bin_names is None:
                bin_names = ['ds_' + str(i + 1) for i in range(len(args))]
            return xr.merge(
                [
                    xhist(
                        *(arg[var].rename(bin_names[i]) for i, arg in enumerate(args)),
                        bins=bins,
                        **kwargs,
                    ).rename(var)
                    for var in overlapping_vars
                ]
            )
        else:
            raise ValueError('No common variables exist across input Datasets')
    else:
        if bin_names:
            args = (arg.rename(bin_names[i]) for i, arg in enumerate(args))
        return xhist(*args, bins=bins, **kwargs)
