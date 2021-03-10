import contextlib
import warnings

import numpy as np
import xarray as xr
from xhistogram.xarray import histogram as xhist

__all__ = ["histogram"]


@contextlib.contextmanager
def suppress_warnings(msg=None):
    """Catch warnings with message msg. From
    https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/
    _utils.py#L23."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", msg)
        yield


def _preprocess_dims(dim, a):
    """Preprocesses dimensions to prep for stacking.
    Parameters
    ----------
    dim : str, list
        The dimension(s) to apply the function along.
    """
    if dim is None:
        dim = list(a.dims)
    elif isinstance(dim, str):
        dim = [dim]
    axis = tuple(range(-1, -len(dim) - 1, -1))
    return dim, axis


def _fail_if_dim_empty(dim):
    if dim == []:
        raise ValueError(
            "metric must be applied along one dimension, therefore "
            f"requires `dim` not being empty, found dim={dim}"
        )


def _stack_input_if_needed(a, b, dim, weights):
    """
    Stack input arrays a, b if needed in correlation metrics.
    Adapt dim and weights accordingly.
    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray or None
        Weights matching dimensions of ``dim`` to apply during the function.
    Returns
    -------
    a : xarray.Dataset or xarray.DataArray stacked with new_dim
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray stacked with new_dim
        Labeled array(s) over which to apply the function.
    new_dim : str
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray stacked with new_dim or None
        Weights matching dimensions of ``dim`` to apply during the function.
    """
    if len(dim) > 1:
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        if weights is not None:
            weights = weights.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
    return a, b, new_dim, weights


def _preprocess_weights(a, dim, new_dim, weights):
    """Preprocesses weights array to prepare for numpy computation.
    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        One of the arrays over which the function will be applied.
    dim : str, list
        The original dimension(s) to apply the function along.
    new_dim : str
        The newly named dimension after running ``_preprocess_dims``
    weights : xarray.Dataset or xarray.DataArray or None
        Weights to apply to function, matching the dimension size of
        ``new_dim``.
    """
    if weights is None:
        return None
    else:
        # Scale weights to vary from 0 to 1.
        weights = weights / weights.max()
        # Check that the weights array has the same size
        # dimension(s) as those being applied over.
        drop_dims = {k: 0 for k in a.dims if k not in new_dim}
        if dict(weights.sizes) != dict(a.isel(drop_dims).sizes):
            raise ValueError(
                f"weights dimension(s) {dim} of size {dict(weights.sizes)} "
                f"does not match DataArray's size "
                f"{dict(a.isel(drop_dims).sizes)}"
            )
        if dict(weights.sizes) != dict(a.sizes):
            # Broadcast weights to full size of main object.
            _, weights = xr.broadcast(a, weights)
        return weights


def _add_as_coord(ds1, ds2, coordinate_suffix):
    """Add ds2 as a coordinate of ds1.
    Assumes that ds1 and ds2 are the same type of xarray object
    """
    if isinstance(ds1, xr.Dataset):
        for var in ds1.data_vars:
            ds1 = ds1.assign_coords({f"{ds2[var].name}_{coordinate_suffix}": ds2[var]})
    elif isinstance(ds1, xr.DataArray):
        ds1 = ds1.assign_coords({coordinate_suffix: ds2})
    else:
        raise ValueError("Inputs ds1 and ds2 must be xarray objects")
    return ds1


def _get_bin_centers(bin_edges):
    """Return the arithmetic mean of the bin_edges"""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def histogram(*args, bins=None, bin_names=None, **kwargs):
    """Wrapper on xhistogram to deal with Datasets appropriately"""
    # xhistogram expects a list for the dim input
    if "dim" in kwargs:
        if isinstance(kwargs["dim"], str):
            kwargs["dim"] = [kwargs["dim"]]
    for bin in bins:
        assert isinstance(
            bin, np.ndarray
        ), f"all bins must be numpy arrays, found {type(bin)}"

    if isinstance(args[0], xr.Dataset):
        # Get list of variables that are shared across all Datasets
        overlapping_vars = set.intersection(*map(set, [arg.data_vars for arg in args]))
        if overlapping_vars:
            # If bin_names not provided, use default ----
            if bin_names is None:
                bin_names = ["ds_" + str(i + 1) for i in range(len(args))]
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
            raise ValueError("No common variables exist across input Datasets")
    else:
        if bin_names:
            args = (arg.rename(bin_names[i]) for i, arg in enumerate(args))
        return xhist(*args, bins=bins, **kwargs)


def _bool_to_int(ds):
    """convert xr.object of dtype bool to int to evade:
    TypeError: numpy boolean subtract, the `-` operator, is not supported"""

    def _helper_bool_to_int(da):
        if da.dtype == "bool":
            da = da.astype("int")
        return da

    if isinstance(ds, xr.Dataset):
        ds = ds.map(_helper_bool_to_int)
    else:
        ds = _helper_bool_to_int(ds)
    return ds


def _check_identical_xr_types(a, b):
    """Check that a and b are both xr.Dataset or both xr.DataArray."""
    if type(a) != type(b):
        raise ValueError(f"a and b must be same type, found {type(a)} and {type(b)}")
    for d in [a, b]:
        if not isinstance(d, (xr.Dataset, xr.DataArray)):
            raise ValueError("inputs must be xr.DataArray or xr.Dataset")


def _keep_nans_masked(ds_before, ds_after, dim=None, ignore=None):
    """Preserve all NaNs from ds_before for ds_after over while ignoring some dimensions optionally."""
    if dim is None:
        dim = list(ds_before.dims)
    elif isinstance(dim, str):
        dim = [dim]
    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = list(ignore)
    all_dim = set(dim) ^ set(ignore)
    all_dim = [d for d in all_dim if d in ds_before.dims]
    mask = ds_before.isnull().all(all_dim)
    ds_after = ds_after.where(~mask.astype("bool"), other=np.nan)
    for d in dim:
        assert d not in ds_after.dims
    if ignore is not None:
        for d in ignore:
            assert d not in ds_after.dims
    return ds_after
