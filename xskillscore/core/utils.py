from __future__ import annotations

from typing import List, Tuple

import numpy as np
import xarray as xr
from xhistogram.xarray import histogram as xhist

from .types import Dim, XArray

__all__ = ["histogram"]


def _preprocess_dims(dim: Dim | None, a: XArray) -> Tuple[List[str], Tuple[int, ...]]:
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
        assert isinstance(bin, np.ndarray), f"all bins must be numpy arrays, found {type(bin)}"

    if isinstance(args[0], xr.Dataset):
        # Get a list of variables that are shared across all Datasets
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
    if not isinstance(a, type(b)):
        raise ValueError(f"a and b must be same type, found {type(a)} and {type(b)}")
    for d in [a, b]:
        if not isinstance(d, (xr.Dataset, xr.DataArray)):
            raise ValueError("inputs must be xr.DataArray or xr.Dataset")


def _keep_nans_masked(observations, forecasts, res, dim=None, member_dim="member"):
    """Preserve all NaNs from inputs (observations, forecasts) on output (res)."""
    if dim is None:
        forecasts_mask_dim, observations_mask_dim = None, None
    else:
        forecasts_mask_dim = dim + [member_dim] if member_dim in forecasts.dims else dim
        if "category_edge" in forecasts.dims:
            forecasts_mask_dim = forecasts_mask_dim + ["category_edge"]
        if "category" in forecasts.dims:
            forecasts_mask_dim = forecasts_mask_dim + ["category"]
        observations_mask_dim = dim
        if "category_edge" in observations.dims:
            observations_mask_dim = observations_mask_dim + ["category_edge"]
        if "category" in observations.dims:
            observations_mask_dim = observations_mask_dim + ["category"]
    res = res.where(observations.notnull().any(observations_mask_dim)).where(
        forecasts.notnull().any(forecasts_mask_dim)
    )
    return res
