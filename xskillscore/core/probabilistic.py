import xarray as xr
from properscoring import crps_ensemble, crps_gaussian, threshold_brier_score


def xr_crps_gaussian(observations, mu, sig):
    """
    xarray version of properscoring.crps_gaussian.

    Parameters
    ----------
    observations : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled observations arrays.
    mu : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled forecasts mean arrays.
    sig : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled forecasts mean arrays.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
     numpy.ndarray, the first type on that list to appear on an input.

    See Also
    --------
    properscoring.crps_gaussian
    xarray.apply_ufunc
    """
    # check if same dimensions
    if mu.dims != observations.dims:
        observations, mu = xr.broadcast(observations, mu)
    if sig.dims != observations.dims:
        observations, sig = xr.broadcast(observations, sig)
    return xr.apply_ufunc(crps_gaussian,
                          observations,
                          mu,
                          sig,
                          input_core_dims=[[], [], []],
                          dask='parallelized',
                          output_dtypes=[float])


def xr_crps_ensemble(observations, forecasts):
    """
    xarray version of properscoring.crps_ensemble.

    Parameters
    ----------
    observations : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled observations arrays.
    forecasts : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled forecasts arrays.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.

    See Also
    --------
    properscoring.crps_ensemble
    xarray.apply_ufunc
    """
    if forecasts.dims != observations.dims:
        observations, forecasts = xr.broadcast(observations, forecasts)
    return xr.apply_ufunc(crps_ensemble,
                          observations,
                          forecasts,
                          input_core_dims=[[], []],
                          dask='parallelized',
                          output_dtypes=[float])


def xr_threshold_brier_score(observations,
                             forecasts,
                             threshold,
                             issorted=False,
                             axis=-1):
    """
    xarray version of properscoring.threshold_brier_score: Calculate the Brier
     scores of an ensemble for exceeding given thresholds.

    Parameters
    ----------
    observations : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled observations arrays.
    forecasts : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or
     scalars, Mix of labeled and/or unlabeled forecasts arrays.
    threshold : scalar (not yet implemented: or 1d scalar threshold value(s) at
     which to calculate) exceedence Brier scores.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    axis : int, optional
        Axis in forecasts which corresponds to different ensemble members,
        along which to calculate the threshold decomposition.


    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input. (If
    ``threshold`` is a scalar, the result will have the same shape as
    observations. Otherwise, it will have an additional final dimension
    corresponding to the threshold levels.)

    References
    ----------
    Gneiting, T. and Ranjan, R. Comparing density forecasts using threshold-
       and quantile-weighted scoring rules. J. Bus. Econ. Stat. 29, 411-422
       (2011). http://www.stat.washington.edu/research/reports/2008/tr533.pdf

    See Also
    --------
    properscoring.threshold_brier_score
    xarray.apply_ufunc
    """
    if forecasts.dims != observations.dims:
        observations, forecasts = xr.broadcast(observations, forecasts)
    return xr.apply_ufunc(threshold_brier_score,
                          observations,
                          forecasts,
                          threshold,
                          input_core_dims=[[], [], []],
                          kwargs={
                              'axis': axis,
                              'issorted': issorted
                          },
                          dask='parallelized',
                          output_dtypes=[float])
