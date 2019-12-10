import numpy as np
import xarray as xr
from properscoring import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    threshold_brier_score,
)

__all__ = [
    "brier_score",
    "crps_ensemble",
    "crps_gaussian",
    "crps_quadrature",
    "threshold_brier_score",
]


def xr_crps_gaussian(observations, mu, sig):
    """
    xarray version of properscoring.crps_gaussian: Continuous Ranked
     Probability Score with a Gaussian distribution.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    mu : xarray.Dataset or xarray.DataArray
        The mean of the forecast normal distribution.
    sig : xarray.Dataset or xarray.DataArray
        The standard deviation of the forecast distribution.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    See Also
    --------
    properscoring.crps_gaussian
    xarray.apply_ufunc
    """
    # check if same dimensions
    if isinstance(mu, (int, float)):
        mu = xr.DataArray(mu)
    if isinstance(sig, (int, float)):
        sig = xr.DataArray(sig)
    if mu.dims != observations.dims:
        observations, mu = xr.broadcast(observations, mu)
    if sig.dims != observations.dims:
        observations, sig = xr.broadcast(observations, sig)
    return xr.apply_ufunc(
        crps_gaussian,
        observations,
        mu,
        sig,
        input_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=[float],
    )


def xr_crps_quadrature(x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
    """
    xarray version of properscoring.crps_quadrature: Continuous Ranked
     Probability Score with numerical integration of the normal distribution

    Parameters
    ----------
    x : xarray.Dataset or xarray.DataArray
        Observations associated with the forecast distribution ``cdf_or_dist``.
    cdf_or_dist : callable or scipy.stats.distribution
        Function which returns the cumulative density of the forecast
        distribution at value x.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    See Also
    --------
    properscoring.crps_quadrature
    xarray.apply_ufunc
    """
    return xr.apply_ufunc(
        crps_quadrature,
        x,
        cdf_or_dist,
        xmin,
        xmax,
        tol,
        input_core_dims=[[], [], [], [], []],
        dask="parallelized",
        output_dtypes=[float],
    )


def xr_crps_ensemble(
    observations, forecasts, weights=None, issorted=False, dim="member"
):
    """
    xarray version of properscoring.crps_ensemble: Continuous Ranked
     Probability Score with the ensemble distribution

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        Forecast with required member dimension ``dim``.
    weights : xarray.Dataset or xarray.DataArray
        If provided, the CRPS is calculated exactly with the assigned
        probability weights to each forecast. Weights should be positive,
        but do not need to be normalized. By default, each forecast is
        weighted equally.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    dim : str, optional
        Name of ensemble member dimension. By default, 'member'.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    See Also
    --------
    properscoring.crps_ensemble
    xarray.apply_ufunc
    """
    return xr.apply_ufunc(
        crps_ensemble,
        observations,
        forecasts,
        input_core_dims=[[], [dim]],
        kwargs={"axis": -1, "issorted": issorted, "weights": weights},
        dask="parallelized",
        output_dtypes=[float],
    )


def xr_brier_score(observations, forecasts):
    """
    xarray version of properscoring.brier_score: Calculate Brier score (BS).

    ..math:
        BS(p, k) = (p_1 - k)^2,

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        The forecasts associated with the observations.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    References
    ----------
    Gneiting, Tilmann, and Adrian E Raftery. “Strictly Proper Scoring Rules,
      Prediction, and Estimation.” Journal of the American Statistical
      Association 102, no. 477 (March 1, 2007): 359–78.
      https://doi.org/10/c6758w.

    See Also
    --------
    properscoring.brier_score
    xarray.apply_ufunc
    """
    return xr.apply_ufunc(
        brier_score,
        observations,
        forecasts,
        input_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[float],
    )


def xr_threshold_brier_score(
    observations, forecasts, threshold, issorted=False, dim="member"
):
    """
    xarray version of properscoring.threshold_brier_score: Calculate the Brier
     scores of an ensemble for exceeding given thresholds.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        Forecast with required member dimension ``dim``.
    threshold : scalar or 1d scalar
        Threshold values at which to calculate exceedence Brier scores.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    dim : str, optional
        Name of ensemble member dimension. By default, 'member'.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        (If ``threshold`` is a scalar, the result will have the same shape as
        observations. Otherwise, it will have an additional final dimension
        corresponding to the threshold levels. Not implemented yet.)

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
    if isinstance(threshold, list):
        threshold.sort()
        threshold = xr.DataArray(threshold, dims="threshold")
        threshold["threshold"] = np.arange(1, 1 + threshold.threshold.size)

    if isinstance(threshold, (xr.DataArray, xr.Dataset)):
        if "threshold" not in threshold.dims:
            raise ValueError(
                "please provide threshold with threshold dim, found", threshold.dims,
            )
        input_core_dims = [[], [dim], ["threshold"]]
        output_core_dims = [["threshold"]]
    elif isinstance(threshold, (int, float)):
        input_core_dims = [[], [dim], []]
        output_core_dims = [[]]
    else:
        raise ValueError(
            "Please provide threshold as list, int, float \
            or xr.object with threshold dimension; found",
            type(threshold),
        )
    return xr.apply_ufunc(
        threshold_brier_score,
        observations,
        forecasts,
        threshold,
        input_core_dims=input_core_dims,
        kwargs={"axis": -1, "issorted": issorted},
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
    )
