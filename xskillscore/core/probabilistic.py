import bottleneck as bn
import numpy as np
import xarray as xr
from properscoring import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    threshold_brier_score,
)

from .utils import histogram

__all__ = [
    'brier_score',
    'crps_ensemble',
    'crps_gaussian',
    'crps_quadrature',
    'threshold_brier_score',
    'rank_histogram',
    'discrimination',
]


def xr_crps_gaussian(observations, mu, sig, dim=None, weights=None, keep_attrs=False):
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
    dim : str or list of str, optional
        Dimension to mean over after calculating crps_gaussian.
        Defaults to None implying averaging.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no mean is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray reduced by dimension dim

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
    res = xr.apply_ufunc(
        crps_gaussian,
        observations,
        mu,
        sig,
        input_core_dims=[[], [], []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if dim is None:
        return res
    else:
        if weights is not None:
            return res.weighted(weights).mean(dim)
        else:
            return res.mean(dim)


def xr_crps_quadrature(
    x,
    cdf_or_dist,
    xmin=None,
    xmax=None,
    tol=1e-6,
    dim=None,
    weights=None,
    keep_attrs=False,
):
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
    xmin, xmax, tol: see properscoring.crps_quadrature
    dim : str or list of str, optional
        Dimension to mean over after calculating crps_gaussian.
        Defaults to None implying averaging.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no mean is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    See Also
    --------
    properscoring.crps_quadrature
    xarray.apply_ufunc

    """
    res = xr.apply_ufunc(
        crps_quadrature,
        x,
        cdf_or_dist,
        xmin,
        xmax,
        tol,
        input_core_dims=[[], [], [], [], []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if dim is None:
        return res
    else:
        if weights is not None:
            return res.weighted(weights).mean(dim)
        else:
            return res.mean(dim, keep_attrs=keep_attrs)


def xr_crps_ensemble(
    observations,
    forecasts,
    member_weights=None,
    issorted=False,
    member_dim='member',
    dim=None,
    weights=None,
    keep_attrs=False,
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
    member_weights : xarray.Dataset or xarray.DataArray
        If provided, the CRPS is calculated exactly with the assigned
        probability weights to each forecast. Weights should be positive,
        but do not need to be normalized. By default, each forecast is
        weighted equally.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    member_dim : str, optional
        Name of ensemble member dimension. By default, 'member'.
    dim : str or list of str, optional
        Dimension to mean over after calculating crps_gaussian.
        Defaults to None implying averaging.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no mean is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    See Also
    --------
    properscoring.crps_ensemble
    xarray.apply_ufunc

    """
    res = xr.apply_ufunc(
        crps_ensemble,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        kwargs={'axis': -1, 'issorted': issorted, 'weights': member_weights},
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if dim is None:
        return res
    else:
        if weights is not None:
            return res.weighted(weights).mean(dim)
        else:
            return res.mean(dim, keep_attrs=keep_attrs)


def xr_brier_score(observations, forecasts, dim=None, weights=None, keep_attrs=False):
    """
    xarray version of properscoring.brier_score: Calculate Brier score (BS).

    ..math:
        BS(p, k) = (p_1 - k)^2

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        The forecasts associated with the observations.
    dim : str or list of str, optional
        Dimension to mean over after calculating crps_gaussian.
        Defaults to None implying averaging.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no mean is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

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
    res = xr.apply_ufunc(
        brier_score,
        observations,
        forecasts,
        input_core_dims=[[], []],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if dim is None:
        return res
    else:
        if weights is not None:
            return res.weighted(weights).mean(dim)
        else:
            return res.mean(dim)


def xr_threshold_brier_score(
    observations,
    forecasts,
    threshold,
    issorted=False,
    member_dim='member',
    dim=None,
    weights=None,
    keep_attrs=False,
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
    member_dim : str, optional
        Name of ensemble member dimension. By default, 'member'.
    dim : str or list of str, optional
        Dimension to mean over after calculating crps_gaussian.
        Defaults to None implying averaging.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no mean is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

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
        threshold = xr.DataArray(threshold, dims='threshold')
        threshold['threshold'] = np.arange(1, 1 + threshold.threshold.size)

    if isinstance(threshold, (xr.DataArray, xr.Dataset)):
        if 'threshold' not in threshold.dims:
            raise ValueError(
                'please provide threshold with threshold dim, found', threshold.dims,
            )
        input_core_dims = [[], [member_dim], ['threshold']]
        output_core_dims = [['threshold']]
    elif isinstance(threshold, (int, float)):
        input_core_dims = [[], [member_dim], []]
        output_core_dims = [[]]
    else:
        raise ValueError(
            'Please provide threshold as list, int, float \
            or xr.object with threshold dimension; found',
            type(threshold),
        )
    res = xr.apply_ufunc(
        threshold_brier_score,
        observations,
        forecasts,
        threshold,
        input_core_dims=input_core_dims,
        kwargs={'axis': -1, 'issorted': issorted},
        output_core_dims=output_core_dims,
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if dim is None:
        return res
    else:
        if weights is not None:
            return res.weighted(weights).mean(dim)
        else:
            return res.mean(dim)


def rank_histogram(observations, forecasts, dim=None, member_dim='member'):
    """Returns the rank histogram (Talagrand diagram) along the specified dimensions.

        Parameters
        ----------
        observations : xarray.Dataset or xarray.DataArray
            The observations or set of observations.
        forecasts : xarray.Dataset or xarray.DataArray
            Forecast with required member dimension ``member_dim``.
        dim : str or list of str, optional
            Dimension(s) over which to compute the histogram of ranks.
            Defaults to None meaning compute over all dimensions
        member_dim : str, optional
            Name of ensemble member dimension. By default, 'member'.

        Returns
        -------
        rank_histogram : xarray.Dataset or xarray.DataArray
            New object containing the histogram of ranks

        Examples
        --------
        >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
        ...                             coords=[('x', np.arange(3)),
        ...                                     ('y', np.arange(3))])
        >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
        ...                          coords=[('x', np.arange(3)),
        ...                                  ('y', np.arange(3)),
        ...                                  ('member', np.arange(3))])
        >>> rank_histogram(observations, forecasts, dim='x')
        <xarray.DataArray 'histogram_rank' (y: 3, rank: 4)>
        array([[0, 1, 1, 1],
               [0, 1, 0, 2],
               [1, 0, 1, 1]])
        Coordinates:
          * y        (y) int64 0 1 2
          * rank     (rank) float64 1.0 2.0 3.0 4.0

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    def _rank_first(x, y):
        """ Concatenates x and y and returns the rank of the first element along the last axes """
        xy = np.concatenate((x[..., np.newaxis], y), axis=-1)
        return bn.nanrankdata(xy, axis=-1)[..., 0]

    if dim is not None:
        if len(dim) == 0:
            raise ValueError(
                'At least one dimension must be supplied to compute rank histogram over'
            )
        if member_dim in dim:
            raise ValueError(f'"{member_dim}" cannot be specified as an input to dim')

    ranks = xr.apply_ufunc(
        _rank_first,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        dask='parallelized',
        output_dtypes=[int],
    )

    bin_edges = np.arange(0.5, len(forecasts[member_dim]) + 2)
    return histogram(
        ranks, bins=[bin_edges], bin_names=['rank'], dim=dim, bin_dim_suffix=''
    )


def discrimination(
    observations,
    forecasts,
    dim=None,
    probability_bin_edges=np.linspace(-1 / 8, 1 + 1 / 8, 6),
):
    """Returns the data required to construct the discrimination diagram for an event; the \
            histogram of forecasts likelihood when observations indicate an event has occurred \
            and has not occurred.

        Parameters
        ----------
        observations : xarray.Dataset or xarray.DataArray
            The observations or set of observations of the event. Data should be boolean (True for event
            occurance, False for non-occurance).
        forecasts : xarray.Dataset or xarray.DataArray
            The forecast likelihoods of the event. Data should be between 0 and 1.
        dim : str or list of str, optional
            Dimension(s) over which to compute the histograms
            Defaults to None meaning compute over all dimensions.
        probability_bin_edges : array_like, optional
            Probability bin edges (right edge inclusive) used to compute the histograms. Defaults to 6 \
            equally spaced edges between -0.125 and 1.125 (i.e. bin centres at [0, 0.25, 0.5, 0.75, 1])

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Histogram of forecast probabilities when the event was observed
        xarray.Dataset or xarray.DataArray
            Histogram of forecast probabilities when the event was not observed

        Examples
        --------
        >>> observations = xr.DataArray(np.random.normal(size=(30,30)),
        ...                             coords=[('x', np.arange(30)),
        ...                                     ('y', np.arange(30))])
        >>> forecasts = xr.DataArray(np.random.normal(size=(30,30,10)),
        ...                          coords=[('x', np.arange(30)),
        ...                                  ('y', np.arange(30)),
        ...                                  ('member', np.arange(10))])
        >>> forecast_event_likelihood = (forecasts > 0).mean('member')
        >>> observed_event = observations > 0
        >>> hist_event, hist_no_event = discrimination(observed_event, forecast_event_likelihood, dim=['x','y'])

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    if dim is not None:
        if len(dim) == 0:
            raise ValueError(
                'At least one dimension must be supplied to compute rank histogram over'
            )

    hist_event = histogram(
        forecasts.where(observations),
        bins=[probability_bin_edges],
        bin_names=['forecast_probability'],
        bin_dim_suffix='',
        dim=dim,
    ) / (observations).sum(dim=dim)

    hist_no_event = histogram(
        forecasts.where(xr.ufuncs.logical_not(observations)),
        bins=[probability_bin_edges],
        bin_names=['forecast_probability'],
        bin_dim_suffix='',
        dim=dim,
    ) / (xr.ufuncs.logical_not(observations)).sum(dim=dim)

    return hist_event, hist_no_event
