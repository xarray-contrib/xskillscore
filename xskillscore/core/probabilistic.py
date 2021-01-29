import bottleneck as bn
import dask.array as darray
import numpy as np
import properscoring
import xarray as xr

from .contingency import Contingency
from .utils import (
    _add_as_coord,
    _fail_if_dim_empty,
    _get_bin_centers,
    _preprocess_dims,
    _stack_input_if_needed,
    histogram,
)

__all__ = [
    "brier_score",
    "crps_ensemble",
    "crps_gaussian",
    "crps_quadrature",
    "threshold_brier_score",
    "rank_histogram",
    "discrimination",
    "reliability",
    "rps",
    "roc",
]

FORECAST_PROBABILITY_DIM = "forecast_probability"


def crps_gaussian(observations, mu, sig, dim=None, weights=None, keep_attrs=False):
    """Continuous Ranked Probability Score with a Gaussian distribution.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    mu : xarray.Dataset or xarray.DataArray
        The mean of the forecast normal distribution.
    sig : xarray.Dataset or xarray.DataArray
        The standard deviation of the forecast distribution.
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``crps_gaussian``.
        Defaults to None implying averaging over all dimensions.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`.
        Defaults to None, such that no weighting is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray reduced by dimension dim

    Examples
    --------
    >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
    ...                             coords=[('x', np.arange(3)),
    ...                                     ('y', np.arange(3))])
    >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
    ...                          coords=[('x', np.arange(3)),
    ...                                  ('y', np.arange(3)),
    ...                                  ('member', np.arange(3))])
    >>> mu = forecasts.mean('member')
    >>> sig = forecasts.std('member')
    >>> crps_gaussian(observations, mu, sig, dim='x')
    <xarray.DataArray (y: 3)>
    array([1.0349773 , 0.36521376, 0.39017126])
    Coordinates:
      * y        (y) int64 0 1 2

    See Also
    --------
    properscoring.crps_gaussian
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
        properscoring.crps_gaussian,
        observations,
        mu,
        sig,
        input_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


def crps_quadrature(
    observations,
    cdf_or_dist,
    xmin=None,
    xmax=None,
    tol=1e-6,
    dim=None,
    weights=None,
    keep_attrs=False,
):
    """Continuous Ranked Probability Score with numerical integration
    of the normal distribution.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        Observations associated with the forecast distribution ``cdf_or_dist``.
    cdf_or_dist : callable or scipy.stats.distribution
        Function which returns the cumulative density of the forecast
        distribution at value x.
    xmin, xmax, tol: see properscoring.crps_quadrature
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``crps_quadrature``.
        Defaults to None implying averaging over all dimensions.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`.
        Defaults to None, such that no weighting is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    Examples
    --------
    >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
    ...                             coords=[('x', np.arange(3)),
    ...                                     ('y', np.arange(3))])
    >>> from scipy.stats import norm
    >>> crps_quadrature(observations, norm)
    <xarray.DataArray (y: 3)>
    array([0.80280921, 0.31818197, 0.32364912])
    Coordinates:
      * y        (y) int64 0 1 2

    See Also
    --------
    properscoring.crps_quadrature
    """
    res = xr.apply_ufunc(
        properscoring.crps_quadrature,
        observations,
        cdf_or_dist,
        xmin,
        xmax,
        tol,
        input_core_dims=[[], [], [], [], []],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


def crps_ensemble(
    observations,
    forecasts,
    member_weights=None,
    issorted=False,
    member_dim="member",
    dim=None,
    weights=None,
    keep_attrs=False,
):
    """Continuous Ranked Probability Score with the ensemble distribution.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        Forecast with required member dimension ``member_dim``.
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
        Dimension over which to compute mean after computing ``crps_ensemble``.
        Defaults to None implying averaging over all dimensions.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`.
        Defaults to None, such that no weighting is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    Examples
    --------
    >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
    ...                             coords=[('x', np.arange(3)),
    ...                                     ('y', np.arange(3))])
    >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
    ...                          coords=[('x', np.arange(3)),
    ...                                  ('y', np.arange(3)),
    ...                                  ('member', np.arange(3))])
    >>> crps_ensemble(observations, forecasts, dim='x')
    <xarray.DataArray (y: 3)>
    array([1.04497153, 0.48997746, 0.47994095])
    Coordinates:
      * y        (y) int64 0 1 2

    See Also
    --------
    properscoring.crps_ensemble
    """
    res = xr.apply_ufunc(
        properscoring.crps_ensemble,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        kwargs={"axis": -1, "issorted": issorted, "weights": member_weights},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


def brier_score(observations, forecasts, dim=None, weights=None, keep_attrs=False):
    """Calculate Brier score (BS).

    .. math:
        BS(p, k) = (p_1 - k)^{2}

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations of the event.
        Data should be boolean or logical \
        (True or 1 for event occurance, False or 0 for non-occurance).
    forecasts : xarray.Dataset or xarray.DataArray
        The forecast likelihoods of the event. Data should be between 0 and 1.
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``brier_score``.
        Defaults to None implying averaging over all dimensions.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`.
        Defaults to None, such that no weighting is applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    Examples
    --------
    >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
    ...                             coords=[('x', np.arange(3)),
    ...                                     ('y', np.arange(3))])
    >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
    ...                          coords=[('x', np.arange(3)),
    ...                                  ('y', np.arange(3)),
    ...                                  ('member', np.arange(3))])
    >>> brier_score(observations > .5, (forecasts > .5).mean('member'))
    <xarray.DataArray (y: 3)>
    array([0.51851852, 0.14814815, 0.37037037])
    Coordinates:
      * y        (y) int64 0 1 2

    See Also
    --------
    properscoring.brier_score

    References
    ----------
    * Gneiting, Tilmann, and Adrian E Raftery. “Strictly Proper Scoring Rules,
      Prediction, and Estimation.” Journal of the American Statistical
      Association 102, no. 477 (March 1, 2007): 359–78.
      https://doi.org/10/c6758w.
    * Brier, Glenn W. "VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY."
      Monthly Weather Review, 78(1): 1-3
      https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2

    """
    res = xr.apply_ufunc(
        properscoring.brier_score,
        observations,
        forecasts,
        input_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


def threshold_brier_score(
    observations,
    forecasts,
    threshold,
    issorted=False,
    member_dim="member",
    dim=None,
    weights=None,
    keep_attrs=False,
):
    """Calculate the Brier scores of an ensemble for exceeding given thresholds.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        Forecast with required member dimension ``member_dim``.
    threshold : scalar or 1d scalar
        Threshold values at which to calculate exceedence Brier scores.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    member_dim : str, optional
        Name of ensemble member dimension. By default, 'member'.
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``threshold_brier_score``.
        Defaults to None implying averaging over all dimensions.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no weighting is
        applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied from the first input to the new
        one. If False (default), the new object will be returned without attributes.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        (If ``threshold`` is a scalar, the result will have the same shape as
        observations. Otherwise, it will have an additional final dimension
        corresponding to the threshold levels. Not implemented yet.)

    Examples
    --------
    >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
    ...                             coords=[('x', np.arange(3)),
    ...                                     ('y', np.arange(3))])
    >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
    ...                          coords=[('x', np.arange(3)),
    ...                                  ('y', np.arange(3)),
    ...                                  ('member', np.arange(3))])
    >>> threshold = [.2, .5, .8]
    >>> threshold_brier_score(observations, forecasts, threshold)
    <xarray.DataArray (y: 3, threshold: 3)>
    array([[0.44444444, 0.51851852, 0.48148148],
           [0.18518519, 0.14814815, 0.03703704],
           [0.18518519, 0.37037037, 0.03703704]])
    Coordinates:
      * y          (y) int64 0 1 2
      * threshold  (threshold) int64 1 2 3

    See Also
    --------
    properscoring.threshold_brier_score

    References
    ----------
    Gneiting, T. and Ranjan, R. Comparing density forecasts using threshold-
      and quantile-weighted scoring rules. J. Bus. Econ. Stat. 29, 411-422
      (2011). http://www.stat.washington.edu/research/reports/2008/tr533.pdf
    """
    if isinstance(threshold, list):
        threshold.sort()
        threshold = xr.DataArray(threshold, dims="threshold")
        threshold["threshold"] = np.arange(1, 1 + threshold.threshold.size)

    if isinstance(threshold, (xr.DataArray, xr.Dataset)):
        if "threshold" not in threshold.dims:
            raise ValueError(
                "please provide threshold with threshold dim, found",
                threshold.dims,
            )
        input_core_dims = [[], [member_dim], ["threshold"]]
        output_core_dims = [["threshold"]]
    elif isinstance(threshold, (int, float)):
        input_core_dims = [[], [member_dim], []]
        output_core_dims = [[]]
    else:
        raise ValueError(
            "Please provide threshold as list, int, float \
            or xr.object with threshold dimension; found",
            type(threshold),
        )
    res = xr.apply_ufunc(
        properscoring.threshold_brier_score,
        observations,
        forecasts,
        threshold,
        input_core_dims=input_core_dims,
        kwargs={"axis": -1, "issorted": issorted},
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


def rps(
    observations,
    forecasts,
    category_edges,
    dim=None,
    weights=None,
    keep_attrs=False,
    member_dim="member",
):
    """Calculate Ranked Probability Score.

     .. math::
        RPS(p, k) = 1/M \\sum_{m=1}^{M}
        [(\\sum_{k=1}^{m} p_k) - (\\sum_{k=1}^{m} o_k)]^{2}

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations of the event.
    forecasts : xarray.Dataset or xarray.DataArray
        The forecasts for the event.
    category_edges : array_like
        Category bin edges used to compute the CDFs. Bins include the left most edge, \
        but not the right.
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``rps``.
        Defaults to None implying averaging over all dimensions.
    weights : xr.DataArray with dimensions from dim, optional
        Weights for `weighted.mean(dim)`. Defaults to None, such that no weighting is
        applied.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied from the first input to the new
        one. If False (default), the new object will be returned without attributes.
    member_dim : str, optional
        Name of ensemble member dimension. By default, 'member'.

    Returns
    -------
    xarray.Dataset or xarray.DataArray:
        ranked probability score

    Examples
    --------
    >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
    ...                             coords=[('x', np.arange(3)),
    ...                                     ('y', np.arange(3))])
    >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
    ...                          coords=[('x', np.arange(3)),
    ...                                  ('y', np.arange(3)),
    ...                                  ('member', np.arange(3))])
    >>> category_edges = np.array([.2, .5, .8])
    >>> rps(observations, forecasts, category_edges)
    <xarray.DataArray 'histogram_category' (y: 3)>
    array([1.        , 1.        , 0.33333333])
    Coordinates:
      * y        (y) int64 0 1 2

    References
    ----------
    https://www.cawcr.gov.au/projects/verification/verif_web_page.html#RPS
    """
    bin_names = ["category"]
    bin_dim = f"{bin_names[0]}_bin"
    # histogram(dim=[]) not allowed therefore add fake member dim
    # to apply over when multi-dim observations
    if len(observations.dims) == 1:
        observations = histogram(
            observations, bins=[category_edges], bin_names=bin_names, dim=None
        )
    else:
        observations = histogram(
            observations.expand_dims(member_dim),
            bins=[category_edges],
            bin_names=bin_names,
            dim=[member_dim],
        )
    forecasts = histogram(
        forecasts, bins=[category_edges], bin_names=bin_names, dim=[member_dim]
    )
    # normalize f.sum()=1
    # # can remove this once density=True
    # https://github.com/xgcm/xhistogram/pull/17
    forecasts = forecasts / forecasts.sum(bin_dim)
    observations = observations / observations.sum(bin_dim)
    # rps formula
    res = ((observations.cumsum(bin_dim) - forecasts.cumsum(bin_dim)) ** 2).sum(bin_dim)
    if weights is not None:
        res = res.weighted(weights)
    return res.mean(dim, keep_attrs=keep_attrs)


def rank_histogram(observations, forecasts, dim=None, member_dim="member"):
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
        """Concatenates x and y and returns the rank of the
        first element along the last axes"""
        xy = np.concatenate((x[..., np.newaxis], y), axis=-1)
        return bn.nanrankdata(xy, axis=-1)[..., 0]

    if dim is not None:
        if len(dim) == 0:
            raise ValueError(
                "At least one dimension must be supplied to compute rank histogram over"
            )
        if member_dim in dim:
            raise ValueError(f'"{member_dim}" cannot be specified as an input to dim')

    ranks = xr.apply_ufunc(
        _rank_first,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        dask="parallelized",
        output_dtypes=[int],
    )

    bin_edges = np.arange(0.5, len(forecasts[member_dim]) + 2)
    return histogram(
        ranks, bins=[bin_edges], bin_names=["rank"], dim=dim, bin_dim_suffix=""
    )


def discrimination(
    observations,
    forecasts,
    dim=None,
    probability_bin_edges=np.linspace(0, 1 + 1e-8, 6),
):
    """Returns the data required to construct the discrimination diagram for an event;
       the histogram of forecasts likelihood when observations indicate an event has
       occurred and has not occurred.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations of the event.
        Data should be boolean or logical \
        (True or 1 for event occurance, False or 0 for non-occurance).
    forecasts : xarray.Dataset or xarray.DataArray
        The forecast likelihoods of the event. Data should be between 0 and 1.
    dim : str or list of str, optional
        Dimension(s) over which to compute the histograms
        Defaults to None meaning compute over all dimensions.
    probability_bin_edges : array_like, optional
        Probability bin edges used to compute the histograms.
        Bins include the left most edge, \
        but not the right. Defaults to 6 equally spaced edges between 0 and 1+1e-8

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Array with added dimension "event" containing the histograms of
        forecast probabilities when the event was observed and not observed

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
    >>> disc = discrimination(observed_event,
    ...                       forecast_event_likelihood,
    ...                       dim=['x','y'])
    <xarray.DataArray (event: 2, forecast_probability: 5)>
    array([[0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])
    Coordinates:
      * forecast_probability  (forecast_probability) float64 0.1 0.3 0.5 0.7 0.9
      * event                 (event) bool True False



    References
    ----------
    http://www.cawcr.gov.au/projects/verification/
    """

    _fail_if_dim_empty(dim)

    hist_event = (
        histogram(
            forecasts.where(observations),
            bins=[probability_bin_edges],
            bin_names=[FORECAST_PROBABILITY_DIM],
            bin_dim_suffix="",
            dim=dim,
        )
        / (observations).sum(dim=dim)
    )

    hist_no_event = (
        histogram(
            forecasts.where(np.logical_not(observations)),
            bins=[probability_bin_edges],
            bin_names=[FORECAST_PROBABILITY_DIM],
            bin_dim_suffix="",
            dim=dim,
        )
        / (np.logical_not(observations)).sum(dim=dim)
    )

    return xr.concat([hist_event, hist_no_event], dim="event").assign_coords(
        {"event": [True, False]}
    )


def reliability(
    observations,
    forecasts,
    dim=None,
    probability_bin_edges=np.linspace(0, 1 + 1e-8, 6),
    keep_attrs=False,
):
    """Returns the data required to construct the reliability diagram for an event;
        the relative frequencies of occurrence of an event
        for a range of forecast probability bins

        Parameters
        ----------
        observations : xarray.Dataset or xarray.DataArray
            The observations or set of observations of the event.
            Data should be boolean or logical \
            (True or 1 for event occurance, False or 0 for non-occurance).
        forecasts : xarray.Dataset or xarray.DataArray
            The forecast likelihoods of the event. Data should be between 0 and 1.
        dim : str or list of str, optional
            Dimension(s) over which to compute the histograms
            Defaults to None meaning compute over all dimensions.
        probability_bin_edges : array_like, optional
            Probability bin edges used to compute the reliability.
            Bins include the left most edge, \
            but not the right. Defaults to 6 equally spaced edges between 0 and 1+1e-8
        keep_attrs : bool, optional
            If True, the attributes (attrs) will be copied from the first input to
            the new one.
            If False (default), the new object will be returned without attributes.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The relative frequency of occurrence for each probability bin

        Examples
        --------
        >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
        ...                          coords=[('x', np.arange(3)),
        ...                                  ('y', np.arange(3)),
        ...                                  ('member', np.arange(3))])
        >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
        ...                            coords=[('x', np.arange(3)),
        ...                                    ('y', np.arange(3))])
        >>> rel = reliability(observations > 0.1,
        ...                   (forecasts > 0.1).mean('member'),
        ...                   dim='x')
        <xarray.DataArray (y: 3, forecast_probability: 5)>
        array([[       nan, 1.        ,        nan,        nan,        nan],
               [0.        , 0.5       ,        nan,        nan,        nan],
               [       nan, 0.33333333,        nan,        nan,        nan]])
        Coordinates:
          * y                     (y) int64 0 1 2
          * forecast_probability  (forecast_probability) float64 0.1 0.3 0.5 0.7 0.9
            samples               (y, forecast_probability) float64 0.0 3.0 ... 0.0 0.0

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    def _reliability(o, f, bin_edges):
        """Return the reliability and number of samples per bin"""
        # I couldn't get dask='parallelized' working in this case
        # so dealing with dask arrays explicitly
        is_dask_array = isinstance(o, darray.core.Array) | isinstance(
            f, darray.core.Array
        )

        if is_dask_array:
            r = []
            N = []
        else:
            r = np.zeros((*o.shape[:-1], len(bin_edges) - 1), dtype=float)
            N = np.zeros_like(r)

        for i in range(len(bin_edges) - 1):
            # Follow xhistogram: all bins are half-open
            # https://github.com/xgcm/xhistogram/issues/18
            f_in_bin = (f >= bin_edges[i]) & (f < bin_edges[i + 1])
            o_f_in_bin = o & f_in_bin
            N_f_in_bin = f_in_bin.sum(axis=-1)
            N_o_f_in_bin = o_f_in_bin.sum(axis=-1)
            if is_dask_array:
                r.append(N_o_f_in_bin / N_f_in_bin)
                N.append(N_f_in_bin)
            else:
                r[..., i] = N_o_f_in_bin / N_f_in_bin
                N[..., i] = N_f_in_bin

        if is_dask_array:
            return (
                darray.stack(r, axis=-1).rechunk({-1: -1}),
                darray.stack(N, axis=-1).rechunk({-1: -1}),
            )
        else:
            return r, N

    _fail_if_dim_empty(dim)

    # Compute over all dims if dim is None
    if dim is None:
        dim = list(observations.dims)

    dim, _ = _preprocess_dims(dim, observations)
    observations, forecasts, stack_dim, _ = _stack_input_if_needed(
        observations, forecasts, dim, weights=None
    )

    rel, samp = xr.apply_ufunc(
        _reliability,
        observations,
        forecasts,
        probability_bin_edges,
        input_core_dims=[[stack_dim], [stack_dim], []],
        dask="allowed",
        output_core_dims=[[FORECAST_PROBABILITY_DIM], [FORECAST_PROBABILITY_DIM]],
        keep_attrs=keep_attrs,
    )

    # Add probability bin coordinate
    rel = rel.assign_coords(
        {FORECAST_PROBABILITY_DIM: _get_bin_centers(probability_bin_edges)}
    )
    samp = samp.assign_coords(
        {FORECAST_PROBABILITY_DIM: _get_bin_centers(probability_bin_edges)}
    )

    # Move samples to a coordinate
    return _add_as_coord(rel, samp, coordinate_suffix="samples")


def _drop_intermediate(fpr, tpr):
    """Attempt to drop thresholds corresponding to points in between and
    collinear with other points. These are always suboptimal and do not
    appear on a plotted ROC curve (and thus do not affect the AUC).
    Here xr.diff(_, 2) is used as a "second derivative" to tell if there is a
    corner at the point. (...) This keeps all cases where the point should be
    kept, but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    https://github.com/scikit-learn/scikit-learn/blob/42aff4e2edd8e8887478f6ff1628f27de97be6a3/sklearn/metrics/_ranking.py#L916
    """
    optimal_idxs = xr.concat(
        [
            fpr.isel(probability_bin=0, drop=False).astype("bool"),
            np.logical_or(
                fpr.diff("probability_bin", 2), tpr.diff("probability_bin", 2)
            ),
            fpr.isel(probability_bin=0, drop=False).astype("bool"),
        ],
        "probability_bin",
    )
    optimal_idxs["probability_bin"] = np.arange(optimal_idxs.probability_bin.size)
    if isinstance(optimal_idxs, xr.Dataset):
        optimal_idxs = optimal_idxs.to_array()
    optimal_idxs = optimal_idxs.where(optimal_idxs, drop=True).probability_bin.values
    tpr = tpr.isel(probability_bin=optimal_idxs)
    fpr = fpr.isel(probability_bin=optimal_idxs)
    return fpr, tpr


def _auc(fpr, tpr, dim="probability_bin"):
    """Get area under the curve with trapez method."""
    # reverse tpr, fpr to fpr, tpr, see numpy.trapz(y, x=None)
    area = xr.apply_ufunc(
        np.trapz, tpr, fpr, input_core_dims=[[dim], [dim]], dask="allowed"
    )
    area = np.abs(area)
    if ((area > 1)).any():
        area = np.clip(area, 0, 1)  # allow only values between 0 and 1
    return area


def roc(
    observations,
    forecasts,
    bin_edges="continuous",
    dim=None,
    drop_intermediate=False,
    return_results="area",
):
    """Computes the relative operating characteristic for a range of thresholds.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        If ``bin_edges=='continuous'``, observations are binary.
    forecasts : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        If ``bin_edges=='continuous'``, forecasts are probabilities.
    bin_edges : array_like, str, default='continuous'
        Bin edges for categorising observations and forecasts.
        Bins include the left most edge, but not the right. ``bin_edges`` will be
        sorted in ascending order. If ``bin_edges=='continuous'``, calculate
        ``bin_edges`` from forecasts, equal to
        ``sklearn.metrics.roc_curve(f_boolean, o_prob)``.
    dim : str, list
        The dimension(s) over which to compute the contingency table
    drop_intermediate : bool, default=False
        Whether to drop some suboptimal thresholds which would not appear on a plotted
        ROC curve. This is useful in order to create lighter ROC curves.
        Defaults to ``True`` in ``sklearn.metrics.roc_curve``.
    return_results: str, default='area'
        Specify how return is structed:

            - 'area': return only the ``area under curve`` of ROC

            - 'all_as_tuple': return ``true positive rate`` and ``false positive rate``
              at each bin and area under the curve of ROC as tuple

            - 'all_as_metric_dim': return ``true positive rate`` and
              ``false positive rate`` at each bin and ``area under curve`` of ROC
              concatinated into new ``metric`` dimension

    Returns
    -------
    xarray.Dataset or xarray.DataArray :
        reduced by dimensions ``dim``, see ``return_results`` parameter.
        ``true positive rate`` and ``false positive rate`` contain
        ``probability_bin`` dimension with ascending ``bin_edges`` as coordinates.


    Examples
    --------
    >>> f = xr.DataArray(
    ...     np.random.normal(size=(1000)),
            coords=[('time', np.arange(1000))]
    ... )
    >>> o = xr.DataArray(
    ...    np.random.normal(size=(1000)),
    ...    coords=[('time', np.arange(1000))]
    ... )
    >>> category_edges = np.linspace(-2, 2, 5)
    >>> roc(o, f, category_edges, dim=['time'])
    <xarray.DataArray 'histogram_observations_forecasts' ()>
    array(0.46812223)

    See also
    --------
    xskillscore.Contingency
    sklearn.metrics.roc_curve

    References
    ----------
    http://www.cawcr.gov.au/projects/verification/
    """

    if dim is None:
        dim = list(forecasts.dims)
    if isinstance(dim, str):
        dim = [dim]

    continuous = False
    if isinstance(bin_edges, str):
        if bin_edges == "continuous":
            continuous = True
            # check that o binary
            if isinstance(observations, xr.Dataset):
                o_check = observations.to_array()
            else:
                o_check = observations
            if str(o_check.dtype) != "bool":
                if not ((o_check == 0) | (o_check == 1)).all():
                    raise ValueError(
                        'Input "observations" must represent logical (True/False) outcomes',
                        o_check,
                    )

            # works only for 1var
            if isinstance(forecasts, xr.Dataset):
                varlist = list(forecasts.data_vars)
                if len(varlist) == 1:
                    v = varlist[0]
                else:
                    raise ValueError(
                        f"Only works for `xr.Dataset` with one variable, found {forecasts.data_vars}. Considering looping over `data_vars` or `.to_array()`."
                    )
                f_bin = forecasts[v]
            else:
                f_bin = forecasts
            f_bin = f_bin.stack(ndim=forecasts.dims)
            f_bin = f_bin.sortby(-f_bin)
            bin_edges = np.append(f_bin[0] + 1, f_bin)
            bin_edges = np.unique(bin_edges)  # ensure that in ascending order
        else:
            raise ValueError("If bin_edges is str, it can only be continuous.")
    else:
        bin_edges = np.sort(bin_edges)  # ensure that in ascending order

    # loop over each bin_edge and get true positive rate and false positive rate
    # from contingency
    tpr, fpr = [], []
    for i in bin_edges:
        dichotomous_category_edges = np.array(
            [-np.inf, i, np.inf]
        )  # "dichotomous" means two-category
        dichotomous_contingency = Contingency(
            observations,
            forecasts,
            dichotomous_category_edges,
            dichotomous_category_edges,
            dim=dim,
        )
        fpr.append(dichotomous_contingency.false_alarm_rate())
        tpr.append(dichotomous_contingency.hit_rate())
    tpr = xr.concat(tpr, "probability_bin")
    fpr = xr.concat(fpr, "probability_bin")
    tpr["probability_bin"] = bin_edges
    fpr["probability_bin"] = bin_edges

    fpr = fpr.fillna(1.0)
    tpr = tpr.fillna(0.0)

    # pad (0,0) and (1,1)
    fpr_pad = xr.concat(
        [
            xr.ones_like(fpr.isel(probability_bin=0, drop=False)),
            fpr,
            xr.zeros_like(fpr.isel(probability_bin=-1, drop=False)),
        ],
        "probability_bin",
    )
    tpr_pad = xr.concat(
        [
            xr.ones_like(tpr.isel(probability_bin=0, drop=False)),
            tpr,
            xr.zeros_like(tpr.isel(probability_bin=-1, drop=False)),
        ],
        "probability_bin",
    )

    if drop_intermediate and fpr.probability_bin.size > 2:

        fpr, tpr = _drop_intermediate(fpr, tpr)
        fpr_pad, tpr_pad = _drop_intermediate(fpr_pad, tpr_pad)

    area = _auc(fpr_pad, tpr_pad)

    if continuous:
        # sklearn returns in reversed order
        fpr = fpr.sortby(-fpr.probability_bin)
        tpr = tpr.sortby(-fpr.probability_bin)

    # mask always nan
    def _keep_masked(new, ori, dim):
        """Keep mask from `ori` deprived of dimensions from `dim` in input `new`."""
        isel_dim = {d: 0 for d in forecasts.dims if d in dim}
        mask = ori.isel(isel_dim, drop=True)
        new_masked = new.where(mask.notnull())
        return new_masked

    fpr = _keep_masked(fpr, forecasts, dim=dim)
    tpr = _keep_masked(tpr, forecasts, dim=dim)
    area = _keep_masked(area, forecasts, dim=dim)

    if return_results == "area":
        return area
    elif return_results == "all_as_metric_dim":
        results = xr.concat([fpr, tpr, area], "metric", coords="minimal")
        results["metric"] = [
            "false positive rate",
            "true positive rate",
            "area under curve",
        ]
        return results
    elif return_results == "all_as_tuple":
        return fpr, tpr, area
    else:
        raise NotImplementedError(
            f"expect `return_results` from [all_as_tuple, area, all_as_metric_dim], found {return_results}"
        )
