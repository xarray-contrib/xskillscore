import dask.array as darray
import bottleneck as bn
import numpy as np
import properscoring
import xarray as xr

from .utils import _get_bin_centers, _preprocess_dims, _stack_input_if_needed, histogram

__all__ = [
    'brier_score',
    'crps_ensemble',
    'crps_gaussian',
    'crps_quadrature',
    'threshold_brier_score',
    'rank_histogram',
    'discrimination',
    'reliability',
]

FORECAST_PROBABILITY_DIM = 'forecast_probability'

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
        properscoring.crps_gaussian,
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


def crps_quadrature(
    x,
    cdf_or_dist,
    xmin=None,
    xmax=None,
    tol=1e-6,
    dim=None,
    weights=None,
    keep_attrs=False,
):
    """Continuous Ranked Probability Score with numerical integration of the normal distribution.

    Parameters
    ----------
    x : xarray.Dataset or xarray.DataArray
        Observations associated with the forecast distribution ``cdf_or_dist``.
    cdf_or_dist : callable or scipy.stats.distribution
        Function which returns the cumulative density of the forecast
        distribution at value x.
    xmin, xmax, tol: see properscoring.crps_quadrature
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``crps_quadrature``.
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
        properscoring.crps_quadrature,
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


def crps_ensemble(
    observations,
    forecasts,
    member_weights=None,
    issorted=False,
    member_dim='member',
    dim=None,
    weights=None,
    keep_attrs=False,
):
    """Continuous Ranked Probability Score with the ensemble distribution

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
        properscoring.crps_ensemble,
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


def brier_score(observations, forecasts, dim=None, weights=None, keep_attrs=False):
    """Calculate Brier score (BS).

    ..math:
        BS(p, k) = (p_1 - k)^2

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations of the event. Data should be boolean or logical \
        (True or 1 for event occurance, False or 0 for non-occurance).
    forecasts : xarray.Dataset or xarray.DataArray
        The forecast likelihoods of the event. Data should be between 0 and 1.
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``brier_score``.
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
    properscoring.brier_score
    xarray.apply_ufunc

    References
    ----------
    Gneiting, Tilmann, and Adrian E Raftery. “Strictly Proper Scoring Rules,
      Prediction, and Estimation.” Journal of the American Statistical
      Association 102, no. 477 (March 1, 2007): 359–78.
      https://doi.org/10/c6758w.
    Brier, Glenn W. "VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY."
      Monthly Weather Review, 78(1): 1-3
      https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2
    """
    res = xr.apply_ufunc(
        properscoring.brier_score,
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


def threshold_brier_score(
    observations,
    forecasts,
    threshold,
    issorted=False,
    member_dim='member',
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

    See Also
    --------
    properscoring.threshold_brier_score
    xarray.apply_ufunc

    References
    ----------
    Gneiting, T. and Ranjan, R. Comparing density forecasts using threshold-
      and quantile-weighted scoring rules. J. Bus. Econ. Stat. 29, 411-422
      (2011). http://www.stat.washington.edu/research/reports/2008/tr533.pdf
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
        properscoring.threshold_brier_score,
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
    probability_bin_edges=np.linspace(0, 1 + 1e-8, 6),
):
    """Returns the data required to construct the discrimination diagram for an event; the \
            histogram of forecasts likelihood when observations indicate an event has occurred \
            and has not occurred.

        Parameters
        ----------
        observations : xarray.Dataset or xarray.DataArray
            The observations or set of observations of the event. Data should be boolean or logical \
            (True or 1 for event occurance, False or 0 for non-occurance).
        forecasts : xarray.Dataset or xarray.DataArray
            The forecast likelihoods of the event. Data should be between 0 and 1.
        dim : str or list of str, optional
            Dimension(s) over which to compute the histograms
            Defaults to None meaning compute over all dimensions.
        probability_bin_edges : array_like, optional
            Probability bin edges used to compute the histograms. Bins include the left most edge, \
            but not the right. Defaults to 6 equally spaced edges between 0 and 1+1e-8

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
                'At least one dimension must be supplied to compute discrimination over'
            )

    hist_event = histogram(
        forecasts.where(observations),
        bins=[probability_bin_edges],
        bin_names=[FORECAST_PROBABILITY_DIM],
        bin_dim_suffix='',
        dim=dim,
    ) / (observations).sum(dim=dim)

    hist_no_event = histogram(
        forecasts.where(xr.ufuncs.logical_not(observations)),
        bins=[probability_bin_edges],
        bin_names=[FORECAST_PROBABILITY_DIM],
        bin_dim_suffix='',
        dim=dim,
    ) / (xr.ufuncs.logical_not(observations)).sum(dim=dim)

    return hist_event, hist_no_event


def reliability(
    observations,
    forecasts,
    dim=None,
    probability_bin_edges=np.linspace(0, 1 + 1e-8, 6),
    keep_attrs=False,
)
    """Returns the data required to construct the reliability diagram for an event; the relative frequencies \
            of occurrence of an event for a range of forecast probability bins
        Parameters
        ----------
        observations : xarray.Dataset or xarray.DataArray
            The observations or set of observations of the event. Data should be boolean or logical \
            (True or 1 for event occurance, False or 0 for non-occurance).
        forecasts : xarray.Dataset or xarray.DataArray
            The forecast likelihoods of the event. Data should be between 0 and 1.
        dim : str or list of str, optional
            Dimension(s) over which to compute the histograms
            Defaults to None meaning compute over all dimensions.
        probability_bin_edges : array_like, optional
            Probability bin edges used to compute the reliability. Bins include the left most edge, \
            but not the right. Defaults to 6 equally spaced edges between 0 and 1+1e-8
        keep_attrs : bool
            If True, the attributes (attrs) will be copied
            from the first input to the new one.
            If False (default), the new object will
            be returned without attributes.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The relative frequency of occurrence for each probability bin
        xarray.Dataset or xarray.DataArray
            The sample size of each probability bin

        Examples
        --------
        >>> forecasts = xr.DataArray(np.random.normal(size=(3,3,3)),
        ...                          coords=[('x', np.arange(3)),
        ...                                  ('y', np.arange(3)),
        ...                                  ('ensemble', np.arange(3))])
        >>> observations = xr.DataArray(np.random.normal(size=(3,3)),
        ...                            coords=[('x', np.arange(3)),
        ...                                    ('y', np.arange(3))])
        >>> rel, samples = reliability(observations > 0.1, (forecasts > 0.1).mean('ensemble'), dim='x')

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    def _reliability(o, f, bin_edges):
        """Return the reliability and number of samples per bin
        """
        # I couldn't get dask='parallelized' working in this case so dealing with dask arrays explicitly
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
            # Follow numpy: all but the last (righthand-most) bin is half-open
            # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
            if (i + 1) == len(bin_edges):
                f_in_bin = (f >= bin_edges[i]) & (f <= bin_edges[i + 1])
            else:
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

    # Compute over all dims if dim is None
    if dim is None:
        dim = observations.dims

    dim, _ = _preprocess_dims(dim)
    observations, forecasts, stack_dim, _ = _stack_input_if_needed(
        observations, forecasts, dim, weights=None
    )

    rel, samp = xr.apply_ufunc(
        _reliability,
        observations,
        forecasts,
        probability_bin_edges,
        input_core_dims=[[stack_dim], [stack_dim], []],
        dask='allowed',
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

    return rel, samp
