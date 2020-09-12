import warnings

import scipy.stats as st
import xarray as xr

from .deterministic import mae

metric_larger_is_better = (
    []
)  # no xskillscore metric that allow dim=[] # pearson_r not allowed for sign_test see DelSole and Tippett (2016)
metric_smaller_is_better = [
    'mae',
    'rmse',
    'median_absolute_error',
    'mse',
    'brier_score',
    'threshold_brier_score',
    'crps_ensemble',
    'median_absolute_error',
    'smape',
    'mape',
    'rps',
]


def sign_test(
    forecast1,
    forecast2,
    observation=None,
    dim=None,
    categorical=False,
    alpha=0.05,
    metric=None,
    orientation='negative',
):
    """
        Returns the Delsole and Tippett sign test over the given time dimension.

        Parameters
        ----------
        forecast1 : xarray.Dataset or xarray.DataArray
            forecast1 to be compared to observation
        forecast2 : xarray.Dataset or xarray.DataArray
            forecast2 to be compared to observation
        observation : xarray.Dataset or xarray.DataArray or None
            observation to be compared to both forecasts.
            If ``None``, then assume that forecast1 and forecast2 have already been
            compared to observation. Please adjust ``metric`` and ``orientation``
            accordingly.
        dim : str
            time dimension of dimension over which to compute the random walk.
            This dimension is not reduced, unlike in other xskillscore functions.
        alpha : float
            significance level for random walk.
        categorical : bool, optional
            If ``True``, the winning forecast is only rewarded a point if it exactly
            equals the observation. If False, use metric to compare forecast# with
            observation.
        metric : callable, optional
            metric to compare forecast# and observation if metric is not None. If
            metric is None, assume that forecast# have been compared observation before
            using ``sign_test``. If categorical is True, metric is ignored. Also allows
            strings to be convered to ``xskillscore.{metric}``. Defaults to None.
        orientation : str
            Which skill values correspond to better skill? Smaller values (negative) or
            Larger values (positive). Defaults to 'negative'. Ignored if metric is None
            or categorical.

        Returns
        -------
        xarray.DataArray or xarray.Dataset reduced by dim containing the sign test and
            confidence as new dimension results:

                * ``sign_test``: positive (negative) number shows how many times over
                    ``dim`` ``forecast1`` is better (worse) than ``forecast2``.
                * ``confidence``: Positive boundary for the random walk at significance
                    ``alpha``.


        Examples
        --------
        >>> f1 = xr.DataArray(np.random.normal(size=(30)),
        ...      coords=[('time', np.arange(30))])
        >>> f2 = xr.DataArray(np.random.normal(size=(30)),
        ...      coords=[('time', np.arange(30))])
        >>> o = xr.DataArray(np.random.normal(size=(30)),
        ...      coords=[('time', np.arange(30))])
        >>> st = sign_test(f1, f2, o, dim='time')
        >>> st.sel(results='sign_test').plot()
        >>> st.sel(results='confidence').plot(c='gray')
        >>> (-1*st.sel(results='confidence')).plot(c='gray')

        References
        ----------
            * DelSole, T., & Tippett, M. K. (2016). Forecast Comparison Based on Random
              Walks. Monthly Weather Review, 144(2), 615–626. doi: 10/f782pf
    """
    # make sure metric is a callable
    if metric is not None:
        if isinstance(metric, str):
            import xskillscore as xs

            if hasattr(xs, metric):
                metric = getattr(xs, metric)
            else:
                raise ValueError(
                    f'xskillscore.metric could not be derived from {metric}.'
                )
        elif not callable(metric):
            raise ValueError(
                f'metric needs to be a function/callable or None, found {type(metric)}'
            )

    if metric:
        if observation is not None:
            # Compare the forecasts and observation using metric
            diff1 = metric(forecast1, observation, dim=[])
            diff2 = metric(forecast2, observation, dim=[])
        else:
            raise ValueError(
                'observations must be provided when metric is provided', UserWarning
            )
    else:  # if metric=None, already evaluated
        if observation is not None:
            warnings.warn(
                'Ignoring provided observation because no metric was provided',
                UserWarning,
            )
        if orientation == 'negative':
            diff1 = forecast1
            diff2 = forecast2
        elif orientation == 'positive':
            diff1 = -forecast1
            diff2 = -forecast2
        if orientation not in ['negative', 'positive']:
            raise ValueError(
                '`orientation` requires to be either "positive" or'
                f'"negative"], found {orientation}.'
            )

    if categorical:  # ignores orientation and warns if metric provided
        if metric:
            warnings.warn('Ignoring provided metric because categorical=True')
        if observation is not None:
            diff1 = -1 * (forecast1 == observation)
            diff2 = -1 * (forecast2 == observation)
        else:
            diff1 = ~forecast1
            diff2 = ~forecast2

    sign_test = (1 * (diff1 < diff2) - 1 * (diff2 < diff1)).cumsum(dim)

    # Estimate 95% confidence interval -----
    notnan = 1 * (diff1.notnull() & diff2.notnull())
    N = notnan.cumsum(dim)
    # z_alpha is the value at which the standardized cumulative Gaussian distributed exceeds alpha
    confidence = st.norm.ppf(1 - alpha / 2) * xr.ufuncs.sqrt(N)
    confidence.coords['alpha'] = alpha

    res = xr.concat([sign_test, confidence], dim='results')
    res['results'] = ['sign_test', 'confidence']
    return res
