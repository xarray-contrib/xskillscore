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
    observation,
    dim=None,
    categorical=False,
    alpha=0.05,
    metric=mae,
):
    """
        Returns the Delsole and Tippett sign test over the given time dimension.

        Parameters
        ----------
        forecast1 : xarray.Dataset or xarray.DataArray
            forecast1 to be compared to observation
        forecast2 : xarray.Dataset or xarray.DataArray
            forecast2 to be compared to observation
        observation : xarray.Dataset or xarray.DataArray
            observation to be compared to both forecasts

            if None, then assume that forecast1 and forecast2 have already been compared
            to observation. Decide comparison based on ``metric`` or choose from:

                * ``negatively_oriented_already_evaluated``: metric between forecast1
                  (forecast2) and observations. Distances are positively oriented,
                  therefore the smaller distance wins.
                * ``positively_oriented_already_evaluated``: metric between forecast1
                  (forecast2) and observations. The larger positively oriented metric
                  wins.

        dim : str
            time dimension of dimension over which to compute the random walk.
            This dimension is not reduced, unlike in other xskillscore functions.
        alpha : float
            significance level for random walk.
        categorical : bool, optional
            If True, the winning forecast is only rewarded a point if it exactly equals
            the observations
        metric : callable, optional
            metric to compare forecast# and observation or that has been used to
            compare forecast# and observation before using xs.sign_test.

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
    # TODO: account for forecast, observation ordering in climpred and observation, forecast ordering in xskillscore; do we need to change something? I think NO

    # make sure metric is a callable
    if isinstance(metric, str):
        import xskillscore as xs

        if hasattr(xs, metric):
            metric = getattr(xs, metric)
            metric_str = metric.__name__
    elif callable(metric):
        metric_str = metric.__name__
    else:
        metric_str = None

    if observation is not None:
        if categorical:
            diff1 = -1 * (forecast1 == observation)
            diff2 = -1 * (forecast2 == observation)
        else:
            if metric_str in metric_smaller_is_better:
                diff1 = metric(forecast1, observation, dim=[])
                diff2 = metric(forecast2, observation, dim=[])
            elif metric_str in metric_larger_is_better:
                if metric_larger_is_better == []:
                    raise ValueError(
                        'no metric which is better for larger values applicable in the sign test found.'
                    )
                diff1 = metric(forecast1, observation, dim=[])
                diff2 = metric(forecast2, observation, dim=[])
            else:
                raise ValueError('dont know how to compare')
    else:
        # shortcuts for climpred
        climpred_keys = [
            'negatively_oriented_already_evaluated',
            'positively_oriented_already_evaluated',
        ]
        if categorical:
            diff1 = ~forecast1
            diff2 = ~forecast2
        else:
            if (
                metric_str in metric_smaller_is_better
                or observation == 'negatively_oriented_already_evaluated'
            ):
                # mse, mae, rmse
                diff1 = forecast1
                diff2 = forecast2
            elif (
                metric_str in metric_larger_is_better
                or observation == 'positively_oriented_already_evaluated'
            ):  # 1-mse/std, msss, correlation not applied to time dimension
                diff1 = -forecast1
                diff2 = -forecast2
            else:
                raise ValueError(
                    f'special key not found in {climpred_keys}, or metric not found in xskillscore, found metric {metric}.'
                )

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
