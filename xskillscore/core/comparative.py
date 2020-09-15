import warnings

import scipy.stats as st
import xarray as xr


def sign_test(
    observation,
    forecast1,
    forecast2,
    dim=None,
    alpha=0.05,
    metric=None,
    orientation='negative',
):
    """
        Returns the Delsole and Tippett sign test over the given time dimension.

        Parameters
        ----------
        observation : xarray.Dataset or xarray.DataArray or None
            observation to be compared to both forecasts.
            If ``None``, then assume that forecast1 and forecast2 have already been
            compared to observation. If metric is None, this assumes forecasts to be
            already compared to observation before and ignores observation. Please
            adjust ``orientation`` accordingly.
        forecast1 : xarray.Dataset or xarray.DataArray
            forecast1 to be compared to observation
        forecast2 : xarray.Dataset or xarray.DataArray
            forecast2 to be compared to observation
        dim : str
            time dimension of dimension over which to compute the random walk.
            This dimension is not reduced, unlike in other xskillscore functions.
        alpha : float
            significance level for random walk.
        metric : callable, optional
            metric to compare forecast# with observation if metric is not None. If
            metric is None, assume that forecast# have been compared observation before
            using ``sign_test`` and adjust ``orientation``.
            Use ``metric=categorical``, if the winning forecast should only be rewarded
            a point if it exactly equals the observation. Also allows strings to be
            convered to ``xskillscore.{metric}``. Defaults to None.
        orientation : str
            Which skill values correspond to better skill? Smaller values (negative) or
            Larger values (positive). Defaults to 'negative'. Ignored if
            ``metric== categorical``.

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
        >>> st = sign_test(f1, f2, o, dim='time', metric='mae', orientation='negative')
        >>> st.sel(results='sign_test').plot()
        >>> st.sel(results='confidence').plot(c='gray')
        >>> (-1*st.sel(results='confidence')).plot(c='gray')

        References
        ----------
            * DelSole, T., & Tippett, M. K. (2016). Forecast Comparison Based on Random
              Walks. Monthly Weather Review, 144(2), 615–626. doi: 10/f782pf
    """

    def _categorical_metric(observations, forecasts, dim):
        """Returns True where forecasts exactly equals observations"""
        return observations == forecasts

    if metric is not None:
        # make sure metric is a callable
        if isinstance(metric, str):
            if metric == 'categorical':
                metric = _categorical_metric
                if orientation != 'positive':
                    warnings.warn(
                        'Changing to "orientation=positive" for consistency with "metric=categorical"',
                        UserWarning,
                    )
                orientation = 'positive'
            else:
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
        if observation is not None:
            # Compare the forecasts and observation using metric
            metric_f1o = metric(observation, forecast1, dim=[])
            metric_f2o = metric(observation, forecast2, dim=[])
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
        metric_f1o = forecast1
        metric_f2o = forecast2

    # Adjust for orientation of metric
    if orientation == 'positive':
        if metric == _categorical_metric:
            metric_f1o = ~metric_f1o
            metric_f2o = ~metric_f2o
        else:
            metric_f1o = -metric_f1o
            metric_f2o = -metric_f2o
    elif orientation not in ['negative', 'positive']:
        raise ValueError(
            '`orientation` requires to be either "positive" or'
            f'"negative"], found {orientation}.'
        )

    sign_test = (1 * (metric_f1o < metric_f2o) - 1 * (metric_f2o < metric_f1o)).cumsum(
        dim
    )

    # Estimate 95% confidence interval -----
    notnan = 1 * (metric_f1o.notnull() & metric_f2o.notnull())
    N = notnan.cumsum(dim)
    # z_alpha is the value at which the standardized cumulative Gaussian distributed exceeds alpha
    confidence = st.norm.ppf(1 - alpha / 2) * xr.ufuncs.sqrt(N)
    confidence.coords['alpha'] = alpha

    res = xr.concat([sign_test, confidence], dim='results')
    res['results'] = ['sign_test', 'confidence']
    return res
