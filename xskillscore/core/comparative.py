import warnings

import numpy as np
import scipy.stats as st
import xarray as xr

from .deterministic import mae, pearson_r as pearson_r_func
from .utils import _add_as_coord


def sign_test(
    forecasts1,
    forecasts2,
    observations=None,
    time_dim="time",
    dim=[],
    alpha=0.05,
    metric=None,
    orientation="negative",
):
    """
    Returns the Delsole and Tippett sign test over the given time dimension.

    The sign test can be applied to a wide class of measures of forecast quality,
    including ordered (ranked) categorical data. It is independent of
    distributional assumptions about the forecast errors. This is different than
    alternative measures like correlation and mean square error, which assume that
    the metrics were computed from independent samples. However, skill metrics
    computed over a common period with a common set of observations are not
    independent. For example, different forecasts tend to bust for the same event.
    This procedure is equivalent to testing whether a coin is fair based on the
    frequency of heads. The null hypothesis is that the difference between the
    median scores is zero.

    Parameters
    ----------
    forecasts1 : xarray.Dataset or xarray.DataArray
        forecasts1 to be compared to observations
    forecasts2 : xarray.Dataset or xarray.DataArray
        forecasts2 to be compared to observations
    observations : xarray.Dataset or xarray.DataArray or None
        observation to be compared to both forecasts. Only used if ``metric`` is
        provided, otherwise it is assumed that both forecasts have already been
        compared to observations and this input is ignored. Please
        adjust ``orientation`` accordingly. Defaults to None.
    time_dim : str
        time dimension of dimension over which to compute the random walk.
        This dimension is not reduced, unlike in other xskillscore functions.
        Defaults to ``'time'``.
    dim : str or list of str
        dimensions to apply metric to if ``metric`` is provided. Cannot contain
        ``time_dim``. Ignored if ``metric`` is None. Defaults to [].
    alpha : float
        significance level for random walk.
    metric : callable, str, optional
        metric to compare forecast# with observations if ``metric`` is not None. If
        ``metric`` is None, assume that forecast# have been compared observations
        before using ``sign_test``. Make sure to adjust ``orientation`` if
        ``metric`` is None. Use ``metric=categorical``, if the winning forecast
        should only be rewarded a point if it exactly equals the observations. Also
        allows strings to be convered to ``xskillscore.{metric}``. Defaults to None.
    orientation : str
        One of [``'positive'``, ``'negative'``]. Which skill values correspond to
        better skill? Smaller values (``'negative'``) or larger values
        (``'positive'``)? Defaults to ``'negative'``.
        Ignored if ``metric== categorical``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset : Positive (negative) sign_test values shows
        how often ``forecast1`` is better (worse) than ``forecast2`` according to
        metric computed over ``dim``. A coordinate, confidence, is included showing
        the positive boundary for the random walk at significance level ``alpha``.


    Examples
    --------
    >>> f1 = xr.DataArray(np.random.normal(size=(30)),
    ...      coords=[('time', np.arange(30))])
    >>> f2 = xr.DataArray(np.random.normal(size=(30)),
    ...      coords=[('time', np.arange(30))])
    >>> o = xr.DataArray(np.random.normal(size=(30)),
    ...      coords=[('time', np.arange(30))])
    >>> st = sign_test(f1, f2, o, time_dim'time', metric='mae', orientation='negative')
    >>> st.plot()
    >>> st['confidence'].plot(color='gray')
    >>> (-1*st['confidence']).plot(color='gray')

    References
    ----------
        * DelSole, T., & Tippett, M. K. (2016). Forecast Comparison Based on Random
          Walks. Monthly Weather Review, 144(2), 615–626. doi: 10/f782pf
    """

    def _categorical_metric(observations, forecasts, dim):
        """Returns True where forecasts exactly equals observations"""
        return observations == forecasts

    if orientation not in ["negative", "positive"]:
        raise ValueError(
            '`orientation` requires to be either "positive" or'
            f'"negative"], found {orientation}.'
        )

    if isinstance(dim, str):
        dim = [dim]
    if time_dim in dim:
        raise ValueError("`dim` cannot contain `time_dim`")

    if metric is not None:
        # make sure metric is a callable
        if isinstance(metric, str):
            if metric == "categorical":
                metric = _categorical_metric
                if orientation != "positive":
                    warnings.warn(
                        'Changing to "orientation=positive" \
                            for consistency with "metric=categorical"',
                        UserWarning,
                    )
                orientation = "positive"
            else:
                import xskillscore as xs

                if hasattr(xs, metric):
                    metric = getattr(xs, metric)
                else:
                    raise ValueError(f"xskillscore.{metric} does not exist.")
        elif not callable(metric):
            raise ValueError(
                f'metric needs to be a function/callable, string ["categorical", '
                f"xskillscore.{{metric}}] or None, found {type(metric)}"
            )
        if observations is not None:
            # Compare the forecasts and observations using metric
            metric_f1o = metric(observations, forecasts1, dim=dim)
            metric_f2o = metric(observations, forecasts2, dim=dim)
        else:
            raise ValueError("observations must be provided when metric is provided")

    else:  # if metric=None, already evaluated
        if observations is not None:
            warnings.warn(
                "Ignoring provided observations because no metric was provided",
                UserWarning,
            )
        metric_f1o = forecasts1
        metric_f2o = forecasts2

    # Adjust for orientation of metric
    if orientation == "positive":
        if metric == _categorical_metric:
            metric_f1o = ~metric_f1o
            metric_f2o = ~metric_f2o
        else:
            metric_f1o = -metric_f1o
            metric_f2o = -metric_f2o

    walk = (1 * (metric_f1o < metric_f2o) - 1 * (metric_f2o < metric_f1o)).cumsum(
        time_dim
    )

    # Estimate 1 - alpha confidence interval -----
    notnan = 1 * (metric_f1o.notnull() & metric_f2o.notnull())
    N = notnan.cumsum(time_dim)
    # z_alpha is the value at which the standardized cumulative Gaussian distributed
    # exceeds alpha
    confidence = st.norm.ppf(1 - alpha / 2) * xr.ufuncs.sqrt(N)
    walk.coords["alpha"] = alpha
    walk = _add_as_coord(walk, confidence, "confidence")
    return walk


def mae_test(
    forecasts1,
    forecasts2,
    observations=None,
    pearson_r=None,
    dim=[],
    time_dim="time",
    alpha=0.05,
):
    """
    Returns the Jolliffe and Ebert MAE significance test.

    https://www.cawcr.gov.au/projects/verification/CIdiff/FAQ-CIdiff.html

    Parameters
    ----------
    forecasts1 : xarray.Dataset or xarray.DataArray
        forecasts1 to be compared to observations
    forecasts2 : xarray.Dataset or xarray.DataArray
        forecasts2 to be compared to observations
    observations : xarray.Dataset or xarray.DataArray or None
        observations to be compared to both forecasts. Defaults to None.
    pearson_r : xarray.Dataset or xarray.DataArray or None
        if observations is None use pearson_r.
    time_dim : str
        time dimension of dimension over which to compute the random walk.
        This dimension is not reduced, unlike in other xskillscore functions.
        Defaults to ``'time'``.
    dim : str or list of str
        dimensions to apply metric to if ``metric`` is provided. Cannot contain
        ``time_dim``. Ignored if ``metric`` is None. Defaults to [].
    alpha : float or 'return_p'
        significance level or return the p-value that forecast1 is not better than
        forecast2.

    Returns
    -------
    xarray.DataArray or xarray.Dataset : Difference in xs.mae and half-width of the
        confidence interval at the level ``alpha`` if ``alpha`` is given or the p-value
        that forecast1 is not better than forecast2 if ``alpha=return_p``.


    Examples
    --------
    >>> f1 = xr.DataArray(np.random.normal(size=(30)),
    ...      coords=[('time', np.arange(30))])
    >>> f2 = xr.DataArray(np.random.normal(size=(30)),
    ...      coords=[('time', np.arange(30))])
    >>> o = xr.DataArray(np.random.normal(size=(30)),
    ...      coords=[('time', np.arange(30))])
    >>> res = mae_test(f1, f2, o, time_dim='time', dim=[])

    References
    ----------
        * https://www.cawcr.gov.au/projects/verification/CIdiff/FAQ-CIdiff.html
    """

    if isinstance(dim, str):
        dim = [dim]

    if time_dim in dim:
        raise ValueError("`dim` cannot contain `time_dim`")

    msg = f"`alpha` must be between 0 and 1 or `return_p`, found {alpha}"
    if isinstance(alpha, str):
        if alpha != "return_p":
            raise ValueError(msg)
    elif isinstance(alpha, float):
        if alpha >= 1 or alpha <= 0:
            raise ValueError(msg)
    else:
        raise ValueError(msg)

    if observations is not None and pearson_r is None:
        # Compare the forecasts and observations using metric
        mae_f1o = mae(observations, forecasts1, dim=dim)
        mae_f2o = mae(observations, forecasts2, dim=dim)
        pearson_r_f1f2 = pearson_r_func(mae_f1o, mae_f2o, dim=time_dim)
    elif observations is None and pearson_r is not None:
        mae_f1o = forecasts1
        mae_f2o = forecasts2
        pearson_r_f1f2 = pearson_r
    else:
        raise ValueError("Either provide observations or pearson_r.")

    # diff mae
    diff = np.abs(mae_f1o - mae_f2o).mean(time_dim)

    notnan = 1 * (mae_f1o.notnull() & mae_f2o.notnull())
    N = notnan.sum(time_dim)
    # average variances and take square root instead of averaging standard deviations
    std = ((mae_f1o.var(time_dim) + mae_f2o.var(time_dim)) / 2) ** 0.5

    if not isinstance(alpha, str):
        confidence = st.norm.ppf(1 - alpha / 2)
        # half width of the confidence interval
        hwci = (2 * (1 - pearson_r_f1f2)) ** 0.5 * confidence * std / N ** 0.5
        ret = xr.concat([diff, hwci], "results")
        ret["alpha"] = alpha
        ret["results"] = ["diff", "hwci"]
        return ret

    elif alpha == "return_p":

        def _cdf(ds):
            return xr.apply_ufunc(st.norm.cdf, ds, dask="parallelized")

        alpha = 2 * (
            1 - _cdf(diff / (2 * (1 - pearson_r_f1f2)) ** 0.5 / std * N ** 0.5)
        )
        ret = xr.concat([diff, alpha], "results")
        ret["results"] = ["diff", "alpha"]
        return ret
