import scipy.stats as st
import xarray as xr


def sign_test(
    forecast1, forecast2, observation, dim=None, categorical=False, alpha=0.05
):
    """
        Returns the Delsole and Tippett sign test over the given time dimension.

        Parameters
        ----------
        forecast1 : xarray DataArray, Dataset
            containing data to be compared to forecast1
        forecast2 : xarray DataArray, Dataset
            containing data to be compared to forecast2
        observation : xarray DataArray, Dataset or str
            containing data to use as observation
            if str, then assume that comparison of forecast1 and forecast2 with
            observations has already been done and choose str from:

                - negatively_oriented_already_evaluated: metric between forecast1
                  (forecast2) and observations. Distances are positively oriented,
                  therefore the smaller distance wins.
                - positively_oriented_already_evaluated: metric between forecast1
                  (forecast2) and observations. The larger positively oriented metric
                  wins.
                - categorical_already_evaluated: categorical data following
                  ``logical(forecast1)==logical(forecast2)`` where ``logical`` is a
                  function return binary output

        dim : str
            time dimension of dimension over which to compute the random walk.
            This dimension is not reduced, unlike in other xskillscore functions.
        alpha : float
            significance level for random walk.
        categorical : bool, optional
            If True, the winning forecast is only rewarded a point if it exactly equals
            the observations

        Returns
        -------
        xarray DataArray, Dataset reduced by dim containing the sign test and
            confidence.


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
        >>> st.sel(results='confidence').plot(c='gray')

        References
        ----------
        * DelSole, T., & Tippett, M. K. (2016). Forecast Comparison Based on Random
          Walks. Monthly Weather Review, 144(2), 615–626. doi: 10/f782pf
    """
    # two shortcuts for climpred
    climpred_keys = [
        'negatively_oriented_already_evaluated',
        'positively_oriented_already_evaluated',
        'categorical_already_evaluated',
    ]
    if isinstance(observation, str):
        if observation == 'negatively_oriented_already_evaluated':  # mse, mae, rmse
            diff1 = forecast1
            diff2 = forecast2
        elif observation == 'positively_oriented_already_evaluated':  # 1-mse/std, msss
            diff1 = forecast1
            diff2 = forecast2
        elif observation == 'categorical_already_evaluated':
            diff1 = ~forecast1
            diff2 = ~forecast2
        else:
            raise ValueError(f'special key not found in {climpred_keys}')
    else:
        if categorical:
            diff1 = -1 * (forecast1 == observation)
            diff2 = -1 * (forecast2 == observation)
        else:
            diff1 = abs(
                forecast1 - observation
            )  # is like xs.mae(forecast1,observation,dim=[])
            diff2 = abs(forecast2 - observation)

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
