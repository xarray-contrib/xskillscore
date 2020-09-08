import scipy.stats as st
import xarray as xr


def sign_test(forecast1, forecast2, observation, dim=None, categorical=False, alpha=0.05):
    """
        Returns the Delsole and Tippett sign test over the given time dimension.

        Parameters
        ----------
        forecast1 : xarray DataArray
            Array containing data to be compared to forecast1
        forecast2 : xarray DataArray
            Array containing data to be compared to forecast2
        observation : xarray DataArray, str
            Array containing data to use as reference
        dim : str,
            Name of dimension over which to compute the random walk
        alpha : float
            significance level
        categorical : bool, optional
            If True, the winning forecast is only rewarded a point if it exactly equals the observations

        Returns
        -------
        sign_test : xarray DataArray
            Array containing the results of the sign test
        confidence : xarray DataArray
            Array containing 95% confidence bounds

        Examples
        --------
        >>> x = xr.DataArray(np.random.normal(size=(3,3)),
        ...                  coords=[('t', np.arange(3)), ('x', np.arange(3))])
        >>> y = xr.DataArray(np.random.normal(size=(3,3)),
        ...                 coords=[('t', np.arange(3)), ('x', np.arange(3))])
        >>> o = xr.DataArray(np.random.normal(size=(3,3)),
        ...                  coords=[('t', np.arange(3)), ('x', np.arange(3))])
        >>> st = sign_test(x, y, o, dim='t')

        Notes
        -----
        See Delsole and Tippett 2016 `Forecast Comparison Based on Random Walks`
    """
    # two shortcuts for climpred
    climpred_keys = ['negatively_oriented_already_evaluated','positively_oriented_already_evaluated','categorical_already_evaluated']
    if isinstance(observation,str):
        if observation=='negatively_oriented_already_evaluated': # mse, mae, rmse
            diff1 = forecast1
            diff2 = forecast2
        elif observation=='positively_oriented_already_evaluated': # 1-mse/std, msss
            diff1 = forecast1
            diff2 = forecast2
        elif observation=='categorical_already_evaluated':
            diff1 = ~forecast1
            diff2 = ~forecast2
        else:
            raise ValueError(f'special key not found in {climpred_keys}')
    else:
        if categorical:
            diff1 = -1 * (forecast1 == observation)
            diff2 = -1 * (forecast2 == observation)
        else:
            diff1 = abs(forecast1 - observation) # is like xs.mae(forecast1,observation,dim=[])
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
