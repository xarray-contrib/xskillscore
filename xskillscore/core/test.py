import xarray as xr


def sign_test(da_cmp1, da_cmp2, da_ref, dim=None, categorical=False):
    """
        Returns the Delsole and Tippett sign test over the given time period

        | Author: Dougie Squire
        | Date: 26/03/2019

        Parameters
        ----------
        da_cmp1 : xarray DataArray
            Array containing data to be compared to da_cmp1
        da_cmp2 : xarray DataArray
            Array containing data to be compared to da_cmp2
        da_ref : xarray DataArray
            Array containing data to use as reference
        dim : str,
            Name of dimension over which to compute the random walk
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
        >>> walk, confidence = sign_test(x, y, o, dim='t')
        >>> walk
        <xarray.DataArray (t: 3, x: 3)>
        array([[-1, -1, -1],
               [ 0,  0, -2],
               [-1, -1, -3]])
        Coordinates:
          * t        (t) int64 0 1 2
          * x        (x) int64 0 1 2

        Notes
        -----
        See Delsole and Tippett 2016 `Forecast Comparison Based on Random Walks`
    """

    if categorical:
        cmp1_diff = -1 * (da_cmp1 == da_ref)
        cmp2_diff = -1 * (da_cmp2 == da_ref)
    else:
        cmp1_diff = abs(da_cmp1 - da_ref)
        cmp2_diff = abs(da_cmp2 - da_ref)

    sign_test = (1 * (cmp1_diff < cmp2_diff) - 1 * (cmp2_diff < cmp1_diff)).cumsum(dim)

    # Estimate 95% confidence interval -----
    notnan = 1 * (cmp1_diff.notnull() & cmp2_diff.notnull())
    N = notnan.cumsum(dim)
    # z_alpha is the value at which the standardized cumulative Gaussian distributed exceeds alpha
    confidence = 1.95996496 * xr.ufuncs.sqrt(N)
    # confidence['alpha'] = alpha

    res = xr.concat([sign_test, confidence], dim='results')
    res['results'] = ['sign_test', 'confidence']
    return res
