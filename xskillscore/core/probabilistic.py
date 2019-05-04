import xarray as xr
from properscoring import crps_ensemble, crps_gaussian


def xr_crps_gaussian(observations, mu, sig):
    """


    See Also
    --------
    properscoring.crps_gaussian
    xarray.apply_ufunc
    """
    # check if same dimensions
    if mu.dims != observations.dims:
        observations, mu = xr.broadcast(observations, mu)
    if sig.dims != observations.dims:
        observations, sig = xr.broadcast(observations, sig)
    return xr.apply_ufunc(crps_gaussian, observations, mu, sig,
                          input_core_dims=[[], [], []],
                          dask='parallelized',
                          output_dtypes=[float])


def xr_crps_ensemble(observations, forecasts):
    """


    See Also
    --------
    properscoring.crps_ensemble
    xarray.apply_ufunc
    """
    if forecasts.dims != observations.dims:
        observations, forecasts = xr.broadcast(observations, forecasts)
    return xr.apply_ufunc(crps_ensemble, observations, forecasts,
                          input_core_dims=[[], []],
                          dask='parallelized',
                          output_dtypes=[float])
