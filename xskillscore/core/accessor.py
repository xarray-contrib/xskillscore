import xarray as xr

from .deterministic import pearson_r, pearson_r_p_value, rmse, mse, mae
from .probabilistic import xr_crps_gaussian, xr_crps_ensemble, xr_threshold_brier_score


@xr.register_dataset_accessor("xs")
class XSkillScoreAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _in_ds(self, x):
        """
        If x is not a string, presumably an array, return the array.
        Else x is a string, presumably within ds, return the ds variable.
        """
        if not isinstance(x, str):
            return x
        else:
            return self._obj[x]

    def pearson_r(self, a, b, dim):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r(a, b, dim)

    def pearson_r_p_value(self, a, b, dim):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r_p_value(a, b, dim)

    def rmse(self, a, b, dim):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return rmse(a, b, dim)

    def mse(self, a, b, dim):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mse(a, b, dim)

    def mae(self, a, b, dim):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mae(a, b, dim)

    def crps_gaussian(self, observations, mu, sig):
        observations = self._in_ds(observations)
        mu = self._in_ds(mu)
        sig = self._in_ds(sig)
        return xr_crps_gaussian(observations, mu, sig)

    def crps_ensemble(self, observations, forecasts):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return xr_crps_ensemble(observations, forecasts)

    def threshold_brier_score(
        self, observations, forecasts, threshold, issorted=False, axis=-1
    ):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        threshold = self._in_ds(threshold)
        return xr_threshold_brier_score(
            observations, forecasts, threshold, issorted=issorted, axis=axis
        )
