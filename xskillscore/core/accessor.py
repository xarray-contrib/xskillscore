import xarray as xr

from .deterministic import (
    median_absolute_error,
    mae,
    mape,
    mse,
    pearson_r,
    pearson_r_p_value,
    pearson_r_eff_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
    spearman_r_eff_p_value,
    effective_sample_size,
    r2,
)
from .probabilistic import xr_brier_score as brier_score
from .probabilistic import xr_crps_ensemble as crps_ensemble
from .probabilistic import xr_crps_gaussian as crps_gaussian
from .probabilistic import xr_crps_quadrature as crps_quadrature
from .probabilistic import xr_threshold_brier_score as threshold_brier_score


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

    def pearson_r(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r(a, b, dim, weights=weights, skipna=skipna)

    def r2(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return r2(a, b, dim, weights=weights, skipna=skipna)

    def pearson_r_p_value(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r_p_value(a, b, dim, weights=weights, skipna=skipna)

    def effective_sample_size(self, a, b, dim, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return effective_sample_size(a, b, dim, skipna=skipna)

    def pearson_r_eff_p_value(self, a, b, dim, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r_eff_p_value(a, b, dim, skipna=skipna)

    def spearman_r(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return spearman_r(a, b, dim, weights=weights, skipna=skipna)

    def spearman_r_p_value(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return spearman_r_p_value(a, b, dim, weights=weights, skipna=skipna)

    def spearman_r_eff_p_value(self, a, b, dim, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return spearman_r_eff_p_value(a, b, dim, skipna=skipna)

    def rmse(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return rmse(a, b, dim, weights=weights, skipna=skipna)

    def mse(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mse(a, b, dim, weights=weights, skipna=skipna)

    def mae(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mae(a, b, dim, weights=weights, skipna=skipna)

    def median_absolute_error(self, a, b, dim, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return median_absolute_error(a, b, dim, skipna=skipna)

    def mape(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mape(a, b, dim, weights=weights, skipna=skipna)

    def smape(self, a, b, dim, weights=None, skipna=False):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return smape(a, b, dim, weights=weights, skipna=skipna)

    def crps_gaussian(self, observations, mu, sig):
        observations = self._in_ds(observations)
        mu = self._in_ds(mu)
        sig = self._in_ds(sig)
        return crps_gaussian(observations, mu, sig)

    def crps_ensemble(
        self, observations, forecasts, weights=None, issorted=False, dim="member"
    ):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return crps_ensemble(
            observations, forecasts, weights=weights, issorted=issorted, dim="member"
        )

    def crps_quadrature(self, x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
        x = self._in_ds(x)
        cdf_or_dist = self._in_ds(cdf_or_dist)
        return crps_quadrature(x, cdf_or_dist, xmin=xmin, xmax=xmax, tol=1e-6)

    def threshold_brier_score(
        self, observations, forecasts, threshold, issorted=False, dim="member"
    ):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return threshold_brier_score(
            observations, forecasts, threshold, issorted=issorted, dim="member"
        )

    def brier_score(self, observations, forecasts):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return brier_score(observations, forecasts)
