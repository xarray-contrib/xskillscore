import xarray as xr

from .deterministic import (
    effective_sample_size,
    linslope,
    mae,
    mape,
    me,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_eff_p_value,
    pearson_r_p_value,
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_eff_p_value,
    spearman_r_p_value,
)
from .probabilistic import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    discrimination,
    rank_histogram,
    reliability,
    roc,
    rps,
    threshold_brier_score,
)


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

    def linslope(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return linslope(a, b, *args, **kwargs)

    def pearson_r(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r(a, b, *args, **kwargs)

    def r2(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return r2(a, b, *args, **kwargs)

    def pearson_r_p_value(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r_p_value(a, b, *args, **kwargs)

    def effective_sample_size(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return effective_sample_size(a, b, *args, **kwargs)

    def pearson_r_eff_p_value(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return pearson_r_eff_p_value(a, b, *args, **kwargs)

    def spearman_r(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return spearman_r(a, b, *args, **kwargs)

    def spearman_r_p_value(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return spearman_r_p_value(a, b, *args, **kwargs)

    def spearman_r_eff_p_value(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return spearman_r_eff_p_value(a, b, *args, **kwargs)

    def me(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return me(a, b, *args, **kwargs)

    def rmse(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return rmse(a, b, *args, **kwargs)

    def mse(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mse(a, b, *args, **kwargs)

    def mae(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mae(a, b, *args, **kwargs)

    def median_absolute_error(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return median_absolute_error(a, b, *args, **kwargs)

    def mape(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return mape(a, b, *args, **kwargs)

    def smape(self, a, b, *args, **kwargs):
        a = self._in_ds(a)
        b = self._in_ds(b)
        return smape(a, b, *args, **kwargs)

    def crps_gaussian(self, observations, mu, sig, *args, **kwargs):
        observations = self._in_ds(observations)
        mu = self._in_ds(mu)
        sig = self._in_ds(sig)
        return crps_gaussian(observations, mu, sig, *args, **kwargs)

    def crps_ensemble(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return crps_ensemble(observations, forecasts, *args, **kwargs)

    def crps_quadrature(self, x, cdf_or_dist, *args, **kwargs):
        x = self._in_ds(x)
        return crps_quadrature(x, cdf_or_dist, *args, **kwargs)

    def threshold_brier_score(
        self, observations, forecasts, threshold, *args, **kwargs
    ):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        threshold = self._in_ds(threshold)
        return threshold_brier_score(
            observations, forecasts, threshold, *args, **kwargs
        )

    def brier_score(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return brier_score(observations, forecasts, *args, **kwargs)

    def rps(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return rps(observations, forecasts, *args, **kwargs)

    def rank_histogram(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return rank_histogram(observations, forecasts, *args, **kwargs)

    def discrimination(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return discrimination(observations, forecasts, *args, **kwargs)

    def reliability(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return reliability(observations, forecasts, *args, **kwargs)

    def roc(self, observations, forecasts, *args, **kwargs):
        observations = self._in_ds(observations)
        forecasts = self._in_ds(forecasts)
        return roc(observations, forecasts, *args, **kwargs)
