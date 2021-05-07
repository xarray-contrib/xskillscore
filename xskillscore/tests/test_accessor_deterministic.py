import pytest
import xarray as xr
from xarray.tests import assert_allclose

from xskillscore.core.deterministic import (
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

correlation_metrics = [
    linslope,
    pearson_r,
    r2,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    effective_sample_size,
    pearson_r_eff_p_value,
    spearman_r_eff_p_value,
]

temporal_only_metrics = [
    pearson_r_eff_p_value,
    spearman_r_eff_p_value,
    effective_sample_size,
]

distance_metrics = [
    me,
    mse,
    rmse,
    mae,
    median_absolute_error,
    mape,
    smape,
]

AXES = ("time", "lat", "lon", ["lat", "lon"], ["time", "lat", "lon"])


def _ds(a, b, skipna_bool):
    ds = xr.Dataset()
    ds["a"] = a
    ds["b"] = b
    if skipna_bool is True:
        ds["b"] = b.where(b < 0.5)
    return ds


def adjust_weights(dim, weight_bool, weights):
    """
    Adjust the weights test data to only span the core dimension
    that the function is being applied over.
    """
    if weight_bool:
        drop_dims = [i for i in weights.dims if i not in dim]
        drop_dims = {k: 0 for k in drop_dims}
        return weights.isel(drop_dims)
    else:
        return None


@pytest.mark.parametrize("outer_bool", [False, True])
@pytest.mark.parametrize("metric", correlation_metrics + distance_metrics)
@pytest.mark.parametrize("dim", AXES)
@pytest.mark.parametrize("weight_bool", [False, True])
@pytest.mark.parametrize("skipna_bool", [False, True])
def test_deterministic_metrics_accessor(
    a, b, dim, skipna_bool, weight_bool, weights_cos_lat, metric, outer_bool
):

    # Update dim to time if testing temporal only metrics
    if (dim != "time") and (metric in temporal_only_metrics):
        dim = "time"

    _weights = adjust_weights(dim, weight_bool, weights_cos_lat)
    ds = _ds(a, b, skipna_bool)
    b = ds["b"]  # Update if populated with nans
    if outer_bool:
        ds = ds.drop_vars("b")

    accessor_func = getattr(ds.xs, metric.__name__)
    if metric in temporal_only_metrics or metric == median_absolute_error:
        actual = metric(a, b, dim=dim, skipna=skipna_bool)
        if outer_bool:
            expected = accessor_func("a", b, dim=dim, skipna=skipna_bool)
        else:
            expected = accessor_func("a", "b", dim=dim, skipna=skipna_bool)
    else:
        actual = metric(a, b, dim=dim, weights=_weights, skipna=skipna_bool)
        if outer_bool:
            expected = accessor_func(
                "a", b, dim=dim, weights=_weights, skipna=skipna_bool
            )
        else:
            expected = accessor_func(
                "a", "b", dim=dim, weights=_weights, skipna=skipna_bool
            )
    assert_allclose(actual, expected)
