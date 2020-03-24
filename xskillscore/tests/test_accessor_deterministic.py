import pytest
import numpy as np
import pandas as pd
import xarray as xr
from xarray.tests import assert_allclose

from xskillscore.core.deterministic import (
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


correlation_metrics = [
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
    mse,
    rmse,
    mae,
    median_absolute_error,
    mape,
    smape,
]

AXES = ("time", "lat", "lon", ["lat", "lon"], ["time", "lat", "lon"])


@pytest.fixture
def a():
    times = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(times), len(lats), len(lons))
    return xr.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])


@pytest.fixture
def b(a):
    b = a.copy()
    b.values = np.random.rand(a.shape[0], a.shape[1], a.shape[2])
    return b


@pytest.fixture
def weights(a):
    """Weighting array by cosine of the latitude."""
    a_weighted = a.copy()
    cos = np.abs(np.cos(a.lat))
    data = np.tile(cos, (a.shape[0], a.shape[2], 1)).reshape(
        a.shape[0], a.shape[1], a.shape[2]
    )
    a_weighted.values = data
    return a_weighted


def _ds(a, b, skipna_bool, dask_bool):
    ds = xr.Dataset()
    ds["a"] = a
    ds["b"] = b
    if skipna_bool is True:
        ds["b"] = b.where(b < 0.5)
    if dask_bool is True:
        ds["a"] = a.chunk()
        ds["b"] = b.chunk()
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
@pytest.mark.parametrize("dask_bool", [False, True])
@pytest.mark.parametrize("skipna_bool", [False, True])
def test_deterministic_metrics_accessor(
    a, b, dim, skipna_bool, dask_bool, weight_bool, weights, metric, outer_bool
):

    # Update dim to time if testing temporal only metrics
    if (dim != "time") and (metric in temporal_only_metrics):
        dim = "time"

    _weights = adjust_weights(dim, weight_bool, weights)
    ds = _ds(a, b, skipna_bool, dask_bool)
    b = ds["b"]  # Update if populated with nans
    if outer_bool:
        ds = ds.drop_vars("b")

    accessor_func = getattr(ds.xs, metric.__name__)
    if metric in temporal_only_metrics or metric == median_absolute_error:
        actual = metric(a, b, dim, skipna=skipna_bool)
        if outer_bool:
            expected = accessor_func("a", b, dim, skipna=skipna_bool)
        else:
            expected = accessor_func("a", "b", dim, skipna=skipna_bool)
    else:
        actual = metric(a, b, dim, weights=_weights, skipna=skipna_bool)
        if outer_bool:
            expected = accessor_func("a", b, dim, weights=_weights, skipna=skipna_bool)
        else:
            expected = accessor_func(
                "a", "b", dim, weights=_weights, skipna=skipna_bool
            )
    assert_allclose(actual, expected)
