import dask
import numpy as np
import pytest
import xarray as xr

import xskillscore as xs
from xskillscore.xr import xr_mae, xr_mse, xr_pearson_r, xr_rmse, xr_spearman_r

METRICS = ["mse", "mae", "rmse", "pearson_r", "spearman_r"]
DIMS = [["lon", "lat"], ["time"]]


@pytest.mark.parametrize("weights", [True, False])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("metric", METRICS)
def test_xr_metrics_equal_xs_metrics(metric, a, b, dim, weights, skipna):
    """Test that xr metrics yield same results as xs metrics"""
    if isinstance(dim, str):
        dim = list(dim)
    if weights:
        if set(dim) == set(["lon", "lat"]):
            weights = a.isel(time=0, drop=True) * np.abs(a.lat)
        elif "time" in dim:
            weights = a.time.dt.day
        else:
            weights = None
    else:
        weights = None
    print(f"{metric}(a,b,dim={dim},skipna={skipna},weights={weights})")
    if metric in ["pearson_r", "spearman_r"] and (weights is not None or skipna):
        print(f"pass {metric}(a,b,dim={dim},skipna={skipna},weights={weights})")
        print("raise NotImplementedError")
        pass
    else:
        res = eval("xr_" + metric)(a, b, skipna=skipna, dim=dim, weights=weights)
        xs_res = getattr(xs, metric)(a, b, skipna=skipna, dim=dim, weights=weights)
        xr.testing.assert_allclose(res, xs_res)
