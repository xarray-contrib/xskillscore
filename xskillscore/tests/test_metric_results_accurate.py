import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xarray as xr


from xskillscore.core.deterministic import (
    mad,
    mae,
    mape,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
)

METRICS = [
    pearson_r,
    pearson_r_p_value,
    spearman_r,
    spearman_r_p_value,
    mae,
    mse,
    mad,
    mape,
    smape,
    rmse,
]


@pytest.fixture
def a():
    time = pd.date_range("1/1/2000", "1/5/2000", freq="D")
    da = xr.DataArray(np.random.rand(len(time)), dims=["time"], coords=[time])
    return da


@pytest.fixture
def b():
    time = pd.date_range("1/1/2000", "1/5/2000", freq="D")
    da = xr.DataArray(np.random.rand(len(time)), dims=["time"], coords=[time])
    return da


def test_pearsonr_same_as_scipy(a, b):
    """Tests that pearson r correlation and pvalue is same as computed from
    scipy."""
    xs_corr = pearson_r(a, b, "time")
    xs_p = pearson_r_p_value(a, b, "time")
    scipy_corr, scipy_p = pearsonr(a, b)
    assert np.allclose(xs_corr, scipy_corr)
    assert np.allclose(xs_p, scipy_p)


def test_spearmanr_same_as_scipy(a, b):
    """Tests that spearman r correlation and pvalue is same as computed from
    scipy."""
    xs_corr = spearman_r(a, b, "time")
    xs_p = spearman_r_p_value(a, b, "time")
    scipy_corr, scipy_p = spearmanr(a, b)
    assert np.allclose(xs_corr, scipy_corr)
    assert np.allclose(xs_p, scipy_p)


def test_rmse_same_as_sklearn(a, b):
    """Tests that root mean squared error is same as computed from sklearn."""
    xs_rmse = rmse(a, b, "time")
    sklearn_rmse = np.sqrt(mean_squared_error(a, b))
    assert np.allclose(xs_rmse, sklearn_rmse)


def test_mse_same_as_sklearn(a, b):
    """Tests that mean squared error is same as computed from sklearn."""
    xs_mse = mse(a, b, "time")
    sklearn_mse = mean_squared_error(a, b)
    assert np.allclose(xs_mse, sklearn_mse)


def test_mae_same_as_sklearn(a, b):
    """Tests that mean absolute error is same as computed from sklearn."""
    xs_mse = mae(a, b, "time")
    sklearn_mse = mean_absolute_error(a, b)
    assert np.allclose(xs_mse, sklearn_mse)


def test_mape_same_as_numpy(a, b):
    """Tests that mean absolute percent error is same as computed from numpy."""
    xs_mape = mape(a, b, "time")
    np_mape = np.mean(np.abs((a - b) / a))
    assert np.allclose(xs_mape, np_mape)


def test_smape_same_as_numpy(a, b):
    """Tests that symmetric mean absolute percent error is same as computed
    from numpy."""
    xs_smape = smape(a, b, "time")
    np_smape = 1 / len(a) * np.sum(np.abs(b - a) / (np.abs(a) + np.abs(b)))
    assert np.allclose(xs_smape, np_smape)
