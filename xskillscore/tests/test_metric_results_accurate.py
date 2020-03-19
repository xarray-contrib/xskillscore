import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.metrics import median_absolute_error as sklearn_med_abs
import xarray as xr


from xskillscore.core.deterministic import (
    median_absolute_error,
    mae,
    mape,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
    r2,
)


@pytest.fixture
def a():
    time = pd.date_range("1/1/2000", "1/3/2000", freq="D")
    da = xr.DataArray(np.random.rand(len(time)), dims=["time"], coords=[time])
    return da


@pytest.fixture
def b(a):
    b = a.copy()
    b.values = np.random.rand(b.shape[0])
    return b


def test_pearsonr_same_as_scipy(a, b):
    """Tests that pearson r correlation and pvalue is same as computed from
    scipy."""
    xs_corr = pearson_r(a, b, "time")
    xs_p = pearson_r_p_value(a, b, "time")
    scipy_corr, scipy_p = pearsonr(a, b)
    assert np.allclose(xs_corr, scipy_corr)
    assert np.allclose(xs_p, scipy_p)


def test_r2_same_as_sklearn(a, b):
    """Tests that r2 is same as computed from sklearn."""
    xs_r2 = r2(a, b, "time")
    sklearn_r2 = r2_score(a, b)
    assert np.allclose(xs_r2, sklearn_r2)


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
    xs_mae = mae(a, b, "time")
    sklearn_mae = mean_absolute_error(a, b)
    assert np.allclose(xs_mae, sklearn_mae)


def test_median_absolute_error_same_as_sklearn(a, b):
    """Tests that median absolute error is same as computed from sklearn."""
    xs_median_absolute_error = median_absolute_error(a, b, "time")
    sklearn_median_absolute_error = sklearn_med_abs(a, b)
    assert np.allclose(xs_median_absolute_error, sklearn_median_absolute_error)


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
