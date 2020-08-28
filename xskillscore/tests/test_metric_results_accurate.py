import numpy as np
import pytest
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error as sklearn_med_abs,
    r2_score,
)

from xskillscore.core.deterministic import (
    mae,
    mape,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_p_value,
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
)


def test_pearsonr_same_as_scipy(a_1d, b_1d):
    """Tests that pearson r correlation and pvalue is same as computed from
    scipy."""
    xs_corr = pearson_r(a_1d, b_1d, 'time')
    xs_p = pearson_r_p_value(a_1d, b_1d, 'time')
    scipy_corr, scipy_p = pearsonr(a_1d, b_1d)
    assert np.allclose(xs_corr, scipy_corr)
    assert np.allclose(xs_p, scipy_p)


def test_r2_same_as_sklearn(a_1d, b_1d):
    """Tests that r2 is same as computed from sklearn."""
    xs_r2 = r2(a_1d, b_1d, 'time')
    sklearn_r2 = r2_score(a_1d, b_1d)
    assert np.allclose(xs_r2, sklearn_r2)


def test_spearmanr_same_as_scipy(a_1d, b_1d):
    """Tests that spearman r correlation and pvalue is same as computed from
    scipy."""
    xs_corr = spearman_r(a_1d, b_1d, 'time')
    xs_p = spearman_r_p_value(a_1d, b_1d, 'time')
    scipy_corr, scipy_p = spearmanr(a_1d, b_1d)
    assert np.allclose(xs_corr, scipy_corr)
    assert np.allclose(xs_p, scipy_p)


def test_rmse_same_as_sklearn(a_1d, b_1d):
    """Tests that root mean squared error is same as computed from sklearn."""
    xs_rmse = rmse(a_1d, b_1d, 'time')
    sklearn_rmse = np.sqrt(mean_squared_error(a_1d, b_1d))
    assert np.allclose(xs_rmse, sklearn_rmse)


def test_mse_same_as_sklearn(a_1d, b_1d):
    """Tests that mean squared error is same as computed from sklearn."""
    xs_mse = mse(a_1d, b_1d, 'time')
    sklearn_mse = mean_squared_error(a_1d, b_1d)
    assert np.allclose(xs_mse, sklearn_mse)


def test_mae_same_as_sklearn(a_1d, b_1d):
    """Tests that mean absolute error is same as computed from sklearn."""
    xs_mae = mae(a_1d, b_1d, 'time')
    sklearn_mae = mean_absolute_error(a_1d, b_1d)
    assert np.allclose(xs_mae, sklearn_mae)


def test_median_absolute_error_same_as_sklearn(a_1d, b_1d):
    """Tests that median absolute error is same as computed from sklearn."""
    xs_median_absolute_error = median_absolute_error(a_1d, b_1d, 'time')
    sklearn_median_absolute_error = sklearn_med_abs(a_1d, b_1d)
    assert np.allclose(xs_median_absolute_error, sklearn_median_absolute_error)


def test_mape_same_as_numpy(a_1d, b_1d):
    """Tests that mean absolute percent error is same as computed from numpy."""
    xs_mape = mape(a_1d, b_1d, 'time')
    np_mape = np.mean(np.abs((a_1d - b_1d) / a_1d))
    assert np.allclose(xs_mape, np_mape)


def test_smape_same_as_numpy(a_1d, b_1d):
    """Tests that symmetric mean absolute percent error is same as computed
    from numpy."""
    xs_smape = smape(a_1d, b_1d, 'time')
    np_smape = (
        1 / len(a_1d) * np.sum(np.abs(b_1d - a_1d) / (np.abs(a_1d) + np.abs(b_1d)))
    )
    assert np.allclose(xs_smape, np_smape)
