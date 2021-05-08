import numpy as np
import pytest
import sklearn.metrics
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

import xskillscore as xs
from xskillscore.core.deterministic import (
    linslope,
    mae,
    mape,
    me,
    mse,
    pearson_r,
    pearson_r_p_value,
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
)

xs_skl_metrics = [
    (r2, r2_score),
    (mse, mean_squared_error),
    (mae, mean_absolute_error),
    (mape, mean_absolute_percentage_error),
]

xs_skl_metrics_with_zeros = [
    (mape, mean_absolute_percentage_error),
]

xs_scipy_metrics = [
    (linslope, linregress, 0),
    (pearson_r, pearsonr, 0),
    (spearman_r, spearmanr, 0),
    (pearson_r_p_value, pearsonr, 1),
    (spearman_r_p_value, spearmanr, 1),
]


xs_np_metrics = [
    (me, lambda x, y: np.mean(x - y)),
    (smape, lambda x, y: np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y)))),
]


@pytest.mark.parametrize("xs_skl_metrics", xs_skl_metrics)
def test_xs_same_as_skl(a_1d, b_1d, xs_skl_metrics):
    """Tests xskillscore metric is same as scikit-learn metric."""
    xs_metric, skl_metric = xs_skl_metrics
    actual = xs_metric(a_1d, b_1d, "time")
    expected = skl_metric(a_1d, b_1d)
    assert np.allclose(actual, expected)


def test_xs_same_as_skl_rmse(a_1d, b_1d):
    actual = rmse(a_1d, b_1d, "time")
    expected = mean_squared_error(a_1d, b_1d, squared=False)
    assert np.allclose(actual, expected)


def test_xs_same_as_skl_same_name(a_1d, b_1d):
    """Tests xskillscore metric is same as scikit-learn metric
    for metrics with same name."""
    xs_metric, skl_metric = (
        getattr(xs, "median_absolute_error"),
        getattr(sklearn.metrics, "median_absolute_error"),
    )
    actual = xs_metric(a_1d, b_1d, "time")
    expected = skl_metric(a_1d, b_1d)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("xs_skl_metrics", xs_skl_metrics_with_zeros)
def test_xs_same_as_skl_with_zeros(a_1d_with_zeros, b_1d, xs_skl_metrics):
    """Tests xskillscore metric is same as scikit-learn metric."""
    xs_metric, skl_metric = xs_skl_metrics
    actual = xs_metric(a_1d_with_zeros, b_1d, "time")
    expected = skl_metric(a_1d_with_zeros, b_1d)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("xs_scipy_metrics", xs_scipy_metrics)
def test_xs_same_as_scipy(a_1d, b_1d, xs_scipy_metrics):
    """Tests xskillscore metric is same as scipy metric."""
    xs_metric, scipy_metric, i = xs_scipy_metrics
    actual = xs_metric(a_1d, b_1d, "time")
    expected = scipy_metric(a_1d, b_1d)[i]
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("xs_np_metrics", xs_np_metrics)
def test_xs_same_as_numpy(a_1d, b_1d, xs_np_metrics):
    """Tests xskillscore metric is same as metric using numpy."""
    xs_metric, np_metric = xs_np_metrics
    actual = xs_metric(a_1d, b_1d, "time")
    expected = np_metric(a_1d, b_1d)
    assert np.allclose(actual, expected)
