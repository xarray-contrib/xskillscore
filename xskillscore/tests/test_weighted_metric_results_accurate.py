import numpy as np
import pytest
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


def weighted_pearsonr(x, y, w):
    xm = x - (np.sum(x * w) / np.sum(w))
    ym = y - (np.sum(y * w) / np.sum(w))
    r_num = np.sum(w * xm * ym)
    r_den = np.sqrt(np.sum(w * xm * xm) * np.sum(w * ym * ym))
    return r_num / r_den


def weighted_linslope(x, y, w):
    xm = x - (np.sum(x * w) / np.sum(w))
    ym = y - (np.sum(y * w) / np.sum(w))
    s_num = np.sum(w * xm * ym)
    s_den = np.sum(w * xm * xm)
    return s_num / s_den


xs_scipy_metrics = [(pearson_r, weighted_pearsonr), (linslope, weighted_linslope)]


xs_np_metrics = [
    (me, lambda x, y, w: np.sum((x - y) * w) / np.sum(w)),
    (
        smape,
        lambda x, y, w: np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)) * w) / np.sum(w),
    ),
]


@pytest.mark.parametrize("xs_skl_metrics", xs_skl_metrics)
def test_xs_same_as_skl_weighted(a_1d, b_1d, weights_linear_time_1d, xs_skl_metrics):
    """Tests weighted xskillscore metric is same as weighted scikit-learn metric."""
    xs_metric, skl_metric = xs_skl_metrics
    actual = xs_metric(a_1d, b_1d, "time", weights_linear_time_1d)
    expected = skl_metric(a_1d, b_1d, sample_weight=weights_linear_time_1d)
    assert np.allclose(actual, expected)


def test_xs_same_as_skl_rmse_weighted(a_1d, b_1d, weights_linear_time_1d):
    actual = rmse(a_1d, b_1d, "time", weights_linear_time_1d)
    expected = mean_squared_error(
        a_1d, b_1d, squared=False, sample_weight=weights_linear_time_1d
    )
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("xs_skl_metrics", xs_skl_metrics_with_zeros)
def test_xs_same_as_skl_with_zeros_weighted(
    a_1d_with_zeros, b_1d, xs_skl_metrics, weights_linear_time_1d
):
    """Tests weighted xskillscore metric is same as weighted scikit-learn metric."""
    xs_metric, skl_metric = xs_skl_metrics
    actual = xs_metric(a_1d_with_zeros, b_1d, "time", weights_linear_time_1d)
    expected = skl_metric(a_1d_with_zeros, b_1d, sample_weight=weights_linear_time_1d)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("xs_scipy_metrics", xs_scipy_metrics)
def test_xs_same_as_scipy(a_1d, b_1d, xs_scipy_metrics, weights_linear_time_1d):
    """Tests weighted xskillscore metric is same as weighted scipy metric."""
    xs_metric, scipy_metric = xs_scipy_metrics
    actual = xs_metric(a_1d, b_1d, "time", weights_linear_time_1d)
    expected = scipy_metric(a_1d.values, b_1d.values, weights_linear_time_1d.values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("xs_np_metrics", xs_np_metrics)
def test_xs_same_as_numpy_weighted(a_1d, b_1d, xs_np_metrics, weights_linear_time_1d):
    """Tests weighted xskillscore metric is same as weighted metric using numpy."""
    xs_metric, np_metric = xs_np_metrics
    actual = xs_metric(a_1d, b_1d, "time", weights_linear_time_1d)
    expected = np_metric(a_1d, b_1d, weights_linear_time_1d)
    assert np.allclose(actual, expected)
