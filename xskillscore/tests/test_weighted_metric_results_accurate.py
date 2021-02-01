import numpy as np
import pytest
import sklearn.metrics
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

import xskillscore as xs
from xskillscore.core.deterministic import (
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
    (pearson_r, pearsonr, 0),
    (spearman_r, spearmanr, 0),
    (pearson_r_p_value, pearsonr, 1),
    (spearman_r_p_value, spearmanr, 1),
]


xs_np_metrics = [
    (me, lambda x, y: np.mean(x - y)),
    (smape, lambda x, y: 1 / len(x) * np.sum(np.abs(y - x) / (np.abs(x) + np.abs(y)))),
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
def test_xs_same_as_skl_with_zeros(
    a_1d_with_zeros, b_1d, xs_skl_metrics, weights_linear_time_1d
):
    """Tests xskillscore metric is same as scikit-learn metric."""
    xs_metric, skl_metric = xs_skl_metrics
    actual = xs_metric(a_1d_with_zeros, b_1d, "time", weights_linear_time_1d)
    expected = skl_metric(a_1d_with_zeros, b_1d, sample_weight=weights_linear_time_1d)
    assert np.allclose(actual, expected)