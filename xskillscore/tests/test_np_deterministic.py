import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from xskillscore.core.np_deterministic import (
    _median_absolute_error,
    _mae,
    _mape,
    _mse,
    _pearson_r,
    _pearson_r_p_value,
    _rmse,
    _smape,
    _spearman_r,
    _spearman_r_p_value,
    _r2,
)


@pytest.fixture
def a():
    return np.random.rand(3, 4, 5)


@pytest.fixture
def b():
    return np.random.rand(3, 4, 5)


# standard params in this testing file
skipna = False
weights = None


def test_pearson_r_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j], p = stats.pearsonr(_a, _b)
    actual = _pearson_r(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j], p = stats.pearsonr(_a, _b)
    actual = _pearson_r(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j], p = stats.pearsonr(_a, _b)
    actual = _pearson_r(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_r2_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = r2_score(_a, _b)
    actual = _r2(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = r2_score(_a, _b)
    actual = _r2(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = r2_score(_a, _b)
    actual = _r2(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_pearson_r_p_value_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            r, expected[i, j] = stats.pearsonr(_a, _b)
    actual = _pearson_r_p_value(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            p, expected[i, j] = stats.pearsonr(_a, _b)
    actual = _pearson_r_p_value(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            r, expected[i, j] = stats.pearsonr(_a, _b)
    actual = _pearson_r_p_value(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_spearman_r_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j], p = stats.spearmanr(_a, _b)
    actual = _spearman_r(a, b, weights, axis, skipna)
    assert_allclose(actual, expected, atol=1e-5)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j], p = stats.spearmanr(_a, _b)
    actual = _spearman_r(a, b, weights, axis, skipna)
    assert_allclose(actual, expected, atol=1e-5)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j], p = stats.spearmanr(_a, _b)
    actual = _spearman_r(a, b, weights, axis, skipna)
    assert_allclose(actual, expected, atol=1e-5)


def test_spearman_r_p_value_nd(a, b):
    nan_policy = "propagate"  # default
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            r, expected[i, j] = stats.spearmanr(_a, _b, nan_policy=nan_policy)
    actual = _spearman_r_p_value(a, b, weights, axis, skipna)
    assert_allclose(actual, expected, atol=1e-5)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            p, expected[i, j] = stats.spearmanr(_a, _b, nan_policy=nan_policy)
    actual = _spearman_r_p_value(a, b, weights, axis, skipna)
    assert_allclose(actual, expected, atol=1e-5)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            r, expected[i, j] = stats.spearmanr(_a, _b, nan_policy=nan_policy)
    actual = _spearman_r_p_value(a, b, weights, axis, skipna)
    assert_allclose(actual, expected, atol=1e-5)


def test_rmse_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = np.sqrt(mean_squared_error(_a, _b))
    actual = _rmse(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = np.sqrt(mean_squared_error(_a, _b))
    actual = _rmse(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = np.sqrt(mean_squared_error(_a, _b))
    actual = _rmse(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_mse_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = mean_squared_error(_a, _b)
    actual = _mse(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = mean_squared_error(_a, _b)
    actual = _mse(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = mean_squared_error(_a, _b)
    actual = _mse(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_mae_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = mean_absolute_error(_a, _b)
    actual = _mae(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = mean_absolute_error(_a, _b)
    actual = _mae(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = mean_absolute_error(_a, _b)
    actual = _mae(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_median_absolute_error_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = median_absolute_error(_a, _b)
    actual = _median_absolute_error(a, b, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = median_absolute_error(_a, _b)
    actual = _median_absolute_error(a, b, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = median_absolute_error(_a, _b)
    actual = _median_absolute_error(a, b, axis, skipna)
    assert_allclose(actual, expected)


def test_mape_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = mean_absolute_error(_a / _a, _b / _a)
    actual = _mape(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = mean_absolute_error(_a / _a, _b / _a)
    actual = _mape(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = mean_absolute_error(_a / _a, _b / _a)
    actual = _mape(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)


def test_smape_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = mean_absolute_error(
                _a / (np.absolute(_a) + np.absolute(_b)),
                _b / (np.absolute(_a) + np.absolute(_b)),
            )
    actual = _smape(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = mean_absolute_error(
                _a / (np.absolute(_a) + np.absolute(_b)),
                _b / (np.absolute(_a) + np.absolute(_b)),
            )
    actual = _smape(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = mean_absolute_error(
                _a / (np.absolute(_a) + np.absolute(_b)),
                _b / (np.absolute(_a) + np.absolute(_b)),
            )
    actual = _smape(a, b, weights, axis, skipna)
    assert_allclose(actual, expected)
