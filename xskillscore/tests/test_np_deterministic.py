import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special, stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


@pytest.fixture
def a():
    return np.random.rand(3, 4, 5)


@pytest.fixture
def b():
    return np.random.rand(3, 4, 5)


def test_pearson_r_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j], p = stats.pearsonr(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    actual = np.clip(r, -1.0, 1.0)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j], p = stats.pearsonr(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    actual = np.clip(r, -1.0, 1.0)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j], p = stats.pearsonr(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    actual = np.clip(r, -1.0, 1.0)
    assert_allclose(actual, expected)


def test_pearson_r_p_value_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            r, expected[i, j] = stats.pearsonr(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    df = a.shape[0] - 2
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    _x = df/(df+t_squared)
    _x = np.asarray(_x)
    _x = np.where(_x < 1.0, _x, 1.0)
    _a = 0.5*df
    _b = 0.5
    actual = special.betainc(_a, _b, _x)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            p, expected[i, j] = stats.pearsonr(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    df = a.shape[0] - 2
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    _x = df/(df+t_squared)
    _x = np.asarray(_x)
    _x = np.where(_x < 1.0, _x, 1.0)
    _a = 0.5*df
    _b = 0.5
    actual = special.betainc(_a, _b, _x)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            r, expected[i, j] = stats.pearsonr(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    ma = np.mean(a, axis=0)
    mb = np.mean(b, axis=0)
    am, bm = a - ma, b - mb
    r_num = np.sum(am * bm, axis=0)
    r_den = np.sqrt(np.sum(am*am, axis=0) * np.sum(bm*bm, axis=0))
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    df = a.shape[0] - 2
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    _x = df/(df+t_squared)
    _x = np.asarray(_x)
    _x = np.where(_x < 1.0, _x, 1.0)
    _a = 0.5*df
    _b = 0.5
    actual = special.betainc(_a, _b, _x)
    assert_allclose(actual, expected)


def test_rmse_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = np.sqrt(mean_squared_error(_a, _b))
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = np.sqrt(((a - b) ** 2).mean(axis=0))
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = np.sqrt(mean_squared_error(_a, _b))
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = np.sqrt(((a - b) ** 2).mean(axis=0))
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = np.sqrt(mean_squared_error(_a, _b))
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = np.sqrt(((a - b) ** 2).mean(axis=0))
    assert_allclose(actual, expected)


def test_mse_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = mean_squared_error(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = ((a - b) ** 2).mean(axis=0)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = mean_squared_error(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = ((a - b) ** 2).mean(axis=0)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = mean_squared_error(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = ((a - b) ** 2).mean(axis=0)
    assert_allclose(actual, expected)


def test_mae_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0, :, :]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:, i, j]
            _b = b[:, i, j]
            expected[i, j] = mean_absolute_error(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = (np.absolute(a - b)).mean(axis=0)
    assert_allclose(actual, expected)

    axis = 1
    expected = np.squeeze(a[:, 0, :]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i, :, j]
            _b = b[i, :, j]
            expected[i, j] = mean_absolute_error(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = (np.absolute(a - b)).mean(axis=0)
    assert_allclose(actual, expected)

    axis = 2
    expected = np.squeeze(a[:, :, 0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i, j, :]
            _b = b[i, j, :]
            expected[i, j] = mean_absolute_error(_a, _b)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    actual = (np.absolute(a - b)).mean(axis=0)
    assert_allclose(actual, expected)
