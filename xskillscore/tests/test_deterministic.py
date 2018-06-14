from numpy.testing import assert_allclose
import numpy as np
from scipy import stats
import pytest


@pytest.fixture
def a():
    return np.random.rand(3,4,5)

@pytest.fixture
def b():
    return np.random.rand(3,4,5)

def test_pearson_r_nd(a, b):
    axis = 0
    expected = np.squeeze(a[0,:,:]).copy()
    for i in range(np.shape(a)[1]):
        for j in range(np.shape(a)[2]):
            _a = a[:,i,j]
            _b = b[:,i,j]
            expected[i,j], p = stats.pearsonr(_a, _b)
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
    expected = np.squeeze(a[:,0,:]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[2]):
            _a = a[i,:,j]
            _b = b[i,:,j]
            expected[i,j], p = stats.pearsonr(_a, _b)
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
    expected = np.squeeze(a[:,:,0]).copy()
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            _a = a[i,j,:]
            _b = b[i,j,:]
            expected[i,j], p = stats.pearsonr(_a, _b)
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
