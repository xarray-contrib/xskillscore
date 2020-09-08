import pytest

from xskillscore import sign_test
import numpy as np
np.random.seed(42)

OFFSET=-2

@pytest.fixture
def a_1d_worse(a_1d):
    return a_1d +OFFSET

def test_sign_test_raw(a_1d, a_1d_worse, b_1d):
    actual = sign_test(a_1d, a_1d_worse, b_1d, dim='time')
    walk_larger_significance = actual.sel(result='sign_test') > actual.sel(result='significance')
    crossing = walk_larger_significance.argmin(dim='time')
    assert crossing>7


def test_sign_test_categorical(a_1d, a_1d_worse, b_1d):
    def logical(ds):
        return ds > 0.5

    a_1d = logical(a_1d)
    a_1d_worse = logical(a_1d_worse)
    b_1d = logical(b_1d)
    actual = sign_test(a_1d, a_1d_worse, b_1d, dim='time', categorical=True)
    print(actual)
    assert False


#def test_sign_test_alpha(a_1d, a_1d_worse, b_1d,alpha)
