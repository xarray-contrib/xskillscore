from xskillscore.core.deterministic import (
    effective_sample_size,
    pearson_r_eff_p_value,
    pearson_r_p_value,
    spearman_r_eff_p_value,
    spearman_r_p_value,
)

DIM = "time"


def test_eff_sample_size_smaller_than_n(a, b):
    """Tests that the effective sample size is less than or equal to the normal
    sample size."""
    # Can just use `.size` here since we don't
    # have any NaNs in the test.
    N = a[DIM].size
    eff_N = effective_sample_size(a, b, DIM)
    assert (eff_N <= N).all()


def test_eff_pearson_p_greater_or_equal_to_normal_p(a, b):
    """Tests that the effective Pearson p value is greater than or equal to the
    normal Pearson p value."""
    print(a)
    print(b)
    normal_p = pearson_r_p_value(a, b, DIM)
    eff_p = pearson_r_eff_p_value(a, b, DIM)
    print("\n" * 3)
    print(normal_p)
    print(eff_p)
    assert (eff_p >= normal_p).all()


def test_eff_spearman_p_greater_or_equal_to_normal_p(a, b):
    """Tests that the effective Spearman p value is greater than or equal to the
    normal Spearman p value."""
    normal_p = spearman_r_p_value(a, b, DIM)
    eff_p = spearman_r_eff_p_value(a, b, DIM)
    assert (eff_p >= normal_p).all()
