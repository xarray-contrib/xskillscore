# flake8: noqa
from importlib.metadata import PackageNotFoundError, version as _get_version

from .core import resampling
from .core.accessor import XSkillScoreAccessor
from .core.comparative import halfwidth_ci_test, sign_test
from .core.contingency import Contingency
from .core.deterministic import (
    effective_sample_size,
    linslope,
    mae,
    mape,
    me,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_eff_p_value,
    pearson_r_p_value,
    r2,
    rmse,
    smape,
    spearman_r,
    spearman_r_eff_p_value,
    spearman_r_p_value,
)
from .core.probabilistic import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    discrimination,
    rank_histogram,
    reliability,
    roc,
    rps,
    threshold_brier_score,
)
from .core.resampling import resample_iterations, resample_iterations_idx
from .core.stattests import multipletests
from .versioning.print_versions import show_versions

__version__ = "0.0.25"
