from importlib.metadata import version, PackageNotFoundError

# ruff: noqa
from xskillscore.core import resampling
from xskillscore.core.accessor import XSkillScoreAccessor
from xskillscore.core.comparative import halfwidth_ci_test, sign_test
from xskillscore.core.contingency import Contingency
from xskillscore.core.deterministic import (
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
from xskillscore.core.probabilistic import (
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
from xskillscore.core.resampling import resample_iterations, resample_iterations_idx
from xskillscore.core.stattests import multipletests
from xskillscore.versioning.print_versions import show_versions

try:
    __version__ = version("xskillscore")
except PackageNotFoundError:
    # package is not installed
    pass
