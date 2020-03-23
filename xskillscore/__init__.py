# flake8: noqa
from pkg_resources import DistributionNotFound, get_distribution

from .core.accessor import XSkillScoreAccessor
from .core.deterministic import (
    effective_sample_size,
    mae,
    mape,
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
    xr_brier_score as brier_score,
    xr_crps_ensemble as crps_ensemble,
    xr_crps_gaussian as crps_gaussian,
    xr_crps_quadrature as crps_quadrature,
    xr_threshold_brier_score as threshold_brier_score,
)
from .versioning.print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
