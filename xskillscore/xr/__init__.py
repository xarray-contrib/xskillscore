"""These metrics in xskillscore.xr, entirely written in xarray functions, are identical
to the well documented metrics in xskillscore.core, which are based on numpy functions
applied to xarray objects by xarray.apply_ufunc. As the xr metrics are only faster for
small data, their use is not encouraged, as the numpy-based metrics are 20-40% faster
on large data."""
from .deterministic import xr_mae, xr_mse, xr_pearson_r, xr_rmse, xr_spearman_r
