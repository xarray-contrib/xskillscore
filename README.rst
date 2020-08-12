xskillscore: Metrics for verifying forecasts
============================================

.. image:: https://travis-ci.org/raybellwaves/xskillscore.svg?branch=master
   :target: https://travis-ci.org/raybellwaves/xskillscore
.. image:: https://img.shields.io/pypi/v/xskillscore.svg
   :target: https://pypi.python.org/pypi/xskillscore/
.. image:: https://anaconda.org/conda-forge/xskillscore/badges/version.svg
   :target: https://anaconda.org/conda-forge/xskillscore/
.. image:: https://coveralls.io/repos/github/raybellwaves/xskillscore/badge.svg?branch=master
   :target: https://coveralls.io/github/raybellwaves/xskillscore?branch=master
.. image:: https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
   :target: https://raybellwaves.github.io/xskillscore/
.. image:: https://img.shields.io/conda/dn/conda-forge/xskillscore.svg
   :target: https://anaconda.org/conda-forge/xskillscore
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/raybellwaves/xskillscore-tutorial/master?urlpath=lab

**xskillscore** is an open source project and Python package that provides verification
metrics of deterministic (and probabilistic from `properscoring`) forecasts with `xarray`.

Installing
----------

``$ conda install -c conda-forge xskillscore``

or

``$ pip install xskillscore``

or

``$ pip install git+https://github.com/raybellwaves/xskillscore``

See also
--------

- If you are interested in using **xskillscore** for data science where you data is mostly in
  ``pandas.DataFrames``'s check out the `xskillscore-tutorial <https://github.com/raybellwaves/xskillscore-tutorial>`_
- If you are interested in using **xskillscore** for climate prediction check out
  `climpred <https://climpred.readthedocs.io/en/stable/>`_.

Examples
--------

.. code-block:: python

   import xarray as xr
   import numpy as np
   from scipy.stats import norm
   import xskillscore as xs


   obs = xr.DataArray(
       np.random.rand(3, 4, 5),
       coords=[
           xr.cftime_range("2000-01-01", "2000-01-03", freq="D"),
           np.arange(4),
           np.arange(5),
       ],
       dims=["time", "lat", "lon"],
   )
   fct = obs.copy()
   fct.values = np.random.rand(3, 4, 5)

   ### Deterministic metrics
   # Pearson's correlation coefficient
   r = xs.pearson_r(obs, fct, "time")
   # >>> r
   # <xarray.DataArray (lat: 4, lon: 5)>
   # array([[ 0.395493, -0.979171,  0.998584, -0.758511,  0.583867],
   #       [ 0.456191,  0.992705,  0.999728, -0.209711,  0.984332],
   #       [-0.738775, -0.820627,  0.190332, -0.780365,  0.27864 ],
   #       [ 0.908445,  0.744518,  0.348995, -0.993572, -0.999234]])
   # Coordinates:
   #  * lat      (lat) int64 0 1 2 3
   #  * lon      (lon) int64 0 1 2 3 4

   # 2-tailed p-value of Pearson's correlation coefficient
   r_p_value = xs.pearson_r_p_value(obs, fct, "time")

   # R^2 (coefficient of determination) score
   r2 = xr.r2(obs, fct, "time")

   # Spearman's correlation coefficient
   rs = xs.spearman_r(obs, fct, "time")

   # 2-tailed p-value associated with Spearman's correlation coefficient
   rs_p_value = xs.spearman_r_p_value(obs, fct, "time")

   # Root Mean Squared Error
   rmse = xs.rmse(obs, fct, "time")

   # Mean Squared Error
   mse = xs.mse(obs, fct, "time")

   # Mean Absolute Error
   mae = xs.mae(obs, fct, "time")

   # Median Absolute Error
   median_absolute_error = xs.median_absolute_error(obs, fct, "time")

   # Mean Absolute Percentage Error
   mape = xs.mape(obs, fct, "time")

   # Symmetric Mean Absolute Percentage Error
   smape = xs.smape(obs, fct, "time")

   # You can also specify multiple axes for deterministic metrics:
   # Apply Pearson's correlation coefficient
   # over the latitude and longitude dimension
   r = xs.pearson_r(obs, fct, ["lat", "lon"])

   # You can weight over the dimensions the function is being applied
   # to by passing the argument ``weights=weight`` with a xr.DataArray
   # containing the dimension(s) being reduced.
   #
   # This is a common practice when working with observations and model
   # simulations of the Earth system. When working with rectilinear grids,
   # one can weight the data by the cosine of the latitude, which is maximum
   # at the equator and minimum at the poles (as in the below example). More
   # complicated model grids tend to be accompanied by a cell area coordinate,
   # which could also be passed into this function.
   obs2 = xr.DataArray(
       np.random.rand(3, 180, 360),
       coords=[
           xr.cftime_range("2000-01-01", "2000-01-03", freq="D"),
           np.linspace(-89.5, 89.5, 180),
           np.linspace(-179.5, 179.5, 360),
       ],
       dims=["time", "lat", "lon"],
    )
   fct2 = obs2.copy()
   fct2.values = np.random.rand(3, 180, 360)

   # make weights as cosine of the latitude and broadcast
   weights = np.cos(np.deg2rad(obs2.lat))
   _, weights = xr.broadcast(obs2, weights)
   # Remove the time dimension from weights
   weights = weights.isel(time=0)

   # Pearson's correlation coefficient with weights
   r_weighted = xs.pearson_r(obs2, fct2, ["lat", "lon"], weights=weights)
   # >>> r_weighted
   # <xarray.DataArray (time: 3)>
   # array([0.00601718, 0.00364946, 0.00213547])
   # Coordinates:
   # * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03
   r = xs.pearson_r(obs2, fct2, ["lat", "lon"])
   # >>> r
   # <xarray.DataArray (time: 3)>
   # array([ 5.02325347e-03, -6.75266864e-05, -3.00668282e-03])
   # Coordinates:
   # * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03

   # You can also pass the optional keyword `skipna=True`
   # to ignore any NaNs on the input data:
   obs_with_nans = obs.where(obs.lat > 1)
   fct_with_nans = fct.where(fct.lat > 1)
   mae_with_skipna = xs.mae(obs_with_nans, fct_with_nans, ['lat', 'lon'], skipna=True)
   # >>> mae_with_skipna
   # <xarray.DataArray (time: 3)>
   # array([0.29007757, 0.29660133, 0.38978561])
   # Coordinates:
   # * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03
   mae_with_no_skipna = xs.mae(obs_with_nans, fct_with_nans, ['lat', 'lon'])
   # >>> mae_with_no_skipna
   # <xarray.DataArray (time: 3)>
   # array([nan, nan, nan])
   # Coordinates:
   # * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03

   ### Probabilistic
   obs3 = xr.DataArray(
       np.random.rand(4, 5),
       coords=[np.arange(4), np.arange(5)],
       dims=["lat", "lon"]
   )
   fct3 = xr.DataArray(
       np.random.rand(3, 4, 5),
       coords=[np.arange(3), np.arange(4), np.arange(5)],
       dims=["member", "lat", "lon"],
   )

   # Continuous Ranked Probability Score with the ensemble distribution
   crps_ensemble = xs.crps_ensemble(obs3, fct3)

   # Continuous Ranked Probability Score with a Gaussian distribution
   crps_gaussian = xs.crps_gaussian(obs3, fct3.mean("member"), fct3.std("member"))

   # Continuous Ranked Probability Score with numerical integration
   # of the normal distribution
   crps_quadrature = xs.crps_quadrature(obs3, norm)

   # Brier scores of an ensemble for exceeding given thresholds
   threshold_brier_score = xs.threshold_brier_score(obs3, fct3, 0.7)

   # Brier score
   brier_score = xs.brier_score(obs3 > 0.5, (fct3 > 0.5).mean("member"))

   ### Contingency-based
   dichotomous_category_edges = np.array([0, 0.5, 1]) # "dichotomous" mean two-category
   dichotomous_contingency = xs.Contingency(obs, fct,
                                            dichotomous_category_edges,
                                            dichotomous_category_edges,
                                            dim=['lat','lon'])

   # Contingency table
   dichotomous_contingency_table = dichotomous_contingency.table

   # Bias score
   bias_score = dichotomous_contingency.bias_score()

   # Hit rate
   hit_rate = dichotomous_contingency.hit_rate()

   # False alarm ratio
   false_alarm_ratio = dichotomous_contingency.false_alarm_ratio()

   # False alarm rate
   false_alarm_rate = dichotomous_contingency.false_alarm_rate()

   # Success ratio
   success_ratio = dichotomous_contingency.success_ratio()

   # Threat score
   threat_score = dichotomous_contingency.threat_score()

   # Equitable threat score
   equit_threat_score = dichotomous_contingency.equit_threat_score()

   # Odds ratio
   odds_ratio = dichotomous_contingency.odds_ratio()

   # Odds ratio skill score
   odds_ratio_skill_score = dichotomous_contingency.odds_ratio_skill_score()

   multi_category_edges = np.array([0, 0.25, 0.75, 1])
   multicategory_contingency = xs.Contingency(obs, fct,
                                              multi_category_edges,
                                              multi_category_edges,
                                              dim=['lat','lon'])

   # Accuracy
   accuracy = multicategory_contingency.accuracy()

   # Heidke score
   heidke_score = multicategory_contingency.heidke_score()

   # Peirce score
   peirce_score = multicategory_contingency.peirce_score()

   # Gerrity score
   gerrity_score = multicategory_contingency.gerrity_score()

   # You can also use xskillscore as a method of your dataset:
   ds = xr.Dataset()
   ds["obs_var"] = obs
   ds["fct_var"] = fct

   # This is the equivalent of r = xs.pearson_r(obs, fct, 'time')
   r = ds.xs.pearson_r("obs_var", "fct_var", "time")

   # If fct is not a part of the dataset, inputting a separate
   # DataArray as an argument works as well:
   ds = ds.drop("fct_var")
   r = ds.xs.pearson_r("obs_var", fct, "time")

What other projects leverage xskillscore?
-----------------------------------------

- `esmlab <https://esmlab.readthedocs.io>`_: Tools for working with earth system multi-model analyses with xarray.
- A `Google Colab notebook <https://colab.research.google.com/drive/1wWHz_SMCHNuos5fxWRUJTcB6wqkTJQCR>`_
  by `Matteo De Felice <https://github.com/matteodefelice>`_.

History
-------

**xskillscore** was orginally developed to parallelize forecast metrics of the multi-model-multi-ensemble
forecasts associated with the `SubX <https://journals.ametsoc.org/doi/pdf/10.1175/BAMS-D-18-0270.1>`_ project.

We are indebted to the **xarray** community for their
`advice <https://groups.google.com/forum/#!searchin/xarray/xskillscore%7Csort:date/xarray/z8ue0G-BLc8/Cau-dY_ACAAJ>`_
in getting this package started.
