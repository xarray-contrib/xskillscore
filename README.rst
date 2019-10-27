xskillscore: Metrics for verifying forecasts
============================================

.. image:: https://travis-ci.org/raybellwaves/xskillscore.svg?branch=master
   :target: https://travis-ci.org/raybellwaves/xskillscore
.. image:: https://img.shields.io/pypi/v/xskillscore.svg
   :target: https://pypi.python.org/pypi/xskillscore/
.. image:: https://anaconda.org/conda-forge/xskillscore/badges/version.svg
   :target: https://anaconda.org/conda-forge/xskillscore/
.. image:: https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
  :target: https://raybellwaves.github.io/xskillscore/


**xskillscore** is an open source project and Python package that provides verification metrics of deterministic (and probabilistic from `properscoring`) forecasts with `xarray`.

Installing
----------

``$ conda install -c conda-forge xskillscore``

or

``$ pip install xskillscore``

or

``$ pip install git+https://github.com/raybellwaves/xskillscore``

Examples
--------

.. code-block:: python

   import xarray as xr
   import pandas as pd
   import numpy as np
   from scipy.stats import norm
   import xskillscore as xs


   obs = xr.DataArray(
       np.random.rand(3, 4, 5),
       coords=[
           pd.date_range("1/1/2000", "1/3/2000", freq="D"),
           np.arange(4),
           np.arange(5),
       ],
       dims=["time", "lat", "lon"],
   )
   fct = xr.DataArray(
       np.random.rand(3, 4, 5),
       coords=[
           pd.date_range("1/1/2000", "1/3/2000", freq="D"),
           np.arange(4),
           np.arange(5),
       ],
       dims=["time", "lat", "lon"],
   )

   # deterministic
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

   r_p_value = xs.pearson_r_p_value(obs, fct, "time")

   rs = xs.spearman_r(obs, fct, "time")

   rs_p_value = xs.spearman_r_p_value(obs, fct, "time")

   rmse = xs.rmse(obs, fct, "time")

   mse = xs.mse(obs, fct, "time")

   mae = xs.mae(obs, fct, "time")

   mad = xs.mad(obs, fct, "time")

   mape = xs.mape(obs, fct, "time")

   smape = xs.smape(obs, fct, "time")

   # You can also specify multiple axes for deterministic metrics:
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
   dims = ('lat', 'lon')
   base_data = np.ones((30, 180, 360))
   a = xr.DataArray(
       base_data + np.random.rand(30, 180, 360),
       dims=['time', 'lat', 'lon']
   )
   b = xr.DataArray(
       base_data + np.random.rand(30, 180, 360),
       dims=['time', 'lat', 'lon']
   )
   x = np.linspace(-179.5, 179.5, 360)
   y = np.linspace(-89.5, 89.5, 180)
   lon, lat = np.meshgrid(x, y)
   a['latitude'] = (dims, lat)
   a['longitude'] = (dims, lon)
   b['latitude'] = (dims, lat)
   b['longitude'] = (dims, lon)

   # make weights
   weights = np.cos(np.deg2rad(a.latitude))
   a, weights = xr.broadcast(a, weights)
   weights = weights.isel(time=0) # remove time from weights

   # example
   weighted = xs.pearson_r(a, b, dims, weights=weights)
   non_weighted = xs.pearson_r(a, b, dims, weights=None)

   # You can also pass the optional keyword `skipna=True` to ignore any NaNs on the
   # input data. This is useful in the case that you are computing these functions
   # over space and have a mask applied to the grid or have NaNs over land.

   skipna_res = xs.mae(obs.where(obs.lat > 1), fct.where(fct.lat > 1), ['lat', 'lon'], skipna=True)
   # >>> skipna_res
   # <xarray.DataArray (time: 3)>
   # array([0.29007757, 0.29660133, 0.38978561])
   # Coordinates:
   # * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03

   no_skipna_res = xs.mae(obs.where(obs.lat > 1), fct.where(fct.lat > 1), ['lat', 'lon'], skipna=False)
   # >>> no_skipna_res
   # <xarray.DataArray (time: 3)>
   # array([nan, nan, nan])
   # Coordinates:
   # * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03

   # probabilistic
   obs = xr.DataArray(
       np.random.rand(4, 5),
       coords=[np.arange(4), np.arange(5)],
       dims=["lat", "lon"]
   )
   fct = xr.DataArray(
       np.random.rand(3, 4, 5),
       coords=[np.arange(3), np.arange(4), np.arange(5)],
       dims=["member", "lat", "lon"],
   )

   crps_ensemble = xs.crps_ensemble(obs, fct)

   crps_gaussian = xs.crps_gaussian(obs, fct.mean("member"), fct.std("member"))

   crps_quadrature = xs.crps_quadrature(obs, norm)

   threshold_brier_score = xs.threshold_brier_score(obs, fct, 0.7)

   brier_score = xs.brier_score(obs > 0.5, (fct > 0.5).mean("member"))


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

What projects leverage xskillscore?
-----------------------------------

- `climpred <https://climpred.readthedocs.io>`_: An xarray wrapper for analysis of ensemble forecast models for climate prediction.
- `esmlab <https://esmlab.readthedocs.io>`_: Tools for working with earth system multi-model analyses with xarray.
- A `Google Colab notebook <https://colab.research.google.com/drive/1wWHz_SMCHNuos5fxWRUJTcB6wqkTJQCR>`_ by `Matteo De Felice <https://github.com/matteodefelice>`_.

History
-------

**xskillscore** was orginally developed to parallelize forecast metrics of the multi-model-multi-ensemble forecasts associated with the `SubX <https://journals.ametsoc.org/doi/pdf/10.1175/BAMS-D-18-0270.1>`_ project. We are indebted to the **xarray** community for their `advice <https://groups.google.com/forum/#!searchin/xarray/xskillscore%7Csort:date/xarray/z8ue0G-BLc8/Cau-dY_ACAAJ>`_ in getting this package started.
