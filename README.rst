xskillscore: Metrics for verifying forecasts
============================================

.. image:: https://travis-ci.org/raybellwaves/xskillscore.svg?branch=master
   :target: https://travis-ci.org/raybellwaves/xskillscore

**xskillscore** is an open source project and Python package that provides verification metrics of deterministic (and probabilistic from `properscoring`) forecasts with `xarray`.

Installing
----------

``pip install xskillscore``

or

``pip install git+https://github.com/raybellwaves/xskillscore``

Examples
--------

.. code-block:: python

   import xarray as xr
   import pandas as pd
   import numpy as np
   import xskillscore as xs


   obs = xr.DataArray(np.random.rand(3, 4, 5),
                      coords=[pd.date_range('1/1/2000', '1/3/2000', freq='D'),
                              np.arange(4), np.arange(5)],
                      dims=['time', 'lat', 'lon'])
   fct = xr.DataArray(np.random.rand(3, 4, 5),
                      coords=[pd.date_range('1/1/2000', '1/3/2000', freq='D'),
                              np.arange(4), np.arange(5)],
                      dims=['time', 'lat', 'lon'])

   # deterministic
   r = xs.pearson_r(obs, fct, 'time')
   #>>> r
   #<xarray.DataArray (lat: 4, lon: 5)>
   #array([[ 0.395493, -0.979171,  0.998584, -0.758511,  0.583867],
   #       [ 0.456191,  0.992705,  0.999728, -0.209711,  0.984332],
   #       [-0.738775, -0.820627,  0.190332, -0.780365,  0.27864 ],
   #       [ 0.908445,  0.744518,  0.348995, -0.993572, -0.999234]])
   #Coordinates:
   #  * lat      (lat) int64 0 1 2 3
   #  * lon      (lon) int64 0 1 2 3 4

   r_p_value = xs.pearson_r_p_value(obs, fct, 'time')

   rmse = xs.rmse(obs, fct, 'time')

   mse = xs.mse(obs, fct, 'time')

   mae = xs.mae(obs, fct, 'time') 
   
   # You can also specify multiple axes for deterministic metrics
   r = xs.pearson_r(obs, fct, ['lat', 'lon'])

   # probabilistic
   crps_ensemble = xs.crps_ensemble(obs, fct)

   crps_gaussian = xs.crps_gaussian(obs, fct.mean('time'), fct.std('time'))

   threshold_brier_score = xs.threshold_brier_score(obs, fct, 0.7)
