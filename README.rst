xskillscore: Verification of forecasts
======================================

.. image:: https://travis-ci.org/raybellwaves/xskillscore.svg?branch=master
   :target: https://travis-ci.org/raybellwaves/xskillscore

**xskillscore** is an open source project and Python package that provides verification metrics of deterministic (and probabilistic) forecasts with xarray.

Installing
----------

``pip install git+https://github.com/raybellwaves/xskillscore``

Examples
--------

.. code-block:: python

   import xarray as xr
   import pandas as pd
   import numpy as np
   import xskillscore as xs

   obs = xr.DataArray(np.random.rand(len(3), len(4), len(5)),
                      coords=[pd.date_range('1/1/2000', '1/3/2000', freq='D'),
                              np.arange(4), np.arange(3]],
                      dims=['time', 'lat', 'lon'])
   fct = xr.DataArray(np.random.rand(len(3), len(4), len(5)),
                      coords=[pd.date_range('1/1/2000', '1/3/2000', freq='D'),
                              np.arange(4), np.arange(3]],         
                      dims=['time', 'lat', 'lon'])
                      
   r = xs.pearson_r(obs, fct, 'time')
   
   rmse = xs.rmse(obs, fct, 'time')
