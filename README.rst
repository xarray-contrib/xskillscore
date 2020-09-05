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
.. image:: https://img.shields.io/readthedocs/xskillscore/stable.svg?style=flat
   :target: https://xskillscore.readthedocs.io/en/stable/?badge=stable
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

Documentation
-------------
Documentation can be found on `readthedocs <https://xskillscore.readthedocs.io/en/latest/>`_.

See also
--------

- If you are interested in using **xskillscore** for data science where you data is mostly in
  ``pandas.DataFrames``'s check out the `xskillscore-tutorial <https://github.com/raybellwaves/xskillscore-tutorial>`_
- If you are interested in using **xskillscore** for climate prediction check out
  `climpred <https://climpred.readthedocs.io/en/stable/>`_.

History
-------

**xskillscore** was orginally developed to parallelize forecast metrics of the multi-model-multi-ensemble
forecasts associated with the `SubX <https://journals.ametsoc.org/doi/pdf/10.1175/BAMS-D-18-0270.1>`_ project.

We are indebted to the **xarray** community for their
`advice <https://groups.google.com/forum/#!searchin/xarray/xskillscore%7Csort:date/xarray/z8ue0G-BLc8/Cau-dY_ACAAJ>`_
in getting this package started.
