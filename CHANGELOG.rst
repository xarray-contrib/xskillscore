=================
Changelog History
=================

xskillscore v0.0.10 (2019-12-21)
================================

Deprecations
------------
- ``mad`` no longer works and is replaced by ``median_absolute_error``. `Riley X. Brady`_

Features
--------

Bug Fixes
---------
- ``skipna`` for ``pearson_r`` and ``spearman_r`` and their p-values now reports accurate results when there are pairwise nans (i.e., nans that occur in different indices in ``a`` and ``b``) `Riley X. Brady`_

Testing
-------
- Test that results from grid cells in a gridded product match the same value if their time series were input directly into functions. `Riley X. Brady`_
- Test that metric results from ``xskillscore`` are the same value as an external package (e.g. ``numpy``, ``scipy``, ``sklearn``). `Riley X. Brady`_
- Test that ``skipna=True`` works properly with pairwise nans. `Riley X. Brady`_

.. _`Riley X. Brady`: https://github.com/bradyrx
