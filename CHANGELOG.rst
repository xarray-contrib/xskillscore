=================
Changelog History
=================

xskillscore v0.0.12 (2020-01-xx)
================================

Internal Changes
----------------
- ~30-50% speedup for deterministic metrics when `weights=None`. `Aaron Spring`_

xskillscore v0.0.11 (2020-01-06)
================================

Features
--------
- Add ``effective_sample_size``, ``pearson_r_eff_p_value``, and ``spearman_r_eff_p_value`` for computing statistical significance for temporally correlated data with autocorrelation. `Riley X. Brady`_

xskillscore v0.0.10 (2019-12-21)
================================

Deprecations
------------
- ``mad`` no longer works and is replaced by ``median_absolute_error``. `Riley X. Brady`_

Bug Fixes
---------
- ``skipna`` for ``pearson_r`` and ``spearman_r`` and their p-values now reports accurate results when there are pairwise nans (i.e., nans that occur in different indices in ``a`` and ``b``) `Riley X. Brady`_

Testing
-------
- Test that results from grid cells in a gridded product match the same value if their time series were input directly into functions. `Riley X. Brady`_
- Test that metric results from ``xskillscore`` are the same value as an external package (e.g. ``numpy``, ``scipy``, ``sklearn``). `Riley X. Brady`_
- Test that ``skipna=True`` works properly with pairwise nans. `Riley X. Brady`_

.. _`Riley X. Brady`: https://github.com/bradyrx
.. _`Aaron Spring`: https://github.com/aaronspring
