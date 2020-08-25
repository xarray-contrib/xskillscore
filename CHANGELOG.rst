=================
Changelog History
=================

xskillscore v0.0.17 (2020-08-xx)
================================

Internal Changes
----------------
- Rename ``dim`` to ``member_dim`` in probabilistic metrics. (:pr:`121`) `Aaron Spring`_
- Added ``utils`` module to house utilities shared across multiple modules
  (:pr:`119`). `Dougie Squire`_

Documentation
-------------
- Added ``sphinx`` documentation with full API and a quick start notebook.
  (:pr:`127`) `Riley X. Brady`_ and `Ray Bell`_.

Features
--------
- Added contingency table and associated metrics (:pr:`119`, :pr:`153`). `Dougie Squire`_
- Added ``rank_histogram`` and ``discrimination`` to probabilistic metrics (:pr:`136`).
  `Dougie Squire`_
- Added ``dim`` and ``weights`` for probabilistic metrics. (:pr:`121`) `Aaron Spring`_
- Added ``keep_attrs`` for all metrics. (:pr:`122`) `Andrew Huang`_

Breaking Changes
----------------
- Rename `dim` to `member_dim` in probabilistic metrics. (:pr:`121`) `Aaron Spring`_


xskillscore v0.0.16 (2020-07-18)
================================

Internal Changes
----------------
- Add community support documents: ``HOWTOCONTRIBUTE.rst``, issue template and pull request
  template. `Aaron Spring`_ and `Ray Bell`_
- Replace ``pandas`` with ``cftime`` in examples and tests. `Aaron Spring`_ and `Ray Bell`_
- Add coveralls for tests coverage. `Aaron Spring`_ and `Ray Bell`_
- Add ``black``, ``flake8``, ``isort``, ``doc8`` and ``pre-commit`` for formatting
  similar to ``climpred``. `Aaron Spring`_ and `Ray Bell`_

Bug Fixes
---------
- Avoid mutating inputted arrays when `skipna=True`. (:pr:`111`) `Riley X. Brady`_.
- Avoid read-only error that appeared due to not copying input arrays when dealing
  with NaNs. (:pr:`111`) `Riley X. Brady`_.


xskillscore v0.0.15 (2020-03-24)
================================

Features
--------
- Update the ``XSkillScoreAccessor`` with all metrics. `Ray Bell`_


xskillscore v0.0.14 (2020-03-20)
================================

Features
--------
- Add ``r2`` as an implementation of ``sklearn.metrics.r2_score``. `Ray Bell`_


xskillscore v0.0.13 (2020-03-17)
================================

Bug Fixes
---------
- Fixes https://github.com/raybellwaves/xskillscore/issues/79 `assignment destination is read-only`
  error when ``skipna=True`` and weights are passed. `Andrew Huang`_


xskillscore v0.0.12 (2020-01-09)
================================

Internal Changes
----------------
- ~30-50% speedup for deterministic metrics when ``weights=None``. `Aaron Spring`_


xskillscore v0.0.11 (2020-01-06)
================================

Features
--------
- Add ``effective_sample_size``, ``pearson_r_eff_p_value``, and ``spearman_r_eff_p_value``
  for computing statistical significance for temporally correlated data with
  autocorrelation. `Riley X. Brady`_


xskillscore v0.0.10 (2019-12-21)
================================

Deprecations
------------
- ``mad`` no longer works and is replaced by ``median_absolute_error``. `Riley X. Brady`_


Bug Fixes
---------
- ``skipna`` for ``pearson_r`` and ``spearman_r`` and their p-values now reports
  accurate results when there are pairwise nans (i.e., nans that occur in different
  indices in ``a`` and ``b``) `Riley X. Brady`_


Testing
-------
- Test that results from grid cells in a gridded product match the same value if their time
  series were input directly into functions. `Riley X. Brady`_
- Test that metric results from ``xskillscore`` are the same value as an external package
  (e.g. ``numpy``, ``scipy``, ``sklearn``). `Riley X. Brady`_
- Test that ``skipna=True`` works properly with pairwise nans. `Riley X. Brady`_


.. _`Aaron Spring`: https://github.com/aaronspring
.. _`Andrew Huang`: https://github.com/ahuang11
.. _`Dougie Squire`: https://github.com/dougiesquire
.. _`Riley X. Brady`: https://github.com/bradyrx
.. _`Ray Bell`: https://github.com/raybellwaves
