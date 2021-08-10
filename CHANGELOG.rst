=================
Changelog History
=================

xskillscore v0.0.23 (2021-08-09)
--------------------------------

Bug Fixes
~~~~~~~~~
- :py:func:`~xskillscore.crps_ensemble` broadcasts
  (:issue:`345`, :pr:`346`) `Aaron Spring`_.

Internal Changes
~~~~~~~~~~~~~~~~
- :py:func:`~xskillscore.resampling.resample_iterations_idx` do not break when ``dim`` is
  not coordinate. (:issue:`303`, :pr:`339`) `Aaron Spring`_.
- Allow ``float`` or ``integer`` forecasts in :py:func:`~xskillscore.brier_score`
  (:issue:`285`, :pr:`341`) `Aaron Spring`_.


xskillscore v0.0.22 (2021-06-29)
--------------------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Created ``np_probabilistic.py`` (:pr:`333`) `Ray Bell`_.
- Require ``xhistogram>=0.3.0`` (:pr:`337`).


xskillscore v0.0.21 (2021-06-13)
--------------------------------

- Allow ``float`` or ``integer`` forecasts in :py:func:`~xskillscore.brier_score`
  (:issue:`285`, :pr:`342`) `Aaron Spring`_


Internal Changes
~~~~~~~~~~~~~~~~
- Added mypy to linting (:pr:`320`) `Zachary Blackwood`_.

Documentation
~~~~~~~~~~~~~
- Added more info in ``quick-start.ipynb`` (:pr:`316`) `Ray Bell`_.
- Created ``tabular-data.ipynb`` (:pr:`330`) `Ray Bell`_.

Breaking changes
~~~~~~~~~~~~~~~~
- Renamed `mae_test` to `halfwidth_ci_test` to make this comparative metric
  generic. Now, it accepts any of the distance metrics functions except for
  `mape`. The new function has an additional argument called `metric` which is a
  string and name of the target distance metric. (:pr:`319`) `Taher Chegini`_.


xskillscore v0.0.20 (2021-05-08)
--------------------------------

Features
~~~~~~~~
- Specify category distribution type with ``input_distributions`` in
  :py:func:`~xskillscore.rps` if ``category_edges==None`` that forecasts and
  observations are probability distributions ``p`` or cumulative
  distributionss ``c``. See :py:func:`~xskillscore.rps` docstrings and doctests for
  examples. (:pr:`300`) `Aaron Spring`_
- Added slope of linear fit :py:func:`~xskillscore.linslope`. (:pr:`307`) `Ray Bell`_

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``pytest-xdist`` and ``matplotlib-base`` in environments to speed up CI.
  (:pr:`283`) `Aaron Spring`_
- :py:func:`~xskillscore.rps` does not break from masking NaNs anymore.
  :py:func:`~xskillscore.rps` expilicty checks for ``bin_dim`` if
  ``category_edges==None``. (:pr:`287`) `Aaron Spring`_
- Add doctest on the docstring examples. (:pr:`302`) `Ray Bell`_
- Removed a call to compute weights in testing. (:pr:`306`) `Ray Bell`_
- Use built in ``xarray`` clip method. (:pr:`309`) `Ray Bell`_


xskillscore v0.0.19 (2021-03-12)
--------------------------------

Features
~~~~~~~~
- Added mean error
  :py:func:`~xskillscore.me`. (:issue:`202`, :pr:`200`)
  `Andrew Huang`_
- :py:func:`~xskillscore.brier_score` and :py:func:`~xskillscore.rps` now contain
  keyword ``fair`` to account for ensemble-size adjustments, but defaults to ``False``.
  :py:func:`~xskillscore.brier_score` also accepts binary or boolean forecasts when a
  ``member_dim`` dimension is present. (:issue:`162`, :pr:`211`) `Aaron Spring`_
- Added MAE significance test :py:func:`~xskillscore.mae_test` from Jolliffe and Ebert
  https://www.cawcr.gov.au/projects/verification/CIdiff/FAQ-CIdiff.html
  (:issue:`192`, :pr:`209`) `Aaron Spring`_
- :py:func:`~xskillscore.resampling.resample_iterations` and faster
  :py:func:`~xskillscore.resampling.resample_iterations_idx` for resampling with and
  without replacement. (:issue:`215`, :pr:`225`) `Aaron Spring`_
- Added receiver operating characteristic (ROC) :py:func:`~xskillscore.roc`.
  (:issue:`114`, :issue:`256`, :pr:`236`, :pr:`259`) `Aaron Spring`_
- Added many options for ``category_edges`` in :py:func:`~xskillscore.rps`, which
  allows multi-dimensional edges. :py:func:`~xskillscore.rps` now
  requires dimension ``member_dim`` in forecasts. (:issue:`275`, :pr:`277`)
  `Aaron Spring`_

Breaking changes
~~~~~~~~~~~~~~~~
- Aligned output of :py:func:`~xskillscore.sign_test` with
  :py:func:`~xskillscore.mae_test`. Now tests from comparative.py return more than
  one object including a boolean indicating ``signficance`` based on ``alpha``.
  (:pr:`209`) `Aaron Spring`_
- Drop support for python 3.6. (:issue:`237`, :pr:`276`) `Ray Bell`_

Bug Fixes
~~~~~~~~~
- :py:func:`~xskillscore.sign_test` now works for ``xr.Dataset`` inputs.
  (:issue:`198`, :pr:`199`) `Aaron Spring`_
- :py:func:`~xskillscore.threshold_brier_score` does not average over thresholds when
  ``dim==None``. Now also carries ``threshold`` as coordinate.
  (:issue:`255`, :pr:`211`) `Aaron Spring`_
- Passing weights no longer triggers eager computation.
  (:issue:`218`, :pr:`224`). `Andrew Huang`_
- :py:func:`~xskillscore.rps` not restricted to ``[0, 1]``.
  (:issue:`266`, :pr:`277`) `Aaron Spring`_

Internal Changes
~~~~~~~~~~~~~~~~
- Added Python 3.7 and Python 3.8 to the CI. Use the latest version of Python 3
  for development. (:issue:`21`, :pr:`189`) `Aaron Spring`_
- Lint with the latest black. (:issue:`179`, :pr:`191`) `Ray Bell`_
- Update mape algorithm from scikit-learn v0.24.0 and test against it.
  (:issue:`160`, :pr:`230`) `Ray Bell`_
- Pin ``numba`` to ``>=0.52`` to fix CI (:issue:`233`, :pr:`234`) `Ray Bell`_
- Refactor ``asv`` benchmarks. (:pr:`231`) `Aaron Spring`_
- Added tests for nans in correlation metrics (:issue:`246`, :pr:`247`) `Ray Bell`_
- Added tests for weighted metrics against scikit-learn (:pr:`257`) `Ray Bell`_
- Pin ``xhistogram`` to ``>=0.1.2`` and adjust code/documentation so that, as in
  np.histogram, right-most bin is right-edge inclusive where bins are specified
  (:pr:`269`) `Dougie Squire`_
- Reduce warnings. (:issue:`41`, :pr:`268`) `Aaron Spring`_
- Use ``raise_if_dask_computes`` from xarray. (:issue:`272`, :pr:`273`) `Ray Bell`_
- :py:func:`~xskillscore.threshold_brier_score` now carries threshold values as
  coordinates. (:pr:`279`) `Aaron Spring`_


xskillscore v0.0.18 (2020-09-23)
--------------------------------

Features
~~~~~~~~
- Added the sign test described in DelSole and Tippett 2016:
  :py:func:`~xskillscore.sign_test`. (:issue:`133`, :pr:`176`)
  `Aaron Spring`_ and `Dougie Squire`_

Internal Changes
~~~~~~~~~~~~~~~~
- Removed an unused variable in ``_rmse``, resulting in 2x speedup
  (:pr:`182`). `Andrew Huang`_
- Require ``xarray=0.16.1`` (:issue:`183`, :pr:`184`) `Aaron Spring`_

Bug Fixes
~~~~~~~~~
- Fix incompatibility with ``xarray=0.16.1`` in ``apply_ufunc``
  (:issue:`183`, :pr:`184`) `Aaron Spring`_

Documentation
~~~~~~~~~~~~~
- Added ``CONTRIBUTING.md`` to trigger built-in Github
  contribution guide reference (:pr:`181`) `mcsitter`_.


xskillscore v0.0.17 (2020-09-06)
--------------------------------

Features
~~~~~~~~
- Added contingency table :py:func:`~xskillscore.Contingency` and associated metrics
  (:pr:`119`, :pr:`153`). `Dougie Squire`_
- Added :py:func:`~xskillscore.rank_histogram` and :py:func:`~xskillscore.discrimination`
  to probabilistic metrics (:pr:`136`). `Dougie Squire`_
- Added :py:func:`~xskillscore.reliability` to probabilistic metrics (:pr:`164`). `Dougie Squire`_
- Added ``dim`` and ``weights`` kwargs for probabilistic metrics. (:pr:`121`) `Aaron Spring`_
- Added ``keep_attrs`` kwarg for all metrics. (:pr:`122`) `Andrew Huang`_
- Added ranked probability score :py:func:`~xskillscore.rps`. (:pr:`163`) `Aaron Spring`_
- Deterministic metrics now automatically broadcast any non-core dimensions. E.g., a single
  time series can be compared to a gridded product spanning that same time span.
  (:issue:`165`, :issue:`71`, :issue:`156`, :pr:`166`) `Aaron Spring`_

Breaking Changes
~~~~~~~~~~~~~~~~
- Renamed ``dim`` to ``member_dim`` in probabilistic metrics. (:pr:`121`) `Aaron Spring`_
- Argument ``dim`` becomes keyword ``dim=None`` in all metrics.
  (:issue:`137`, :pr:`143`) `Aaron Spring`_
- ``dim=None`` reduces all dimensions as in ``xr.mean(dim=None)``.
  (:issue:`137`, :pr:`143`) `Aaron Spring`_

Bug Fixes
~~~~~~~~~
- Fixes ``weights=None`` type issue with latest version of ``dask``.
  (:issue:`168`, :pr:`171`) `Andrew Huang`_

Documentation
~~~~~~~~~~~~~
- Added ``sphinx`` documentation with full API and a `quick start <quick-start.html>`__ notebook.
  (:pr:`127`) `Riley X. Brady`_ and `Ray Bell`_.

Internal Changes
~~~~~~~~~~~~~~~~
- Added ``utils`` module to house utilities shared across multiple modules
  (:pr:`119`). `Dougie Squire`_
- Added ``conftest.py`` to gather all ``pytest.fixtures``. (:issue:`126`, :pr:`159`).
  `Aaron Spring`_ and `Ray Bell`_
- Removed ``test_np_deterministic`` covered by ``test_metric_results_accurate``.
  (:pr:`159`) `Aaron Spring`_


xskillscore v0.0.16 (2020-07-18)
--------------------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add community support documents: ``HOWTOCONTRIBUTE.rst``, issue template and pull request
  template. `Aaron Spring`_ and `Ray Bell`_
- Replace ``pandas`` with ``cftime`` in examples and tests. `Aaron Spring`_ and `Ray Bell`_
- Add coveralls for tests coverage. `Aaron Spring`_ and `Ray Bell`_
- Add ``black``, ``flake8``, ``isort``, ``doc8`` and ``pre-commit`` for formatting
  similar to ``climpred``. `Aaron Spring`_ and `Ray Bell`_

Bug Fixes
~~~~~~~~~
- Avoid mutating inputted arrays when `skipna=True`. (:pr:`111`) `Riley X. Brady`_.
- Avoid read-only error that appeared due to not copying input arrays when dealing
  with NaNs. (:pr:`111`) `Riley X. Brady`_.


xskillscore v0.0.15 (2020-03-24)
--------------------------------

Features
~~~~~~~~
- Update the ``XSkillScoreAccessor`` with all metrics. `Ray Bell`_


xskillscore v0.0.14 (2020-03-20)
--------------------------------

Features
~~~~~~~~
- Add ``r2`` as an implementation of ``sklearn.metrics.r2_score``. `Ray Bell`_


xskillscore v0.0.13 (2020-03-17)
--------------------------------

Bug Fixes
~~~~~~~~~
- Fixes https://github.com/xarray-contrib/xskillscore/issues/79 `assignment destination is read-only`
  error when ``skipna=True`` and weights are passed. `Andrew Huang`_


xskillscore v0.0.12 (2020-01-09)
--------------------------------

Internal Changes
~~~~~~~~~~~~~~~~
- ~30-50% speedup for deterministic metrics when ``weights=None``. `Aaron Spring`_


xskillscore v0.0.11 (2020-01-06)
--------------------------------

Features
~~~~~~~~
- Add ``effective_sample_size``, ``pearson_r_eff_p_value``, and ``spearman_r_eff_p_value``
  for computing statistical significance for temporally correlated data with
  autocorrelation. `Riley X. Brady`_


xskillscore v0.0.10 (2019-12-21)
--------------------------------

Deprecations
~~~~~~~~~~~~
- ``mad`` no longer works and is replaced by ``median_absolute_error``. `Riley X. Brady`_


Bug Fixes
~~~~~~~~~
- ``skipna`` for ``pearson_r`` and ``spearman_r`` and their p-values now reports
  accurate results when there are pairwise nans (i.e., nans that occur in different
  indices in ``a`` and ``b``) `Riley X. Brady`_


Testing
~~~~~~~
- Test that results from grid cells in a gridded product match the same value if their time
  series were input directly into functions. `Riley X. Brady`_
- Test that metric results from ``xskillscore`` are the same value as an external package
  (e.g. ``numpy``, ``scipy``, ``sklearn``). `Riley X. Brady`_
- Test that ``skipna=True`` works properly with pairwise nans. `Riley X. Brady`_


.. _`Aaron Spring`: https://github.com/aaronspring
.. _`Andrew Huang`: https://github.com/ahuang11
.. _`Dougie Squire`: https://github.com/dougiesquire
.. _`mcsitter`: https://github.com/mcsitter
.. _`Riley X. Brady`: https://github.com/bradyrx
.. _`Ray Bell`: https://github.com/raybellwaves
.. _`Taher Chegini`: https://github.com/cheginit
.. _`Zachary Blackwood`: https://github.com/blackary
