API Reference
=============

This page provides an auto-generated summary of ``xskillscore``'s API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

.. currentmodule:: xskillscore

Deterministic Metrics
---------------------

Correlation Metrics
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    effective_sample_size
    pearson_r
    pearson_r_p_value
    pearson_r_eff_p_value
    linslope
    spearman_r
    spearman_r_p_value
    spearman_r_eff_p_value


Distance Metrics
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    mae
    mape
    me
    median_absolute_error
    mse
    r2
    rmse
    smape


Probabilistic Metrics
---------------------

Currently, most of our probabilistic metrics are ported over from
`properscoring <https://github.com/TheClimateCorporation/properscoring>`__ to work with
``xarray`` DataArrays and Datasets.

.. autosummary::
    :toctree: api/

    brier_score
    crps_ensemble
    crps_gaussian
    crps_quadrature
    discrimination
    rank_histogram
    reliability
    roc
    rps
    threshold_brier_score


Contingency-based Metrics
-------------------------

These metrics rely upon the construction of a ``Contingency`` object. The user calls the
individual methods to access metrics based on the table.

.. autosummary::
    :toctree: api/

    Contingency

Contingency table
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Contingency.table

Dichotomous-Only (yes/no) Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Contingency.bias_score
    Contingency.correct_negatives
    Contingency.equit_threat_score
    Contingency.false_alarm_rate
    Contingency.false_alarm_ratio
    Contingency.false_alarms
    Contingency.hit_rate
    Contingency.hits
    Contingency.misses
    Contingency.odds_ratio
    Contingency.odds_ratio_skill_score
    Contingency.success_ratio
    Contingency.threat_score


Multi-Category Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Contingency.accuracy
    Contingency.gerrity_score
    Contingency.heidke_score
    Contingency.peirce_score
    roc


Comparative
-----------

Tests to compare whether one forecast is significantly better than another one.

.. autosummary::
    :toctree: api/

    halfwidth_ci_test
    sign_test


Resampling
----------

Functions for resampling from a dataset with or without replacement that create a new
``iteration`` dimension.

.. currentmodule:: xskillscore.core.resampling

.. autosummary::
    :toctree: api/

    resample_iterations
    resample_iterations_idx
