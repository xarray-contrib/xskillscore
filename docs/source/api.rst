API Reference
=============

This page provides an auto-generated summary of ``xskillscore``'s API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

Deterministic Metrics
---------------------

.. currentmodule:: xskillscore.core.deterministic


Correlation Metrics
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    pearson_r
    pearson_r_p_value
    pearson_r_eff_p_value
    spearman_r
    spearman_r_p_value
    spearman_r_eff_p_value
    effective_sample_size
    r2

Distance Metrics
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    rmse
    mse
    mae
    median_absolute_error
    smape
    mape


Probabilistic Metrics
---------------------

.. currentmodule:: xskillscore.core.probabilistic

Currently, our probabilistic metrics are ported over from
`properscoring <https://github.com/TheClimateCorporation/properscoring>`__ to work with
``xarray`` DataArrays and Datasets.

.. autosummary::
    :toctree: api/

    brier_score
    crps_ensemble
    crps_gaussian
    crps_quadrature
    threshold_brier_score

Dichotomous Metrics
-------------------

.. currentmodule:: xskillscore

These metrics rely upon the construction of a ``Contingency`` object for dichotomous (yes/no)
forecasts. The user calls the individual methods to access metrics based on the table.

.. autosummary::
    :toctree: api/

    Contingency

Dichotomous-Only Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Contingency.hits
    Contingency.misses
    Contingency.false_alarms
    Contingency.correct_negatives
    Contingency.bias_score
    Contingency.hit_rate
    Contingency.false_alarm_ratio
    Contingency.false_alarm_rate
    Contingency.success_ratio
    Contingency.threat_score
    Contingency.equit_threat_score
    Contingency.odds_ratio
    Contingency.odds_ratio_skill_score

Multi-Category Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Contingency.accuracy
    Contingency.heidke_score
    Contingency.peirce_score
    Contingency.gerrity_score
