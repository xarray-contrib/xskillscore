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
