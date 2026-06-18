xskillscore: Metrics for verifying forecasts
============================================

+---------------------------+-------------------------------------------+
| Documentation and Support | |docs| |context7| |binder|                |
+---------------------------+-------------------------------------------+
| Open Source               | |pypi| |conda-forge| |conda-downloads|    |
|                           | |license| |zenodo|                        |
+---------------------------+-------------------------------------------+
| Coding Standards          | |codecov| |pre-commit|                    |
+---------------------------+-------------------------------------------+
| Development Status        | |status| |testing| |upstream|             |
+---------------------------+-------------------------------------------+

Installation
============

You can install the latest release of ``xskillscore`` using ``pip`` or ``conda``:

.. code-block:: bash

    pip install xskillscore

.. code-block:: bash

    conda install -c conda-forge xskillscore

You can also install the bleeding edge (pre-release versions) by running:

.. code-block:: bash

    pip install git+https://github.com/xarray-contrib/xskillscore@main --upgrade

**Getting Started**

* :doc:`quick-start`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    quick-start.ipynb
    tabular-data.ipynb

**Help & Reference**

* :doc:`api`
* :doc:`contributing`
* :doc:`changelog`
* :doc:`release_procedure`
* :doc:`related-projects`
* :doc:`contributors`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & Reference

   api
   contributing
   changelog
   release_procedure
   related-projects
   contributors

.. |binder| image:: https://mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/raybellwaves/xskillscore-tutorial/master?urlpath=lab
        :alt: Binder

.. |codecov| image:: https://codecov.io/gh/xarray-contrib/xskillscore/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/xarray-contrib/xskillscore
        :alt: Codecov

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/xskillscore.svg
        :target: https://anaconda.org/conda-forge/xskillscore
        :alt: conda-forge

.. |conda-downloads| image:: https://img.shields.io/conda/dn/conda-forge/xskillscore.svg
        :target: https://anaconda.org/conda-forge/xskillscore
        :alt: conda-forge downloads

.. |context7| image:: https://img.shields.io/badge/Context7-Docs-6366f1?logo=readthedocs&logoColor=white
        :target: https://context7.com/xarray-contrib/xskillscore
        :alt: Context7 Documentation

.. |docs| image:: https://img.shields.io/readthedocs/xskillscore/stable.svg?style=flat
        :target: https://xskillscore.readthedocs.io/en/stable/?badge=stable
        :alt: Documentation Status

.. |license| image:: https://img.shields.io/github/license/xarray-contrib/xncml.svg
        :target: https://github.com/xarray-contrib/xncml/blob/main/LICENSE
        :alt: License

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/xarray-contrib/xskillscore/main.svg
        :target: https://results.pre-commit.ci/latest/github/xarray-contrib/xskillscore/main
        :alt: Pre-Commit

.. |pypi| image:: https://img.shields.io/pypi/v/xskillscore.svg
        :target: https://pypi.python.org/pypi/xskillscore/
        :alt: PyPI

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |testing| image:: https://github.com/xarray-contrib/xskillscore/actions/workflows/xskillscore_testing.yml/badge.svg
        :target: https://github.com/xarray-contrib/xskillscore/actions/workflows/xskillscore_testing.yml
        :alt: Testing

.. |upstream| image:: https://github.com/xarray-contrib/xskillscore/actions/workflows/upstream-dev-ci.yml/badge.svg
        :target: https://github.com/xarray-contrib/xskillscore/actions/workflows/upstream-dev-ci.yml
        :alt: Upstream Testing

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5173152.svg
        :target: https://doi.org/10.5281/zenodo.5173152
        :alt: Zenodo DOI
