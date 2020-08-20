=====================
Contribution Guide
=====================

Contributions are highly welcomed and appreciated.  Every little help counts,
so do not hesitate! You can make a high impact on ``xskillscore`` just by using it and
reporting `issues <https://github.com/raybellwaves/xskillscore/issues>`__.

The following sections cover some general guidelines
regarding development in ``xskillscore`` for maintainers and contributors.


Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.



.. contents:: Contribution links
   :depth: 2



.. _submitfeedback:

Feature requests and feedback
-----------------------------

We are eager to hear about your requests for new features and any suggestions about the
API, infrastructure, and so on. Feel free to submit these as
`issues <https://github.com/raybellwaves/xskillscore/issues/new>`__ with the label "feature request."

Please make sure to explain in detail how the feature should work and keep the scope as
narrow as possible. This will make it easier to implement in small PRs.


.. _reportbugs:

Report bugs
-----------

Report bugs for ``xskillscore`` in the `issue tracker <https://github.com/raybellwaves/xskillscore/issues>`_
with the label "bug".

If you are reporting a bug, please include:

* Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and ``xskillscore``
  version.
* Detailed steps to reproduce the bug, ideally a Minimal, Complete and Verifiable Example (http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports)

If you can write a demonstration test that currently fails but should passm
that is a very useful commit to make as well, even if you cannot fix the bug itself.


.. _fixbugs:

Fix bugs
--------

Look through the `GitHub issues for bugs <https://github.com/raybellwaves/xskillscore/labels/bug>`_.

Talk to developers to find out how you can fix specific bugs.


Write documentation
-------------------

``xskillscore`` could always use more documentation. What could you add?

* More complementary documentation. Have you perhaps found something unclear?
* Docstrings.
* Example notebooks of ``xskillscore`` being used in real analyses.

Our documentation is written in reStructuredText. You can follow our conventions in already written
documents. Some helpful guides are located
`here <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`__ and
`here <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`__.

.. note::
    Build the documentation locally with the following command:

    .. code:: bash

        $ conda env update -f ci/doc.yml
        $ conda activate xskillscore-docs
        $ cd docs
        $ make html

    The build documentation should be available in the ``docs/build/`` folder.

If you are adding new functions to the API, run ``sphinx-autogen -o api api.rst`` from the
``docs/source`` directory and add the functions to ``api.rst``.

Preparing Pull Requests
-----------------------


#. Fork the
   `xskillscore GitHub repository <https://github.com/raybellwaves/xskillscore>`__.  It's
   fine to use ``xskillscore`` as your fork repository name because it will live
   under your user.

#. Clone your fork locally using `git <https://git-scm.com/>`_, connect your repository
   to the upstream (main project), and create a branch::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/xskillscore.git
    $ cd xskillscore
    $ git remote add upstream git@github.com:raybellwaves/xskillscore.git

    # now, to fix a bug or add feature create your own branch off "master":

    $ git checkout -b your-bugfix-feature-branch-name master

   If you need some help with Git, follow this quick start
   guide: https://git.wiki.kernel.org/index.php/QuickStart

#. Install dependencies into a new conda environment::

    $ conda env update -f ci/requirements-py36.yml
    $ conda activate xskillscore-dev

#. Make an editable install of xskillscore by running::

    $ pip install --no-deps -e .

#. Run `pre-commit <https://pre-commit.com>`_::

     $ pre-commit run --all-files

   https://pre-commit.com/ is a framework for managing and maintaining multi-language pre-commit
   hooks to ensure code-style and code formatting is consistent.

#. Break your edits up into reasonably sized commits::

    $ git commit -a -m "<commit message>"
    $ git push -u

#. Run all the tests

   Now running tests is as simple as issuing this command::

    $ pytest xskillscore

   Check that your contribution is covered by tests and therefore increases the overall test coverage::

    $ coverage run --source xskillscore -m py.test
    $ coverage report
    $ coveralls

  Please stick to `xarray <http://xarray.pydata.org/en/stable/contributing.html>`_'s testing recommendations.

#. Running the performance test suite

Performance matters and it is worth considering whether your code has introduced
performance regressions. `xskillscore` is starting to write a suite of benchmarking tests
using `asv <https://asv.readthedocs.io/en/stable/>`_
to enable easy monitoring of the performance of critical `xskillscore` operations.
These benchmarks are all found in the ``asv_bench`` directory.

If you need to run a benchmark, change your directory to ``asv_bench/`` and run::

    $ asv continuous -f 1.1 upstream/master HEAD

You can replace ``HEAD`` with the name of the branch you are working on,
and report benchmarks that changed by more than 10%.
The command uses ``conda`` by default for creating the benchmark
environments.

Running the full benchmark suite can take up to half an hour and use up a few GBs of
RAM. Usually it is sufficient to paste only a subset of the results into the pull
request to show that the committed changes do not cause unexpected performance
regressions.  You can run specific benchmarks using the ``-b`` flag, which
takes a regular expression.  For example, this will only run tests from a
``asv_bench/benchmarks/deterministic.py`` file::

    $ asv continuous -f 1.1 upstream/master HEAD -b ^deterministic

If you want to only run a specific group of tests from a file, you can do it
using ``.`` as a separator. For example::

    $ asv continuous -f 1.1 upstream/master HEAD -b deterministic.Compute_small.time_xskillscore_metric_small

will only run the ``time_xskillscore_metric_small`` benchmark of class ``Compute_small``
defined in ``deterministic.py``.

#. Create a new changelog entry in ``CHANGELOG.rst``:

   - The entry should be entered as:

    <description> (``:pr:`#<pull request number>```) ```<author's names>`_``

    where ``<description>`` is the description of the PR related to the change and
    ``<pull request number>`` is the pull request number and ``<author's names>`` are your first
    and last names.

   - Add yourself to list of authors at the end of ``CHANGELOG.rst`` file if not there yet, in
     alphabetical order.

#. Add yourself to the contributors list via ``docs/source/contributors.rst``.

#. Finally, submit a pull request through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/xskillscore
    compare: your-branch-name

    base-fork: raybellwaves/xskillscore
    base: master

Note that you can create the Pull Request while you're working on this. The PR will update
as you add more commits. ``xskillscore`` developers and contributors can then review your code
and offer suggestions.
